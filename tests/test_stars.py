"""Star ratings and the star-range filter: rating, caching, scanning, training."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from maniabot import beatmaps
from maniabot.beatmaps import StarCache, check_beatmap, estimate_stars, extract_osz
from maniabot.training import TrainingSpec, run_training, validate_spec
from osu_parser import Beatmap, Note, parse_osu
from tests.conftest import DEMO_MAPS, tiny_spec


def _bm(times: list[int], md5: str = "") -> Beatmap:
    return Beatmap(title="t", version="v", keys=4, od=8.0, md5=md5,
                   notes=[Note(time=t, lane=i % 4, end_time=t) for i, t in enumerate(times)])


def test_denser_maps_estimate_higher():
    sparse = _bm(list(range(0, 60_000, 500)))   # 2 notes/s
    dense = _bm(list(range(0, 60_000, 100)))    # 10 notes/s
    assert 0 < estimate_stars(sparse) < estimate_stars(dense)


def test_real_rating_for_demo_map_is_exact_and_cached(tmp_path):
    cache = StarCache(tmp_path / "stars.json")
    path = DEMO_MAPS / "demo_000.osu"
    bm = parse_osu(path)
    stars, exact = beatmaps.rate_map(path, bm, cache)
    assert exact and stars > 0
    cache.flush()
    again = StarCache(tmp_path / "stars.json")
    assert again.get(bm.md5) == pytest.approx(stars)


def test_cache_is_discarded_when_the_rating_engine_changes(tmp_path):
    f = tmp_path / "stars.json"
    f.write_text(json.dumps({"engine": "rosu-pp-py 0.0.0", "stars": {"abc": 9.9}}))
    assert StarCache(f).get("abc") is None


def test_falls_back_to_an_estimate_when_the_library_cannot_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(beatmaps, "_rosu_stars", lambda _p: None)
    path = DEMO_MAPS / "demo_000.osu"
    stars, exact = beatmaps.rate_map(path, parse_osu(path), StarCache(tmp_path / "s.json"))
    assert not exact and stars > 0


def test_scan_rates_every_valid_map_and_filters_inclusively(data_root):
    scan = beatmaps.scan_maps(DEMO_MAPS)
    assert scan.valid and all(c.stars is not None for c in scan.valid)
    lo_b, hi_b = scan.star_bounds()
    assert len(scan.in_star_range()) == len(scan.valid)                      # no limits = everything
    assert len(scan.in_star_range(lo_b, hi_b)) == len(scan.valid)            # bounds are inclusive
    assert scan.in_star_range(hi_b + 1, None) == []
    cut = sorted(c.stars for c in scan.valid)[len(scan.valid) // 2]
    below = scan.in_star_range(None, cut)
    above = scan.in_star_range(cut, None)
    assert all(c.stars <= cut for c in below) and all(c.stars >= cut for c in above)
    assert {c.path for c in below} | {c.path for c in above} == {c.path for c in scan.valid}


def test_rejected_maps_have_no_rating(tmp_path):
    seven_k = tmp_path / "7k.osu"
    seven_k.write_text("osu file format v14\n[General]\nMode: 3\n[Difficulty]\nCircleSize:7\n[HitObjects]\n")
    c = check_beatmap(seven_k)
    assert not c.ok and c.stars is None


def test_validate_spec_rejects_bad_star_ranges(small_maps):
    ok = tiny_spec("m", small_maps, min_stars=1.0, max_stars=3.0)
    assert validate_spec(ok) == []
    assert any("higher than the maximum" in e for e in validate_spec(tiny_spec("m", small_maps, min_stars=3, max_stars=1)))
    assert any("Minimum" in e for e in validate_spec(tiny_spec("m", small_maps, min_stars=-1)))
    assert any("Maximum" in e for e in validate_spec(tiny_spec("m", small_maps, max_stars=float("nan"))))


def test_old_specs_without_star_fields_still_load():
    spec = TrainingSpec.from_dict({"model_id": "m", "timesteps": 2000})
    assert spec.min_stars is None and spec.max_stars is None


def test_training_uses_only_maps_in_range(store, tmp_path):
    folder = tmp_path / "all demo maps"
    folder.mkdir()
    for f in sorted(DEMO_MAPS.glob("*.osu")):
        (folder / f.name).write_bytes(f.read_bytes())
    scan = beatmaps.scan_maps(folder)
    ordered = sorted(c.stars for c in scan.valid)
    lo, hi = ordered[3], ordered[8]
    expected = {Path(c.path).name for c in scan.in_star_range(lo, hi)}
    assert 3 <= len(expected) < len(scan.valid)

    rec = store.create_model("Stars Bot")
    s = run_training(store, tiny_spec(rec.id, folder, min_stars=lo, max_stars=hi, val_count=1))
    assert s["status"] == "completed", s.get("error")
    meta = store.get_version(rec.id, "v001").meta
    used = {Path(m["path"]).name for m in meta["training_maps"] + meta["validation_maps"]}
    assert used == expected                                  # validation maps obey the range too
    assert meta["star_range"] == {"min": lo, "max": hi, "maps_in_range": len(expected),
                                  "maps_in_folder": len(scan.valid)}
    assert all(lo <= m["stars"] <= hi for m in meta["training_maps"] + meta["validation_maps"])


def test_training_fails_cleanly_when_no_map_is_in_range(store, small_maps):
    rec = store.create_model("Nothing In Range")
    s = run_training(store, tiny_spec(rec.id, small_maps, min_stars=90.0))
    assert s["status"] == "failed" and "★" in s["error"] and "3 usable map" in s["error"]
    assert store.list_versions(rec.id) == []


def test_extracts_mapsets_whose_paths_exceed_the_windows_limit(tmp_path):
    """A very long file name inside a deep cache folder used to make the whole mapset unreadable."""
    osu_bytes = (DEMO_MAPS / "demo_000.osu").read_bytes()
    long_name = "Some Artist - " + "A Very Long Song Title " * 7 + "[4K Insane].osu"
    osz = tmp_path / "long.osz"
    with zipfile.ZipFile(osz, "w") as z:
        z.writestr(long_name, osu_bytes)
    cache = tmp_path / ("deep_" * 12) / "cache" / "mapsets"
    files = extract_osz(osz, cache)
    assert len(files) == 1 and files[0].name == long_name
    assert check_beatmap(files[0]).ok
    assert extract_osz(osz, cache) == files          # reused on the next scan
