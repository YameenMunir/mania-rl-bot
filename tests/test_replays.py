"""4K validation, inference, evaluation metrics, replay generation and history."""
from __future__ import annotations

import lzma
import struct

import pytest

from maniabot.beatmaps import check_beatmap, scan_maps, split_maps
from maniabot.evaluation import POLICIES, compare, evaluate, play_map
from maniabot.models import ModelError
from maniabot.replays import ReplayHistory, generate_replay, model_status
from osu_parser import parse_osu
from tests.conftest import DEMO_MAPS

SEVEN_K = """osu file format v14

[General]
AudioFilename: audio.mp3
Mode: 3

[Difficulty]
CircleSize:7
OverallDifficulty:8

[HitObjects]
""" + "".join(f"{36 + 73 * (i % 7)},192,{1000 + i * 100},1,0,0:0:0:0:\n" for i in range(80))


def test_4k_validation(tmp_path):
    ok = check_beatmap(DEMO_MAPS / "demo_000.osu")
    assert ok.ok and ok.beatmap.keys == 4
    assert any("Audio" in w for w in ok.warnings)  # demo .osu has no audio next to it
    seven = tmp_path / "seven key.osu"
    seven.write_text(SEVEN_K)
    bad = check_beatmap(seven)
    assert not bad.ok and "7K" in bad.reason
    std = tmp_path / "std.osu"
    std.write_text(SEVEN_K.replace("Mode: 3", "Mode: 0"))
    assert "Not an osu!mania" in check_beatmap(std).reason
    assert "not found" in check_beatmap(tmp_path / "missing.osu").reason.lower()
    corrupt = tmp_path / "corrupt.osu"
    corrupt.write_text("osu file format v14\n[General]\nMode: 3\n[Difficulty]\nCircleSize:4\n[HitObjects]\nx,y,z,w,v\n")
    assert "corrupt" in check_beatmap(corrupt).reason


def test_scan_reports_rejections(tmp_path):
    (tmp_path / "a.osu").write_bytes((DEMO_MAPS / "demo_000.osu").read_bytes())
    (tmp_path / "b.osu").write_text(SEVEN_K)
    res = scan_maps(tmp_path)
    assert len(res.valid) == 1 and len(res.rejected) == 1
    train, val = split_maps(res.beatmaps * 3, 1, seed=0)
    assert len(train) == 2 and len(val) == 1


def test_lanes_and_holds_parsed():
    bm = parse_osu(DEMO_MAPS / "demo_003.osu")
    assert {n.lane for n in bm.notes} == {0, 1, 2, 3}
    assert any(n.is_hold for n in bm.notes)


def test_inference_and_metrics(store, trained):
    v = store.get_version(trained, "v001")
    model = POLICIES.load(v.weights)
    assert POLICIES.load(v.weights) is model  # cached, not reloaded
    bm = parse_osu(DEMO_MAPS / "demo_003.osu")
    res = play_map(model, bm, v.meta["env_config"], seed=1, record=True)
    m = res.metrics
    assert m["total_notes"] == len(bm.notes)
    assert sum(m["counts"]) == m["total_notes"]
    assert len(m["lanes"]) == 4 and sum(l["notes"] for l in m["lanes"]) == m["total_notes"]
    assert m["holds_total"] == sum(1 for n in bm.notes if n.is_hold)
    assert res.events and all(lane in range(4) for _t, lane, _d in res.events)
    summary = evaluate(model, [bm, parse_osu(DEMO_MAPS / "demo_004.osu")], v.meta["env_config"])
    assert summary["n_maps"] == 2 and "_deltas" not in summary
    cmp = compare(summary, {**summary, "misses": summary["misses"] + 1})
    row = next(r for r in cmp.rows if r["key"] == "misses")
    assert row["winner"] == "a"


def _read_string(data: bytes, i: int):
    if data[i] == 0:
        return "", i + 1
    assert data[i] == 0x0B
    i += 1
    n, shift = 0, 0
    while True:
        b = data[i]
        i += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            break
    return data[i:i + n].decode(), i + n


def parse_osr(data: bytes) -> dict:
    """Minimal .osr reader following the osu! wiki layout."""
    mode, _version = struct.unpack_from("<Bi", data, 0)
    i = 5
    md5, i = _read_string(data, i)
    player, i = _read_string(data, i)
    _rmd5, i = _read_string(data, i)
    counts = struct.unpack_from("<6H", data, i)
    i += 12
    score, combo, _perfect, mods = struct.unpack_from("<iHBi", data, i)
    i += 11
    _life, i = _read_string(data, i)
    _ticks, length = struct.unpack_from("<qi", data, i)
    i += 12
    blob = data[i:i + length]
    # osr_writer stores the real size *and* an end marker (osu!'s 7-Zip decoder accepts that);
    # Python's stricter decoder wants "size unknown" when an end marker is present.
    declared = struct.unpack_from("<Q", blob, 5)[0]
    frames = lzma.decompress(blob[:5] + bytes([255]) * 8 + blob[13:], format=lzma.FORMAT_ALONE).decode()
    assert declared == len(frames)
    return {"mode": mode, "md5": md5, "player": player, "counts": counts, "score": score,
            "combo": combo, "frames": [f for f in frames.split(",") if f]}


def test_generate_replay_and_history(store, storage, trained, tmp_path):
    hist = ReplayHistory(storage)
    osu = DEMO_MAPS / "demo_002.osu"
    rec = generate_replay(store, hist, trained, "v001", osu, player="Tester", tries=2)
    path = rec["path"]
    osr = parse_osr(open(path, "rb").read())
    assert osr["mode"] == 3 and osr["md5"] == parse_osu(osu).md5 and osr["player"] == "Tester"
    assert sum(osr["counts"]) == rec["metrics"]["total_notes"]
    assert str(storage.replays_dir) in path  # replays live in the data root, not the repo
    listed = hist.list()
    assert listed[0]["id"] == rec["id"] and listed[0]["exists"]
    assert listed[0]["model_name"] == "Test Bot" and listed[0]["version"] == "v001"
    assert model_status(store, listed[0]) == ""
    # Deleting the model keeps the replay and its history entry
    store.delete_model(trained)
    listed = hist.list()
    assert listed[0]["exists"] and model_status(store, listed[0]) == "model deleted"
    hist.delete(rec["id"])
    assert hist.list() == [] and not (tmp_path / path).exists()


def test_replay_rejects_non_4k(store, storage, trained, tmp_path):
    seven = tmp_path / "seven.osu"
    seven.write_text(SEVEN_K)
    with pytest.raises(ModelError, match="7K"):
        generate_replay(store, ReplayHistory(storage), trained, "v001", seven)


def test_osr_frames_roundtrip():
    """Decompress the replay frames and check they are well-formed mania frames."""
    from osr_writer import build_osr

    events = [(1000.0, 0, 1), (1050.0, 0, 0), (1100.0, 3, 1), (1300.0, 3, 0)]
    data = build_osr(beatmap_md5="a" * 32, player="p", counts=[1, 0, 0, 0, 0, 0], score=1, max_combo=1, events=events)
    frames = parse_osr(data)["frames"]
    assert frames[-1].startswith("-12345")
    masks = [int(f.split("|")[1]) for f in frames[2:-1]]
    assert 1 in masks and 8 in masks


def test_legacy_replay_import(storage, tmp_path):
    src = tmp_path / "old replays"
    src.mkdir()
    (src / "a b.osr").write_bytes(b"\x03fake")
    hist = ReplayHistory(storage)
    assert hist.import_files(list(src.glob("*.osr"))) == 1
    assert hist.import_files(list(src.glob("*.osr"))) == 0  # no duplicates
    assert hist.list()[0]["kind"] == "imported"
