"""Model library: creation, discovery, versions, lineage, flags, deletion."""
from __future__ import annotations

import json
import shutil

import pytest

from maniabot.models import (
    IN_USE, InUseError, InvalidCheckpoint, ModelError, ModelNotFound, ProtectedError, VersionNotFound, verify_weights,
)
from tests.conftest import ROOT


SAMPLE = {}


def _fake_version(store, model_id, parent=None, val=None, **meta):
    """Commit a version from a sample model zip (fast, no training)."""
    return store.version_from_file(
        model_id, SAMPLE["zip"],
        {"parent_version": parent, "validation": ({"accuracy": val} if val is not None else None), **meta},
        set_active=True,
    )


@pytest.fixture()
def legacy_zip(sample_zip):
    SAMPLE["zip"] = sample_zip
    return sample_zip


def test_create_and_discover(store):
    rec = store.create_model("ManiaBot 01", "first")
    assert rec.id == "ManiaBot-01"
    assert [m.id for m in store.list_models()] == ["ManiaBot-01"]
    meta = store.get_model(rec.id).meta
    assert meta["mode"] == "osu!mania 4K" and meta["active_version"] is None
    with pytest.raises(ModelError):
        store.create_model("maniabot 01")  # names are unique (case-insensitive)


def test_missing_model_and_metadata(store):
    with pytest.raises(ModelNotFound):
        store.get_model("nope")
    rec = store.create_model("X")
    (store.model_dir(rec.id) / "model.json").unlink()
    assert store.list_models() == []  # a folder without metadata is not shown as a model


def test_versions_lineage_and_flags(store, legacy_zip):
    rec = store.create_model("Lineage")
    _fake_version(store, rec.id, val=0.5)
    _fake_version(store, rec.id, parent="v001", val=0.7)
    _fake_version(store, rec.id, parent="v002", val=0.6)
    v4 = _fake_version(store, rec.id, parent="v001", val=0.4)  # branch from an older version
    assert [v.version for v in store.list_versions(rec.id)] == ["v001", "v002", "v003", "v004"]
    assert v4.parent == "v001"
    assert store.lineage(rec.id, "v003") == [("v001", True), ("v002", True), ("v003", True)]
    m = store.get_model(rec.id).meta
    assert m["latest_version"] == "v004" and m["active_version"] == "v004"
    assert m["best_version"] == "v002"  # best is by validation, not recency
    store.set_active(rec.id, "v002")  # restore an old version: nothing is deleted
    assert len(store.list_versions(rec.id)) == 4
    assert store.resolve_version(rec.id, "active").version == "v002"
    store.set_best(rec.id, "v003")
    _fake_version(store, rec.id, val=0.99)
    assert store.get_model(rec.id).meta["best_version"] == "v003"  # manual best is respected
    store.set_best(rec.id, None)
    assert store.get_model(rec.id).meta["best_version"] == "v005"
    store.update_version(rec.id, "v001", protected=True, favourite=True, notes="keep me")
    v = store.get_version(rec.id, "v001")
    assert v.protected and v.favourite and v.meta["notes"] == "keep me"
    assert "Protected" in v.badges()


def test_version_numbers_never_reused(store, legacy_zip):
    rec = store.create_model("Numbers")
    _fake_version(store, rec.id)
    _fake_version(store, rec.id)
    store.delete_version(rec.id, "v002")
    assert _fake_version(store, rec.id).version == "v003"


def test_delete_version_updates_pointers(store, legacy_zip):
    rec = store.create_model("Del")
    _fake_version(store, rec.id, val=0.9)
    _fake_version(store, rec.id, parent="v001", val=0.1)
    _fake_version(store, rec.id, parent="v002", val=0.2)
    plan = store.plan_delete_version(rec.id, "v001")
    assert any("Best" in w for w in plan.warnings)
    assert any("v002" in w for w in plan.warnings)  # child lineage warning
    store.delete_version(rec.id, "v001")
    m = store.get_model(rec.id).meta
    assert m["best_version"] == "v003"
    assert store.lineage(rec.id, "v003") == [("v001", False), ("v002", True), ("v003", True)]
    store.delete_version(rec.id, "v003")  # active + latest
    m = store.get_model(rec.id).meta
    assert m["active_version"] == "v002" and m["latest_version"] == "v002"
    store.get_version(rec.id, "v002")  # remaining version still loads
    verify_weights(store.get_version(rec.id, "v002").weights)
    with pytest.raises(VersionNotFound):
        store.get_version(rec.id, "v003")


def test_protected_version_needs_extra_confirmation(store, legacy_zip):
    rec = store.create_model("Prot")
    _fake_version(store, rec.id)
    store.update_version(rec.id, "v001", protected=True)
    plan = store.plan_delete_version(rec.id, "v001")
    assert plan.needs_protected_confirmation
    with pytest.raises(ProtectedError):
        store.delete_version(rec.id, "v001")
    store.delete_version(rec.id, "v001", confirm_protected=True)  # still deletable
    assert store.list_versions(rec.id) == []


def test_in_use_blocks_deletion(store, legacy_zip):
    rec = store.create_model("Busy")
    _fake_version(store, rec.id)
    with IN_USE.use(rec.id, "v001", "generating a replay"):
        assert store.plan_delete_version(rec.id, "v001").blocked_reason
        with pytest.raises(InUseError):
            store.delete_version(rec.id, "v001")
        with pytest.raises(InUseError):
            store.delete_model(rec.id)
    store.delete_version(rec.id, "v001")


def test_delete_full_model(store, legacy_zip):
    rec = store.create_model("Gone")
    _fake_version(store, rec.id)
    store.update_version(rec.id, "v001", protected=True)
    plan = store.plan_delete_model(rec.id)
    assert plan.needs_protected_confirmation and plan.size_bytes > 0
    with pytest.raises(ProtectedError):
        store.delete_model(rec.id)
    store.delete_model(rec.id, confirm_protected=True)
    assert not store.exists(rec.id)
    assert store.list_models() == []
    assert not any(store.root.iterdir())  # no stale folders left


def test_rename_keeps_id(store):
    rec = store.create_model("Old Name")
    store.rename_model(rec.id, "New Name")
    assert store.get_model(rec.id).name == "New Name"
    assert store.get_model(rec.id).id == rec.id


def test_invalid_and_corrupt_weights(store, tmp_path):
    rec = store.create_model("Bad")
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not a zip")
    with pytest.raises(InvalidCheckpoint):
        verify_weights(junk)
    with pytest.raises(InvalidCheckpoint):
        store.version_from_file(rec.id, junk, {})
    assert store.list_versions(rec.id) == []
    # an aborted commit leaves no temp folders and no version
    assert not list(store.versions_dir(rec.id).iterdir())


def test_commit_failure_preserves_previous(store, legacy_zip, monkeypatch):
    rec = store.create_model("Atomic")
    _fake_version(store, rec.id)
    pending = store.begin_version(rec.id)
    pending.weights.write_bytes(b"half written")
    with pytest.raises(InvalidCheckpoint):
        store.commit_version(pending, {})
    assert [v.version for v in store.list_versions(rec.id)] == ["v001"]
    assert store.get_model(rec.id).meta["active_version"] == "v001"
    assert not pending.tmp_dir.exists()


def test_missing_version_metadata_is_tolerated(store, legacy_zip):
    rec = store.create_model("Meta")
    _fake_version(store, rec.id)
    (store.version_dir(rec.id, "v001") / "version.json").unlink()
    v = store.list_versions(rec.id)[0]
    assert v.meta.get("missing_metadata")


def test_legacy_import_copies_without_touching_original(store, legacy_zip, tmp_path):
    from maniabot.legacy import already_imported, import_legacy_run

    run = tmp_path / "runs" / "old run"
    run.mkdir(parents=True)
    shutil.copy2(legacy_zip, run / "model.zip")
    (run / "config.json").write_text(json.dumps({"dt_ms": 10.0, "clip_seconds": 20.0, "jitter_ms": 0.0, "maps": "maps/demo"}))
    (run / "episodes.csv").write_text("play,timesteps,accuracy\n1,10,0.1\n")
    before = {p.name: p.stat().st_size for p in run.iterdir()}
    rec = import_legacy_run(store, run)
    assert {p.name: p.stat().st_size for p in run.iterdir()} == before
    v = store.get_version(rec.id, "v001")
    assert v.meta["total_timesteps"] == 0 and v.meta["architecture"]["net_arch"] == [16, 16]
    assert v.meta["env_config"] == {"dt_ms": 10.0, "clip_seconds": 20.0, "jitter_ms": 0.0}
    assert already_imported(store, run)
    assert store.get_model(rec.id).meta["active_version"] == "v001"


def test_storage_validation_rejects_repo(tmp_path):
    from maniabot.storage import validate_storage_dir

    ok, msg = validate_storage_dir(ROOT / "somewhere")
    assert not ok and "repository" in msg
    ok, _ = validate_storage_dir(tmp_path / "new place" / "data")
    assert ok


def test_storage_migration_never_loses_models(store, storage, legacy_zip, tmp_path):
    from maniabot.storage import Storage, migrate_storage

    rec = store.create_model("Mover")
    _fake_version(store, rec.id)
    dst = Storage(tmp_path / "new root")
    report = migrate_storage(storage, dst)
    assert report.ok and report.copied_models == [rec.id]
    assert (store.model_dir(rec.id) / "versions" / "v001" / "model.zip").exists()  # source kept
    assert (dst.models_dir / rec.id / "versions" / "v001" / "model.zip").exists()
    again = migrate_storage(storage, dst)
    assert again.skipped_models == [rec.id]  # never overwrites existing data at the destination


def test_settings_persist(data_root):
    from maniabot.storage import Settings, settings_path

    s = Settings.load()
    s["player_name"] = "Tester"
    s.save()
    assert json.loads(settings_path().read_text())["player_name"] == "Tester"
    assert Settings.load()["player_name"] == "Tester"
    assert str(settings_path()).startswith(str(data_root))


def test_scan_reads_osz_mapsets_without_touching_source(tmp_path):
    import shutil
    from pathlib import Path
    from maniabot import beatmaps

    src = tmp_path / "songs"
    src.mkdir()
    shutil.copy(Path(__file__).resolve().parent.parent / "maps" / "osz" / "demo_000.osz", src / "a.osz")
    (src / "broken.osz").write_bytes(b"not a zip")
    before = sorted(p.name for p in src.iterdir())

    cache = tmp_path / "cache"
    files = []
    for f in sorted(src.glob("*.osz")):
        try:
            files += beatmaps.extract_osz(f, cache)
        except Exception:
            pass
    assert [f.suffix for f in files] == [".osu"]
    assert beatmaps.check_beatmap(files[0]).ok
    assert sorted(p.name for p in src.iterdir()) == before
