"""Training sessions: new model, resume, checkpoints, stop/cancel/fail, lineage."""
from __future__ import annotations

import zipfile

import pytest

from maniabot.training import Control, TrainingSpec, compatibility_warnings, run_training, validate_spec
from tests.conftest import tiny_spec


def test_new_model_training_creates_v001(store, trained):
    versions = store.list_versions(trained)
    assert [v.version for v in versions] == ["v001"]
    v = versions[0]
    m = v.meta
    assert m["parent_version"] is None and m["status"] == "completed"
    assert m["trained_timesteps"] >= 1024 and m["total_timesteps"] == m["trained_timesteps"]
    assert m["hyperparameters"]["learning_rate"] == 3e-4
    assert m["architecture"]["net_arch"] == [16, 16]
    assert len(m["training_maps"]) == 2 and len(m["validation_maps"]) == 1
    assert m["validation"]["heldout"] is True
    assert 0.0 <= m["validation"]["accuracy"] <= 1.0
    with zipfile.ZipFile(v.weights) as z:
        assert "policy.optimizer.pth" in z.namelist()  # optimiser state is saved for resuming
    model_meta = store.get_model(trained).meta
    assert model_meta["active_version"] == "v001" and model_meta["latest_version"] == "v001"
    sessions = store.list_sessions(trained)
    assert sessions[-1]["status"] == "completed" and sessions[-1]["result_version"] == "v001"
    assert sessions[-1]["checkpoints"], "a checkpoint should have been written"
    assert store.list_evaluations(trained, "v001")


def test_continue_training_creates_new_version_and_keeps_old(store, trained, small_maps):
    v1 = store.get_version(trained, "v001")
    v1_hash = v1.meta["weights_sha256"]
    session = run_training(store, tiny_spec(trained, small_maps, kind="continue", source_version="v001",
                                            learning_rate=1e-4))
    assert session["status"] == "completed", session.get("error")
    v2 = store.get_version(trained, "v002")
    assert v2.parent == "v001"
    assert v2.meta["total_timesteps"] > v1.meta["total_timesteps"]  # step counter restored and continued
    assert v2.meta["hyperparameters"]["learning_rate"] == 1e-4
    assert store.get_version(trained, "v001").meta["weights_sha256"] == v1_hash  # untouched


def test_branch_from_older_version(store, trained, small_maps):
    for _ in range(2):
        latest = store.get_model(trained).meta["latest_version"]
        run_training(store, tiny_spec(trained, small_maps, kind="continue", source_version=latest))
    assert [v.version for v in store.list_versions(trained)] == ["v001", "v002", "v003"]
    session = run_training(store, tiny_spec(trained, small_maps, kind="continue", source_version="v001"))
    assert session["result_version"] == "v004"
    assert store.get_version(trained, "v004").parent == "v001"
    assert len(store.list_versions(trained)) == 4  # nothing overwritten


def test_resume_from_checkpoint(store, trained, small_maps):
    ckpt = store.list_checkpoints(trained)[-1]
    assert ckpt["parent_version"] is None  # checkpoint of the initial (new-model) session
    session = run_training(store, tiny_spec(trained, small_maps, kind="continue", source_checkpoint=ckpt["path"]))
    assert session["status"] == "completed", session.get("error")
    v = store.get_version(trained, session["result_version"])
    assert v.meta["source_checkpoint"] == ckpt["path"]


def test_invalid_checkpoint_fails_cleanly(store, trained, small_maps, tmp_path):
    bad = tmp_path / "bad ckpt.zip"
    bad.write_bytes(b"garbage")
    before = [v.version for v in store.list_versions(trained)]
    session = run_training(store, tiny_spec(trained, small_maps, kind="continue", source_checkpoint=str(bad)))
    assert session["status"] == "failed" and "corrupt" in session["error"]
    assert [v.version for v in store.list_versions(trained)] == before


class _StopAfter(Control):
    def __init__(self, n_polls, cancel=False):
        super().__init__()
        self.n, self.do_cancel = n_polls, cancel

    def poll(self):
        self.n -= 1
        if self.n == 0:
            self.cancel() if self.do_cancel else self.stop()


def test_stop_saves_version_cancel_does_not(store, trained, small_maps):
    spec = tiny_spec(trained, small_maps, kind="continue", source_version="v001", timesteps=50_000)
    s = run_training(store, spec, control=_StopAfter(300))
    assert s["status"] == "stopped" and s["result_version"] == "v002"
    assert store.get_version(trained, "v002").meta["status"] == "stopped"
    spec = tiny_spec(trained, small_maps, kind="continue", source_version="v001", timesteps=50_000)
    s = run_training(store, spec, control=_StopAfter(300, cancel=True))
    assert s["status"] == "cancelled" and s["result_version"] is None
    assert [v.version for v in store.list_versions(trained)] == ["v001", "v002"]


def test_failed_training_on_no_maps(store, tmp_path):
    rec = store.create_model("No Maps")
    empty = tmp_path / "empty maps"
    empty.mkdir()
    (empty / "7k.osu").write_text("osu file format v14\n[General]\nMode: 3\n[Difficulty]\nCircleSize:7\n[HitObjects]\n")
    s = run_training(store, tiny_spec(rec.id, empty))
    assert s["status"] == "failed" and "4K" in s["error"]
    assert store.list_versions(rec.id) == []


def test_interrupted_session_detection(store, trained):
    sessions = store.list_sessions(trained)
    s = dict(sessions[-1], session_id="crashed", status="running")
    store.save_session(trained, s)
    assert store.mark_interrupted_sessions() == 1
    assert next(x for x in store.list_sessions(trained) if x["session_id"] == "crashed")["status"] == "interrupted"


def test_spec_validation_and_warnings(small_maps):
    spec = TrainingSpec(model_id="x", maps_folder=str(small_maps), timesteps=10, learning_rate=5)
    errs = validate_spec(spec)
    assert any("1,000" in e for e in errs) and any("Learning rate" in e for e in errs)
    spec = TrainingSpec(model_id="x", maps_folder=str(small_maps), dt_ms=5, jitter_ms=8, learning_rate=0.01,
                        n_envs=1, n_steps=100, batch_size=64)
    w = compatibility_warnings({"env_config": {"dt_ms": 10.0, "jitter_ms": 0.0},
                                "hyperparameters": {"learning_rate": 3e-4}}, spec)
    assert len(w) == 4


def test_spec_roundtrip():
    spec = TrainingSpec(model_id="m", maps_folder="x")
    assert TrainingSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.slow
def test_training_job_subprocess(store, small_maps):
    """The real worker-process path used by the UI."""
    import time

    from maniabot.training import TrainingJob

    rec = store.create_model("Proc Bot")
    job = TrainingJob(store, tiny_spec(rec.id, small_maps)).start()
    deadline = time.time() + 240
    events = []
    while not job.finished and time.time() < deadline:
        events += job.poll()
        time.sleep(0.2)
    job.join(30)
    assert job.finished and job.status == "completed", job.error
    assert any(e["type"] == "progress" for e in events)
    assert store.get_version(rec.id, "v001")
