"""Runtime data must never end up as files to commit in the source repository."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from maniabot.storage import base_root, platform_default_root, settings_path
from tests.conftest import ROOT

git = shutil.which("git")
pytestmark = pytest.mark.skipif(git is None or not (ROOT / ".git").exists(), reason="needs a git checkout")


def _status() -> set[str]:
    out = subprocess.run([git, "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return {line[3:].strip().strip('"') for line in out.splitlines()}


def test_default_data_root_is_outside_repo(monkeypatch):
    monkeypatch.delenv("OSUMANIAAI_HOME", raising=False)
    root = platform_default_root()
    assert ROOT.resolve() not in root.resolve().parents
    assert base_root() == root
    assert ROOT.resolve() not in settings_path().resolve().parents


@pytest.mark.parametrize("path", [
    "runs/x/model.zip", "runs/x/checkpoints/ppo_100_steps.zip", "replays/a b.osr", "models/m/versions/v001/model.zip",
    "checkpoints/c.pt", "anything/weights.pth", "logs/app.log", "user_data/models/x/model.json",
    "runtime/cache/x.bin", "foo/model.zip", "a/b/thing.safetensors", "x.tmp", "datasets/m.osu",
    "some/dir/events.out.tfevents.123", ".venv/Lib/x.py",
])
def test_gitignore_covers_runtime_files(path):
    r = subprocess.run([git, "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert r.returncode == 0, f"{path} is not ignored"


@pytest.mark.parametrize("path", ["maniabot/models.py", "maps/demo/demo_000.osu", "maps/osz/demo_000.osz", "tests/test_models.py"])
def test_gitignore_keeps_source(path):
    r = subprocess.run([git, "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert r.returncode == 1, f"{path} should not be ignored"


def test_runtime_artifacts_do_not_dirty_repo(tmp_path, monkeypatch):
    """Simulate a model, checkpoint, replay and logs written by mistake into the repo."""
    before = _status()
    fake = ROOT / "user_data" / "_git_safety_probe"
    files = [
        fake / "models" / "m" / "versions" / "v001" / "model.zip",
        fake / "models" / "m" / "checkpoints" / "s" / "ppo_1_steps.zip",
        fake / "replays" / "bot.osr",
        fake / "logs" / "app.log",
        ROOT / "runs" / "_git_safety_probe" / "model.zip",
        ROOT / "replays" / "_git_safety_probe.osr",
    ]
    try:
        for f in files:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"x")
        after = _status()
        assert after == before, f"new files to commit: {after - before}"
    finally:
        shutil.rmtree(fake, ignore_errors=True)
        shutil.rmtree(ROOT / "runs" / "_git_safety_probe", ignore_errors=True)
        (ROOT / "replays" / "_git_safety_probe.osr").unlink(missing_ok=True)
        ud = ROOT / "user_data"
        if ud.exists() and not any(ud.iterdir()):
            ud.rmdir()
