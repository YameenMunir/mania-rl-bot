from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO_MAPS = ROOT / "maps" / "demo"


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    """An isolated data root whose path contains spaces (like many Windows user folders)."""
    root = tmp_path / "App Data With Spaces" / "OsuManiaAI"
    monkeypatch.setenv("OSUMANIAAI_HOME", str(root))
    return root


@pytest.fixture()
def storage(data_root):
    from maniabot.storage import Storage

    return Storage(data_root).ensure()


@pytest.fixture()
def store(storage):
    from maniabot.models import ModelStore

    return ModelStore(storage)


@pytest.fixture()
def small_maps(tmp_path):
    """Three demo maps copied into a folder with spaces in its name."""
    folder = tmp_path / "my maps" / "4k set"
    folder.mkdir(parents=True)
    for name in ("demo_000.osu", "demo_001.osu", "demo_002.osu"):
        (folder / name).write_bytes((DEMO_MAPS / name).read_bytes())
    return folder


def tiny_spec(model_id: str, maps: Path, **kw):
    from maniabot.training import TrainingSpec

    base = dict(model_id=model_id, maps_folder=str(maps), timesteps=1024, n_steps=256, batch_size=256,
                n_epochs=1, n_envs=1, checkpoint_every=1000, clip_seconds=5.0, device="cpu", val_count=1,
                net_arch=[16, 16])
    base.update(kw)
    return TrainingSpec(**base)


@pytest.fixture()
def trained(store, small_maps):
    """A model with one trained version (v001)."""
    from maniabot.training import run_training

    rec = store.create_model("Test Bot", "unit test")
    session = run_training(store, tiny_spec(rec.id, small_maps))
    assert session["status"] == "completed", session.get("error")
    return rec.id


@pytest.fixture(scope="session")
def sample_zip(tmp_path_factory):
    """A real (untrained) stable-baselines3 PPO zip for the 4K env, created once."""
    from stable_baselines3 import PPO

    from mania_env import ManiaEnv
    from osu_parser import load_maps

    env = ManiaEnv(load_maps(DEMO_MAPS)[:1], clip_seconds=5.0)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, policy_kwargs=dict(net_arch=[16, 16]), device="cpu")
    path = tmp_path_factory.mktemp("sample model") / "model.zip"
    model.save(path)
    return path
