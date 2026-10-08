"""Import models trained with the original `train.py` CLI (runs/<name>/model.zip).

Importing *copies* the run into the model library as a new model with version v001. The
original run folder is never modified, moved or deleted.
"""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any

from maniabot import REPO_ROOT
from maniabot.logs import event
from maniabot.models import ModelError, ModelRecord, ModelStore, now_iso
from maniabot.storage import read_json

log = logging.getLogger("maniabot.legacy")

ENV_DEFAULTS = {"dt_ms": 10.0, "clip_seconds": 20.0, "jitter_ms": 0.0}


def read_sb3_data(zip_path: Path) -> dict[str, Any]:
    """Plain-JSON fields of a stable-baselines3 zip (no torch needed)."""
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("data"))
    return {k: v for k, v in data.items() if not (isinstance(v, dict) and ":serialized:" in v)}


def find_legacy_runs(runs_dir: Path | None = None) -> list[Path]:
    runs_dir = Path(runs_dir or REPO_ROOT / "runs")
    if not runs_dir.exists():
        return []
    return sorted(p for p in runs_dir.iterdir() if (p / "model.zip").exists())


def already_imported(store: ModelStore, run_dir: Path) -> bool:
    target = str(Path(run_dir).resolve())
    for m in store.list_models():
        if m.meta.get("imported_from") == target:
            return True
    return False


def import_legacy_run(store: ModelStore, run_dir: Path, name: str | None = None) -> ModelRecord:
    run_dir = Path(run_dir)
    weights = run_dir / "model.zip"
    if not weights.exists():
        raise ModelError(f"No model.zip in {run_dir}")
    cfg = {**ENV_DEFAULTS, **(read_json(run_dir / "config.json", {}) or {})}
    maps_folder = cfg.pop("maps", None)
    data = read_sb3_data(weights)
    net_arch = (data.get("policy_kwargs") or {}).get("net_arch", [64, 64])

    name = name or run_dir.name
    base, n = name, 2
    while store.name_taken(name):
        name, n = f"{base} ({n})", n + 1
    architecture = {"algorithm": "PPO", "policy": "MlpPolicy", "net_arch": net_arch, "n_upcoming": 4, "lookahead_ms": 1000.0}
    rec = store.create_model(name, f"Imported from {run_dir}", architecture, cfg)

    session_id = f"import-{now_iso().replace(':', '')}"
    sessions_dir = store.sessions_dir(rec.id)
    if (run_dir / "episodes.csv").exists():
        shutil.copy2(run_dir / "episodes.csv", sessions_dir / f"{session_id}_episodes.csv")
    if (run_dir / "learning_curve.png").exists():
        shutil.copy2(run_dir / "learning_curve.png", sessions_dir / f"{session_id}_curve.png")
    store.save_session(rec.id, {
        "session_id": session_id,
        "status": "completed",
        "kind": "import",
        "started_at": now_iso(),
        "ended_at": now_iso(),
        "start_version": None,
        "result_version": "v001",
        "notes": f"Imported from legacy run folder {run_dir}",
    })
    hp = {k: data.get(k) for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "n_envs", "seed")}
    store.version_from_file(rec.id, weights, {
        "parent_version": None,
        "session_id": session_id,
        "status": "imported",
        "source": f"Imported from {run_dir}",
        "trained_timesteps": int(data.get("num_timesteps", 0)),
        "total_timesteps": int(data.get("num_timesteps", 0)),
        "hyperparameters": hp,
        "device": "cpu",
        "env_config": cfg,
        "architecture": architecture,
        "maps_folder": maps_folder,
        "training_maps": [],
        "validation_maps": [],
        "validation": None,
        "sb3_version": _sb3_version(weights),
    }, set_active=True)
    with store._edit_model(rec.id) as mm:
        mm["imported_from"] = str(run_dir.resolve())
    event(log, "legacy.imported", run=run_dir, model=rec.id)
    return store.get_model(rec.id)


def _sb3_version(weights: Path) -> str | None:
    try:
        with zipfile.ZipFile(weights) as z:
            return z.read("_stable_baselines3_version").decode().strip()
    except (KeyError, OSError, zipfile.BadZipFile):
        return None
