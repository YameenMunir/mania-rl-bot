"""Replay generation (via osr_writer.build_osr) and the persistent replay history.

History lives in <data root>/replays/history.json. Each record stores a snapshot of the
model name and version, so the history still makes sense after the model is deleted.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from maniabot.beatmaps import check_beatmap, map_summary
from maniabot.evaluation import POLICIES, PlayResult, play_map, strip_private
from maniabot.logs import event
from maniabot.models import IN_USE, ModelError, ModelStore, now_iso
from maniabot.storage import Storage, atomic_write_bytes, atomic_write_json, read_json
from osr_writer import build_osr

log = logging.getLogger("maniabot.replays")


def safe_filename(name: str, limit: int = 150) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return name[:limit] or "replay"


class ReplayHistory:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.dir = storage.replays_dir
        self.path = self.dir / "history.json"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        return read_json(self.path, {"replays": []}) or {"replays": []}

    def list(self) -> list[dict[str, Any]]:
        recs = self._load()["replays"]
        for r in recs:
            r["exists"] = Path(r["path"]).exists()
        return sorted(recs, key=lambda r: r.get("created_at") or "", reverse=True)

    def get(self, replay_id: str) -> dict[str, Any] | None:
        return next((r for r in self.list() if r["id"] == replay_id), None)

    def add(self, record: dict[str, Any]) -> None:
        data = self._load()
        data["replays"].append(record)
        atomic_write_json(self.path, data)

    def delete(self, replay_id: str, delete_file: bool = True) -> None:
        data = self._load()
        rec = next((r for r in data["replays"] if r["id"] == replay_id), None)
        if rec is None:
            return
        if delete_file:
            Path(rec["path"]).unlink(missing_ok=True)
        data["replays"] = [r for r in data["replays"] if r["id"] != replay_id]
        atomic_write_json(self.path, data)
        event(log, "replay.deleted", id=replay_id, path=rec["path"])

    def delete_for_model(self, model_id: str) -> int:
        ids = [r["id"] for r in self.list() if r.get("model_id") == model_id]
        for rid in ids:
            self.delete(rid)
        return len(ids)

    def import_files(self, files: list[Path], note: str = "Imported replay") -> int:
        """Add existing .osr files (e.g. from the legacy replays/ folder) by copying them in."""
        n = 0
        known = {Path(r["path"]).name for r in self._load()["replays"]}
        for f in files:
            f = Path(f)
            if not f.exists() or f.suffix.lower() != ".osr" or f.name in known:
                continue
            target = self.dir / f.name
            if not target.exists():
                atomic_write_bytes(target, f.read_bytes())
            self.add({
                "id": uuid.uuid4().hex,
                "path": str(target),
                "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
                "kind": "imported",
                "beatmap": {"title": f.stem},
                "model_id": None,
                "model_name": note,
                "version": None,
                "metrics": None,
                "processing_s": None,
            })
            n += 1
        return n


def model_status(store: ModelStore, rec: dict[str, Any]) -> str:
    """'' if the model/version still exists, else a short note for the history view."""
    if not rec.get("model_id"):
        return ""
    if not store.exists(rec["model_id"]):
        return "model deleted"
    if rec.get("version") and not store.version_dir(rec["model_id"], rec["version"]).exists():
        return "version deleted"
    return ""


def write_replay(
    store: ModelStore,
    history: ReplayHistory,
    model_id: str,
    version: str,
    result: PlayResult,
    player: str,
    kind: str = "generated",
    tries: int = 1,
    seed: int = 0,
    processing_s: float | None = None,
) -> dict[str, Any]:
    rec_model = store.get_model(model_id)
    bm = result.beatmap
    info = result.info
    data = build_osr(
        beatmap_md5=bm.md5,
        player=player,
        counts=info["counts"],
        score=int(info["accuracy"] * 1_000_000),
        max_combo=info["max_combo"],
        events=result.events,
    )
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = safe_filename(f"{player} - {bm.artist} - {bm.title} [{bm.version}] ({rec_model.name} {version}) {stamp}")
    target = history.dir / f"{base}.osr"
    if target.exists():
        target = history.dir / f"{base} {uuid.uuid4().hex[:4]}.osr"
    atomic_write_bytes(target, data)
    record = {
        "id": uuid.uuid4().hex,
        "path": str(target),
        "created_at": now_iso(),
        "kind": kind,
        "beatmap": map_summary(bm),
        "model_id": model_id,
        "model_name": rec_model.name,
        "version": version,
        "player": player,
        "tries": tries,
        "seed": seed,
        "metrics": {k: v for k, v in strip_private(result.metrics).items()},
        "processing_s": round(processing_s if processing_s is not None else result.seconds, 2),
    }
    history.add(record)
    event(log, "replay.generated", model=model_id, version=version, map=bm.name, path=target,
          acc=round(info["accuracy"], 4))
    return record


def generate_replay(
    store: ModelStore,
    history: ReplayHistory,
    model_id: str,
    version: str,
    beatmap_path: str | Path,
    player: str = "ManiaBot",
    tries: int = 1,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run inference on a 4K map without watching and save an .osr replay."""
    progress = progress or (lambda _m: None)
    t0 = time.perf_counter()
    check = check_beatmap(beatmap_path)
    if not check.ok:
        raise ModelError(f"Can't use this beatmap: {check.reason}")
    ver = store.get_version(model_id, version)
    with IN_USE.use(model_id, version, "generating a replay"):
        progress("Loading model…")
        model = POLICIES.load(ver.weights)
        best = None
        for k in range(max(1, tries)):
            progress(f"Playing {check.beatmap.name} ({k + 1}/{tries})…" if tries > 1 else f"Playing {check.beatmap.name}…")
            res = play_map(model, check.beatmap, ver.meta.get("env_config"), seed=seed + k, record=True,
                           should_stop=should_stop)
            if best is None or res.info["accuracy"] > best.info["accuracy"]:
                best = res
        progress("Writing replay…")
        return write_replay(store, history, model_id, version, best, player, "generated", tries, seed,
                            time.perf_counter() - t0)
