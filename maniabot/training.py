"""Training engine: new models, continued training, sessions, checkpoints, safe version saves.

`run_training()` does the work and can run anywhere; `TrainingJob` runs it in a separate
process so the UI stays responsive, and relays progress events and pause/stop commands
through multiprocessing queues.

Training never overwrites an existing version. It always ends in one of:
  completed / stopped  -> a NEW version (vNNN) is saved with parent_version = the source
  cancelled            -> no version; the session's checkpoints are kept
  failed / interrupted -> no version; checkpoints are kept so training can be resumed
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
import shutil
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from maniabot.beatmaps import default_val_count, map_summary, scan_maps, split_maps, star_range_label
from maniabot.logs import event, setup_logging
from maniabot.models import IN_USE, InvalidCheckpoint, ModelError, ModelStore, now_iso, verify_weights
from maniabot.storage import Storage

log = logging.getLogger("maniabot.training")

DEFAULT_NET_ARCH = [128, 128]
FIXED_PPO = {"gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.003}
VAL_MAPS_PER_CHECKPOINT = 3


def default_n_envs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


@dataclass
class TrainingSpec:
    model_id: str
    kind: str = "new"  # "new" | "continue"
    source_version: str | None = None  # continue from this version...
    source_checkpoint: str | None = None  # ...or from this checkpoint file
    maps_folder: str = ""
    val_count: int = -1  # -1 = automatic
    min_stars: float | None = None  # only use maps rated in [min_stars, max_stars]; None = no limit
    max_stars: float | None = None
    timesteps: int = 300_000
    learning_rate: float = 3e-4
    batch_size: int = 1024
    n_steps: int = 1024
    n_epochs: int = 6
    n_envs: int = field(default_factory=default_n_envs)
    net_arch: list[int] = field(default_factory=lambda: list(DEFAULT_NET_ARCH))
    dt_ms: float = 10.0
    clip_seconds: float = 20.0
    jitter_ms: float = 0.0
    checkpoint_every: int = 100_000
    device: str = "auto"
    seed: int = 0
    set_active: bool = True
    notes: str = ""
    session_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])

    @property
    def env_config(self) -> dict[str, float]:
        return {"dt_ms": float(self.dt_ms), "clip_seconds": float(self.clip_seconds), "jitter_ms": float(self.jitter_ms)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ------------------------------------------------------------------ devices
def device_status() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": False, "cuda": False, "name": None, "message": "PyTorch is not installed."}
    cuda = bool(torch.cuda.is_available())
    return {
        "torch": True,
        "torch_version": torch.__version__,
        "cuda": cuda,
        "name": torch.cuda.get_device_name(0) if cuda else None,
        "message": f"CUDA GPU: {torch.cuda.get_device_name(0)}" if cuda else "No CUDA GPU available, training uses the CPU.",
    }


def resolve_device(device: str) -> str:
    device = (device or "auto").lower()
    if device == "cpu":
        return "cpu"
    import torch

    if device == "cuda":
        if not torch.cuda.is_available():
            raise ModelError("CUDA was selected but no CUDA GPU is available. Choose Auto or CPU.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- validation
def validate_spec(spec: TrainingSpec) -> list[str]:
    errors = []
    if spec.timesteps < 1000:
        errors.append("Training length must be at least 1,000 timesteps.")
    if not (0 < spec.learning_rate < 1):
        errors.append("Learning rate must be between 0 and 1 (typical: 0.0003).")
    if spec.batch_size < 16:
        errors.append("Batch size must be at least 16.")
    if spec.n_steps < 16:
        errors.append("Rollout length must be at least 16 steps.")
    if spec.n_envs < 1 or spec.n_envs > 64:
        errors.append("Parallel games must be between 1 and 64.")
    if spec.n_epochs < 1 or spec.n_epochs > 50:
        errors.append("Optimisation passes per update must be between 1 and 50.")
    if spec.dt_ms <= 0 or spec.dt_ms > 50:
        errors.append("Decision tick must be between 0 and 50 ms.")
    if spec.jitter_ms < 0:
        errors.append("Timing noise can't be negative.")
    if spec.checkpoint_every < 1000:
        errors.append("Checkpoint frequency must be at least 1,000 timesteps.")
    if not spec.maps_folder or not Path(spec.maps_folder).exists():
        errors.append("Choose a beatmap folder that exists.")
    for label, v in (("Minimum", spec.min_stars), ("Maximum", spec.max_stars)):
        if v is not None and not (math.isfinite(v) and v >= 0):
            errors.append(f"{label} star rating must be a number of 0 or more.")
    if (spec.min_stars is not None and spec.max_stars is not None
            and spec.min_stars > spec.max_stars):
        errors.append("Minimum star rating can't be higher than the maximum.")
    if spec.kind == "new" and (not spec.net_arch or any(int(n) < 1 for n in spec.net_arch)):
        errors.append("Network size must be a list of positive layer widths, e.g. 128,128.")
    if spec.kind == "continue" and not (spec.source_version or spec.source_checkpoint):
        errors.append("Choose a version or checkpoint to continue from.")
    if spec.device == "cuda":
        try:
            resolve_device("cuda")
        except ModelError as exc:
            errors.append(str(exc))
    return errors


def compatibility_warnings(source_meta: dict[str, Any], spec: TrainingSpec) -> list[str]:
    """Things that changed versus the checkpoint and may hurt or surprise. Never blocks."""
    warn = []
    env = {"dt_ms": 10.0, "clip_seconds": 20.0, "jitter_ms": 0.0, **(source_meta.get("env_config") or {})}
    hp = source_meta.get("hyperparameters") or {}
    if abs(float(env["dt_ms"]) - spec.dt_ms) > 1e-9:
        warn.append(f"Decision tick changes from {env['dt_ms']:g} ms to {spec.dt_ms:g} ms. The model's timing was learned "
                    "at the old tick, so expect accuracy to drop before it re-adapts.")
    if abs(float(env["jitter_ms"]) - spec.jitter_ms) > 1e-9:
        warn.append(f"Timing noise changes from {env['jitter_ms']:g} ms to {spec.jitter_ms:g} ms; results won't be "
                    "directly comparable with earlier versions.")
    old_lr = hp.get("learning_rate")
    if old_lr and spec.learning_rate > 3 * float(old_lr):
        warn.append(f"Learning rate is much higher than before ({old_lr:g} -> {spec.learning_rate:g}); training may become unstable.")
    if spec.batch_size > spec.n_steps * spec.n_envs:
        warn.append("Batch size is larger than one rollout (rollout length x parallel games); it will be capped.")
    elif (spec.n_steps * spec.n_envs) % spec.batch_size:
        warn.append("Rollout length x parallel games isn't a multiple of the batch size; the last mini-batch will be smaller.")
    old_maps = source_meta.get("maps_folder")
    if old_maps and spec.maps_folder and Path(old_maps).resolve() != Path(spec.maps_folder).resolve():
        warn.append("Training on a different beatmap folder than the previous version.")
    return warn


# ------------------------------------------------------------------ control
class Control:
    """Pause/stop/cancel/checkpoint flags polled by the training loop."""

    def __init__(self):
        self._paused = False
        self._stop = False
        self._cancel = False
        self._checkpoint = False

    def poll(self) -> None:  # overridden by queue-backed control
        pass

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._stop = True

    def cancel(self) -> None:
        self._cancel = True
        self._stop = True

    def request_checkpoint(self) -> None:
        self._checkpoint = True

    @property
    def paused(self) -> bool:
        self.poll()
        return self._paused and not self._stop

    @property
    def stopping(self) -> bool:
        self.poll()
        return self._stop

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def take_checkpoint_request(self) -> bool:
        self.poll()
        req, self._checkpoint = self._checkpoint, False
        return req


class QueueControl(Control):
    def __init__(self, cmd_q):
        super().__init__()
        self.cmd_q = cmd_q

    def poll(self) -> None:
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            except (EOFError, OSError):
                self._stop = True
                return
            getattr(self, {"pause": "pause", "resume": "resume", "stop": "stop", "cancel": "cancel",
                           "checkpoint": "request_checkpoint"}.get(cmd, "poll"))()


# ----------------------------------------------------------------- training
class TrainingFailed(ModelError):
    pass


def _friendly_error(exc: BaseException) -> str:
    name = type(exc).__name__
    text = str(exc)
    if name == "OutOfMemoryError" or "out of memory" in text.lower():
        return "The GPU ran out of memory. Lower the batch size or parallel games, or train on the CPU."
    if isinstance(exc, ModelError):
        return text
    if isinstance(exc, PermissionError):
        return f"Permission denied writing training files: {exc.filename or text}"
    if isinstance(exc, OSError):
        return f"File error: {exc.strerror or text}"
    return f"Training failed: {name}: {text}"


def run_training(
    store: ModelStore,
    spec: TrainingSpec,
    emit: Callable[[dict[str, Any]], None] | None = None,
    control: Control | None = None,
) -> dict[str, Any]:
    """Run one training session. Returns the final session record."""
    emit = emit or (lambda _e: None)
    control = control or Control()
    rec = store.get_model(spec.model_id)
    model_name = rec.name
    session: dict[str, Any] = {
        "session_id": spec.session_id,
        "model_id": spec.model_id,
        "model_name": model_name,
        "kind": spec.kind,
        "status": "running",
        "started_at": now_iso(),
        "ended_at": None,
        "start_version": spec.source_version,
        "start_checkpoint": spec.source_checkpoint,
        "result_version": None,
        "spec": spec.to_dict(),
        "checkpoints": [],
        "best_checkpoint": None,
        "latest_checkpoint": None,
        "loss_history": [],
        "error": None,
    }
    store.save_session(spec.model_id, session)
    event(log, "training.started", model=spec.model_id, session=spec.session_id, kind=spec.kind,
          source=spec.source_version or spec.source_checkpoint or "new", timesteps=spec.timesteps)
    t_start = time.time()
    env = None
    try:
        errors = validate_spec(spec)
        if errors:
            raise TrainingFailed(" ".join(errors))
        emit({"type": "status", "status": "loading maps", "message": "Checking beatmaps…"})
        scan = scan_maps(spec.maps_folder)
        if not scan.valid:
            reasons = {c.reason for c in scan.rejected}
            raise TrainingFailed(
                f"No usable osu!mania 4K maps in {spec.maps_folder}."
                + (f" Rejected {len(scan.rejected)} file(s): {'; '.join(sorted(reasons)[:3])}" if scan.rejected else "")
            )
        in_range = scan.in_star_range(spec.min_stars, spec.max_stars)
        if not in_range:
            lo, hi = scan.star_bounds() or (0.0, 0.0)
            raise TrainingFailed(
                f"No maps with {star_range_label(spec.min_stars, spec.max_stars)} in {spec.maps_folder}. "
                f"It has {len(scan.valid)} usable map(s), rated {lo:.1f}★ to {hi:.1f}★."
            )
        stars_by_path = {c.beatmap.path: c.stars for c in in_range}
        maps = [c.beatmap for c in in_range]
        n_val = default_val_count(len(maps)) if spec.val_count < 0 else spec.val_count
        train_maps, val_maps = split_maps(maps, n_val, spec.seed)
        val_is_heldout = bool(val_maps)
        if not val_maps:  # too few maps to hold any out: validate on (a few) training maps
            val_maps = train_maps[: min(4, len(train_maps))]
        session.update({
            "training_maps": [{**map_summary(m), "stars": stars_by_path.get(m.path)} for m in train_maps],
            "validation_maps": [{**map_summary(m), "stars": stars_by_path.get(m.path)} for m in val_maps],
            "star_range": {"min": spec.min_stars, "max": spec.max_stars,
                           "maps_in_range": len(in_range), "maps_in_folder": len(scan.valid)},
            "validation_heldout": val_is_heldout,
            "rejected_maps": len(scan.rejected),
        })

        device = resolve_device(spec.device)
        session["device"] = device
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        from mania_env import ManiaEnv
        from maniabot.sb3_compat import load_ppo

        env_kwargs = dict(dt_ms=spec.dt_ms, clip_seconds=spec.clip_seconds, jitter_ms=spec.jitter_ms)

        def make_env():
            return ManiaEnv(train_maps, **env_kwargs)

        emit({"type": "status", "status": "starting", "message": f"Starting {spec.n_envs} simulator(s) on {device}…"})
        vec_cls = SubprocVecEnv if spec.n_envs > 1 else DummyVecEnv
        env = make_vec_env(make_env, n_envs=spec.n_envs, seed=spec.seed, vec_env_cls=vec_cls)

        source_meta: dict[str, Any] = {}
        start_timesteps = 0
        if spec.kind == "new":
            if store.list_versions(spec.model_id) and not spec.source_version:
                log.info("training a fresh network inside existing model %s", spec.model_id)
            model = PPO(
                "MlpPolicy", env,
                learning_rate=spec.learning_rate, n_steps=spec.n_steps, batch_size=spec.batch_size,
                n_epochs=spec.n_epochs, policy_kwargs=dict(net_arch=[int(n) for n in spec.net_arch]),
                seed=spec.seed, device=device, verbose=0, **FIXED_PPO,
            )
            source_desc = "new model"
        else:
            if spec.source_checkpoint:
                src_path = Path(spec.source_checkpoint)
                source_desc = f"checkpoint {src_path.name}"
                parent = next((c.get("parent_version") for c in store.list_checkpoints(spec.model_id)
                               if Path(c["path"]) == src_path), None)
                if spec.source_version is None:
                    spec.source_version = parent
                    session["start_version"] = parent
                if spec.source_version:
                    try:
                        source_meta = store.get_version(spec.model_id, spec.source_version).meta
                    except ModelError:
                        source_meta = {}
            else:
                src = store.get_version(spec.model_id, spec.source_version)
                src_path = src.weights
                source_meta = src.meta
                source_desc = f"version {spec.source_version}"
            verify_weights(src_path)
            emit({"type": "status", "status": "loading model", "message": f"Restoring {source_desc}…"})
            try:
                model = load_ppo(
                    src_path, env=env, device=device,
                    custom_objects={"learning_rate": spec.learning_rate, "batch_size": spec.batch_size,
                                    "n_steps": spec.n_steps, "n_epochs": spec.n_epochs},
                )
            except Exception as exc:
                raise InvalidCheckpoint(f"Couldn't restore {source_desc}: {exc}") from exc
            model.set_random_seed(spec.seed)
            start_timesteps = int(model.num_timesteps)
        event(log, "checkpoint.loaded" if spec.kind == "continue" else "model.initialised",
              model=spec.model_id, source=source_desc, device=device)

        target_version = store.reserve_version(spec.model_id)
        session["result_version_planned"] = target_version
        store.save_session(spec.model_id, session)

        callback = _make_callback(store, spec, session, model, val_maps, emit, control, t_start,
                                  start_timesteps, target_version, model_name, device)
        emit({"type": "status", "status": "training", "message": "Training…"})
        model.learn(total_timesteps=spec.timesteps, callback=callback, reset_num_timesteps=(spec.kind == "new"))
        callback.finish()

        if control.cancelled:
            session["status"] = "cancelled"
            emit({"type": "status", "status": "cancelled", "message": "Cancelled; no new version was saved."})
        else:
            session["status"] = "stopped" if control.stopping else "completed"
            emit({"type": "status", "status": "saving", "message": f"Validating and saving {target_version}…"})
            version = _save_version(store, spec, session, model, val_maps, val_is_heldout, source_desc,
                                    source_meta, target_version, start_timesteps, t_start, device, emit, control)
            session["result_version"] = version
    except BaseException as exc:  # noqa: BLE001 - report every failure, including KeyboardInterrupt
        session["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        session["error"] = _friendly_error(exc)
        log.error("training failed: %s\n%s", exc, traceback.format_exc())
        emit({"type": "error", "message": session["error"], "trace": traceback.format_exc()})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                log.exception("closing envs")
        session["ended_at"] = now_iso()
        session["duration_s"] = round(time.time() - t_start, 1)
        try:
            store.save_session(spec.model_id, session)
        except Exception:  # noqa: BLE001
            log.exception("could not save session record")
        event(log, "training.ended", model=spec.model_id, session=spec.session_id, status=session["status"],
              version=session.get("result_version"))
        emit({"type": "done", "status": session["status"], "version": session.get("result_version"),
              "session": session, "error": session.get("error")})
    return session


def _save_checkpoint_file(model, target: Path, tmp_root: Path) -> None:
    """Save to a temp folder, verify, then move into place (never leaves a half-written zip)."""
    tmp_dir = tmp_root / f"ckpt-{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        tmp_file = tmp_dir / "model.zip"
        model.save(tmp_file)
        verify_weights(tmp_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_file), str(target))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _make_callback(store, spec, session, model, val_maps, emit, control, t_start, start_timesteps,
                   target_version, model_name, device):
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback

    from maniabot.evaluation import evaluate
    from train import EpisodeLogger

    ckpt_dir = store.checkpoints_dir(spec.model_id) / spec.session_id
    csv_path = store.sessions_dir(spec.model_id) / f"{spec.session_id}_episodes.csv"
    tmp_root = store.storage.tmp_dir

    class AppCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes = EpisodeLogger(csv_path, print_every=10**9)
            self.next_ckpt = spec.checkpoint_every
            self.last_emit = 0.0
            self.maps_seen: set[str] = set()
            self.best_val = -1.0
            self.paused_for = 0.0
            self.loss = None

        def _init_callback(self) -> None:
            self.episodes.init_callback(self.model)

        def _on_training_start(self) -> None:
            self.episodes.on_training_start(self.locals, self.globals)

        def _on_rollout_start(self) -> None:
            # train() for the previous rollout has just run, so its losses are available here.
            vals = self.model.logger.name_to_value
            if "train/loss" in vals:
                self.loss = {
                    "loss": float(vals["train/loss"]),
                    "value_loss": float(vals.get("train/value_loss", float("nan"))),
                    "policy_loss": float(vals.get("train/policy_gradient_loss", float("nan"))),
                    "entropy": float(vals.get("train/entropy_loss", float("nan"))),
                }
                session["loss_history"].append({"timesteps": self.session_steps, **self.loss})

        @property
        def session_steps(self) -> int:
            return int(self.num_timesteps - start_timesteps)

        def _on_step(self) -> bool:
            self.episodes.update_locals(self.locals)
            self.episodes.on_step()
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done and "map" in info:
                    self.maps_seen.add(info["map"])
            if control.paused:
                self._pause_loop()
            if control.stopping:
                return False
            if self.session_steps >= self.next_ckpt or control.take_checkpoint_request():
                self.checkpoint(validate=True)
                while self.next_ckpt <= self.session_steps:
                    self.next_ckpt += spec.checkpoint_every
            now = time.time()
            if now - self.last_emit > 0.5:
                self.last_emit = now
                self.emit_progress()
            return True

        def _pause_loop(self) -> None:
            t0 = time.time()
            session["status"] = "paused"
            store.save_session(spec.model_id, session)
            emit({"type": "status", "status": "paused", "message": "Paused"})
            event(log, "training.paused", session=spec.session_id)
            while control.paused:
                if control.take_checkpoint_request():
                    self.checkpoint(validate=True)
                time.sleep(0.1)
            self.paused_for += time.time() - t0
            if not control.stopping:
                session["status"] = "running"
                store.save_session(spec.model_id, session)
                emit({"type": "status", "status": "training", "message": "Training…"})
                event(log, "training.resumed", session=spec.session_id)

        def checkpoint(self, validate: bool) -> None:
            path = ckpt_dir / f"ppo_{self.num_timesteps}_steps.zip"
            emit({"type": "status", "status": "checkpoint", "message": "Saving checkpoint…"})
            _save_checkpoint_file(self.model, path, tmp_root)
            entry = {"path": str(path), "timesteps": int(self.num_timesteps), "session_timesteps": self.session_steps,
                     "created_at": now_iso(), "val_accuracy": None}
            if validate and val_maps:
                emit({"type": "status", "status": "validating", "message": "Validating checkpoint…"})
                res = evaluate(self.model, val_maps[:VAL_MAPS_PER_CHECKPOINT], spec.env_config, seed=12345)
                entry["val_accuracy"] = res["accuracy"]
                if res["accuracy"] > self.best_val:
                    self.best_val = res["accuracy"]
                    session["best_checkpoint"] = str(path)
            session["checkpoints"].append(entry)
            session["latest_checkpoint"] = str(path)
            store.save_session(spec.model_id, session)
            event(log, "checkpoint.created", model=spec.model_id, path=path, val_acc=entry["val_accuracy"])
            emit({"type": "checkpoint", **entry, "best": session["best_checkpoint"] == str(path)})
            emit({"type": "status", "status": "training", "message": "Training…"})

        def emit_progress(self) -> None:
            elapsed = time.time() - t_start - self.paused_for
            done = self.session_steps
            accs = self.episodes.accs
            rate = done / elapsed if elapsed > 0 else 0.0
            best = next((c for c in session["checkpoints"] if c["path"] == session["best_checkpoint"]), None)
            emit({
                "type": "progress",
                "model_name": model_name,
                "version": target_version,
                "parent_version": spec.source_version,
                "timesteps": done,
                "target": spec.timesteps,
                "total_timesteps": int(self.num_timesteps),
                "plays": len(accs),
                "maps_seen": len(self.maps_seen),
                "recent_accuracy": float(np.mean(accs[-100:])) if accs else None,
                "loss": self.loss,
                "learning_rate": float(self.model.lr_schedule(self.model._current_progress_remaining)),
                "device": device,
                "elapsed": elapsed,
                "steps_per_s": rate,
                "eta": (spec.timesteps - done) / rate if rate > 0 else None,
                "best_checkpoint": best,
                "latest_checkpoint": session["latest_checkpoint"],
            })

        def finish(self) -> None:
            self.emit_progress()
            self.episodes.on_training_end()

    return AppCallback()


def _save_version(store, spec, session, model, val_maps, heldout, source_desc, source_meta,
                  target_version, start_timesteps, t_start, device, emit, control) -> str:
    from maniabot.evaluation import evaluate, strip_private

    pending = store.begin_version(spec.model_id, target_version)
    try:
        model.save(pending.weights)
        verify_weights(pending.weights)
    except BaseException:
        store.abort_version(pending)
        raise
    validation = None
    if val_maps:
        try:
            validation = strip_private(evaluate(model, val_maps, spec.env_config, seed=12345))
            validation["heldout"] = heldout
            validation["maps_used"] = [m.name for m in val_maps]
        except Exception:  # noqa: BLE001 - validation failure must not lose the trained weights
            log.exception("final validation failed")
    curve = store.sessions_dir(spec.model_id) / f"{spec.session_id}_curve.png"
    try:
        from train import plot_curve

        csv_path = store.sessions_dir(spec.model_id) / f"{spec.session_id}_episodes.csv"
        if csv_path.exists():
            plot_curve(csv_path, curve)
    except Exception:  # noqa: BLE001
        log.exception("plotting learning curve")
    hp = {"learning_rate": spec.learning_rate, "n_steps": spec.n_steps, "batch_size": spec.batch_size,
          "n_epochs": spec.n_epochs, "n_envs": spec.n_envs, "seed": spec.seed, **FIXED_PPO}
    net_arch = spec.net_arch if spec.kind == "new" else (source_meta.get("architecture") or {}).get("net_arch") or (
        store.get_model(spec.model_id).meta.get("architecture") or {}).get("net_arch")
    last_loss = session["loss_history"][-1] if session["loss_history"] else None
    meta = {
        "parent_version": spec.source_version if spec.kind == "continue" else None,
        "session_id": spec.session_id,
        "status": session["status"],
        "source": source_desc,
        "source_checkpoint": spec.source_checkpoint,
        "trained_timesteps": int(model.num_timesteps - start_timesteps),
        "total_timesteps": int(model.num_timesteps),
        "training_duration_s": round(time.time() - t_start, 1),
        "hyperparameters": hp,
        "device": device,
        "env_config": spec.env_config,
        "architecture": {"algorithm": "PPO", "policy": "MlpPolicy", "net_arch": net_arch, "n_upcoming": 4,
                         "lookahead_ms": 1000.0},
        "maps_folder": str(Path(spec.maps_folder).resolve()),
        "training_maps": session.get("training_maps", []),
        "validation_maps": session.get("validation_maps", []),
        "star_range": session.get("star_range"),
        "validation": validation,
        "final_loss": last_loss,
        "best_checkpoint": session.get("best_checkpoint"),
        "notes": spec.notes,
    }
    rec = store.commit_version(pending, meta, set_active=spec.set_active)
    if validation:
        store.save_evaluation(spec.model_id, rec.version, {"kind": "validation", **validation})
    return rec.version


# --------------------------------------------------------- worker process
def _worker_main(storage_root: str, spec_dict: dict[str, Any], event_q, cmd_q) -> None:
    storage = Storage(Path(storage_root))
    setup_logging(storage.logs_dir, filename="training-worker.log")
    store = ModelStore(storage)
    spec = TrainingSpec.from_dict(spec_dict)

    def emit(e: dict[str, Any]) -> None:
        try:
            event_q.put(e)
        except Exception:  # noqa: BLE001
            pass

    run_training(store, spec, emit, QueueControl(cmd_q))


class TrainingJob:
    """A training session running in a child process. Call poll() regularly from the UI thread."""

    def __init__(self, store: ModelStore, spec: TrainingSpec):
        self.store = store
        self.spec = spec
        ctx = mp.get_context("spawn")
        self.event_q = ctx.Queue()
        self.cmd_q = ctx.Queue()
        self.proc = ctx.Process(
            target=_worker_main,
            args=(str(store.storage.root), spec.to_dict(), self.event_q, self.cmd_q),
            name=f"training-{spec.session_id}",
            daemon=False,  # SubprocVecEnv needs to start its own child processes
        )
        self.status = "starting"
        self.message = "Starting…"
        self.last_progress: dict[str, Any] = {}
        self.checkpoints: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.started_at = time.time()
        self._tokens: list[int] = []

    def start(self) -> "TrainingJob":
        self._tokens.append(IN_USE.acquire(self.spec.model_id, self.spec.source_version or "__new__", "being trained"))
        self.proc.start()
        event(log, "training.job_started", model=self.spec.model_id, session=self.spec.session_id, pid=self.proc.pid)
        return self

    def _send(self, cmd: str) -> None:
        if self.proc.is_alive():
            self.cmd_q.put(cmd)

    def pause(self) -> None:
        self._send("pause")

    def resume(self) -> None:
        self._send("resume")

    def stop(self) -> None:
        self._send("stop")

    def cancel(self) -> None:
        self._send("cancel")

    def save_checkpoint(self) -> None:
        self._send("checkpoint")

    @property
    def finished(self) -> bool:
        return self.result is not None

    def poll(self) -> list[dict[str, Any]]:
        events = []
        while True:
            try:
                e = self.event_q.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError):
                break
            events.append(e)
            if e["type"] == "progress":
                self.last_progress = e
            elif e["type"] == "status":
                self.status, self.message = e["status"], e.get("message", "")
            elif e["type"] == "checkpoint":
                self.checkpoints.append(e)
            elif e["type"] == "error":
                self.error = e["message"]
            elif e["type"] == "done":
                self.result = e
                self.status = e["status"]
        if self.result is None and not self.proc.is_alive() and self.proc.exitcode is not None:
            # The worker died without reporting (crash, killed): record it as interrupted.
            self.store.mark_interrupted_sessions()
            self.error = self.error or f"The training process exited unexpectedly (code {self.proc.exitcode})."
            self.result = {"type": "done", "status": "interrupted", "version": None, "error": self.error}
            self.status = "interrupted"
            events.append(self.result)
        if self.result is not None:
            self._release()
        return events

    def _release(self) -> None:
        for t in self._tokens:
            IN_USE.release(t)
        self._tokens.clear()

    def join(self, timeout: float | None = None) -> None:
        self.proc.join(timeout)

    def kill(self) -> None:
        if self.proc.is_alive():
            self.proc.terminate()
        self._release()
