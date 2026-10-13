"""Inference and evaluation of a trained policy on full osu!mania 4K maps.

All metrics are measured by the simulator (`ManiaEnv`), which uses osu!mania stable hit
windows. They are *not* osu! scores: osu! re-judges a replay with its own rules, so the
in-game result can differ slightly (see README).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mania_env import ACC_WEIGHT, JUDGEMENTS, ManiaEnv
from maniabot.logs import event
from maniabot.models import InvalidCheckpoint, verify_weights
from maniabot.mods import mod_list_label, mods_label
from osu_parser import Beatmap

log = logging.getLogger("maniabot.evaluation")
ENV_DEFAULTS = {"dt_ms": 10.0, "jitter_ms": 0.0}


# ----------------------------------------------------------------- loading
class PolicyCache:
    """Keeps the last few loaded policies so testing several maps doesn't reload from disk."""

    def __init__(self, size: int = 3):
        self.size = size
        self._items: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.Lock()

    def load(self, weights: Path, device: str = "cpu"):
        weights = Path(weights)
        verify_weights(weights)
        key = (str(weights.resolve()), weights.stat().st_mtime_ns, device)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        from maniabot.sb3_compat import load_ppo

        t0 = time.perf_counter()
        try:
            model = load_ppo(weights, device=device)
        except Exception as exc:  # torch/pickle errors from damaged or foreign files
            log.exception("failed to load %s", weights)
            raise InvalidCheckpoint(f"Couldn't load {weights.name}: {exc}") from exc
        event(log, "model.loaded", path=weights, device=device, seconds=round(time.perf_counter() - t0, 2))
        with self._lock:
            self._items[key] = model
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return model

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


POLICIES = PolicyCache()


def check_compatible(model, env: ManiaEnv) -> None:
    obs_shape = tuple(model.observation_space.shape)
    if obs_shape != tuple(env.observation_space.shape):
        raise InvalidCheckpoint(
            f"This model expects observations of shape {obs_shape} but the 4K simulator produces "
            f"{tuple(env.observation_space.shape)}; it was trained with different settings."
        )
    n_act = getattr(model.action_space, "n", None)
    if n_act is not None and int(np.prod(n_act)) != env.KEYS:
        raise InvalidCheckpoint("This model was not trained for 4 keys.")


# ---------------------------------------------------------------- playing
@dataclass
class PlayResult:
    beatmap: Beatmap
    info: dict[str, Any]
    events: list[tuple[float, int, int]]
    metrics: dict[str, Any]
    seconds: float = 0.0
    mods: int = 0  # osu! mod bit flags the map was played with


def make_play_env(bm: Beatmap, env_cfg: dict[str, Any], record: bool = False, mods: int = 0) -> ManiaEnv:
    cfg = {**ENV_DEFAULTS, **(env_cfg or {})}
    return ManiaEnv([bm], dt_ms=float(cfg["dt_ms"]), clip_seconds=None, jitter_ms=float(cfg["jitter_ms"]), record=record,
                    mods=mods)


def final_events(env: ManiaEnv) -> list[tuple[float, int, int]]:
    """The recorded key events (song time), with anything still held released at the end."""
    events = list(env.events)
    for lane in range(env.KEYS):
        if env.keys_down[lane]:
            events.append((env.t * env.rate, lane, 0))
    return events


def play_map(
    model,
    bm: Beatmap,
    env_cfg: dict[str, Any] | None = None,
    seed: int = 0,
    record: bool = False,
    should_stop: Callable[[], bool] | None = None,
    mods: int = 0,
) -> PlayResult:
    """Play a full map with the policy (actions sampled like in training)."""
    t0 = time.perf_counter()
    env = make_play_env(bm, env_cfg or {}, record, mods)
    check_compatible(model, env)
    model.set_random_seed(seed)
    obs, _ = env.reset(seed=seed)
    done, info = False, {}
    while not done:
        if should_stop and env.steps % 200 == 0 and should_stop():
            raise InterruptedError("stopped")
        # Sampled, not argmax: a press is a rare event, so the most likely action is always "no press".
        action = model.predict(obs, deterministic=False)[0]
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return PlayResult(bm, info, final_events(env) if record else [], compute_metrics(env, info),
                      time.perf_counter() - t0, env.mods)


def compute_metrics(env: ManiaEnv, info: dict[str, Any]) -> dict[str, Any]:
    counts = np.array(info["counts"], dtype=np.int64)
    total = int(info["total_notes"])
    deltas = np.array([d for (_l, d, j) in env.hit_log if j < 5], dtype=np.float64)
    lanes = []
    for lane in range(env.KEYS):
        n = len(env.times[lane])
        lc = env.lane_counts[lane].copy()
        lc[5] += max(0, n - int(lc.sum()))  # notes never judged count as misses
        ld = np.array([d for (l, d, j) in env.hit_log if l == lane and j < 5], dtype=np.float64)
        lanes.append({
            "lane": lane + 1,
            "notes": n,
            "accuracy": float((lc * ACC_WEIGHT).sum() / (300.0 * n)) if n else None,
            "misses": int(lc[5]),
            "extra_presses": int(env.lane_strays[lane]),
            "mean_offset_ms": float(ld.mean()) if len(ld) else None,
        })
    return {
        "accuracy": float(info["accuracy"]),
        "score": int(info["score"]),
        "mods": mods_label(env.mods),
        "counts": counts.tolist(),
        "judgements": dict(zip(JUDGEMENTS, counts.tolist())),
        "total_notes": total,
        "misses": int(counts[5]),
        "hit_rate": float((total - counts[5]) / total) if total else None,
        "extra_presses": int(info["stray_presses"]),
        "broken_holds": int(info["broken_holds"]),
        "holds_total": int(env.holds_total),
        "holds_completed": int(env.holds_ok),
        "hold_accuracy": float(env.holds_ok / env.holds_total) if env.holds_total else None,
        "max_combo": int(info["max_combo"]),
        "mean_offset_ms": float(deltas.mean()) if len(deltas) else None,
        "timing_std_ms": float(deltas.std()) if len(deltas) else None,
        "mean_abs_error_ms": float(np.abs(deltas).mean()) if len(deltas) else None,
        "lanes": lanes,
        "_deltas": deltas.tolist(),  # used for pooling, stripped before saving
    }


def aggregate(results: list[PlayResult]) -> dict[str, Any]:
    """Pool per-map results into one evaluation summary (note-weighted)."""
    if not results:
        return {}
    counts = np.sum([r.metrics["counts"] for r in results], axis=0)
    total = int(sum(r.metrics["total_notes"] for r in results))
    deltas = np.concatenate([np.array(r.metrics["_deltas"]) for r in results]) if results else np.array([])
    holds_total = sum(r.metrics["holds_total"] for r in results)
    holds_ok = sum(r.metrics["holds_completed"] for r in results)
    lanes = []
    for lane in range(4):
        notes = sum(r.metrics["lanes"][lane]["notes"] for r in results)
        weighted = sum((r.metrics["lanes"][lane]["accuracy"] or 0.0) * r.metrics["lanes"][lane]["notes"] for r in results)
        lanes.append({
            "lane": lane + 1,
            "notes": notes,
            "accuracy": weighted / notes if notes else None,
            "misses": sum(r.metrics["lanes"][lane]["misses"] for r in results),
            "extra_presses": sum(r.metrics["lanes"][lane]["extra_presses"] for r in results),
        })
    return {
        "accuracy": float((counts * ACC_WEIGHT).sum() / (300.0 * total)) if total else 0.0,
        "mean_map_accuracy": float(np.mean([r.metrics["accuracy"] for r in results])),
        "counts": counts.tolist(),
        "judgements": dict(zip(JUDGEMENTS, counts.tolist())),
        "total_notes": total,
        "misses": int(counts[5]),
        "hit_rate": float((total - counts[5]) / total) if total else None,
        "extra_presses": int(sum(r.metrics["extra_presses"] for r in results)),
        "broken_holds": int(sum(r.metrics["broken_holds"] for r in results)),
        "holds_total": int(holds_total),
        "hold_accuracy": float(holds_ok / holds_total) if holds_total else None,
        "mean_offset_ms": float(deltas.mean()) if len(deltas) else None,
        "timing_std_ms": float(deltas.std()) if len(deltas) else None,
        "mean_abs_error_ms": float(np.abs(deltas).mean()) if len(deltas) else None,
        "lanes": lanes,
        "maps": [
            {"title": r.beatmap.name, "md5": r.beatmap.md5, "mods": mods_label(r.mods), "accuracy": r.metrics["accuracy"],
             "misses": r.metrics["misses"], "max_combo": r.metrics["max_combo"]}
            for r in results
        ],
        "n_maps": len(results),
    }


def evaluate(
    model,
    maps: list[Beatmap],
    env_cfg: dict[str, Any] | None = None,
    seed: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    mods: int | list[int] = 0,
) -> dict[str, Any]:
    """Play every map once. With a list of mod combinations, map i is played with mods[i % len(mods)]."""
    t0 = time.perf_counter()
    mod_cycle = list(mods) if isinstance(mods, (list, tuple)) else [mods]
    mod_cycle = mod_cycle or [0]
    results = []
    for i, bm in enumerate(maps):
        if progress:
            progress(i, len(maps), bm.name)
        results.append(play_map(model, bm, env_cfg, seed=seed + i, should_stop=should_stop,
                                mods=mod_cycle[i % len(mod_cycle)]))
    summary = aggregate(results)
    summary["seconds"] = round(time.perf_counter() - t0, 2)
    summary["seed"] = seed
    summary["env_config"] = {**ENV_DEFAULTS, **(env_cfg or {})}
    summary["mods"] = mod_list_label(mod_cycle)
    return summary


def strip_private(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


COMPARE_ROWS: list[tuple[str, str, str]] = [
    # (key, label, "higher" | "lower" | "closer0" | "" for neutral)
    ("accuracy", "Accuracy (simulator)", "higher"),
    ("hit_rate", "Notes hit", "higher"),
    ("misses", "Missed notes", "lower"),
    ("extra_presses", "Extra presses", "lower"),
    ("hold_accuracy", "Hold notes completed", "higher"),
    ("broken_holds", "Broken holds", "lower"),
    ("mean_offset_ms", "Mean timing offset (ms)", "closer0"),
    ("timing_std_ms", "Timing deviation (ms)", "lower"),
    ("mean_abs_error_ms", "Mean timing error (ms)", "lower"),
]


@dataclass
class Comparison:
    rows: list[dict[str, Any]] = field(default_factory=list)


def compare(a: dict[str, Any] | None, b: dict[str, Any] | None) -> Comparison:
    """Row-by-row comparison of two evaluation summaries; only metrics both actually have."""
    out = Comparison()
    a, b = a or {}, b or {}
    for key, label, better in COMPARE_ROWS:
        va, vb = a.get(key), b.get(key)
        winner = None
        if va is not None and vb is not None and va != vb and better:
            if better == "higher":
                winner = "a" if va > vb else "b"
            elif better == "lower":
                winner = "a" if va < vb else "b"
            else:
                winner = "a" if abs(va) < abs(vb) else "b"
        out.rows.append({"key": key, "label": label, "a": va, "b": vb, "winner": winner})
    for lane in range(4):
        la = (a.get("lanes") or [{}] * 4)[lane].get("accuracy") if a.get("lanes") else None
        lb = (b.get("lanes") or [{}] * 4)[lane].get("accuracy") if b.get("lanes") else None
        winner = None
        if la is not None and lb is not None and la != lb:
            winner = "a" if la > lb else "b"
        out.rows.append({"key": f"lane{lane + 1}", "label": f"Lane {lane + 1} accuracy", "a": la, "b": lb, "winner": winner})
    return out
