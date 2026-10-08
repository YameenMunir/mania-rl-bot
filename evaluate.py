"""Evaluate a trained agent (or the scripted baseline) on full maps.

Usage:
  python evaluate.py --run runs/demo --maps maps/demo
  python evaluate.py --baseline --maps maps/demo          # sanity check of the simulator
  python evaluate.py --baseline --jitter-ms 8 --maps maps/demo   # ceiling with human-like noise
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mania_env import ACC_WEIGHT, JUDGEMENTS, ManiaEnv
from osu_parser import load_maps


def scripted_action(env: ManiaEnv) -> np.ndarray:
    """Oracle that peeks at the note times and presses at the ideal moment."""
    act = np.zeros(env.KEYS, dtype=np.int8)
    half = env.dt / 2
    for lane in range(env.KEYS):
        if env.holding[lane]:
            act[lane] = 1 if env.hold_end[lane] - env.t > half else 0
        elif env.keys_down[lane]:
            act[lane] = 0  # release first so the next note can be pressed
        else:
            p = env.ptr[lane]
            if p < len(env.times[lane]) and env.times[lane][p] - env.t <= half:
                act[lane] = 1
    return act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="maps/demo")
    ap.add_argument("--run", default=None, help="training output folder containing model.zip and config.json")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--jitter-ms", type=float, default=None)
    ap.add_argument("--dt-ms", type=float, default=None)
    args = ap.parse_args()
    if not args.run and not args.baseline:
        ap.error("pass --run <folder> or --baseline")

    cfg = {"dt_ms": 10.0, "jitter_ms": 0.0}
    model = None
    if args.run:
        from stable_baselines3 import PPO

        cfg.update(json.loads((Path(args.run) / "config.json").read_text()))
        model = PPO.load(Path(args.run) / "model", device="cpu")
    if args.jitter_ms is not None:
        cfg["jitter_ms"] = args.jitter_ms
    if args.dt_ms is not None:
        cfg["dt_ms"] = args.dt_ms

    maps = load_maps(args.maps)
    env = ManiaEnv(maps, dt_ms=cfg["dt_ms"], clip_seconds=None, jitter_ms=cfg["jitter_ms"])

    totals = np.zeros(6, dtype=np.int64)
    accs = []
    print(f"{'map':32s} {'acc%':>7s}  " + " ".join(f"{j:>5s}" for j in JUDGEMENTS))
    for i in range(len(maps)):
        obs, _ = env.reset(seed=i, options={"map_index": i})
        done = False
        info = {}
        while not done:
            # Sample actions (like training). Taking the most likely action per tick would
            # never press, because a press is a rare event (p < 0.5 on almost every tick).
            action = scripted_action(env) if model is None else model.predict(obs, deterministic=False)[0]
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        counts = np.array(info["counts"])
        totals += counts
        accs.append(info["accuracy"])
        print(f"{info['map'][:32]:32s} {info['accuracy'] * 100:7.2f}  " + " ".join(f"{c:5d}" for c in counts))
    overall = (totals * ACC_WEIGHT).sum() / (300.0 * totals.sum()) * 100
    print(f"\nmean accuracy {np.mean(accs) * 100:.2f}%  (note-weighted {overall:.2f}%)")


if __name__ == "__main__":
    main()
