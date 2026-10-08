"""Export the agent's plays as osu! stable replay files (.osr).

Usage:
  python export_replay.py --run runs/demo --maps maps/demo --out replays
  python export_replay.py --run runs/demo --maps "C:/Users/you/AppData/Local/osu!/Songs" --match "Some Song" --tries 5

The beatmap MD5 is written into the replay, so the .osu file you pass in here must be
byte-identical to the one in your osu! library (point --maps at your Songs folder, or
import the .osz files from maps/osz for the demo maps). To watch a replay, double-click
the .osr or drag it onto the osu! window.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from mania_env import ManiaEnv
from maniabot.sb3_compat import load_ppo
from osr_writer import build_osr
from osu_parser import load_maps


def play_once(model, bm, dt_ms: float, jitter_ms: float, seed: int):
    env = ManiaEnv([bm], dt_ms=dt_ms, clip_seconds=None, jitter_ms=jitter_ms, record=True)
    obs, _ = env.reset(seed=seed)
    done, info = False, {}
    while not done:
        action = model.predict(obs, deterministic=False)[0]  # sampled, like training
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    events = list(env.events)
    for lane in range(env.KEYS):  # release anything still held at the end
        if env.keys_down[lane]:
            events.append((env.t, lane, 0))
    return info, events


def safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training folder with model.zip and config.json")
    ap.add_argument("--maps", default="maps/demo", help="folder (searched recursively) or a single .osu file")
    ap.add_argument("--out", default="replays")
    ap.add_argument("--player", default="ManiaBot")
    ap.add_argument("--match", default="", help="only export maps whose title/version/path contains this text")
    ap.add_argument("--limit", type=int, default=None, help="export at most this many maps")
    ap.add_argument("--tries", type=int, default=1, help="play each map this many times and keep the best")
    ap.add_argument("--jitter-ms", type=float, default=None, help="override the training-time timing noise")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = {"dt_ms": 10.0, "jitter_ms": 0.0}
    cfg.update(json.loads((run / "config.json").read_text()))
    jitter = cfg["jitter_ms"] if args.jitter_ms is None else args.jitter_ms
    model = load_ppo(run / "model", device="cpu")

    maps = load_maps(args.maps)
    if args.match:
        needle = args.match.lower()
        maps = [m for m in maps if needle in f"{m.name} {m.artist} {m.path}".lower()]
    if args.limit:
        maps = maps[: args.limit]
    if not maps:
        raise SystemExit("no maps matched")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for bm in maps:
        best = None
        for k in range(args.tries):
            info, events = play_once(model, bm, cfg["dt_ms"], jitter, args.seed + k)
            if best is None or info["accuracy"] > best[0]["accuracy"]:
                best = (info, events)
        info, events = best
        data = build_osr(
            beatmap_md5=bm.md5,
            player=args.player,
            counts=info["counts"],
            score=int(info["accuracy"] * 1_000_000),
            max_combo=info["max_combo"],
            events=events,
        )
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = safe(f"{args.player} - {bm.artist} - {bm.title} [{bm.version}] ({stamp}) Mania.osr")
        (out / fname).write_bytes(data)
        print(f"{info['accuracy'] * 100:6.2f}%  combo {info['max_combo']:4d}  {out / fname}")
        time.sleep(1.0)  # keep timestamps in file names unique


if __name__ == "__main__":
    main()
