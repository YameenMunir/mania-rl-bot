"""Train a PPO agent to play osu!mania 4K in the simulator.

Usage:
  python train.py --maps maps/demo --timesteps 300000 --out runs/demo
  python train.py --maps maps/real --jitter-ms 8 --out runs/human_like
  python train.py --maps maps/demo --mods "NM, DT, HR" --out runs/mods   # each play picks one at random

Every finished play (episode) is appended to <out>/episodes.csv, so you can watch the
agent get better with each play. A learning_curve.png is written at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mania_env import ManiaEnv
from maniabot.mods import mods_label, parse_mod_list
from osu_parser import load_maps


class EpisodeLogger(BaseCallback):
    """Logs accuracy of every completed play and prints a rolling average."""

    def __init__(self, csv_path: Path, print_every: int = 100):
        super().__init__()
        self.csv_path = csv_path
        self.print_every = print_every
        self.accs: list[float] = []
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        append = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._file = open(self.csv_path, "a", newline="")
        self._writer = csv.writer(self._file)
        if not append:
            self._writer.writerow(["play", "timesteps", "accuracy", "misses", "broken_holds", "stray_presses", "map"])

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "accuracy" in info:
                self.accs.append(info["accuracy"])
                self._writer.writerow(
                    [len(self.accs), self.num_timesteps, f"{info['accuracy']:.5f}", info["counts"][5],
                     info["broken_holds"], info["stray_presses"], info["map"]]
                )
                if len(self.accs) % self.print_every == 0:
                    recent = np.mean(self.accs[-self.print_every:]) * 100
                    print(f"plays={len(self.accs):6d}  steps={self.num_timesteps:9d}  acc(last {self.print_every})={recent:6.2f}%", flush=True)
        return True

    def _on_training_end(self) -> None:
        if self._file:
            self._file.close()


def plot_curve(csv_path: Path, out_png: Path, window: int = 50) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    acc = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=2)
    if acc.ndim == 0 or len(acc) < 2:
        return
    w = min(window, len(acc))
    smooth = np.convolve(acc, np.ones(w) / w, mode="valid") * 100
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(w, len(acc) + 1), smooth)
    plt.xlabel("play")
    plt.ylabel(f"accuracy % (rolling {w})")
    plt.title("Learning curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="maps/demo")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--dt-ms", type=float, default=10.0)
    ap.add_argument("--clip-seconds", type=float, default=20.0)
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="Gaussian press-timing noise (human-like limit)")
    ap.add_argument("--mods", default="NM", help='mod combinations to practise, e.g. "NM, DT, DTHR" (one per play)')
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="path to a model .zip to continue training")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    maps = load_maps(args.maps)
    print(f"loaded {len(maps)} 4K maps from {args.maps}")

    try:
        mods = parse_mod_list(args.mods)
    except ValueError as exc:
        ap.error(str(exc))
    env_kwargs = dict(dt_ms=args.dt_ms, clip_seconds=args.clip_seconds, jitter_ms=args.jitter_ms)
    (out / "config.json").write_text(json.dumps(
        {**env_kwargs, "maps": args.maps, "mods": [mods_label(m) for m in mods]}, indent=2))

    def make_env():
        return ManiaEnv(maps, mods=mods, **env_kwargs)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls)

    if args.resume:
        from maniabot.sb3_compat import load_ppo

        model = load_ppo(args.resume, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=1024,
            n_epochs=6,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.003,
            policy_kwargs=dict(net_arch=[128, 128]),
            seed=args.seed,
            device="cpu",
            verbose=0,
        )

    callbacks = [
        EpisodeLogger(out / "episodes.csv"),
        CheckpointCallback(save_freq=max(1, 100_000 // args.n_envs), save_path=str(out / "checkpoints"), name_prefix="ppo"),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=not args.resume)
    model.save(out / "model")
    env.close()
    plot_curve(out / "episodes.csv", out / "learning_curve.png")
    print(f"saved model to {out / 'model.zip'}")


if __name__ == "__main__":
    main()
