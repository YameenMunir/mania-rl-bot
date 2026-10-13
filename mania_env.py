"""Gymnasium environment that simulates osu!mania 4K scoring.

Every tick (dt_ms) the agent chooses which of the 4 keys are held down. A key going
from up to down is a press and is judged against the next unhit note in that lane
using osu!mania hit windows (scaled by the map's OD). Releasing a key ends a hold.

Observation (per lane, flattened):
  n_upcoming coarse times-to-hit (sqrt-scaled so precision is highest near 0),
  n_upcoming "is hold note" flags, key_down, holding, hold_remaining.

Mods (maniabot.mods): DT / NC / HT change the song speed and EZ / HR the hit windows. The
env runs in real time, so with DT the note times are divided by 1.5 and the agent simply sees
denser notes (mania keeps the windows the same size in real time). `mods` is one bit mask, or
a list to pick from at random every episode; reset(options={"mods": ...}) overrides it.

With record=True the env logs every key press/release as (time_ms, lane, down) in
`env.events`, using the same (possibly jittered) timestamps the judging used, converted to
song time (what an .osr replay stores). That log is what export_replay.py turns into an
osu! .osr replay.

For evaluation the env also keeps `hit_log` ((lane, delta_ms, judgement_index) for every
judged press, negative delta = early), `lane_counts` (per-lane judgement counts),
`lane_strays` (presses that hit no note) and `holds_total` (hold notes in the clip).
`scorer` keeps the osu!mania ScoreV1 score (maniabot.scoring) as the play goes.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from maniabot.mods import check_mods, mods_label, speed, window_scale
from maniabot.scoring import ScoreV1
from osu_parser import Beatmap

JUDGEMENTS = ("MAX", "300", "200", "100", "50", "miss")
ACC_WEIGHT = np.array([300, 300, 200, 100, 50, 0], dtype=np.float64)


def hit_windows(od: float) -> np.ndarray:
    """Half-widths in ms for MAX, 300, 200, 100, 50 and miss (osu!mania stable)."""
    return np.array([16.0, 64 - 3 * od, 97 - 3 * od, 127 - 3 * od, 151 - 3 * od, 188 - 3 * od])


class ManiaEnv(gym.Env):
    KEYS = 4

    def __init__(
        self,
        beatmaps: list[Beatmap],
        dt_ms: float = 10.0,
        clip_seconds: float | None = 20.0,
        lookahead_ms: float = 1000.0,
        n_upcoming: int = 4,
        jitter_ms: float = 0.0,
        stray_penalty: float = 0.02,
        record: bool = False,
        mods: int | list[int] = 0,
    ):
        super().__init__()
        if not beatmaps:
            raise ValueError("need at least one beatmap")
        if any(b.keys != self.KEYS for b in beatmaps):
            raise ValueError("ManiaEnv only supports 4K maps")
        self.beatmaps = list(beatmaps)
        self.dt = float(dt_ms)
        self.clip_ms = None if clip_seconds is None else clip_seconds * 1000.0
        self.lookahead = float(lookahead_ms)
        self.n_up = int(n_upcoming)
        self.jitter_ms = float(jitter_ms)
        self.stray_penalty = float(stray_penalty)
        self.record = bool(record)
        self.events: list[tuple[float, int, int]] = []
        self.mod_choices = [check_mods(m) for m in (mods if isinstance(mods, (list, tuple)) else [mods])] or [0]
        self.mods, self.rate = 0, 1.0

        self.n_features = 2 * self.n_up + 3
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.KEYS * self.n_features,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.KEYS)

    # ------------------------------------------------------------------ setup
    def _pick_clip(self, bm: Beatmap):
        """(start, end, notes) of the part to play; start and end are in real time."""
        notes, r = bm.notes, self.rate
        first = notes[0].time / r
        last = max(n.end_time for n in notes) / r
        if self.clip_ms is None:
            return first - 2000.0, last + 1000.0, list(notes)
        lo = first - 1000.0
        hi = max(lo, last - self.clip_ms)
        for _ in range(20):
            start = float(self.np_random.uniform(lo, hi))
            end = start + self.clip_ms
            sel = [n for n in notes if start + 500.0 <= n.time / r and n.end_time / r <= end]
            if len(sel) >= 10:
                return start, end, sel
        end = first + self.clip_ms
        sel = [n for n in notes if n.end_time / r <= end] or list(notes[:10])
        return first - 2000.0, end, sel

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        idx = options.get("map_index")
        if idx is None:
            idx = int(self.np_random.integers(len(self.beatmaps)))
        self.beatmap = self.beatmaps[idx]
        if options.get("mods") is not None:
            self.mods = check_mods(options["mods"])
        elif len(self.mod_choices) == 1:
            self.mods = self.mod_choices[0]
        else:
            self.mods = self.mod_choices[int(self.np_random.integers(len(self.mod_choices)))]
        self.rate = speed(self.mods)
        self.windows = hit_windows(self.beatmap.od) * window_scale(self.mods)

        start, end, sel = self._pick_clip(self.beatmap)
        self.times, self.ends, self.is_hold = [], [], []
        for lane in range(self.KEYS):  # in real time from here on
            ln = [n for n in sel if n.lane == lane]
            self.times.append(np.array([n.time / self.rate for n in ln], dtype=np.float64))
            self.ends.append(np.array([n.end_time / self.rate for n in ln], dtype=np.float64))
            self.is_hold.append(np.array([n.is_hold for n in ln], dtype=bool))
        self.total = len(sel)
        self.holds_total = sum(1 for n in sel if n.is_hold)
        self.scorer = ScoreV1(self.total, self.mods)

        self.t = float(start)
        self.ptr = [0] * self.KEYS
        self.holding = [False] * self.KEYS
        self.hold_end = [0.0] * self.KEYS
        self.keys_down = [0] * self.KEYS
        self.counts = np.zeros(6, dtype=np.int64)
        self.broken = 0
        self.holds_ok = 0
        self.strays = 0
        self.combo = 0
        self.max_combo = 0
        self.events = []
        self.hit_log: list[tuple[int, float, int]] = []
        self.lane_counts = np.zeros((self.KEYS, 6), dtype=np.int64)
        self.lane_strays = [0] * self.KEYS
        self.steps = 0
        self.max_steps = int((end - start) / self.dt) + int(3000 / self.dt)
        return self._obs(), {}

    # ---------------------------------------------------------------- judging
    def _now(self) -> float:
        if self.jitter_ms > 0:
            return self.t + float(self.np_random.normal(0.0, self.jitter_ms))
        return self.t

    def _press(self, lane: int, now: float) -> float:
        ptr = self.ptr[lane]
        times = self.times[lane]
        if ptr >= len(times):
            self.strays += 1
            self.lane_strays[lane] += 1
            return -self.stray_penalty
        delta = now - times[ptr]
        w = self.windows
        if delta < -w[5]:  # too early to count for this note
            self.strays += 1
            self.lane_strays[lane] += 1
            return -self.stray_penalty
        ad = abs(delta)
        if ad <= w[4]:
            j = int(np.argmax(ad <= w[:5]))
        else:
            j = 5  # inside the miss window: counts as a miss and eats the note
        self.counts[j] += 1
        self.lane_counts[lane, j] += 1
        self.scorer.add(j)
        self.hit_log.append((lane, float(delta), j))
        self.ptr[lane] += 1
        if j == 5:
            self.combo = 0
            return -1.0
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)
        if self.is_hold[lane][ptr]:
            self.holding[lane] = True
            self.hold_end[lane] = self.ends[lane][ptr]
        return ACC_WEIGHT[j] / 300.0 + 0.25 * max(0.0, 1.0 - ad / w[1])

    def _release(self, lane: int, now: float) -> float:
        if not self.holding[lane]:
            return 0.0
        self.holding[lane] = False
        if now < self.hold_end[lane] - self.windows[3]:
            self.broken += 1
            return -0.5
        self.holds_ok += 1
        return 0.3

    def _expire(self) -> float:
        r = 0.0
        w5 = self.windows[5]
        for lane in range(self.KEYS):
            times = self.times[lane]
            ptr = self.ptr[lane]
            while ptr < len(times) and self.t > times[ptr] + w5:
                self.counts[5] += 1
                self.lane_counts[lane, 5] += 1
                self.scorer.add(5)
                self.combo = 0
                r -= 1.0
                ptr += 1
            self.ptr[lane] = ptr
            if self.holding[lane] and self.t > self.hold_end[lane] + self.windows[3]:
                self.holding[lane] = False
                self.holds_ok += 1
                r += 0.3
        return r

    # ------------------------------------------------------------------- step
    def step(self, action):
        act = np.asarray(action).reshape(-1)
        reward = 0.0
        for lane in range(self.KEYS):
            down = int(act[lane] > 0)
            was = self.keys_down[lane]
            if down != was:
                now = self._now()
                if self.record:
                    self.events.append((now * self.rate, lane, down))
                reward += self._press(lane, now) if down else self._release(lane, now)
            self.keys_down[lane] = down
        self.t += self.dt
        self.steps += 1
        reward += self._expire()

        done = all(self.ptr[i] >= len(self.times[i]) and not self.holding[i] for i in range(self.KEYS))
        truncated = (not done) and self.steps >= self.max_steps
        info = self._final_info() if (done or truncated) else {}
        return self._obs(), float(reward), bool(done), bool(truncated), info

    def _final_info(self) -> dict:
        counts = self.counts.copy()
        counts[5] += max(0, self.total - int(counts.sum()))
        acc = float((counts * ACC_WEIGHT).sum() / (300.0 * max(1, self.total)))
        return {
            "accuracy": acc,
            "score": self.scorer.value,
            "counts": counts.tolist(),
            "map": self.beatmap.name,
            "mods": mods_label(self.mods),
            "total_notes": self.total,
            "broken_holds": self.broken,
            "stray_presses": self.strays,
            "max_combo": self.max_combo,
        }

    # ------------------------------------------------------------ observation
    def _obs(self) -> np.ndarray:
        K, L = self.n_up, self.lookahead
        obs = np.ones((self.KEYS, self.n_features), dtype=np.float32)
        obs[:, K : 2 * K] = 0.0
        for lane in range(self.KEYS):
            times = self.times[lane]
            p = self.ptr[lane]
            for k in range(K):
                if p + k >= len(times):
                    break
                x = (times[p + k] - self.t) / L
                if x > 1.0:
                    break
                x = max(-1.0, x)
                obs[lane, k] = np.sign(x) * np.sqrt(abs(x))
                obs[lane, K + k] = 1.0 if self.is_hold[lane][p + k] else 0.0
            obs[lane, 2 * K] = self.keys_down[lane]
            obs[lane, 2 * K + 1] = 1.0 if self.holding[lane] else 0.0
            if self.holding[lane]:
                obs[lane, 2 * K + 2] = min(1.0, max(0.0, (self.hold_end[lane] - self.t) / L))
            else:
                obs[lane, 2 * K + 2] = 0.0
        return obs.reshape(-1)
