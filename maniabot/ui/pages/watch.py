"""Watch Bot Play: the model plays the map live, in real time, in a local 4K playfield.

Inference runs step-by-step on the Tk thread, paced to the wall clock, using the same
simulator and hit windows as training. When the play ends (or is stopped) it can be saved
as an .osr replay and opened in osu! stable.
"""
from __future__ import annotations

import logging
import time
import tkinter as tk
from tkinter import ttk

from maniabot import osu_integration
from maniabot.evaluation import POLICIES, PlayResult, check_compatible, compute_metrics, final_events, make_play_env
from maniabot.logs import event
from maniabot.models import IN_USE
from maniabot.ranks import RANK_COLOURS, mania_rank
from maniabot.replays import write_replay
from maniabot.ui import theme
from maniabot.ui.widgets import error, fmt_pct, message

log = logging.getLogger("maniabot.watch")

JUDGE_COLOURS = ("#9ef0ff", "#ffe066", "#7cf29c", "#6cb6ff", "#c792ea", "#ff6b6b")
JUDGE_TEXT = ("MAX", "300", "200", "100", "50", "MISS")


def _dim(hex_colour: str, f: float) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}"


class WatchWindow(tk.Toplevel):
    LANE_W = 78  # at 96 DPI; scaled in __init__
    HIT_Y_FROM_BOTTOM = 110
    SCROLL = 0.9  # px per ms

    def __init__(self, parent, ctx, model_id: str, version: str, check, speed: float = 1.0):
        super().__init__(parent)
        self.ctx, self.model_id, self.version, self.check = ctx, model_id, version, check
        self.bm = check.beatmap
        self.speed = speed
        self.title(f"Watch Bot Play — {self.bm.name}")
        self.configure(bg=theme.BG)
        k = theme.SCALE
        self.LANE_W, self.HIT_Y_FROM_BOTTOM, self.SCROLL = int(78 * k), int(110 * k), 0.9 * k
        self.geometry(f"{int(760 * k)}x{min(int(820 * k), self.winfo_screenheight() - 80)}")
        self.minsize(int(640 * k), int(560 * k))
        self.protocol("WM_DELETE_WINDOW", self.stop_and_close)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        head = ttk.Frame(self, padding=(16, 12))
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(0, weight=1)
        rec = ctx.store.get_model(model_id)
        ttk.Label(head, text=f"{rec.name} {version}  ·  {self.bm.artist} - {self.bm.title} [{self.bm.version}]",
                  style="H2.TLabel").grid(row=0, column=0, sticky="w")
        self.status = ttk.Label(head, text="Loading model…", style="Muted.TLabel")
        self.status.grid(row=1, column=0, sticky="w")
        self.canvas = tk.Canvas(self, bg="#0b0b10", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        foot = ttk.Frame(self, padding=(16, 10))
        foot.grid(row=2, column=0, sticky="ew")
        self.btn_stop = ttk.Button(foot, text="Stop", style="Danger.TButton", command=self.stop)
        self.btn_stop.pack(side="left")
        self.btn_save = ttk.Button(foot, text="Save as replay", command=self.save_replay)
        self.btn_osu = ttk.Button(foot, text="Save & watch in osu!", style="Accent.TButton", command=self.save_and_open)
        self.osu_lbl = ttk.Label(foot, text="", style="Muted.TLabel")
        self.osu_lbl.pack(side="right")
        self._osu_status()

        self.running = False
        self.finished = False
        self.stopped_early = False
        self.rank: str | None = None  # set when a full play finishes
        self.saved: dict | None = None
        self.token = IN_USE.acquire(model_id, version, "watching a live test")
        self.last_counts = None
        self.flash: tuple[int, float] | None = None
        self.after(50, self._load)

    def _osu_status(self) -> None:
        o = self.ctx.osu
        if o is None:
            text = ""
        elif not o["installed"]:
            text = "osu! stable not found (replay will still be saved)"
        else:
            text = "osu! stable is running" if o["running"] else "osu! stable installed (not running)"
        self.osu_lbl.configure(text=text)

    def _load(self) -> None:
        def work():
            v = self.ctx.store.get_version(self.model_id, self.version)
            model = POLICIES.load(v.weights)
            env = make_play_env(self.bm, v.meta.get("env_config"), record=True)
            check_compatible(model, env)
            return model, env

        self.ctx.tasks.run(work, self._start, self._load_failed, name="watch-load")

    def _load_failed(self, exc, tb) -> None:
        self._release()
        if self.winfo_exists():
            self.destroy()
        error(self.master, str(exc), "Couldn't start the test", tb)

    def _start(self, loaded) -> None:
        if not self.winfo_exists():
            return
        self.model, self.env = loaded
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(0)
        self.obs, _ = self.env.reset(seed=0)
        self.info = {}
        self.wall0 = time.perf_counter()
        self.t0 = self.env.t
        self.last_counts = self.env.counts.copy()
        self.running = True
        event(log, "watch.started", model=self.model_id, version=self.version, map=self.bm.name)
        self._tick()

    # ---------------------------------------------------------- playing
    def _tick(self) -> None:
        if not self.running or not self.winfo_exists():
            return
        target = self.t0 + (time.perf_counter() - self.wall0) * 1000.0 * self.speed
        steps = 0
        env = self.env
        while env.t < target and steps < 400:
            action = self.model.predict(self.obs, deterministic=False)[0]
            self.obs, _r, term, trunc, info = env.step(action)
            steps += 1
            if term or trunc:
                self.info = info
                self._finish()
                break
        self._judgement_flash()
        self._draw()
        if self.running:
            self.after(15, self._tick)

    def _judgement_flash(self) -> None:
        diff = self.env.counts - self.last_counts
        if diff.any():
            j = int(max(i for i in range(6) if diff[i] > 0))
            self.flash = (j, time.perf_counter())
        self.last_counts = self.env.counts.copy()

    def _acc(self) -> float:
        from mania_env import ACC_WEIGHT

        judged = int(self.env.counts.sum())
        return float((self.env.counts * ACC_WEIGHT).sum() / (300.0 * judged)) if judged else 1.0

    def _draw(self) -> None:
        c, env = self.canvas, self.env
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        lw = self.LANE_W
        x0 = (w - 4 * lw) / 2
        hit_y = h - self.HIT_Y_FROM_BOTTOM
        c.create_rectangle(x0, 0, x0 + 4 * lw, h, fill="#111118", outline="")
        for i in range(5):
            c.create_line(x0 + i * lw, 0, x0 + i * lw, h, fill="#24243a")
        t = env.t
        for lane in range(4):
            lx = x0 + lane * lw
            col = theme.LANE_COLOURS[lane]
            if env.holding[lane]:  # body of the hold currently being held
                top = hit_y - (env.hold_end[lane] - t) * self.SCROLL
                c.create_rectangle(lx + 18, max(0, top), lx + lw - 18, hit_y, fill=_dim(col, 0.75), outline="")
            times, ends, holds = env.times[lane], env.ends[lane], env.is_hold[lane]
            for k in range(env.ptr[lane], len(times)):
                y = hit_y - (times[k] - t) * self.SCROLL
                if y < -40:
                    break
                if holds[k]:
                    ytop = hit_y - (ends[k] - t) * self.SCROLL
                    c.create_rectangle(lx + 18, max(-10, ytop), lx + lw - 18, y, fill=_dim(col, 0.45), outline="")
                c.create_rectangle(lx + 4, y - 12, lx + lw - 4, y + 2, fill=col, outline="")
            down = env.keys_down[lane]
            c.create_rectangle(lx + 3, hit_y, lx + lw - 3, hit_y + theme.px(26), outline=col, width=2,
                               fill=col if down else "#181822")
            c.create_text(lx + lw / 2, hit_y + theme.px(44), text=f"Lane {lane + 1}", fill=theme.MUTED,
                          font=theme.fonts()["small"])
        if self.flash and time.perf_counter() - self.flash[1] < 0.45:
            j = self.flash[0]
            c.create_text(w / 2, hit_y - theme.px(150), text=JUDGE_TEXT[j], fill=JUDGE_COLOURS[j], font=(theme.FAMILY, 26, "bold"))
        c.create_text(w / 2, hit_y - theme.px(200), text=f"{env.combo}x" if env.combo else "", fill=theme.TEXT,
                      font=(theme.FAMILY, 18, "bold"))
        counts = " ".join(f"{JUDGE_TEXT[i]} {int(env.counts[i])}" for i in range(6))
        c.create_text(theme.px(16), theme.px(12), anchor="nw", text=f"{self._acc() * 100:.2f}%", fill=theme.TEXT,
                      font=(theme.FAMILY, 22, "bold"))
        c.create_text(theme.px(16), theme.px(58), anchor="nw", text=counts, fill=theme.MUTED,
                      font=theme.fonts()["small"])
        if self.rank:
            cx, cy = w / 2, h * 0.24
            c.create_text(cx, cy - theme.px(62), text="RANK", fill=theme.MUTED, font=(theme.FAMILY, 12, "bold"))
            c.create_text(cx, cy, text=self.rank, fill=RANK_COLOURS[self.rank], font=(theme.FAMILY, 72, "bold"))
            c.create_text(cx, cy + theme.px(62), text=f"{self._acc() * 100:.2f}% accuracy", fill=theme.TEXT,
                          font=(theme.FAMILY, 13))
            c.create_text(cx, cy + theme.px(88), text="simulator rank; osu! may differ slightly", fill=theme.FAINT,
                          font=theme.fonts()["small"])
        last = max((e[-1] for e in env.ends if len(e)), default=1.0)
        first = min((x[0] for x in env.times if len(x)), default=0.0)
        frac = min(1.0, max(0.0, (t - first) / max(1.0, last - first)))
        c.create_rectangle(0, h - 4, w * frac, h, fill=theme.ACCENT, outline="")
        if self.running:
            self.status.configure(text=f"Bot playing live (simulator, {self.speed:g}x)  ·  "
                                       f"{max(0.0, (t - first) / 1000):.1f}s / {(last - first) / 1000:.1f}s")

    def _finish(self) -> None:
        self.running = False
        self.finished = True
        self.btn_stop.pack_forget()
        self.btn_save.pack(side="left")
        self.btn_osu.pack(side="left", padx=6)
        acc = self.info.get("accuracy")
        counts = self.info.get("counts")
        self.rank = None if self.stopped_early else mania_rank(acc, counts)
        shown = f"Rank {self.rank}  ·  " if self.rank else ""
        self.status.configure(text=f"Finished  ·  {shown}accuracy {fmt_pct(acc)} (simulator)  ·  "
                                   f"max combo {self.info.get('max_combo')}  ·  {(counts or [0] * 6)[5]} misses")
        self._release()
        event(log, "watch.finished", model=self.model_id, version=self.version, acc=acc, rank=self.rank)
        if self.winfo_exists():
            self._draw()  # show the rank on the final frame

    def stop(self) -> None:
        """Stop early: the partial play is judged up to this point and can still be saved."""
        if not self.running:
            return
        self.running = False
        self.stopped_early = True  # a partial play gets no rank
        self.info = self.env._final_info()
        self.status.configure(text="Stopped.")
        self._finish()
        self.status.configure(text=f"Stopped early  ·  accuracy so far {fmt_pct(self.info['accuracy'])} "
                                   "(unplayed notes count as misses)")

    def _result(self) -> PlayResult:
        return PlayResult(self.bm, self.info, final_events(self.env), compute_metrics(self.env, self.info))

    def save_replay(self) -> dict | None:
        if self.saved:
            return self.saved
        try:
            self.saved = write_replay(self.ctx.store, self.ctx.history, self.model_id, self.version, self._result(),
                                      self.ctx.settings["player_name"], kind="watched")
        except Exception as exc:  # noqa: BLE001
            error(self, str(exc), "Couldn't save the replay")
            return None
        self.ctx.emit("replays_changed")
        self.btn_save.configure(text="Saved ✓")
        self.btn_save.state(["disabled"])
        return self.saved

    def save_and_open(self) -> None:
        rec = self.save_replay()
        if not rec:
            return
        try:
            osu_integration.open_in_osu(rec["path"], self.ctx.settings["osu_path"])
        except osu_integration.OsuError as exc:
            message(self, "osu! not available", f"{exc}\n\nThe replay was saved to:\n{rec['path']}", "warning")

    def _release(self) -> None:
        if self.token is not None:
            IN_USE.release(self.token)
            self.token = None

    def stop_and_close(self) -> None:
        self.running = False
        self._release()
        self.destroy()
