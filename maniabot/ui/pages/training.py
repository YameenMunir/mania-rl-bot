"""Live training monitor with pause / save checkpoint / stop / cancel, plus session history."""
from __future__ import annotations

from pathlib import Path
from tkinter import ttk

from maniabot.ui import theme
from maniabot.ui.pages.base import Page
from maniabot.ui.widgets import (
    Card, EmptyState, KeyValueGrid, LineChart, Table, ask, badge, fmt_date, fmt_duration, fmt_int, fmt_pct,
)


class TrainingPage(Page):
    title = "Training"
    subtitle = "Live progress of the current training session"

    def build(self) -> None:
        self.empty = EmptyState(self.body, "No training running",
                                "Start a new model, or continue training one you already have.",
                                [("Create New Model", lambda: self.ctx.app.show("create")),
                                 ("Continue Training", lambda: self.ctx.app.show("continue"))])
        self.live = ttk.Frame(self.body)
        self.live.columnconfigure(0, weight=1)

        head = Card(self.live)
        head.grid(row=0, column=0, sticky="ew")
        hb = head.body
        hb.columnconfigure(0, weight=1)
        self.h_title = ttk.Label(hb, text="", style="CardH2.TLabel")
        self.h_title.grid(row=0, column=0, sticky="w")
        self.h_status = badge(hb, "running")
        self.h_status.grid(row=0, column=1, sticky="e")
        self.h_sub = ttk.Label(hb, text="", style="CardMuted.TLabel")
        self.h_sub.grid(row=1, column=0, columnspan=2, sticky="w")
        self.prog = ttk.Progressbar(hb, maximum=1.0)
        self.prog.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 4))
        self.prog_lbl = ttk.Label(hb, text="", style="CardMuted.TLabel")
        self.prog_lbl.grid(row=3, column=0, columnspan=2, sticky="w")
        btns = ttk.Frame(hb, style="Card.TFrame")
        btns.grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.b_pause = ttk.Button(btns, text="Pause", command=self._pause)
        self.b_pause.pack(side="left")
        self.b_ckpt = ttk.Button(btns, text="Save checkpoint now", command=self._checkpoint)
        self.b_ckpt.pack(side="left", padx=6)
        self.b_stop = ttk.Button(btns, text="Stop & save version", command=self._stop)
        self.b_stop.pack(side="left")
        self.b_cancel = ttk.Button(btns, text="Cancel (discard)", style="Danger.TButton", command=self._cancel)
        self.b_cancel.pack(side="left", padx=6)

        self.done_card = Card(self.live)
        self.done_lbl = ttk.Label(self.done_card.body, text="", style="Card.TLabel", wraplength=800, justify="left")
        self.done_lbl.grid(row=0, column=0, sticky="w")
        self.done_btns = ttk.Frame(self.done_card.body, style="Card.TFrame")
        self.done_btns.grid(row=1, column=0, sticky="w", pady=(10, 0))

        stats = Card(self.live, "Progress")
        stats.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.kv = KeyValueGrid(stats.body, columns=3)
        self.kv.grid(row=0, column=0, sticky="ew")
        for key, label, tip in (
            ("steps", "Training step", "Timesteps trained in this session / planned."),
            ("updates", "PPO update", "PPO iterations so far / planned (each update trains on one rollout)."),
            ("total", "Total model steps", "Including all earlier versions in this lineage."),
            ("loss", "Training loss", "PPO total loss of the latest update."),
            ("vloss", "Value loss", None),
            ("entropy", "Entropy", "Negative entropy loss; closer to 0 means the bot is more decisive."),
            ("acc", "Recent accuracy", "Average simulator accuracy of the last 100 practice plays."),
            ("val", "Best checkpoint", "Validation accuracy of the best checkpoint in this session."),
            ("lr", "Learning rate", None),
            ("plays", "Plays (maps processed)", "Practice plays finished / distinct maps seen."),
            ("device", "Device", None),
            ("elapsed", "Elapsed / ETA", None),
            ("latest", "Latest checkpoint", None),
            ("speed", "Speed", None),
        ):
            self.kv.add(key, label, tip=tip)

        charts = ttk.Frame(self.live)
        charts.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        charts.columnconfigure(0, weight=1, uniform="c")
        charts.columnconfigure(1, weight=1, uniform="c")
        c1 = Card(charts, padding=4)
        c1.grid(row=0, column=0, sticky="ew", padx=(0, 7))
        self.acc_chart = LineChart(c1.body, label="Recent accuracy (practice plays)", fmt=lambda v: f"{v * 100:.1f}%")
        self.acc_chart.grid(row=0, column=0, sticky="ew")
        c2 = Card(charts, padding=4)
        c2.grid(row=0, column=1, sticky="ew", padx=(7, 0))
        self.loss_chart = LineChart(c2.body, label="Training loss", colour=theme.INFO)
        self.loss_chart.grid(row=0, column=0, sticky="ew")

        self.ck_card = Card(self.live, "Checkpoints this session")
        self.ck_card.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        self.ck_table = Table(self.ck_card.body, [("file", "Checkpoint", 260), ("steps", "Steps", 100),
                                                  ("val", "Validation", 100), ("when", "Saved", 160), ("note", "", 100)],
                              height=4)
        self.ck_table.grid(row=0, column=0, sticky="ew")

        self.hist = Card(self.body, "Recent training sessions")
        self.hist_table = Table(self.hist.body, [
            ("model", "Model", 160), ("status", "Status", 90), ("from", "From", 80), ("result", "Created", 80),
            ("steps", "Steps", 90), ("started", "Started", 140), ("dur", "Duration", 80), ("err", "Note", 260)],
            height=7)
        self.hist_table.grid(row=0, column=0, sticky="ew")

        self.acc_points: list[tuple[float, float]] = []
        self.loss_points: list[tuple[float, float]] = []
        self._job_id = None
        self.ctx.on("training_event", self._on_event)
        self.ctx.on("training_started", lambda **_: self.on_show())

    # ------------------------------------------------------------ render
    def on_show(self, **_) -> None:
        job = self.ctx.job
        if job is None:
            self.live.grid_forget()
            self.empty.grid(row=0, column=0, sticky="ew")
        else:
            self.empty.grid_forget()
            self.live.grid(row=0, column=0, sticky="ew")
            if self._job_id != job.spec.session_id:
                self._job_id = job.spec.session_id
                self.acc_points, self.loss_points = [], []
                self.ck_table.clear()
                for c in job.checkpoints:
                    self._add_ckpt(c)
            self._render()
        self.hist.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        self._render_history()

    def _on_event(self, event) -> None:
        job = self.ctx.job
        if job is None:
            return
        if self._job_id != job.spec.session_id:
            self.on_show()
            return
        if event["type"] == "progress":
            p = event
            if p.get("recent_accuracy") is not None:
                self.acc_points.append((p["timesteps"], p["recent_accuracy"]))
            if p.get("loss"):
                pt = (p["timesteps"], p["loss"]["loss"])
                if not self.loss_points or self.loss_points[-1] != pt:
                    self.loss_points.append(pt)
        elif event["type"] == "checkpoint":
            self._add_ckpt(event)
        elif event["type"] == "done":
            self._render_history()
        if self.visible():
            self._render()

    def _add_ckpt(self, c) -> None:
        if self.ck_table.tv.exists(c["path"]):
            return
        if c.get("best"):
            for iid in self.ck_table.tv.get_children():
                self.ck_table.tv.set(iid, "note", "")
        self.ck_table.insert(c["path"], [Path(c["path"]).name, fmt_int(c["timesteps"]), fmt_pct(c.get("val_accuracy")),
                                         fmt_date(c.get("created_at")), "best so far" if c.get("best") else ""])

    def _render(self) -> None:
        job = self.ctx.job
        if job is None:
            return
        p = job.last_progress or {}
        spec = job.spec
        parent = spec.source_version or ("checkpoint" if spec.source_checkpoint else None)
        version = p.get("version") or "next version"
        try:
            name = self.ctx.store.get_model(spec.model_id).name
        except Exception:  # noqa: BLE001
            name = spec.model_id
        self.h_title.configure(text=f"{name}  →  {version}")
        self.h_sub.configure(text=(f"Continuing from {parent}" if parent else "New model, initial training")
                             + f"  ·  session {spec.session_id}")
        status = job.status
        colour = theme.BADGE_COLOURS.get(status, theme.INFO)
        self.h_status.configure(text=f" {status} ", fg=colour, highlightbackground=colour)
        done, target = p.get("timesteps", 0), max(1, p.get("target", spec.timesteps))
        self.prog["value"] = min(1.0, done / target)
        self.prog_lbl.configure(text=f"{done / target * 100:.1f}%   ·   {job.message}")
        rollout = spec.n_steps * spec.n_envs
        loss = p.get("loss") or {}
        best = p.get("best_checkpoint")
        el = p.get("elapsed")
        self.kv.set("steps", f"{fmt_int(done)} / {fmt_int(target)}")
        self.kv.set("updates", f"{done // rollout:,} / {-(-target // rollout):,}")
        self.kv.set("total", fmt_int(p.get("total_timesteps")))
        self.kv.set("loss", f"{loss['loss']:.4f}" if loss else "—")
        self.kv.set("vloss", f"{loss['value_loss']:.4f}" if loss else "—")
        self.kv.set("entropy", f"{loss['entropy']:.4f}" if loss else "—")
        self.kv.set("acc", fmt_pct(p.get("recent_accuracy")))
        self.kv.set("val", f"{fmt_pct(best['val_accuracy'])} @ {fmt_int(best['timesteps'])}" if best else "—")
        self.kv.set("lr", f"{p['learning_rate']:.2e}" if p.get("learning_rate") else "—")
        self.kv.set("plays", f"{fmt_int(p.get('plays'))} / {p.get('maps_seen', 0)} maps" if p else "—")
        self.kv.set("device", (p.get("device") or spec.device).upper())
        self.kv.set("elapsed", f"{fmt_duration(el)} / {fmt_duration(p.get('eta'))}" if el is not None else "—")
        self.kv.set("latest", Path(p["latest_checkpoint"]).name if p.get("latest_checkpoint") else "—")
        self.kv.set("speed", f"{p['steps_per_s']:,.0f} steps/s" if p.get("steps_per_s") else "—")
        self.acc_chart.set(self.acc_points)
        self.loss_chart.set(self.loss_points)

        running = not job.finished
        for b in (self.b_pause, self.b_ckpt, self.b_stop, self.b_cancel):
            b.state(["!disabled"] if running else ["disabled"])
        self.b_pause.configure(text="Resume" if status == "paused" else "Pause")
        if job.finished:
            self._render_done(job)
        else:
            self.done_card.grid_forget()

    def _render_done(self, job) -> None:
        r = job.result or {}
        status, version = r.get("status"), r.get("version")
        for w in self.done_btns.winfo_children():
            w.destroy()
        mid = job.spec.model_id
        if version:
            text = f"✓ Training {status}. Saved new version {version}."
            sess = r.get("session") or {}
            try:
                v = self.ctx.store.get_version(mid, version)
                if v.val_accuracy is not None:
                    text += f"  Validation accuracy {fmt_pct(v.val_accuracy)}."
            except Exception:  # noqa: BLE001
                pass
            if sess.get("start_version"):
                text += f"  {sess['start_version']} is unchanged."
            ttk.Button(self.done_btns, text="Test this version", style="Accent.TButton",
                       command=lambda: self.ctx.app.show("test", model_id=mid, version=version, mode="watch")).pack(side="left")
            ttk.Button(self.done_btns, text="Generate replay",
                       command=lambda: self.ctx.app.show("test", model_id=mid, version=version, mode="replay")).pack(
                side="left", padx=6)
            ttk.Button(self.done_btns, text="View version history",
                       command=lambda: self.ctx.app.show("library", model_id=mid, tab="versions")).pack(side="left")
            ttk.Button(self.done_btns, text="Train more",
                       command=lambda: self.ctx.app.show("continue", model_id=mid, version=version)).pack(side="left", padx=6)
        else:
            text = f"Training {status}. No new version was saved."
            if r.get("error"):
                text += f"\n{r['error']}"
            if job.checkpoints or self.ctx.store.list_checkpoints(mid):
                text += "\nCheckpoints from this session are kept; you can continue from one."
                ttk.Button(self.done_btns, text="Continue from a checkpoint",
                           command=lambda: self.ctx.app.show("continue", model_id=mid)).pack(side="left")
        self.done_lbl.configure(text=text, foreground=theme.GOOD if version else (theme.BAD if r.get("error") else theme.TEXT))
        self.done_card.grid(row=1, column=0, sticky="ew", pady=(14, 0))

    def _render_history(self) -> None:
        self.hist_table.clear()
        rows = []
        for m in self.ctx.store.list_models():
            for s in self.ctx.store.list_sessions(m.id):
                rows.append((m, s))
        rows.sort(key=lambda r: r[1].get("started_at") or "", reverse=True)
        for m, s in rows[:25]:
            steps = (s.get("spec") or {}).get("timesteps")
            self.hist_table.insert(f"{m.id}/{s['session_id']}", [
                m.name, s.get("status"), s.get("start_version") or ("—" if s.get("kind") != "new" else "new"),
                s.get("result_version") or "—", fmt_int(steps), fmt_date(s.get("started_at")),
                fmt_duration(s.get("duration_s")), s.get("error") or s.get("notes") or ""],
                tags=("bad",) if s.get("status") in ("failed", "interrupted") else ())

    # ----------------------------------------------------------- controls
    def _pause(self) -> None:
        job = self.ctx.job
        if job.status == "paused":
            job.resume()
        else:
            job.pause()

    def _checkpoint(self) -> None:
        self.ctx.job.save_checkpoint()

    def _stop(self) -> None:
        if ask(self, "Stop training?", "Training stops now and what has been learned so far is validated and saved "
                                       "as a new version. Earlier versions are not affected.", ok="Stop & save"):
            self.ctx.job.stop()

    def _cancel(self) -> None:
        if ask(self, "Cancel training?", "Training stops and NO new version is saved. Checkpoints already written "
                                         "in this session are kept, so you can still continue from them.",
               ok="Cancel training", cancel="Keep training", danger=True):
            self.ctx.job.cancel()
