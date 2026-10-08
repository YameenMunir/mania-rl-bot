"""Dashboard: overview, training status, devices, storage and quick actions."""
from __future__ import annotations

from tkinter import ttk

from maniabot.legacy import already_imported, find_legacy_runs, import_legacy_run
from maniabot.models import ModelError
from maniabot.storage import human_size
from maniabot.ui import theme
from maniabot.ui.pages.base import Page
from maniabot.ui.widgets import Card, StatTile, ask, fmt_date, fmt_pct, message


class DashboardPage(Page):
    title = "Dashboard"
    subtitle = "Train, version and test osu!mania 4K bots"

    def build(self) -> None:
        self.action("Create Model", lambda: self.ctx.app.show("create"), accent=True)
        self.action("Continue Training", lambda: self.ctx.app.show("continue"))
        self.action("Test Model", lambda: self.ctx.app.show("test", mode="watch"))
        self.action("Generate Replay", lambda: self.ctx.app.show("test", mode="replay"))
        self.action("Model Library", lambda: self.ctx.app.show("library"))

        self.banner = ttk.Frame(self.body, style="Card.TFrame", padding=14)
        tiles = ttk.Frame(self.body)
        tiles.grid(row=1, column=0, sticky="ew")
        for i in range(4):
            tiles.columnconfigure(i, weight=1, uniform="t")
        self.t_models = StatTile(tiles, "Models")
        self.t_versions = StatTile(tiles, "Model versions")
        self.t_replays = StatTile(tiles, "Generated replays")
        self.t_storage = StatTile(tiles, "Storage used")
        for i, t in enumerate((self.t_models, self.t_versions, self.t_replays, self.t_storage)):
            t.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 10, 0))

        grid = ttk.Frame(self.body)
        grid.grid(row=2, column=0, sticky="nsew", pady=(14, 0))
        grid.columnconfigure(0, weight=1, uniform="g")
        grid.columnconfigure(1, weight=1, uniform="g")

        self.c_training = Card(grid, "Training status")
        self.c_training.grid(row=0, column=0, sticky="nsew", padx=(0, 7), pady=(0, 14))
        self.training_lbl = ttk.Label(self.c_training.body, text="", style="Card.TLabel", wraplength=480, justify="left")
        self.training_lbl.grid(row=0, column=0, sticky="w")
        self.training_prog = ttk.Progressbar(self.c_training.body, maximum=1.0)
        self.training_btn = ttk.Button(self.c_training.body, text="Open training monitor",
                                       command=lambda: self.ctx.app.show("training"))

        self.c_last = Card(grid, "Last trained")
        self.c_last.grid(row=0, column=1, sticky="nsew", padx=(7, 0), pady=(0, 14))
        self.last_lbl = ttk.Label(self.c_last.body, text="", style="Card.TLabel", wraplength=480, justify="left")
        self.last_lbl.grid(row=0, column=0, sticky="w")
        self.last_btns = ttk.Frame(self.c_last.body, style="Card.TFrame")
        self.last_btns.grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.c_best = Card(grid, "Best model", "Highest validation accuracy (simulator) across all models")
        self.c_best.grid(row=1, column=0, sticky="nsew", padx=(0, 7), pady=(0, 14))
        self.best_lbl = ttk.Label(self.c_best.body, text="", style="Card.TLabel", wraplength=480, justify="left")
        self.best_lbl.grid(row=0, column=0, sticky="w")

        self.c_sys = Card(grid, "System")
        self.c_sys.grid(row=1, column=1, sticky="nsew", padx=(7, 0), pady=(0, 14))
        self.sys_lbl = ttk.Label(self.c_sys.body, text="Checking…", style="Card.TLabel", wraplength=480, justify="left")
        self.sys_lbl.grid(row=0, column=0, sticky="w")

        self.c_recent = Card(grid, "Recently tested maps")
        self.c_recent.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.recent_lbl = ttk.Label(self.c_recent.body, text="", style="Card.TLabel", justify="left")
        self.recent_lbl.grid(row=0, column=0, sticky="w")

        for name in ("models_changed", "replays_changed", "device", "osu", "training_started"):
            self.ctx.on(name, lambda **_: self.refresh() if self.visible() else None)
        self.ctx.on("training_event", self._on_training_event)

    def on_show(self, **kwargs) -> None:
        self.refresh()

    def _on_training_event(self, event) -> None:
        if self.visible():
            self._render_training()

    def _render_training(self) -> None:
        job = self.ctx.job
        if job is not None and not job.finished:
            p = job.last_progress
            frac = p["timesteps"] / max(1, p["target"]) if p else 0.0
            acc = fmt_pct(p.get("recent_accuracy")) if p else "—"
            self.training_lbl.configure(
                text=f"● {job.message}\n{job.spec.model_id} → {p.get('version', '…') if p else '…'}"
                     f"   ·   {frac * 100:.0f}%   ·   recent accuracy {acc}", foreground=theme.ACCENT)
            self.training_prog.grid(row=1, column=0, sticky="ew", pady=(10, 0))
            self.training_prog["value"] = frac
            self.training_btn.grid(row=2, column=0, sticky="w", pady=(10, 0))
        else:
            last = ""
            if job is not None and job.result:
                v = job.result.get("version")
                last = f"\nLast session: {job.result.get('status')}" + (f", saved {v}" if v else "")
            self.training_lbl.configure(text="Idle — no training running." + last, foreground=theme.TEXT)
            self.training_prog.grid_forget()
            self.training_btn.grid_forget()

    def refresh(self) -> None:
        self._render_training()
        store = self.ctx.store
        self._legacy_banner()

        def gather():
            models = store.list_models()
            versions = {m.id: store.list_versions(m.id) for m in models}
            usage = self.ctx.storage.usage()
            replays = self.ctx.history.list()
            return models, versions, usage, replays

        self.ctx.tasks.run(gather, self._render, name="dashboard")
        self._render_system()

    def _render(self, data) -> None:
        models, versions, usage, replays = data
        n_versions = sum(len(v) for v in versions.values())
        self.t_models.set(str(len(models)), f"{sum(1 for m in models if m.archived)} archived")
        self.t_versions.set(str(n_versions))
        self.t_replays.set(str(len(replays)))
        self.t_storage.set(human_size(sum(usage.values())), f"models {human_size(usage['models'])}")

        for w in self.last_btns.winfo_children():
            w.destroy()
        trained = [m for m in models if m.meta.get("last_trained_at")]
        if trained:
            m = max(trained, key=lambda r: r.meta["last_trained_at"])
            v = m.meta.get("latest_version")
            vr = next((x for x in versions[m.id] if x.version == v), None)
            acc = fmt_pct(vr.val_accuracy) if vr else "—"
            self.last_lbl.configure(text=f"{m.name}  ·  {v}\nTrained {fmt_date(m.meta['last_trained_at'])}\n"
                                         f"Validation accuracy {acc}")
            ttk.Button(self.last_btns, text="Test it", command=lambda: self.ctx.app.show(
                "test", model_id=m.id, version=v, mode="watch")).pack(side="left")
            ttk.Button(self.last_btns, text="Continue training", command=lambda: self.ctx.app.show(
                "continue", model_id=m.id, version=v)).pack(side="left", padx=6)
        else:
            self.last_lbl.configure(text="No models trained yet. Create one to get started.")
            ttk.Button(self.last_btns, text="Create New Model", style="Accent.TButton",
                       command=lambda: self.ctx.app.show("create")).pack(side="left")

        best = None
        for m in models:
            for v in versions[m.id]:
                if v.val_accuracy is not None and (best is None or v.val_accuracy > best[1].val_accuracy):
                    best = (m, v)
        if best:
            m, v = best
            maps = len(v.meta.get("validation_maps") or [])
            held = "held-out" if (v.meta.get("validation") or {}).get("heldout") else "training"
            self.best_lbl.configure(text=f"{m.name}  ·  {v.version}\n{fmt_pct(v.val_accuracy)} on {maps} {held} map(s)")
        else:
            self.best_lbl.configure(text="No validated versions yet.")

        seen, lines = set(), []
        for r in replays:
            title = (r.get("beatmap") or {}).get("title")
            key = (r.get("beatmap") or {}).get("md5") or title
            if not title or key in seen:
                continue
            seen.add(key)
            bm = r["beatmap"]
            acc = fmt_pct((r.get("metrics") or {}).get("accuracy"))
            lines.append(f"{bm.get('artist', '')} - {title} [{bm.get('version', '')}]   ·   "
                         f"{r.get('model_name')} {r.get('version') or ''}   ·   {acc}   ·   {fmt_date(r.get('created_at'))}")
            if len(lines) == 6:
                break
        self.recent_lbl.configure(text="\n".join(lines) or "Nothing tested yet.")

    def _render_system(self) -> None:
        d, o = self.ctx.device, self.ctx.osu
        dev = d["message"] if d else "Checking compute devices…"
        if d and d.get("torch_version"):
            dev += f"  (PyTorch {d['torch_version']})"
        if o is None:
            osu = "Checking for osu! stable…"
        elif o["installed"]:
            osu = f"osu! stable found{' and running' if o['running'] else ''}: {o['exe']}"
        else:
            osu = "osu! stable not found. Replays are still saved; set its folder in Settings to open them in osu!."
        self.sys_lbl.configure(text=f"{dev}\n{osu}\nData folder: {self.ctx.storage.root}")

    # ------------------------------------------------------- legacy import
    def _legacy_banner(self) -> None:
        for w in self.banner.winfo_children():
            w.destroy()
        runs = [r for r in find_legacy_runs() if not already_imported(self.ctx.store, r)]
        if not runs or self.ctx.settings["legacy_import_dismissed"]:
            self.banner.grid_forget()
            return
        self.banner.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.banner.columnconfigure(0, weight=1)
        ttk.Label(self.banner, text=f"Found {len(runs)} model(s) trained with the old train.py script in runs/: "
                                    f"{', '.join(r.name for r in runs)}", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self.banner, text="Import copies them into the Model Library as v001. The original files are not changed.",
                  style="CardMuted.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Button(self.banner, text="Import", style="Accent.TButton", command=lambda: self._import(runs)).grid(
            row=0, column=1, rowspan=2, padx=6)
        ttk.Button(self.banner, text="Dismiss", command=self._dismiss).grid(row=0, column=2, rowspan=2)

    def _import(self, runs) -> None:
        def work():
            done = []
            for r in runs:
                done.append(import_legacy_run(self.ctx.store, r).name)
            return done

        def ok(names):
            message(self, "Imported", f"Imported: {', '.join(names)}. They are in the Model Library.", "success")
            self.ctx.emit("models_changed")

        def fail(exc, tb):
            message(self, "Import failed", str(exc) if isinstance(exc, ModelError) else f"{exc}", "error", tb)

        self.ctx.tasks.run(work, ok, fail, name="legacy-import")

    def _dismiss(self) -> None:
        if ask(self, "Hide this notice?", "You can still import old runs later from Settings → Import."):
            self.ctx.settings["legacy_import_dismissed"] = True
            self.ctx.settings.save()
            self.refresh()
