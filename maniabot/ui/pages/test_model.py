"""Test Model: pick a model, version and 4K map, then watch the bot or generate a replay."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from maniabot import osu_integration
from maniabot.beatmaps import check_beatmap, extract_osz
from maniabot.models import ModelError
from maniabot.ranks import mania_rank
from maniabot.replays import generate_replay
from maniabot.ui import theme
from maniabot.ui.pages.base import Page, version_label
from maniabot.ui.pages.map_browser import MapBrowser
from maniabot.ui.pages.watch import WatchWindow
from maniabot.ui.widgets import Card, Dialog, EmptyState, Table, KeyValueGrid, error, fmt_ms, fmt_pct, message


class TestModelPage(Page):
    title = "Test Model"
    subtitle = "Watch a version play an osu!mania 4K map live, or generate a replay without watching"

    def build(self) -> None:
        self.empty = EmptyState(self.body, "No trained models yet",
                                "Train a model first, then come back to test it on any osu!mania 4K map.",
                                [("Create New Model", lambda: self.ctx.app.show("create"))])
        self.main = ttk.Frame(self.body)
        self.main.columnconfigure(0, weight=1)

        c1 = Card(self.main, "1. Model and version")
        c1.grid(row=0, column=0, sticky="ew")
        b = c1.body
        self.model_var, self.version_var = tk.StringVar(), tk.StringVar()
        self.model_cb = ttk.Combobox(b, textvariable=self.model_var, state="readonly", width=40)
        self.model_cb.grid(row=0, column=0, sticky="w")
        self.version_cb = ttk.Combobox(b, textvariable=self.version_var, state="readonly", width=46)
        self.version_cb.grid(row=0, column=1, sticky="w", padx=10)
        self.model_cb.bind("<<ComboboxSelected>>", lambda _e: self._model_changed())
        self.version_cb.bind("<<ComboboxSelected>>", lambda _e: self._version_changed())
        self.v_info = ttk.Label(b, text="", style="CardMuted.TLabel")
        self.v_info.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        c2 = Card(self.main, "2. osu!mania 4K beatmap")
        c2.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        b = c2.body
        b.columnconfigure(0, weight=1)
        self.map_var = tk.StringVar()
        ttk.Entry(b, textvariable=self.map_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(b, text="Choose .osu…", command=self._browse).grid(row=0, column=1, padx=(6, 0))
        self.songs_btn = ttk.Button(b, text="Browse osu! Songs…", command=lambda: self._browse(songs=True))
        self.songs_btn.grid(row=0, column=2, padx=(6, 0))
        self.recent_var = tk.StringVar()
        self.recent_cb = ttk.Combobox(b, textvariable=self.recent_var, state="readonly")
        self.recent_cb.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.recent_cb.bind("<<ComboboxSelected>>", lambda _e: self.map_var.set(self.recent_var.get()))
        ttk.Separator(b, orient="horizontal").grid(row=2, column=0, columnspan=3, sticky="ew", pady=(14, 10))
        ttk.Label(b, text="Or browse a folder of maps", style="CardH3.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w")
        saved = self.ctx.settings["test_star_range"] or [None, None]
        self.browser = MapBrowser(
            b, self.ctx, on_pick=self.map_var.set, on_change=self._browser_changed,
            folder=self.ctx.settings["test_maps_folder"] or self.ctx.settings["default_maps"],
            star_range=(saved[0], saved[1]))
        self.browser.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.map_status = ttk.Label(b, text="", style="CardMuted.TLabel", wraplength=760, justify="left")
        self.map_status.grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.map_var.trace_add("write", lambda *_: self._check_map())

        c3 = Card(self.main, "3. How do you want to test it?")
        c3.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        b = c3.body
        b.columnconfigure(0, weight=1, uniform="m")
        b.columnconfigure(1, weight=1, uniform="m")
        w = ttk.Frame(b, style="Card.TFrame")
        w.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(w, text="Watch Bot Play", style="CardH3.TLabel").pack(anchor="w")
        ttk.Label(w, text="The bot plays the map live in a 4K playfield on your screen, in real time. Afterwards you "
                          "can save the play and watch it in osu! stable.",
                  style="CardMuted.TLabel", wraplength=360, justify="left").pack(anchor="w", pady=(2, 8))
        sp = ttk.Frame(w, style="Card.TFrame")
        sp.pack(anchor="w")
        ttk.Label(sp, text="Speed", style="Card.TLabel").pack(side="left")
        self.speed = tk.StringVar(value="1x")
        ttk.Combobox(sp, textvariable=self.speed, values=("0.5x", "1x", "1.5x", "2x"), state="readonly", width=6).pack(
            side="left", padx=6)
        self.watch_btn = ttk.Button(w, text="Watch Bot Play", style="Accent.TButton", command=self.watch)
        self.watch_btn.pack(anchor="w", pady=(10, 0))

        g = ttk.Frame(b, style="Card.TFrame")
        g.grid(row=0, column=1, sticky="nsew")
        ttk.Label(g, text="Generate Replay", style="CardH3.TLabel").pack(anchor="w")
        ttk.Label(g, text="Runs the bot without showing it and saves an osu! stable replay (.osr) you can open in "
                          "osu! whenever you like.", style="CardMuted.TLabel", wraplength=360, justify="left").pack(
            anchor="w", pady=(2, 8))
        tr = ttk.Frame(g, style="Card.TFrame")
        tr.pack(anchor="w")
        ttk.Label(tr, text="Attempts (keep best)", style="Card.TLabel").pack(side="left")
        self.tries = tk.StringVar(value="1")
        ttk.Spinbox(tr, from_=1, to=20, textvariable=self.tries, width=5).pack(side="left", padx=6)
        ttk.Label(tr, text="Player name", style="Card.TLabel").pack(side="left", padx=(12, 0))
        self.player = tk.StringVar(value=self.ctx.settings["player_name"])
        ttk.Entry(tr, textvariable=self.player, width=14).pack(side="left", padx=6)
        row = ttk.Frame(g, style="Card.TFrame")
        row.pack(anchor="w", pady=(10, 0))
        self.gen_btn = ttk.Button(row, text="Generate Replay", style="Accent.TButton", command=self.generate)
        self.gen_btn.pack(side="left")
        self.cancel_btn = ttk.Button(row, text="Cancel", command=self._cancel)
        self.gen_prog = ttk.Progressbar(g, mode="indeterminate", length=260)
        self.gen_status = ttk.Label(g, text="", style="CardMuted.TLabel", wraplength=360)
        self.gen_status.pack(anchor="w", pady=(6, 0))

        self.result = Card(self.main, "Replay ready")
        self.res_kv = KeyValueGrid(self.result.body, columns=3)
        self.res_kv.grid(row=0, column=0, sticky="ew")
        for k, l in (("acc", "Accuracy (simulator)"), ("rank", "Rank"), ("judg", "MAX / 300 / 200 / 100 / 50 / miss"), ("combo", "Max combo"),
                     ("extra", "Extra presses"), ("holds", "Hold notes completed"), ("offset", "Mean offset / deviation"),
                     ("time", "Processing time"), ("model", "Model"), ("lanes", "Lane accuracy")):
            self.res_kv.add(k, l)
        self.res_path = ttk.Label(self.result.body, text="", style="CardMuted.TLabel", wraplength=760)
        self.res_path.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.res_hint = ttk.Label(self.result.body, text="", style="CardWarn.TLabel", wraplength=760, justify="left")
        self.res_hint.grid(row=2, column=0, sticky="w")
        ra = ttk.Frame(self.result.body, style="Card.TFrame")
        ra.grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Button(ra, text="Open in osu!", style="Accent.TButton", command=self._open_osu).pack(side="left")
        self.import_btn = ttk.Button(ra, text="Import map into osu!", command=self._import_map)
        self.import_btn.pack(side="left", padx=6)
        ttk.Button(ra, text="Open Replay Folder", command=self._open_folder).pack(side="left")
        ttk.Button(ra, text="Copy Replay Path", command=self._copy_path).pack(side="left", padx=6)
        ttk.Button(ra, text="Test Another Map", command=self._another).pack(side="left")
        ttk.Button(ra, text="Replay History", command=lambda: self.ctx.app.show("replays")).pack(side="left", padx=6)

        self.models = []
        self.versions = []
        self.check = None
        self.last = None
        self._stop = False
        self.ctx.on("models_changed", lambda **_: self._reload_models() if self.visible() else None)
        self.ctx.on("osu", lambda **_: self._osu_changed())

    # ---------------------------------------------------------- selection
    def on_show(self, model_id: str | None = None, version: str | None = None, mode: str | None = None, **_) -> None:
        self._reload_models(model_id, version)
        self._osu_changed()
        self.browser.ensure_scanned()
        recent = [p for p in self.ctx.settings["recent_maps"] if Path(p).exists()]
        self.recent_cb.configure(values=recent)
        self.recent_var.set("Recent maps…" if recent else "No recent maps yet")
        if not self.map_var.get() and recent:
            self.map_var.set(recent[0])
        if mode == "replay":
            self.gen_btn.focus_set()
        elif mode == "watch":
            self.watch_btn.focus_set()

    def _reload_models(self, model_id: str | None = None, version: str | None = None) -> None:
        store = self.ctx.store
        self.models = [m for m in store.list_models() if store.list_versions(m.id)]
        if not self.models:
            self.main.grid_forget()
            self.empty.grid(row=0, column=0, sticky="ew")
            return
        self.empty.grid_forget()
        self.main.grid(row=0, column=0, sticky="ew")
        prev_model = self._model_id_selected
        prev_version = self._version()
        names = [m.name for m in self.models]
        self.model_cb.configure(values=names)
        cur = model_id or prev_model
        idx = next((i for i, m in enumerate(self.models) if m.id == cur), 0)
        self.model_cb.current(idx)
        if version is None and self.models[idx].id == prev_model:
            version = prev_version  # keep the user's choice when the list is refreshed
        self._model_changed(version)

    def _model(self):
        i = self.model_cb.current()
        return self.models[i] if 0 <= i < len(self.models) else None

    @property
    def _model_id_selected(self) -> str | None:
        name = self.model_var.get()
        return next((m.id for m in self.models if m.name == name), None) if name else None

    def _version(self) -> str | None:
        i = self.version_cb.current()
        return self.versions[i].version if 0 <= i < len(self.versions) else None

    def _model_changed(self, version: str | None = None) -> None:
        m = self._model()
        if not m:
            return
        self.versions = self.ctx.store.list_versions(m.id)
        if version and not any(v.version == version for v in self.versions):
            version = None
        self.version_cb.configure(values=[version_label(v) for v in self.versions])
        want = version or m.meta.get("active_version") or self.versions[-1].version
        idx = next((i for i, v in enumerate(self.versions) if v.version == want), len(self.versions) - 1)
        self.version_cb.current(idx)
        self._version_changed()

    def _version_changed(self) -> None:
        i = self.version_cb.current()
        if i < 0:
            return
        v = self.versions[i]
        env = v.meta.get("env_config") or {}
        self.v_info.configure(text=f"Trained {v.meta.get('total_timesteps') or 0:,} timesteps  ·  validation "
                                   f"{fmt_pct(v.val_accuracy)}  ·  timing noise {env.get('jitter_ms', 0):g} ms"
                                   f"  ·  parent {v.parent or 'none'}")
        m = self._model()
        evals = self.ctx.store.list_evaluations(m.id, v.version) if m else []
        self.browser.set_context(v.meta, evals)

    def _browser_changed(self) -> None:
        """Remember the folder and star range between visits."""
        try:
            lo, hi = self.browser.stars.get()
        except ModelError:
            lo = hi = None
        self.ctx.settings["test_maps_folder"] = self.browser.folder.get()
        self.ctx.settings["test_star_range"] = [lo, hi]
        self.ctx.settings.save()

    def _osu_changed(self) -> None:
        o = self.ctx.osu or {}
        if o.get("songs_dir"):
            self.songs_btn.state(["!disabled"])
        else:
            self.songs_btn.state(["disabled"])

    def _browse(self, songs: bool = False) -> None:
        initial = (self.ctx.osu or {}).get("songs_dir") if songs else (
            str(Path(self.map_var.get()).parent) if self.map_var.get() else self.ctx.settings["default_maps"])
        p = filedialog.askopenfilename(title="Choose an osu!mania 4K beatmap", initialdir=initial,
                                       filetypes=[("osu! maps", "*.osu *.osz *.zip"), ("osu! beatmap", "*.osu"),
                                                  ("osu! mapset / map pack", "*.osz *.zip"), ("All files", "*.*")])
        if p and Path(p).suffix.lower() in (".osz", ".zip"):
            p = self._pick_from_archive(p)
        if p:
            self.map_var.set(p)

    def _pick_from_archive(self, archive: str) -> str | None:
        """Unpack an .osz/.zip and let the user choose one of its usable 4K difficulties."""
        try:
            files = extract_osz(archive)
        except Exception as exc:
            error(self, f"Couldn't open this archive: {exc}", "Can't open archive")
            return None
        checks = [c for c in map(check_beatmap, files) if c.ok]
        if not checks:
            error(self, "No usable osu!mania 4K maps in this archive (4K only, at least 50 notes).", "No maps found")
            return None
        if len(checks) == 1:
            return checks[0].path
        d = Dialog(self, "Choose a difficulty", width=720)
        ttk.Label(d.body, text=f"{len(checks)} usable 4K maps in {Path(archive).name}",
                  style="CardH2.TLabel").grid(row=0, column=0, sticky="w")
        t = Table(d.body, [("map", "Map", 480), ("notes", "Notes", 80)], height=12)
        t.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        for i, c in enumerate(checks):
            t.insert(str(i), [c.label, len(c.beatmap.notes)])
        t.select("0")
        chosen: list[str] = []
        close = d.close

        def on_close(value):
            sel = t.selected()  # read before the dialog (and its table) is destroyed
            if value == "ok" and sel is not None:
                chosen.append(sel)
            close(value)

        d.close = on_close
        d.buttons([("Cancel", None, "TButton"), ("Use this map", "ok", "Accent.TButton")])
        d.show()
        return checks[int(chosen[0])].path if chosen else None

    def _check_map(self) -> None:
        path = self.map_var.get().strip()
        self.check = None
        if not path:
            self.map_status.configure(text="Choose a .osu file, or an .osz / .zip to pick a difficulty from.", style="CardMuted.TLabel")
            return
        if Path(path).suffix.lower() in (".osz", ".zip"):
            self.map_status.configure(text="Use Browse… to pick a difficulty from this archive.",
                                      style="CardMuted.TLabel")
            return
        c = check_beatmap(path)
        if not c.ok:
            self.map_status.configure(text=f"✗ {c.reason}", style="CardError.TLabel")
            return
        self.check = c
        bm = c.beatmap
        holds = sum(1 for n in bm.notes if n.is_hold)
        length = (max(n.end_time for n in bm.notes) - bm.notes[0].time) / 1000
        text = (f"✓ osu!mania 4K  ·  {bm.artist} - {bm.title} [{bm.version}]  ·  {len(bm.notes)} notes "
                f"({holds} holds)  ·  OD {bm.od:g}  ·  {length:.0f}s")
        if c.warnings:
            text += "\n" + "\n".join(f"⚠ {w}" for w in c.warnings)
        self.map_status.configure(text=text, style="CardGood.TLabel" if not c.warnings else "CardWarn.TLabel")

    def _ready(self) -> bool:
        if not self._model() or not self._version():
            error(self, "Choose a model and version first.", "No model selected")
            return False
        if not self.check:
            error(self, "Choose a valid osu!mania 4K beatmap first.", "No beatmap selected")
            return False
        self.ctx.settings.add_recent_map(self.check.path)
        self.ctx.settings.save()
        return True

    # ------------------------------------------------------------ actions
    def watch(self) -> None:
        if not self._ready():
            return
        speed = float(self.speed.get().rstrip("x"))
        WatchWindow(self, self.ctx, self._model().id, self._version(), self.check, speed)

    def generate(self) -> None:
        if not self._ready():
            return
        try:
            tries = max(1, int(self.tries.get()))
        except ValueError:
            tries = 1
        player = self.player.get().strip() or "ManiaBot"
        if player != self.ctx.settings["player_name"]:
            self.ctx.settings["player_name"] = player
            self.ctx.settings.save()
        mid, ver, path = self._model().id, self._version(), self.check.path
        self._stop = False
        self.gen_btn.state(["disabled"])
        self.cancel_btn.pack(side="left", padx=6)
        self.gen_prog.pack(anchor="w", pady=(8, 0))
        self.gen_prog.start(12)
        self.result.grid_forget()

        def progress(msg):
            self.ctx.tasks.post(lambda: self.gen_status.configure(text=msg, style="CardMuted.TLabel"))

        self.ctx.tasks.run(lambda: generate_replay(self.ctx.store, self.ctx.history, mid, ver, path, player, tries,
                                                   progress=progress, should_stop=lambda: self._stop),
                           self._generated, self._gen_failed, name="generate-replay")

    def _cancel(self) -> None:
        self._stop = True
        self.gen_status.configure(text="Cancelling…")

    def _gen_done_ui(self) -> None:
        self.gen_prog.stop()
        self.gen_prog.pack_forget()
        self.cancel_btn.pack_forget()
        self.gen_btn.state(["!disabled"])

    def _gen_failed(self, exc, tb) -> None:
        self._gen_done_ui()
        if isinstance(exc, InterruptedError):
            self.gen_status.configure(text="Cancelled. No replay was saved.", style="CardMuted.TLabel")
            return
        msg = str(exc) if isinstance(exc, (ModelError, OSError)) else f"Replay generation failed: {exc}"
        self.gen_status.configure(text=msg, style="CardError.TLabel")
        error(self, msg, "Replay generation failed", tb)

    def _generated(self, rec) -> None:
        self._gen_done_ui()
        self.last = rec
        self.ctx.emit("replays_changed")
        self.gen_status.configure(text="✓ Replay saved.", style="CardGood.TLabel")
        m = rec["metrics"]
        self.res_kv.set("acc", fmt_pct(m["accuracy"]))
        self.res_kv.set("rank", mania_rank(m["accuracy"], m["counts"]) or "—")
        self.res_kv.set("judg", " / ".join(str(c) for c in m["counts"]))
        self.res_kv.set("combo", f"{m['max_combo']:,}")
        self.res_kv.set("extra", f"{m['extra_presses']:,}")
        self.res_kv.set("holds", f"{fmt_pct(m['hold_accuracy'])} of {m['holds_total']}" if m["holds_total"] else "no holds")
        self.res_kv.set("offset", f"{fmt_ms(m['mean_offset_ms'])} / {m['timing_std_ms']:.1f} ms" if m["timing_std_ms"] is not None else "—")
        self.res_kv.set("time", f"{rec['processing_s']:.1f}s")
        self.res_kv.set("model", f"{rec['model_name']} {rec['version']}")
        self.res_kv.set("lanes", "  ".join(fmt_pct(l["accuracy"], 0) for l in m["lanes"]))
        self.res_path.configure(text=rec["path"])
        self.result.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        self._library_hint(rec)

    def _library_hint(self, rec) -> None:
        o = self.ctx.osu or {}
        self.res_hint.configure(text="")
        self.import_btn.state(["!disabled"])
        if not o.get("installed"):
            self.res_hint.configure(text="osu! stable wasn't found on this PC. The replay is saved; set the osu! folder "
                                         "in Settings to open it from here.")
            return
        songs = Path(o["songs_dir"]) if o.get("songs_dir") else None

        def work():
            return osu_integration.map_in_library(rec["beatmap"]["path"], rec["beatmap"]["md5"], songs)

        def done(found):
            if found is False:
                self.res_hint.configure(text="This exact beatmap isn't in your osu! Songs folder yet, so osu! would say "
                                             "the beatmap is missing. Click “Import map into osu!” first.")
            elif found:
                self.res_hint.configure(text="✓ The beatmap is in your osu! library.", foreground=theme.GOOD)

        self.ctx.tasks.run(work, done, name="map-in-library")

    def _open_osu(self) -> None:
        if not self.last:
            return
        try:
            osu_integration.open_in_osu(self.last["path"], self.ctx.settings["osu_path"])
        except osu_integration.OsuError as exc:
            error(self, str(exc), "osu! not available")

    def _import_map(self) -> None:
        if not self.last:
            return
        try:
            osz = osu_integration.importable_mapset(self.last["beatmap"]["path"], self.ctx.storage.cache_dir)
            osu_integration.open_in_osu(osz, self.ctx.settings["osu_path"])
        except osu_integration.OsuError as exc:
            error(self, str(exc), "Can't import the map")
            return
        message(self, "Importing", "osu! is importing the mapset. When it's done, click “Open in osu!” to watch the replay.")

    def _open_folder(self) -> None:
        if self.last:
            osu_integration.open_folder(self.ctx.storage.replays_dir, select=self.last["path"])

    def _copy_path(self) -> None:
        if self.last:
            self.clipboard_clear()
            self.clipboard_append(self.last["path"])
            self.gen_status.configure(text="Replay path copied to the clipboard.", style="CardMuted.TLabel")

    def _another(self) -> None:
        self.result.grid_forget()
        self.map_var.set("")
        self._browse()
