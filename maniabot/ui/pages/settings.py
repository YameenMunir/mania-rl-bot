"""Settings: storage (location, usage, migration, cleanup), osu! stable, defaults, imports, logs."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from maniabot import REPO_ROOT, osu_integration
from maniabot.evaluation import POLICIES
from maniabot.legacy import already_imported, find_legacy_runs, import_legacy_run
from maniabot.storage import Storage, base_root, dir_size, human_size, migrate_storage, validate_storage_dir
from maniabot.ui.pages.base import DEVICE_CHOICES, MapsFolderField, Page, form_row
from maniabot.ui.widgets import Card, Table, ask, error, message


class SettingsPage(Page):
    title = "Settings"
    subtitle = "Storage, osu! stable and defaults"

    def build(self) -> None:
        s = self.ctx.settings
        # ----------------------------------------------------------- storage
        st = Card(self.body, "Storage", "Models, versions, checkpoints, replays and logs live here, outside the "
                                        "source-code repository, so they never end up in Git.")
        st.grid(row=0, column=0, sticky="ew")
        b = st.body
        b.columnconfigure(1, weight=1)
        self.path_lbl = ttk.Label(b, text="", style="Card.TLabel")
        form_row(b, 0, "Data folder", self.path_lbl)
        self.models_lbl = ttk.Label(b, text="", style="CardMuted.TLabel")
        form_row(b, 1, "Model storage", self.models_lbl)
        row = ttk.Frame(b, style="Card.TFrame")
        row.grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Button(row, text="Open Storage Folder", command=lambda: osu_integration.open_folder(self.ctx.storage.root)).pack(side="left")
        ttk.Button(row, text="View Storage Usage", command=self.refresh_usage).pack(side="left", padx=6)
        ttk.Button(row, text="Change Location…", command=self.change_location).pack(side="left")
        ttk.Button(row, text="Clean Temporary Files", command=self.clean_tmp).pack(side="left", padx=6)
        self.usage_lbl = ttk.Label(b, text="", style="CardMuted.TLabel", justify="left")
        self.usage_lbl.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self.usage = Table(b, [("model", "Model", 220), ("versions", "Versions", 80), ("ckpts", "Checkpoints", 90),
                               ("archived", "Archived", 70), ("size", "Size", 90), ("vsizes", "Per version", 330)], height=6)
        self.usage.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(b, text="To free space, open a model in the Model Library → Storage tab. Automatic cleanup never "
                          "deletes versions, the active/best version, protected versions or checkpoints.",
                  style="CardMuted.TLabel", wraplength=800).grid(row=5, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # -------------------------------------------------------------- osu!
        osu = Card(self.body, "osu! stable", "Used only to open replays (.osr) and import mapsets (.osz). Optional.")
        osu.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        b = osu.body
        b.columnconfigure(1, weight=1)
        self.osu_path = tk.StringVar(value=s["osu_path"] or "")
        r = ttk.Frame(b, style="Card.TFrame")
        r.columnconfigure(0, weight=1)
        ttk.Entry(r, textvariable=self.osu_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(r, text="Browse…", command=self.browse_osu).grid(row=0, column=1, padx=6)
        ttk.Button(r, text="Detect", command=self.detect_osu).grid(row=0, column=2)
        form_row(b, 0, "osu! folder", r, "Leave empty to auto-detect (%LOCALAPPDATA%\\osu! or the osu:// file association).")
        self.osu_lbl = ttk.Label(b, text="", style="CardMuted.TLabel", wraplength=700, justify="left")
        self.osu_lbl.grid(row=1, column=1, sticky="w")

        # ---------------------------------------------------------- defaults
        df = Card(self.body, "Defaults")
        df.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        b = df.body
        b.columnconfigure(1, weight=1)
        self.player = tk.StringVar(value=s["player_name"])
        form_row(b, 0, "Replay player name", ttk.Entry(b, textvariable=self.player, width=24))
        self.maps = MapsFolderField(b, self.ctx, s["default_maps"])
        form_row(b, 1, "Default beatmap folder", self.maps)
        self.device = tk.StringVar(value=s["default_device"])
        form_row(b, 2, "Default device", ttk.Combobox(b, textvariable=self.device, values=DEVICE_CHOICES,
                                                       state="readonly", width=10))
        self.device_lbl = ttk.Label(b, text="", style="CardMuted.TLabel")
        self.device_lbl.grid(row=3, column=1, sticky="w")
        ttk.Button(b, text="Save", style="Accent.TButton", command=self.save).grid(row=4, column=1, sticky="w", pady=(10, 0))

        # ----------------------------------------------------------- imports
        im = Card(self.body, "Import from the old command-line scripts",
                  "Copies models from runs/ and replays from replays/ in the project folder into your library. "
                  "The originals are left untouched.")
        im.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        r = ttk.Frame(im.body, style="Card.TFrame")
        r.grid(row=0, column=0, sticky="w")
        ttk.Button(r, text="Import models from runs/", command=self.import_runs).pack(side="left")
        ttk.Button(r, text="Import replays from replays/", command=self.import_replays).pack(side="left", padx=6)

        lg = Card(self.body, "Logs", "Technical details of errors are written here.")
        lg.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(lg.body, text="Open Logs Folder", command=lambda: osu_integration.open_folder(self.ctx.storage.logs_dir)).grid(
            row=0, column=0, sticky="w")
        self.ctx.on("osu", lambda **_: self._render_osu())
        self.ctx.on("device", lambda **_: self._render_device())

    def on_show(self, **_) -> None:
        self.path_lbl.configure(text=str(self.ctx.storage.root))
        self.models_lbl.configure(text=str(self.ctx.store.root))
        self._render_osu()
        self._render_device()
        self.refresh_usage()

    def _render_device(self) -> None:
        d = self.ctx.device
        self.device_lbl.configure(text=d["message"] if d else "Checking…")

    def _render_osu(self) -> None:
        o = self.ctx.osu
        if o is None:
            self.osu_lbl.configure(text="Checking…")
        elif o["installed"]:
            self.osu_lbl.configure(text=f"Found {o['exe']}\nSongs folder: {o['songs_dir'] or 'not found'}\n"
                                        f"{'Running now' if o['running'] else 'Not running'}")
        else:
            self.osu_lbl.configure(text="osu! stable not found. Everything else still works; replays are saved to the "
                                        "data folder and can be opened later.")

    # ----------------------------------------------------------- storage
    def refresh_usage(self) -> None:
        store = self.ctx.store

        def work():
            usage = self.ctx.storage.usage()
            rows = []
            for m in store.list_models():
                vs = store.list_versions(m.id)
                rows.append((m, vs, len(store.list_checkpoints(m.id)), dir_size(m.path),
                             [(v.version, dir_size(v.path)) for v in vs]))
            return usage, rows, len(self.ctx.history.list())

        def done(res):
            usage, rows, n_replays = res
            total = sum(usage.values())
            n_ckpt = sum(r[2] for r in rows)
            n_arch = sum(1 for r in rows if r[0].archived)
            self.usage_lbl.configure(text=(
                f"Total {human_size(total)}  ·  models {human_size(usage['models'])}  ·  replays {human_size(usage['replays'])} "
                f"({n_replays})  ·  logs {human_size(usage['logs'])}  ·  temp/cache {human_size(usage['cache'])}\n"
                f"{len(rows)} model(s), {sum(len(r[1]) for r in rows)} version(s), {n_ckpt} checkpoint(s), {n_arch} archived"))
            self.usage.clear()
            for m, vs, nck, size, vsz in rows:
                self.usage.insert(m.id, [m.name, len(vs), nck, "yes" if m.archived else "", human_size(size),
                                         ", ".join(f"{v} {human_size(s)}" for v, s in vsz[:8]) + (" …" if len(vsz) > 8 else "")])

        self.ctx.tasks.run(work, done, name="storage-usage")

    def clean_tmp(self) -> None:
        if self.ctx.training_running:
            message(self, "Training is running", "Wait until training finishes before cleaning temporary files.", "warning")
            return
        n = self.ctx.store.cleanup_temp()
        before = dir_size(self.ctx.storage.tmp_dir) + dir_size(self.ctx.storage.cache_dir)
        self.ctx.storage.clear_tmp()
        for f in self.ctx.storage.cache_dir.glob("*.osz"):
            f.unlink(missing_ok=True)
        message(self, "Cleaned up", f"Removed {n} leftover temporary save folder(s) and {human_size(before)} of temp "
                                    "files. No models, versions, checkpoints or replays were touched.", "success")
        self.refresh_usage()

    def change_location(self) -> None:
        if self.ctx.training_running:
            message(self, "Training is running", "Wait until training finishes before moving storage.", "warning")
            return
        d = filedialog.askdirectory(title="Choose a new data folder")
        if not d:
            return
        new = Path(d)
        if new.resolve() == self.ctx.storage.root.resolve():
            return
        ok, msg = validate_storage_dir(new)
        if not ok:
            error(self, msg, "Can't use that folder")
            return
        old = self.ctx.storage
        if not ask(self, "Move data to the new folder?",
                   f"Your models and replays will be COPIED from\n{old.root}\nto\n{new}\n\nThe app will then use the new "
                   "folder. The old folder is left as it is (nothing is deleted), so you can remove it yourself once "
                   "you've checked everything is there.", ok="Copy and switch"):
            return
        self.usage_lbl.configure(text="Copying…")

        def work():
            return migrate_storage(old, Storage(new), progress=lambda m: self.ctx.tasks.post(
                lambda: self.usage_lbl.configure(text=m)))

        def done(report):
            if not report.ok:
                error(self, "Some files couldn't be copied, so the storage location was NOT changed:\n"
                            + "\n".join(report.errors[:6]), "Migration failed")
                self.refresh_usage()
                return
            self.ctx.settings["storage_root"] = None if new.resolve() == base_root().resolve() else str(new)
            self.ctx.settings.save()
            POLICIES.clear()
            self.ctx.load()
            msg = f"Copied {len(report.copied_models)} model(s) and {report.copied_replays} replay(s)."
            if report.skipped_models:
                msg += f"\nAlready present at the destination (kept as is): {', '.join(report.skipped_models)}."
            message(self, "Storage moved", msg + f"\n\nNow using {new}.", "success")
            self.ctx.emit("models_changed")
            self.ctx.emit("replays_changed")
            self.on_show()

        self.ctx.tasks.run(work, done, name="migrate-storage")

    # -------------------------------------------------------------- osu!
    def browse_osu(self) -> None:
        d = filedialog.askdirectory(title="Choose your osu! folder (the one containing osu!.exe)")
        if d:
            self.osu_path.set(d)
            self.save()

    def detect_osu(self) -> None:
        exe = osu_integration.find_osu_exe(None)
        if exe:
            self.osu_path.set(str(exe.parent))
            self.save()
        else:
            message(self, "Not found", "osu! stable couldn't be found automatically. Use Browse to pick its folder.")

    def save(self) -> None:
        s = self.ctx.settings
        p = self.osu_path.get().strip()
        if p and not osu_integration.find_osu_exe(p):
            error(self, f"No osu!.exe in {p}.", "osu! not found there")
            return
        s["osu_path"] = p or None
        s["player_name"] = self.player.get().strip() or "ManiaBot"
        s["default_maps"] = self.maps.get() or str(REPO_ROOT / "maps" / "demo")
        s["default_device"] = self.device.get()
        s.save()
        self.ctx.refresh_osu()

    # ----------------------------------------------------------- imports
    def import_runs(self) -> None:
        runs = [r for r in find_legacy_runs() if not already_imported(self.ctx.store, r)]
        if not runs:
            message(self, "Nothing to import", "No un-imported model.zip files were found under runs/.")
            return

        def work():
            return [import_legacy_run(self.ctx.store, r).name for r in runs]

        self.ctx.tasks.run(work, lambda names: (message(self, "Imported", f"Imported {', '.join(names)}.", "success"),
                                                self.ctx.emit("models_changed")), name="import-runs")

    def import_replays(self) -> None:
        files = sorted((REPO_ROOT / "replays").glob("*.osr"))
        n = self.ctx.history.import_files(files, note="Imported (old CLI)")
        message(self, "Imported", f"Imported {n} replay(s).", "success")
        self.ctx.emit("replays_changed")
