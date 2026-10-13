"""Replay History: every generated / watched replay, kept even after its model is deleted."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from maniabot import osu_integration
from maniabot.models import DeletionPlan
from maniabot.ranks import mania_rank
from maniabot.replays import generate_replay, model_status
from maniabot.ui.pages.base import Page
from maniabot.ui.widgets import (
    Card, Dialog, EmptyState, KeyValueGrid, Table, confirm_delete, error, fmt_date, fmt_pct, fmt_value, message,
)


class ReplaysPage(Page):
    title = "Replay History"
    subtitle = "Replays are stored in your user data folder and open in osu! stable"

    def build(self) -> None:
        self.action("Open Replays Folder", lambda: osu_integration.open_folder(self.ctx.storage.replays_dir))
        self.action("Import .osr files…", self._import)
        self.action("Generate Replay", lambda: self.ctx.app.show("test", mode="replay"), accent=True)
        bar = ttk.Frame(self.body)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(bar, text="Model").pack(side="left")
        self.filter = tk.StringVar(value="All models")
        self.filter_cb = ttk.Combobox(bar, textvariable=self.filter, state="readonly", width=34)
        self.filter_cb.pack(side="left", padx=8)
        self.filter_cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        self.count_lbl = ttk.Label(bar, text="", style="Muted.TLabel")
        self.count_lbl.pack(side="left", padx=10)

        self.card = Card(self.body, padding=0)
        self.card.grid(row=1, column=0, sticky="nsew")
        self.table = Table(self.card.body, [
            ("date", "Generated", 130), ("map", "Beatmap", 230), ("diff", "Difficulty", 130), ("model", "Model", 200),
            ("acc", "Accuracy", 75), ("rank", "Rank", 50), ("miss", "Misses", 60), ("time", "Processing", 75), ("kind", "Type", 70),
            ("file", "File", 70)], height=16, selectmode="extended")
        self.table.grid(row=0, column=0, sticky="nsew")
        self.table.on_double(self._details)
        self.empty = EmptyState(self.body, "No replays yet",
                                "Generate a replay from Test Model, or watch the bot play and save the result.",
                                [("Test Model", lambda: self.ctx.app.show("test", mode="replay"))])
        acts = ttk.Frame(self.body)
        acts.grid(row=2, column=0, sticky="w", pady=(10, 0))
        for text, fn, style in (("Open in osu!", self._open, "Accent.TButton"), ("Show in Folder", self._folder, "TButton"),
                                ("Copy Path", self._copy, "TButton"), ("Details", self._details, "TButton"),
                                ("Generate Again", self._again, "TButton"), ("Delete Replay", self._delete, "Danger.TButton")):
            ttk.Button(acts, text=text, style=style, command=fn).pack(side="left", padx=(0, 6))
        self.status = ttk.Label(self.body, text="", style="Muted.TLabel")
        self.status.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.records: dict[str, dict] = {}
        self.ctx.on("replays_changed", lambda **_: self.refresh() if self.visible() else None)
        self.ctx.on("models_changed", lambda **_: self.refresh() if self.visible() else None)

    def on_show(self, **_) -> None:
        self.refresh()

    def refresh(self) -> None:
        recs = self.ctx.history.list()
        names = sorted({r.get("model_name") or "—" for r in recs})
        self.filter_cb.configure(values=["All models", *names])
        f = self.filter.get()
        if f != "All models":
            recs = [r for r in recs if (r.get("model_name") or "—") == f]
        self.table.clear()
        self.records = {r["id"]: r for r in recs}
        if not self.ctx.history.list():
            self.card.grid_remove()
            self.empty.grid(row=1, column=0, sticky="ew")
        else:
            self.empty.grid_forget()
            self.card.grid()
        for r in recs:
            bm = r.get("beatmap") or {}
            m = r.get("metrics") or {}
            st = model_status(self.ctx.store, r)
            model = f"{r.get('model_name') or '—'} {r.get('version') or ''}".strip() + (f"  ({st})" if st else "")
            self.table.insert(r["id"], [
                fmt_date(r.get("created_at")), f"{bm.get('artist', '')} - {bm.get('title', '')}".strip(" -"),
                bm.get("version", ""), model, fmt_pct(m.get("accuracy")),
                mania_rank(m.get("accuracy"), m.get("counts")) or "—", m.get("misses", "—"),
                f"{r['processing_s']:.1f}s" if r.get("processing_s") is not None else "—", r.get("kind", ""),
                "ok" if r["exists"] else "missing"], tags=("muted",) if not r["exists"] else ())
        self.count_lbl.configure(text=f"{len(recs)} replay(s)")

    def _sel(self) -> dict | None:
        sel = self.table.selected_all()
        if len(sel) != 1:
            message(self, "Choose a replay", "Select one replay first.")
            return None
        return self.records[sel[0]]

    def _open(self) -> None:
        r = self._sel()
        if not r:
            return
        if not Path(r["path"]).exists():
            error(self, "The replay file is missing (it may have been moved or deleted outside the app).")
            return
        try:
            osu_integration.open_in_osu(r["path"], self.ctx.settings["osu_path"])
            self.status.configure(text="Sent to osu!. If osu! says the beatmap is missing, import the map first "
                                       "(Test Model → Import map into osu!).")
        except osu_integration.OsuError as exc:
            error(self, str(exc), "osu! not available")

    def _folder(self) -> None:
        r = self._sel()
        if r:
            osu_integration.open_folder(self.ctx.storage.replays_dir, select=r["path"])

    def _copy(self) -> None:
        r = self._sel()
        if r:
            self.clipboard_clear()
            self.clipboard_append(r["path"])
            self.status.configure(text="Path copied.")

    def _details(self) -> None:
        r = self._sel()
        if not r:
            return
        d = Dialog(self, "Replay details", width=640)
        bm = r.get("beatmap") or {}
        ttk.Label(d.body, text=f"{bm.get('artist', '')} - {bm.get('title', '')} [{bm.get('version', '')}]",
                  style="CardH2.TLabel", wraplength=580).grid(row=0, column=0, sticky="w")
        kv = KeyValueGrid(d.body, columns=2)
        kv.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        st = model_status(self.ctx.store, r)
        m = r.get("metrics") or {}
        rows = [("Model", f"{r.get('model_name')} {r.get('version') or ''}" + (f" ({st})" if st else "")),
                ("Generated", fmt_date(r.get("created_at"))), ("Type", r.get("kind")),
                ("Attempts", r.get("tries", "—")), ("Player", r.get("player", "—")),
                ("Notes / holds", f"{bm.get('notes', '—')} / {bm.get('holds', '—')}"), ("OD", bm.get("od", "—"))]
        rows.append(("Rank", mania_rank(m.get("accuracy"), m.get("counts")) or "—"))
        for key in ("accuracy", "misses", "extra_presses", "hold_accuracy", "mean_offset_ms", "timing_std_ms", "max_combo"):
            rows.append((key.replace("_", " ").capitalize(), fmt_value(key, m.get(key))))
        if m.get("lanes"):
            rows.append(("Lane accuracy", "  ".join(fmt_pct(l["accuracy"], 0) for l in m["lanes"])))
        rows.append(("File", r["path"]))
        for i, (k, v) in enumerate(rows):
            kv.add(str(i), k, str(v))
        ttk.Label(d.body, text="Accuracy and judgements come from the simulator, not an osu! score.",
                  style="CardMuted.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0))
        d.buttons([("Close", None, "Accent.TButton")])
        d.show()

    def _again(self) -> None:
        r = self._sel()
        if not r:
            return
        if not r.get("model_id") or model_status(self.ctx.store, r):
            error(self, f"The model used for this replay ({r.get('model_name')} {r.get('version') or ''}) no longer "
                        "exists, so it can't be generated again.", "Model deleted")
            return
        path = (r.get("beatmap") or {}).get("path")
        if not path or not Path(path).exists():
            error(self, "The beatmap file used for this replay can't be found any more.", "Beatmap missing")
            return
        self.status.configure(text="Generating…")
        self.ctx.tasks.run(lambda: generate_replay(self.ctx.store, self.ctx.history, r["model_id"], r["version"], path,
                                                   r.get("player") or self.ctx.settings["player_name"], r.get("tries", 1),
                                                   seed=int(r.get("seed", 0)) + 1),
                           lambda rec: (self.status.configure(text=f"✓ New replay: {Path(rec['path']).name}"),
                                        self.ctx.emit("replays_changed")),
                           lambda exc, tb: (self.status.configure(text=str(exc)),
                                            error(self, str(exc), "Replay generation failed", tb)), name="regenerate")

    def _delete(self) -> None:
        sel = self.table.selected_all()
        if not sel:
            return
        recs = [self.records[s] for s in sel]
        size = sum(Path(r["path"]).stat().st_size for r in recs if Path(r["path"]).exists())
        plan = DeletionPlan(title=f"Delete {len(recs)} replay(s)?",
                            items=[Path(r["path"]).name for r in recs[:8]] + ([f"…and {len(recs) - 8} more"] if len(recs) > 8 else []),
                            size_bytes=size)
        ok, _ = confirm_delete(self, plan, "Delete Replays")
        if ok:
            for r in recs:
                self.ctx.history.delete(r["id"])
            self.ctx.emit("replays_changed")

    def _import(self) -> None:
        files = filedialog.askopenfilenames(title="Import osu! replays", filetypes=[("osu! replay", "*.osr")])
        if files:
            n = self.ctx.history.import_files([Path(f) for f in files])
            self.status.configure(text=f"Imported {n} replay(s) (copied into the replays folder).")
            self.ctx.emit("replays_changed")
