"""Map browser for Test Model: pick a folder, narrow by star range, see what the model can and can't play.

The folder scan, star ratings and rejected-map details are the same ones Create / Continue Training use
(`MapsFolderField`, `StarRangeField`). On top of that the list knows the selected model version, so each
map is tagged as trained on / held out / new, with the version's recorded accuracy where there is one.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

from maniabot.beatmaps import ScanResult
from maniabot.models import ModelError
from maniabot.ui.pages.base import MapsFolderField, StarRangeField
from maniabot.ui.widgets import Table

FILTERS = {
    "Playable maps": "playable",
    "Unseen by this model": "unseen",
    "Held-out maps": "heldout",
    "Trained on": "trained",
    "Can't play (not 4K)": "cant",
}
FILTER_HELP = {
    "playable": "Every osu!mania 4K map in the folder.",
    "unseen": "Maps this version never trained on (held-out validation maps and brand-new ones): the fair test.",
    "heldout": "Maps kept out of training and used only to measure this version.",
    "trained": "Maps this version trained on.",
    "cant": "Maps the bot can't play: only 4K osu!mania maps are supported. The reason is shown for each.",
}
SORTABLE = {
    "status": lambda r: r["status"],
    "map": lambda r: r["title"].casefold(),
    "stars": lambda r: r["stars"],
    "notes": lambda r: r["notes"],
    "length": lambda r: r["length"],
    "result": lambda r: r["acc"],
}


def fmt_length(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


class MapBrowser(ttk.Frame):
    def __init__(self, parent, ctx, on_pick: Callable[[str], None], folder: str = "",
                 star_range: tuple[float | None, float | None] = (None, None),
                 on_change: Callable[[], None] | None = None):
        super().__init__(parent, style="Card.TFrame")
        self.ctx = ctx
        self.on_pick = on_pick
        self.on_change: Callable[[], None] | None = None  # wired up once construction is finished
        self.columnconfigure(1, weight=1)
        self._base: list[dict[str, Any]] = []
        self._trained: set[str] = set()
        self._heldout: set[str] = set()
        self._known = False  # do we know what the selected version trained on?
        self._acc: dict[str, float] = {}
        self._sort = ("stars", False)  # (column, descending)
        self._rows_by_iid: dict[str, dict[str, Any]] = {}
        self._selected: str | None = None

        ttk.Label(self, text="Folder", style="Card.TLabel").grid(row=0, column=0, sticky="nw", pady=(0, 6), padx=(0, 14))
        self.folder = MapsFolderField(self, ctx, folder, on_scan=self._scanned)
        self.folder.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Label(self, text="Star range", style="Card.TLabel").grid(row=1, column=0, sticky="nw", pady=6, padx=(0, 14))
        self.stars = StarRangeField(self, on_change=self._stars_changed)
        self.stars.grid(row=1, column=1, sticky="ew", pady=6)

        ttk.Label(self, text="Show", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=6, padx=(0, 14))
        bar = ttk.Frame(self, style="Card.TFrame")
        bar.grid(row=2, column=1, sticky="ew", pady=6)
        self.show_var = tk.StringVar(value="Playable maps")
        cb = ttk.Combobox(bar, textvariable=self.show_var, values=list(FILTERS), state="readonly", width=24)
        cb.grid(row=0, column=0, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh())
        ttk.Label(bar, text="Search", style="Card.TLabel").grid(row=0, column=1, padx=(18, 6))
        self.search = tk.StringVar()
        ttk.Entry(bar, textvariable=self.search, width=28).grid(row=0, column=2)
        self.search.trace_add("write", lambda *_: self._refresh())
        self.hint = ttk.Label(self, text="", style="CardMuted.TLabel", wraplength=760, justify="left")
        self.hint.grid(row=3, column=1, sticky="w")

        self.table = Table(self, [("status", "Status", 95), ("map", "Map", 400), ("stars", "Stars", 60),
                                  ("notes", "Notes", 60), ("length", "Length", 60), ("result", "Result / reason", 220)],
                           height=10)
        self.table.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        for key in SORTABLE:
            self.table.tv.heading(key, command=lambda k=key: self._sort_by(k))
        self.table.on_select(self._on_select)
        self.note = ttk.Label(self, text="", style="CardMuted.TLabel", wraplength=760, justify="left")
        self.note.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._update_hint()
        self.stars.set(*star_range)  # last: this triggers a refresh, which needs the table above
        self.on_change = on_change

    # ------------------------------------------------------------ data in
    def ensure_scanned(self) -> None:
        """Scan the remembered folder the first time the page is shown."""
        path = self.folder.get()
        if path and Path(path).exists() and self.folder.result is None:
            self.folder.scan()

    def set_context(self, meta: dict[str, Any] | None, evaluations: list[dict[str, Any]] | None = None) -> None:
        """Tell the browser which version is selected: what it trained on and how it scored per map."""
        meta = meta or {}
        validation = meta.get("validation") or {}
        heldout = {m.get("md5") for m in meta.get("validation_maps") or [] if m.get("md5")}
        trained = {m.get("md5") for m in meta.get("training_maps") or [] if m.get("md5")}
        if not validation.get("heldout"):  # validated on training maps: nothing was truly held out
            trained |= heldout
            heldout = set()
        self._trained, self._heldout = trained, heldout
        self._known = bool(trained or heldout)
        acc: dict[str, float] = {}
        results = [validation, *sorted(evaluations or [], key=lambda e: e.get("created_at") or "")]
        for res in results:  # later evaluations win
            for m in res.get("maps") or []:
                if m.get("md5") and m.get("accuracy") is not None:
                    acc[m["md5"]] = float(m["accuracy"])
        self._acc = acc
        self._update_hint()
        self._refresh()

    def _scanned(self, res: ScanResult | None) -> None:
        self.stars.show_scan(res)
        self._base = []
        if res:
            for c in res.valid:
                notes = c.beatmap.notes if c.beatmap else []
                length = (max(n.end_time for n in notes) - notes[0].time) / 1000.0 if notes else 0.0
                self._base.append({"path": c.path, "playable": True, "md5": c.beatmap.md5, "title": c.label,
                                   "stars": c.stars, "exact": c.stars_exact, "notes": len(notes), "length": length,
                                   "detail": ""})
            for c in res.rejected:
                self._base.append({"path": c.path, "playable": False, "md5": "", "title": Path(c.path).name,
                                   "stars": None, "exact": True, "notes": None, "length": None, "detail": c.reason})
        self._refresh()
        if self.on_change:
            self.on_change()

    def _stars_changed(self) -> None:
        self._refresh()
        if self.on_change:
            self.on_change()

    # ------------------------------------------------------------ view
    def _status(self, row: dict[str, Any]) -> str:
        if not row["playable"]:
            return "Can't play"
        if not self._known:
            return "Playable"
        if row["md5"] in self._heldout:
            return "Held-out"
        return "Trained on" if row["md5"] in self._trained else "New"

    def _update_hint(self) -> None:
        kind = FILTERS.get(self.show_var.get(), "playable")
        text = FILTER_HELP[kind]
        if kind in ("unseen", "heldout", "trained") and not self._known:
            text = "Pick a model version above first; this list needs to know what it trained on."
        self.hint.configure(text=text)

    def _sort_by(self, key: str) -> None:
        col, desc = self._sort
        self._sort = (key, not desc if key == col else False)
        self._refresh()

    def _refresh(self) -> None:
        self._update_hint()
        kind = FILTERS.get(self.show_var.get(), "playable")
        try:
            lo, hi = self.stars.get()
        except ModelError:
            lo = hi = None
        needle = self.search.get().strip().casefold()
        rows = []
        for r in self._base:
            r = dict(r, status=self._status(r))
            r["acc"] = self._acc.get(r["md5"]) if r["playable"] else None
            if kind == "cant":
                keep = not r["playable"]
            else:
                keep = r["playable"]
                if kind == "unseen":
                    keep = keep and r["status"] in ("Held-out", "New")
                elif kind == "heldout":
                    keep = keep and r["status"] == "Held-out"
                elif kind == "trained":
                    keep = keep and r["status"] == "Trained on"
                if keep and (lo is not None or hi is not None):
                    s = r["stars"] or 0.0
                    keep = (lo is None or s >= lo) and (hi is None or s <= hi)
            if keep and needle and needle not in r["title"].casefold() and needle not in Path(r["path"]).name.casefold():
                keep = False
            if keep:
                rows.append(r)
        col, desc = self._sort
        key = SORTABLE[col]
        present = sorted((r for r in rows if key(r) is not None), key=key, reverse=desc)
        rows = present + [r for r in rows if key(r) is None]  # blanks always last

        self.table.clear()
        self._rows_by_iid = {}
        for i, r in enumerate(rows):
            iid = str(i)
            self._rows_by_iid[iid] = r
            if r["playable"]:
                stars = "" if r["stars"] is None else f"{r['stars']:.2f}" + ("" if r["exact"] else " ~")
                result = f"{r['acc'] * 100:.1f}%" if r["acc"] is not None else ""
                tag = ("muted",) if r["status"] == "Trained on" else ()
            else:
                stars, result, tag = "", r["detail"], ("bad",)
            self.table.insert(iid, [r["status"], r["title"], stars, r["notes"], fmt_length(r["length"]), result], tags=tag)
        total = sum(1 for r in self._base if r["playable"])
        if self._base:
            self.note.configure(text=f"{len(rows)} shown · {total} playable 4K map(s) in this folder · "
                                     f"{len(self._base) - total} can't be played", style="CardMuted.TLabel")
        else:
            self.note.configure(text="Choose a folder of maps (.osu, .osz or .zip) to browse them here.",
                                style="CardMuted.TLabel")
        for iid, r in self._rows_by_iid.items():  # keep the selection across refreshes
            if r["path"] == self._selected:
                self.table.select(iid)
                break

    def _on_select(self) -> None:
        iid = self.table.selected()
        row = self._rows_by_iid.get(iid or "")
        if not row:
            return
        if row["playable"]:
            self._selected = row["path"]
            self.on_pick(row["path"])
        else:
            self.note.configure(text=f"The bot can't play this one: {row['detail']}", style="CardError.TLabel")
