"""Page base class and form helpers shared by the pages."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable

from maniabot.beatmaps import ScanResult, scan_maps
from maniabot.models import ModelError, VersionRecord
from maniabot.ui import theme
from maniabot.ui.widgets import Dialog, ScrollFrame, Table, fmt_pct, help_icon


class Page(ttk.Frame):
    title = ""
    subtitle = ""

    def __init__(self, parent, ctx):
        super().__init__(parent)
        self.ctx = ctx
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.head = ttk.Frame(self, padding=(28, 22, 28, 8))
        self.head.grid(row=0, column=0, sticky="ew")
        self.head.columnconfigure(0, weight=1)
        self.title_lbl = ttk.Label(self.head, text=self.title, style="H1.TLabel")
        self.title_lbl.grid(row=0, column=0, sticky="w")
        self.sub_lbl = ttk.Label(self.head, text=self.subtitle, style="Muted.TLabel")
        self.sub_lbl.grid(row=1, column=0, sticky="w")
        self.actions = ttk.Frame(self.head)
        self.actions.grid(row=0, column=1, rowspan=2, sticky="e")
        self.scroll = ScrollFrame(self)
        self.scroll.grid(row=1, column=0, sticky="nsew")
        self.body = self.scroll.inner
        self.body.configure(padding=(28, 8, 28, 28))
        self.body.columnconfigure(0, weight=1)
        self.build()

    def build(self) -> None:  # override
        pass

    def on_show(self, **kwargs) -> None:  # override
        pass

    def visible(self) -> bool:
        """True if this page is the one currently shown (stacked pages all report winfo_ismapped)."""
        return self.ctx.app.pages.get(self.ctx.app.current) is self

    def set_title(self, title: str, subtitle: str = "") -> None:
        self.title_lbl.configure(text=title)
        self.sub_lbl.configure(text=subtitle)

    def action(self, text: str, cmd: Callable[[], None], accent: bool = False) -> ttk.Button:
        b = ttk.Button(self.actions, text=text, command=cmd, style="Accent.TButton" if accent else "TButton")
        b.pack(side="left", padx=(8, 0))
        return b


def form_row(parent, row: int, label: str, widget: tk.Widget, help_text: str | None = None,
             note: str | None = None, stretch: bool | None = None) -> None:
    """Label (with optional ? tooltip) in column 0, the input in column 1.

    Inputs keep their natural width unless `stretch` (default: True for composite fields).
    """
    parent.columnconfigure(0, weight=0)
    parent.columnconfigure(1, weight=1)
    lab = ttk.Frame(parent, style="Card.TFrame")
    lab.grid(row=row, column=0, sticky="nw", pady=6, padx=(0, 18))
    ttk.Label(lab, text=label, style="Card.TLabel").pack(side="left")
    if help_text:
        help_icon(lab, help_text).pack(side="left", padx=(4, 0))
    if stretch is None:
        stretch = isinstance(widget, (ttk.Frame, tk.Frame)) and not isinstance(widget, ttk.Spinbox)
    widget.grid(row=row, column=1, sticky="ew" if stretch else "w", pady=6)
    if note:
        ttk.Label(parent, text=note, style="CardMuted.TLabel", font=theme.fonts()["small"]).grid(
            row=row, column=2, sticky="w", padx=(10, 0))


def version_label(v: VersionRecord) -> str:
    tags = [b for b in v.badges() if b in ("Active", "Best", "Latest", "Protected")]
    acc = f"  val {fmt_pct(v.val_accuracy, 1)}" if v.val_accuracy is not None else ""
    return f"{v.version}{'  [' + ', '.join(tags) + ']' if tags else ''}{acc}"


class MapsFolderField(ttk.Frame):
    """Folder entry + browse + background 4K validation summary."""

    def __init__(self, parent, ctx, initial: str = "", on_scan: Callable[[ScanResult | None], None] | None = None):
        super().__init__(parent, style="Card.TFrame")
        self.ctx = ctx
        self.on_scan = on_scan
        self.var = tk.StringVar(value=initial)
        self.columnconfigure(0, weight=1)
        ttk.Entry(self, textvariable=self.var).grid(row=0, column=0, sticky="ew")
        ttk.Button(self, text="Browse…", command=self.browse).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(self, text="File…", command=self.browse_file).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(self, text="Check maps", command=self.scan).grid(row=0, column=3, padx=(6, 0))
        self.summary = ttk.Label(self, text="", style="CardMuted.TLabel", wraplength=560, justify="left")
        self.summary.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.details_btn = ttk.Button(self, text="Show rejected maps", style="CardLink.TButton", command=self.details)
        self.result: ScanResult | None = None
        self._gen = 0

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, path: str) -> None:
        self.var.set(path)
        self.scan()

    def browse(self) -> None:
        d = filedialog.askdirectory(title="Choose a folder of osu!mania 4K beatmaps", initialdir=self.get() or None)
        if d:
            self.set(d)

    def browse_file(self) -> None:
        start = Path(self.get())
        f = filedialog.askopenfilename(
            title="Choose a beatmap (.osu), mapset (.osz) or map pack (.zip)",
            initialdir=str(start if start.is_dir() else start.parent) if self.get() else None,
            filetypes=[("osu! maps", "*.osu *.osz *.zip"), ("All files", "*.*")])
        if f:
            self.set(f)

    def scan(self) -> None:
        path = self.get()
        self._gen += 1
        gen = self._gen
        self.result = None
        self.details_btn.grid_forget()
        if not path or not Path(path).exists():
            self.summary.configure(text="Folder or file not found." if path else "", style="CardError.TLabel")
            if self.on_scan:
                self.on_scan(None)
            return
        self.summary.configure(text="Checking beatmaps…", style="CardMuted.TLabel")

        def done(res: ScanResult):
            if gen != self._gen:
                return
            self.result = res
            n, bad = len(res.valid), len(res.rejected)
            if n:
                self.summary.configure(
                    text=f"✓ {n} osu!mania 4K map(s) ready" + (f"  ·  {bad} file(s) skipped (not 4K / unusable)" if bad else ""),
                    style="CardGood.TLabel")
            else:
                self.summary.configure(text=f"No usable osu!mania 4K maps here ({bad} file(s) rejected).",
                                       style="CardError.TLabel")
            if bad:
                self.details_btn.grid(row=2, column=0, sticky="w")
            if self.on_scan:
                self.on_scan(res)

        self.ctx.tasks.run(lambda: scan_maps(path), done, name="scan-maps")

    def details(self) -> None:
        if not self.result:
            return
        d = Dialog(self, "Rejected beatmaps", width=760)
        ttk.Label(d.body, text="These files were not used", style="CardH2.TLabel").grid(row=0, column=0, sticky="w")
        t = Table(d.body, [("file", "File", 300), ("reason", "Reason", 420)], height=12)
        t.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        for i, c in enumerate(self.result.rejected[:2000]):
            t.insert(str(i), [Path(c.path).name, c.reason])
        d.buttons([("Close", None, "Accent.TButton")])
        d.show()


def int_var(value: int) -> tk.StringVar:
    return tk.StringVar(value=str(value))


def parse_int(var: tk.StringVar, label: str) -> int:
    try:
        return int(str(var.get()).replace(",", "").replace("_", "").strip())
    except ValueError:
        raise ModelError(f"{label} must be a whole number.") from None


def parse_float(var: tk.StringVar, label: str) -> float:
    try:
        return float(str(var.get()).strip())
    except ValueError:
        raise ModelError(f"{label} must be a number.") from None


DEVICE_CHOICES = ("auto", "cpu", "cuda")
DEVICE_HELP = ("Auto uses a CUDA GPU when one is available, otherwise the CPU. This network is small, so the "
               "CPU is often as fast; most time is spent in the game simulator.")
LENGTH_PRESETS = {"Quick test (50k)": 50_000, "Short (300k)": 300_000, "Standard (1M)": 1_000_000,
                  "Long (3M)": 3_000_000}
LENGTH_HELP = ("How many simulator steps to train for. One step is one 10 ms decision; a 20-second practice clip "
               "is about 2,000 steps. 300k steps takes a few minutes on a laptop CPU.")


def summary_dialog(parent, title: str, rows: list[tuple[str, str]], warnings: list[str] = (),
                   ok: str = "Start Training") -> bool:
    d = Dialog(parent, title, width=560)
    ttk.Label(d.body, text=title, style="CardH2.TLabel").grid(row=0, column=0, sticky="w")
    g = ttk.Frame(d.body, style="Card.TFrame")
    g.grid(row=1, column=0, sticky="ew", pady=(12, 0))
    for i, (k, v) in enumerate(rows):
        ttk.Label(g, text=k, style="CardMuted.TLabel").grid(row=i, column=0, sticky="nw", padx=(0, 16), pady=2)
        ttk.Label(g, text=v, style="Card.TLabel", wraplength=360, justify="left").grid(row=i, column=1, sticky="w", pady=2)
    r = 2
    for w in warnings:
        ttk.Label(d.body, text=f"⚠  {w}", style="CardWarn.TLabel", wraplength=500, justify="left").grid(
            row=r, column=0, sticky="w", pady=(8, 0))
        r += 1
    d.buttons([("Back", False, "TButton"), (ok, True, "Accent.TButton")])
    return bool(d.show())


def go_button(parent, text: str, cmd: Callable[[], Any], style: str = "TButton") -> ttk.Button:
    b = ttk.Button(parent, text=text, command=cmd, style=style)
    return b
