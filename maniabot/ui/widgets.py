"""Reusable widgets: cards, badges, stat tiles, tables, dialogs, tooltips, background tasks."""
from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
import traceback
from datetime import datetime
from tkinter import ttk
from typing import Any, Callable, Iterable

from maniabot.ui import theme

log = logging.getLogger("maniabot.ui")


# ------------------------------------------------------------ formatting
def fmt_pct(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def fmt_int(x: int | None) -> str:
    return "—" if x is None else f"{int(x):,}"


def fmt_ms(x: float | None) -> str:
    return "—" if x is None else f"{x:+.1f} ms" if x else "0.0 ms"


def fmt_num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y %H:%M")
    except ValueError:
        return iso


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s" if m else f"{s}s"


def fmt_value(key: str, v: Any) -> str:
    if v is None:
        return "—"
    if key in ("accuracy", "hit_rate", "hold_accuracy") or key.startswith("lane"):
        return fmt_pct(v)
    if key.endswith("_ms"):
        return f"{v:.2f} ms"
    if isinstance(v, float):
        return f"{v:,.4g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


# ------------------------------------------------------------- building
class Card(ttk.Frame):
    def __init__(self, parent, title: str | None = None, subtitle: str | None = None, padding=16, **kw):
        super().__init__(parent, style="Card.TFrame", padding=padding, **kw)
        self.columnconfigure(0, weight=1)
        self._row = 0
        if title:
            ttk.Label(self, text=title, style="CardH2.TLabel").grid(row=0, column=0, sticky="w")
            self._row = 1
        if subtitle:
            ttk.Label(self, text=subtitle, style="CardMuted.TLabel", wraplength=700, justify="left").grid(
                row=self._row, column=0, sticky="w", pady=(2, 0))
            self._row += 1
        self.body = ttk.Frame(self, style="Card.TFrame")
        self.body.grid(row=self._row, column=0, sticky="nsew", pady=(10 if (title or subtitle) else 0, 0))
        self.body.columnconfigure(0, weight=1)
        self.rowconfigure(self._row, weight=1)


def badge(parent, text: str, colour: str | None = None, bg: str = theme.SURFACE) -> tk.Label:
    colour = colour or theme.BADGE_COLOURS.get(text, theme.MUTED)
    return tk.Label(parent, text=f" {text} ", fg=colour, bg=bg, font=theme.fonts()["small"],
                    highlightthickness=1, highlightbackground=colour, padx=2, pady=0)


def badge_row(parent, labels: Iterable[str], bg: str = theme.SURFACE) -> ttk.Frame:
    f = tk.Frame(parent, bg=bg)
    for text in labels:
        badge(f, text, bg=bg).pack(side="left", padx=(0, 4))
    return f


class StatTile(ttk.Frame):
    def __init__(self, parent, label: str, value: str = "—", hint: str = ""):
        super().__init__(parent, style="Card.TFrame", padding=(16, 12))
        self.value = ttk.Label(self, text=value, style="Stat.TLabel")
        self.value.pack(anchor="w")
        ttk.Label(self, text=label, style="CardMuted.TLabel").pack(anchor="w")
        self.hint = ttk.Label(self, text=hint, style="CardMuted.TLabel", font=theme.fonts()["small"])
        self.hint.pack(anchor="w")

    def set(self, value: str, hint: str | None = None) -> None:
        self.value.configure(text=value)
        if hint is not None:
            self.hint.configure(text=hint)


class KeyValueGrid(ttk.Frame):
    """Two-column label/value grid on a card background."""

    def __init__(self, parent, columns: int = 2, **kw):
        super().__init__(parent, style="Card.TFrame", **kw)
        self.columns = columns
        self._vars: dict[str, ttk.Label] = {}
        self._n = 0

    def add(self, key: str, label: str, value: str = "—", tip: str | None = None) -> ttk.Label:
        r, c = divmod(self._n, self.columns)
        self._n += 1
        lab = ttk.Label(self, text=label, style="CardMuted.TLabel")
        lab.grid(row=r, column=c * 2, sticky="w", padx=(0, 10), pady=2)
        val = ttk.Label(self, text=value, style="Card.TLabel")
        val.grid(row=r, column=c * 2 + 1, sticky="w", padx=(0, 28), pady=2)
        if tip:
            Tooltip(lab, tip)
        self._vars[key] = val
        return val

    def set(self, key: str, value: str) -> None:
        if key in self._vars:
            self._vars[key].configure(text=value)


class Table(ttk.Frame):
    """Treeview with scrollbar, column specs and convenient selection helpers."""

    def __init__(self, parent, columns: list[tuple[str, str, int]], height: int = 10, tree: bool = False,
                 selectmode: str = "browse"):
        super().__init__(parent, style="Card.TFrame")
        self.columns = [c[0] for c in columns]
        show = "tree headings" if tree else "headings"
        self.tv = ttk.Treeview(self, columns=self.columns, show=show, height=height, selectmode=selectmode)
        if tree:
            self.tv.column("#0", width=theme.px(110), stretch=False)
            self.tv.heading("#0", text=columns[0][1])
            self.columns = self.columns[1:]
            self.tv.configure(columns=self.columns)
            columns = columns[1:]
        for key, label, width in columns:
            self.tv.heading(key, text=label)
            self.tv.column(key, width=theme.px(width), minwidth=theme.px(40), stretch=True, anchor="w")
        sb = ttk.Scrollbar(self, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        self.tv.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.tv.tag_configure("muted", foreground=theme.FAINT)
        self.tv.tag_configure("accent", foreground=theme.ACCENT)
        self.tv.tag_configure("good", foreground=theme.GOOD)
        self.tv.tag_configure("bad", foreground=theme.BAD)

    def clear(self) -> None:
        self.tv.delete(*self.tv.get_children())

    def insert(self, iid: str, values: list[Any], parent: str = "", text: str = "", tags=(), open_=True) -> None:
        self.tv.insert(parent, "end", iid=iid, text=text, values=[("" if v is None else v) for v in values],
                       tags=tags, open=open_)

    def selected(self) -> str | None:
        sel = self.tv.selection()
        return sel[0] if sel else None

    def selected_all(self) -> list[str]:
        return list(self.tv.selection())

    def select(self, iid: str) -> None:
        if self.tv.exists(iid):
            self.tv.selection_set(iid)
            self.tv.see(iid)

    def on_select(self, fn: Callable[[], None]) -> None:
        self.tv.bind("<<TreeviewSelect>>", lambda _e: fn())

    def on_double(self, fn: Callable[[], None]) -> None:
        self.tv.bind("<Double-1>", lambda _e: fn())


class ScrollFrame(ttk.Frame):
    """A vertically scrollable frame; put content in `.inner`."""

    def __init__(self, parent, bg: str = theme.BG):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda _e: self._update_region())
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: (self.canvas.itemconfigure(self._win, width=e.width),
                                                   self._update_region()))
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.sb.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.bind_all_wheel()

    def _scrollable(self) -> bool:
        return self.inner.winfo_reqheight() > self.canvas.winfo_height()

    def _update_region(self) -> None:
        # Never smaller than the visible area, so short pages can't be scrolled off the top.
        h = max(self.inner.winfo_reqheight(), self.canvas.winfo_height())
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), h))
        if not self._scrollable():
            self.canvas.yview_moveto(0)

    def bind_all_wheel(self) -> None:
        def wheel(e):
            try:
                w = self.winfo_containing(e.x_root, e.y_root)
            except (KeyError, tk.TclError):  # pointer over a destroyed/foreign window
                return
            while w is not None:
                if w is self.canvas:
                    if self._scrollable():
                        self.canvas.yview_scroll(int(-e.delta / 120), "units")
                    return
                if isinstance(w, (ttk.Treeview, tk.Text, tk.Listbox)):
                    return
                w = w.master
        self.canvas.bind_all("<MouseWheel>", wheel, add="+")


class Tooltip:
    def __init__(self, widget, text: str, delay: int = 450):
        self.widget, self.text, self.delay = widget, text, delay
        self.tip: tk.Toplevel | None = None
        self._after = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._after = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tip or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, bg=theme.SURFACE_2, fg=theme.TEXT, justify="left", wraplength=340,
                 padx=8, pady=5, font=theme.fonts()["small"], highlightthickness=1,
                 highlightbackground=theme.BORDER).pack()

    def _hide(self, _e=None):
        if self._after:
            self.widget.after_cancel(self._after)
            self._after = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


def help_icon(parent, text: str, bg: str = theme.SURFACE) -> tk.Label:
    lab = tk.Label(parent, text="?", fg=theme.MUTED, bg=bg, font=theme.fonts()["small"], cursor="question_arrow",
                   padx=3)
    Tooltip(lab, text)
    return lab


class EmptyState(ttk.Frame):
    def __init__(self, parent, title: str, text: str, actions: list[tuple[str, Callable[[], None]]] = (),
                 style: str = "Card.TFrame"):
        super().__init__(parent, style=style, padding=30)
        lab_style = "CardH2.TLabel" if style == "Card.TFrame" else "H2.TLabel"
        mut_style = "CardMuted.TLabel" if style == "Card.TFrame" else "Muted.TLabel"
        ttk.Label(self, text=title, style=lab_style).pack()
        ttk.Label(self, text=text, style=mut_style, wraplength=520, justify="center").pack(pady=(6, 14))
        row = ttk.Frame(self, style=style)
        row.pack()
        for i, (label, fn) in enumerate(actions):
            ttk.Button(row, text=label, command=fn, style="Accent.TButton" if i == 0 else "TButton").pack(
                side="left", padx=4)


class LineChart(tk.Canvas):
    """Tiny dependency-free line chart for live training curves."""

    def __init__(self, parent, height: int = 140, colour: str = theme.ACCENT, label: str = "", fmt=lambda v: f"{v:.3g}"):
        super().__init__(parent, height=theme.px(height), bg=theme.SURFACE, highlightthickness=0)
        self.colour, self.label, self.fmt = colour, label, fmt
        self.points: list[tuple[float, float]] = []
        self.bind("<Configure>", lambda _e: self.redraw())

    def set(self, points: list[tuple[float, float]]) -> None:
        self.points = [(x, y) for x, y in points if y == y]  # drop NaN
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        self.create_text(8, 8, text=self.label, anchor="nw", fill=theme.MUTED, font=theme.fonts()["small"])
        if len(self.points) < 2 or w < 40:
            self.create_text(w / 2, h / 2, text="waiting for data…", fill=theme.FAINT, font=theme.fonts()["small"])
            return
        xs, ys = zip(*self.points)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if y1 - y0 < 1e-12:
            y0, y1 = y0 - 1, y1 + 1
        pad_l, pad_r, pad_t, pad_b = 8, 60, 26, 10

        def px(x, y):
            return (pad_l + (x - x0) / ((x1 - x0) or 1) * (w - pad_l - pad_r),
                    pad_t + (1 - (y - y0) / (y1 - y0)) * (h - pad_t - pad_b))

        coords = [c for p in self.points for c in px(*p)]
        self.create_line(*coords, fill=self.colour, width=2, smooth=False)
        lx, ly = px(*self.points[-1])
        self.create_oval(lx - 3, ly - 3, lx + 3, ly + 3, fill=self.colour, outline="")
        self.create_text(w - pad_r + 6, ly, text=self.fmt(self.points[-1][1]), anchor="w", fill=theme.TEXT,
                         font=theme.fonts()["small"])
        self.create_text(w - pad_r + 6, pad_t, text=self.fmt(y1), anchor="w", fill=theme.FAINT, font=theme.fonts()["small"])
        self.create_text(w - pad_r + 6, h - pad_b, text=self.fmt(y0), anchor="w", fill=theme.FAINT,
                         font=theme.fonts()["small"])


# -------------------------------------------------------------- dialogs
class Dialog(tk.Toplevel):
    def __init__(self, parent, title: str, width: int = 520):
        super().__init__(parent)
        self.withdraw()
        self.title(title)
        self.configure(bg=theme.SURFACE)
        self.transient(parent.winfo_toplevel())
        self.resizable(False, False)
        self.result: Any = None
        self.body = ttk.Frame(self, style="Card.TFrame", padding=22)
        self.body.pack(fill="both", expand=True)
        self.body.columnconfigure(0, weight=1)
        self._width = theme.px(width)
        self.bind("<Escape>", lambda _e: self.close(None))
        self.protocol("WM_DELETE_WINDOW", lambda: self.close(None))

    def buttons(self, specs: list[tuple[str, Any, str]]) -> ttk.Frame:
        row = ttk.Frame(self.body, style="Card.TFrame")
        row.grid(row=99, column=0, sticky="e", pady=(18, 0))
        self._buttons = {}
        for label, value, style in specs:
            b = ttk.Button(row, text=label, style=style, command=lambda v=value: self.close(v))
            b.pack(side="left", padx=(8, 0))
            self._buttons[label] = b
        return row

    def show(self) -> Any:
        self.update_idletasks()
        p = self.master.winfo_toplevel()
        w = max(self._width, self.winfo_reqwidth())
        h = self.winfo_reqheight()
        x = p.winfo_rootx() + (p.winfo_width() - w) // 2
        y = p.winfo_rooty() + max(40, (p.winfo_height() - h) // 3)
        self.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")
        self.deiconify()
        self.grab_set()
        self.focus_set()
        self.wait_window()
        return self.result

    def close(self, value: Any) -> None:
        self.result = value
        self.grab_release()
        self.destroy()


def message(parent, title: str, text: str, kind: str = "info", details: str | None = None) -> None:
    d = Dialog(parent, title)
    colour = {"error": theme.BAD, "warning": theme.WARN, "success": theme.GOOD}.get(kind, theme.TEXT)
    ttk.Label(d.body, text=title, style="CardH2.TLabel", foreground=colour).grid(row=0, column=0, sticky="w")
    ttk.Label(d.body, text=text, style="Card.TLabel", wraplength=470, justify="left").grid(row=1, column=0, sticky="w",
                                                                                           pady=(8, 0))
    if details:
        ttk.Label(d.body, text="Technical details are in the log file (Settings → Open logs folder).",
                  style="CardMuted.TLabel", wraplength=470).grid(row=2, column=0, sticky="w", pady=(8, 0))
    d.buttons([("OK", True, "Accent.TButton")])
    d.show()


def error(parent, text: str, title: str = "Something went wrong", details: str | None = None) -> None:
    message(parent, title, text, "error", details)


def ask(parent, title: str, text: str, ok: str = "OK", cancel: str = "Cancel", danger: bool = False) -> bool:
    d = Dialog(parent, title)
    ttk.Label(d.body, text=title, style="CardH2.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(d.body, text=text, style="Card.TLabel", wraplength=470, justify="left").grid(row=1, column=0, sticky="w",
                                                                                           pady=(8, 0))
    d.buttons([(cancel, False, "TButton"), (ok, True, "Danger.TButton" if danger else "Accent.TButton")])
    return bool(d.show())


def ask_string(parent, title: str, label: str, initial: str = "") -> str | None:
    d = Dialog(parent, title, width=440)
    ttk.Label(d.body, text=title, style="CardH2.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(d.body, text=label, style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 4))
    var = tk.StringVar(value=initial)
    e = ttk.Entry(d.body, textvariable=var, width=46)
    e.grid(row=2, column=0, sticky="ew")
    e.focus_set()
    e.select_range(0, "end")
    d.buttons([("Cancel", None, "TButton"), ("Save", "ok", "Accent.TButton")])
    e.bind("<Return>", lambda _e: d.close("ok"))
    return var.get().strip() if d.show() == "ok" else None


def confirm_delete(parent, plan, action_label: str, require_text: str | None = None) -> tuple[bool, bool]:
    """Deletion confirmation for a DeletionPlan. Returns (confirmed, confirmed_protected).

    The destructive button stays disabled until the user ticks the acknowledgement (and types
    the name, for whole models), so one accidental click can never delete anything.
    """
    from maniabot.storage import human_size

    if plan.blocked_reason:
        message(parent, "Can't delete right now", plan.blocked_reason, "warning")
        return False, False
    d = Dialog(parent, plan.title, width=560)
    ttk.Label(d.body, text=plan.title, style="CardH2.TLabel", foreground=theme.BAD).grid(row=0, column=0, sticky="w")
    ttk.Label(d.body, text="This will permanently remove:", style="Card.TLabel").grid(row=1, column=0, sticky="w",
                                                                                     pady=(10, 2))
    lst = "\n".join(f"  •  {i}" for i in plan.items) + f"\n\n  {human_size(plan.size_bytes)} will be freed."
    ttk.Label(d.body, text=lst, style="Card.TLabel", justify="left").grid(row=2, column=0, sticky="w")
    r = 3
    for w in plan.warnings:
        ttk.Label(d.body, text=f"⚠  {w}", style="CardWarn.TLabel", wraplength=500, justify="left").grid(
            row=r, column=0, sticky="w", pady=(6, 0))
        r += 1
    ttk.Label(d.body, text="This cannot be undone.", style="CardError.TLabel").grid(row=r, column=0, sticky="w",
                                                                                    pady=(10, 0))
    r += 1
    ack = tk.BooleanVar(value=False)
    prot = tk.BooleanVar(value=not plan.needs_protected_confirmation)
    typed = tk.StringVar()
    ttk.Checkbutton(d.body, text="I understand these files will be permanently deleted", variable=ack,
                    style="Card.TCheckbutton").grid(row=r, column=0, sticky="w", pady=(10, 0))
    r += 1
    if plan.needs_protected_confirmation:
        ttk.Checkbutton(d.body, text="Yes, also delete PROTECTED versions", variable=prot,
                        style="Card.TCheckbutton").grid(row=r, column=0, sticky="w", pady=(4, 0))
        r += 1
    if require_text:
        ttk.Label(d.body, text=f"Type the model name ({require_text}) to confirm:", style="CardMuted.TLabel").grid(
            row=r, column=0, sticky="w", pady=(10, 2))
        ttk.Entry(d.body, textvariable=typed).grid(row=r + 1, column=0, sticky="ew")
        r += 2
    d.buttons([("Cancel", False, "TButton"), (action_label, True, "Danger.TButton")])
    btn = d._buttons[action_label]

    def refresh(*_):
        ok = ack.get() and prot.get() and (not require_text or typed.get().strip() == require_text)
        btn.state(["!disabled"] if ok else ["disabled"])

    for v in (ack, prot, typed):
        v.trace_add("write", refresh)
    refresh()
    confirmed = bool(d.show())
    return confirmed, confirmed and plan.needs_protected_confirmation


# ------------------------------------------------------- background work
class Tasks:
    """Run callables on worker threads and deliver results on the Tk thread."""

    def __init__(self, root: tk.Misc):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.active = 0
        self.root.after(40, self._pump)

    def run(self, fn: Callable[[], Any], on_done: Callable[[Any], None] | None = None,
            on_error: Callable[[BaseException, str], None] | None = None, name: str = "task") -> None:
        self.active += 1

        def work():
            try:
                res = fn()
                self.q.put((on_done, (res,), None))
            except BaseException as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                log.error("background %s failed: %s\n%s", name, exc, tb)
                self.q.put((on_error or self._default_error, (exc, tb), None))

        threading.Thread(target=work, name=name, daemon=True).start()

    def post(self, fn: Callable, *args) -> None:
        """Call fn(*args) on the Tk thread (safe from worker threads)."""
        self.q.put((fn, args, "post"))

    def _default_error(self, exc: BaseException, tb: str) -> None:
        error(self.root, str(exc) or type(exc).__name__, details=tb)

    def _pump(self) -> None:
        try:
            while True:
                fn, args, kind = self.q.get_nowait()
                if kind is None:
                    self.active -= 1
                if fn:
                    try:
                        fn(*args)
                    except Exception:  # noqa: BLE001
                        log.exception("UI callback failed")
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(40, self._pump)
