"""Colours, fonts and ttk styles (a dark theme with an osu!-pink accent)."""
from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

BG = "#16161d"
SIDEBAR = "#101016"
SURFACE = "#1f1f29"
SURFACE_2 = "#282834"
BORDER = "#34344a"
TEXT = "#ececf3"
MUTED = "#9a9ab0"
FAINT = "#6c6c82"
ACCENT = "#ff66aa"
ACCENT_DARK = "#d94c8c"
GOOD = "#5ad19a"
WARN = "#f2c14e"
BAD = "#ff6b6b"
INFO = "#6cb6ff"
LANE_COLOURS = ("#6cb6ff", "#ff66aa", "#ff66aa", "#6cb6ff")

BADGE_COLOURS = {
    "Active": ACCENT,
    "Best": GOOD,
    "Latest": INFO,
    "Protected": WARN,
    "Favourite": "#c792ea",
    "Archived": FAINT,
    "completed": GOOD,
    "stopped": WARN,
    "cancelled": FAINT,
    "failed": BAD,
    "interrupted": BAD,
    "running": INFO,
    "paused": WARN,
    "imported": FAINT,
}

FAMILY = "Segoe UI" if sys.platform == "win32" else "Helvetica"
MONO = "Consolas" if sys.platform == "win32" else "Courier"


def fonts() -> dict[str, tuple]:
    return {
        "body": (FAMILY, 10),
        "small": (FAMILY, 9),
        "bold": (FAMILY, 10, "bold"),
        "h1": (FAMILY, 20, "bold"),
        "h2": (FAMILY, 13, "bold"),
        "h3": (FAMILY, 11, "bold"),
        "stat": (FAMILY, 22, "bold"),
        "mono": (MONO, 9),
        "nav": (FAMILY, 11),
    }


def enable_dpi_awareness() -> None:
    """Crisp text on scaled Windows displays (must run before the Tk root is created)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


SCALE = 1.0  # set by apply(): screen DPI / 96


def ui_scale(root: tk.Misc) -> float:
    return max(1.0, root.winfo_fpixels("1i") / 96.0)


def px(n: float) -> int:
    """Scale a 96-DPI pixel size to the current display."""
    return int(round(n * SCALE))


ASSETS = Path(__file__).resolve().parents[2] / "assets"


def load_image(name: str) -> tk.PhotoImage | None:
    """A PNG from the repo's assets folder, or None if it's missing (the logo is optional)."""
    try:
        return tk.PhotoImage(file=str(ASSETS / name))
    except (tk.TclError, OSError):
        return None


def logo_image() -> tk.PhotoImage | None:
    """The sidebar logo, in the pre-rendered size closest to a 40 px (96-DPI) badge."""
    size = 40 if SCALE < 1.25 else 60 if SCALE < 1.75 else 80
    return load_image(f"logo_{size}.png")


def apply(root: tk.Tk) -> None:
    global SCALE
    SCALE = ui_scale(root)
    f = fonts()
    tkfont.nametofont("TkDefaultFont").configure(family=FAMILY, size=10)
    root.configure(bg=BG)
    root.option_add("*TCombobox*Listbox.background", SURFACE_2)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=BG, foreground=TEXT, fieldbackground=SURFACE_2, bordercolor=BORDER,
                lightcolor=BORDER, darkcolor=BORDER, troughcolor=SURFACE_2, focuscolor=ACCENT,
                selectbackground=ACCENT, selectforeground="#ffffff", insertcolor=TEXT, font=f["body"])
    s.configure("TFrame", background=BG)
    s.configure("Card.TFrame", background=SURFACE, relief="flat")
    s.configure("Sidebar.TFrame", background=SIDEBAR)
    s.configure("TLabel", background=BG, foreground=TEXT)
    s.configure("Card.TLabel", background=SURFACE)
    s.configure("Muted.TLabel", foreground=MUTED)
    s.configure("CardMuted.TLabel", background=SURFACE, foreground=MUTED)
    s.configure("H1.TLabel", font=f["h1"])
    s.configure("H2.TLabel", font=f["h2"])
    s.configure("CardH2.TLabel", font=f["h2"], background=SURFACE)
    s.configure("CardH3.TLabel", font=f["h3"], background=SURFACE)
    s.configure("Stat.TLabel", font=f["stat"], background=SURFACE)
    s.configure("Error.TLabel", foreground=BAD)
    s.configure("CardError.TLabel", foreground=BAD, background=SURFACE)
    s.configure("CardWarn.TLabel", foreground=WARN, background=SURFACE)
    s.configure("CardGood.TLabel", foreground=GOOD, background=SURFACE)
    s.configure("Brand.TLabel", background=SIDEBAR, foreground=ACCENT, font=(FAMILY, 15, "bold"))
    s.configure("SidebarMuted.TLabel", background=SIDEBAR, foreground=FAINT, font=f["small"])

    s.configure("TButton", background=SURFACE_2, foreground=TEXT, padding=(12, 6), borderwidth=0)
    s.map("TButton", background=[("disabled", SURFACE), ("pressed", BORDER), ("active", BORDER)],
          foreground=[("disabled", FAINT)])
    s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", font=f["bold"])
    s.map("Accent.TButton", background=[("disabled", SURFACE_2), ("pressed", ACCENT_DARK), ("active", ACCENT_DARK)])
    s.configure("Danger.TButton", background="#4a2230", foreground=BAD)
    s.map("Danger.TButton", background=[("disabled", SURFACE), ("active", "#5e2a3c")], foreground=[("disabled", FAINT)])
    s.configure("Link.TButton", background=BG, foreground=ACCENT, padding=(2, 2))
    s.map("Link.TButton", background=[("active", BG)], foreground=[("active", TEXT)])
    s.configure("CardLink.TButton", background=SURFACE, foreground=ACCENT, padding=(2, 2))
    s.map("CardLink.TButton", background=[("active", SURFACE)], foreground=[("active", TEXT)])
    s.configure("Nav.TButton", background=SIDEBAR, foreground=MUTED, anchor="w", padding=(18, 9), font=f["nav"])
    s.map("Nav.TButton", background=[("active", SURFACE)], foreground=[("active", TEXT)])
    s.configure("NavActive.TButton", background=SURFACE, foreground=TEXT, anchor="w", padding=(18, 9),
                font=(FAMILY, 11, "bold"))
    s.configure("SubNav.TButton", background=SIDEBAR, foreground=FAINT, anchor="w", padding=(34, 5), font=f["small"])
    s.map("SubNav.TButton", background=[("active", SURFACE)], foreground=[("active", TEXT)])

    for w in ("TEntry", "TSpinbox", "TCombobox"):
        s.configure(w, fieldbackground=SURFACE_2, foreground=TEXT, background=SURFACE_2, arrowcolor=TEXT,
                    padding=5, bordercolor=BORDER)
        s.map(w, fieldbackground=[("readonly", SURFACE_2), ("disabled", SURFACE)],
              foreground=[("disabled", FAINT)], bordercolor=[("focus", ACCENT)])
    s.configure("TCheckbutton", background=BG, foreground=TEXT, indicatorbackground=SURFACE_2)
    s.map("TCheckbutton", background=[("active", BG)], indicatorbackground=[("selected", ACCENT)])
    s.configure("Card.TCheckbutton", background=SURFACE)
    s.map("Card.TCheckbutton", background=[("active", SURFACE)], indicatorbackground=[("selected", ACCENT)])
    s.configure("TRadiobutton", background=SURFACE, foreground=TEXT, indicatorbackground=SURFACE_2)
    s.map("TRadiobutton", background=[("active", SURFACE)], indicatorbackground=[("selected", ACCENT)])
    s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=SURFACE_2, borderwidth=0, thickness=px(10))
    s.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, foreground=TEXT, rowheight=px(26),
                borderwidth=0)
    s.map("Treeview", background=[("selected", "#3a2a40")], foreground=[("selected", TEXT)])
    s.configure("Treeview.Heading", background=SURFACE_2, foreground=MUTED, relief="flat", font=f["small"],
                padding=(6, 5))
    s.map("Treeview.Heading", background=[("active", BORDER)])
    s.configure("TNotebook", background=BG, borderwidth=0)
    s.configure("TNotebook.Tab", background=SURFACE, foreground=MUTED, padding=(14, 6), borderwidth=0)
    s.map("TNotebook.Tab", background=[("selected", SURFACE_2)], foreground=[("selected", TEXT)])
    s.configure("TSeparator", background=BORDER)
    s.configure("Vertical.TScrollbar", background=SURFACE_2, troughcolor=BG, arrowcolor=MUTED, borderwidth=0)
    s.configure("TLabelframe", background=SURFACE, bordercolor=BORDER)
    s.configure("TLabelframe.Label", background=SURFACE, foreground=MUTED)
