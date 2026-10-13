"""Comparison charts: learning-curve overlay and metric bar charts for two versions.

Pure data + matplotlib `Figure` building (no pyplot, no Tk), so it is safe off the UI thread and
easy to test; the Compare dialog embeds the figure with FigureCanvasTkAgg.
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from maniabot.evaluation import Comparison

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from maniabot.models import ModelStore

log = logging.getLogger("maniabot.charts")

# Colours mirror maniabot.ui.theme; A/B are darker steps of its blue/pink, validated for
# CVD separation and contrast against the dark chart surface.
SURFACE = "#1f1f29"
GRID = "#34344a"
TEXT = "#ececf3"
MUTED = "#9a9ab0"
SERIES = ("#3d94e8", "#e84d93")  # A, B

# (key, short label, panel) - panels share a unit so each chart has a single honest y axis.
BAR_METRICS: list[tuple[str, str, str]] = [
    ("accuracy", "Accuracy", "pct"),
    ("hit_rate", "Notes hit", "pct"),
    ("hold_accuracy", "Holds", "pct"),
    ("lane1", "Lane 1", "pct"),
    ("lane2", "Lane 2", "pct"),
    ("lane3", "Lane 3", "pct"),
    ("lane4", "Lane 4", "pct"),
    ("misses", "Misses", "count"),
    ("extra_presses", "Extra presses", "count"),
    ("broken_holds", "Broken holds", "count"),
    ("mean_offset_ms", "Mean offset", "ms"),
    ("timing_std_ms", "Deviation", "ms"),
    ("mean_abs_error_ms", "Mean error", "ms"),
]
PANELS = {  # panel -> (title, which way is better)
    "pct": ("Accuracy (%)", "higher is better"),
    "count": ("Mistakes", "lower is better"),
    "ms": ("Timing (ms)", "closer to 0 is better"),
}


@dataclass
class Curve:
    timesteps: np.ndarray  # cumulative training timesteps at the end of each play
    accuracy: np.ndarray   # rolling-mean accuracy in %
    window: int
    sessions: int          # how many training sessions (lineage) the curve spans


def read_episodes(csv_path: Path) -> tuple[list[int], list[float]]:
    """(timesteps, accuracy 0..1) per finished play from a session's episodes CSV."""
    steps, accs = [], []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    t, a = int(row["timesteps"]), float(row["accuracy"])
                except (KeyError, TypeError, ValueError):
                    continue  # partially written last line of an interrupted session
                steps.append(t)
                accs.append(a)
    except OSError:
        log.warning("cannot read episode log %s", csv_path)
    return steps, accs


def smooth(steps: list[int], accs: list[float], window: int = 50) -> tuple[np.ndarray, np.ndarray, int]:
    """Rolling mean of accuracy (in %), aligned to the timestep of each window's last play."""
    if len(accs) < 2:
        return np.array([]), np.array([]), 0
    w = max(1, min(window, len(accs)))
    y = np.convolve(np.asarray(accs, dtype=float), np.ones(w) / w, mode="valid") * 100
    x = np.asarray(steps[w - 1:], dtype=float)
    return x, y, w


def lineage_sessions(store: "ModelStore", model_id: str, version: str) -> list[str]:
    """Session ids that trained this version, oldest first (follows parent versions)."""
    out: list[str] = []
    seen: set[str] = set()
    cur: str | None = version
    while cur and cur not in seen:
        seen.add(cur)
        try:
            rec = store.get_version(model_id, cur)
        except Exception:  # noqa: BLE001 - a deleted parent just ends the lineage
            break
        sid = rec.meta.get("session_id")
        if sid:
            out.append(sid)
        cur = rec.parent
    return out[::-1]


def version_curve(store: "ModelStore", model_id: str, version: str, window: int = 50) -> Curve | None:
    """Learning curve of everything that trained `version`, including its parent versions."""
    steps: list[int] = []
    accs: list[float] = []
    used = 0
    for sid in lineage_sessions(store, model_id, version):
        path = store.sessions_dir(model_id) / f"{sid}_episodes.csv"
        if not path.exists():
            continue
        s, a = read_episodes(path)
        if not s:
            continue
        # Branching from an older version restarts the counter at that point; drop any overlap.
        start = next((i for i, t in enumerate(s) if not steps or t > steps[-1]), len(s))
        steps += s[start:]
        accs += a[start:]
        used += 1
    x, y, w = smooth(steps, accs, window)
    if not len(x):
        return None
    return Curve(x, y, w, used)


def _bar_values(cmp: Comparison) -> dict[str, tuple[float | None, float | None, str | None]]:
    rows = {r["key"]: r for r in cmp.rows}
    out = {}
    for key, _label, panel in BAR_METRICS:
        r = rows.get(key)
        if not r or (r["a"] is None and r["b"] is None):
            continue
        scale = 100.0 if panel == "pct" else 1.0
        conv = [None if v is None else float(v) * scale for v in (r["a"], r["b"])]
        out[key] = (conv[0], conv[1], r["winner"])
    return out


def _style_axes(ax, title: str, hint: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=9.5, loc="left", pad=16 if hint else 8)
    if hint:
        ax.text(0, 1.025, hint, transform=ax.transAxes, color=MUTED, fontsize=7.5, va="bottom")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _fmt_bar(v: float, panel: str) -> str:
    if panel in ("pct", "ms"):
        return f"{v:.1f}"
    return f"{v:,.0f}" if abs(v) >= 10 else f"{v:.3g}"


def _fmt_full(v: float, panel: str) -> str:
    return {"pct": f"{v:.2f}%", "ms": f"{v:.2f} ms"}.get(panel, f"{v:,.4g}")


def _margin(a: float, b: float, key: str, panel: str) -> str:
    """How much better the winner is: "+0.9" (points), "−272" (mistakes), "0.8 closer" (offset)."""
    if key == "mean_offset_ms":
        return f"{abs(abs(a) - abs(b)):.1f} closer"
    return f"{'+' if panel == 'pct' else '−'}{_fmt_bar(abs(a - b), panel)}"


def _empty(ax, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)


def _draw_curves(fig, ax, curves: tuple[Curve | None, Curve | None], labels: tuple[str, str]) -> None:
    _style_axes(ax, "Learning curve", "rolling accuracy (%) by training timesteps")
    drawn = False
    for curve, label, colour in zip(curves, labels, SERIES):
        if curve is None:
            continue
        ax.plot(curve.timesteps, curve.accuracy, color=colour, linewidth=2, label=label, solid_capstyle="round")
        # End-of-training dot; the final value is in the legend so no label sits on the other line.
        ax.plot([curve.timesteps[-1]], [curve.accuracy[-1]], "o", color=colour, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        drawn = True
    if not drawn:
        _empty(ax, "No training log for either version (imported, or logs were deleted).")
        return
    missing = [lab for c, lab in zip(curves, labels) if c is None]
    if missing:
        ax.text(0.99, 0.04, f"No training log for {missing[0]}", transform=ax.transAxes, ha="right",
                color=MUTED, fontsize=8)
    windows = sorted({c.window for c in curves if c})
    ax.set_xlabel(f"timesteps  ·  rolling mean over {'/'.join(map(str, windows))} plays", color=MUTED, fontsize=8)
    ax.xaxis.set_major_formatter(_steps_formatter())
    ax.margins(x=0.03)
    fig._mb_curve = (ax, curves, labels)


def _steps(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v / 1e6:.3g}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.3g}k"
    return f"{v:.0f}"


def _steps_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos: _steps(v) if v else "0")


def _draw_bars(fig, ax, panel: str, values: dict, labels: tuple[str, str], detail: bool) -> None:
    _style_axes(ax, *PANELS[panel])
    keys = [(k, lab) for k, lab, p in BAR_METRICS if p == panel and k in values]
    if not keys:
        _empty(ax, "No data")
        return
    x = np.arange(len(keys))
    width = 0.38
    tops = [0.0] * len(keys)
    for side, (colour, name) in enumerate(zip(SERIES, labels)):
        vals = [values[k][side] for k, _ in keys]
        heights = [math.nan if v is None else v for v in vals]
        offs = x + (side - 0.5) * (width + 0.02)  # small gap between the A and B bars
        bars = ax.bar(offs, heights, width, color=colour, label=name, edgecolor=SURFACE, linewidth=1)
        for i, (b, v, (k, metric)) in enumerate(zip(bars, vals, keys)):
            if v is None:
                continue
            tops[i] = max(tops[i], v)
            fig._mb_bars.append((b, ax, f"{metric}\n{name}: {_fmt_full(v, panel)}"))
            if detail:
                won = values[k][2] == ("a", "b")[side]
                y = b.get_height()
                ax.annotate(_fmt_bar(v, panel), (b.get_x() + b.get_width() / 2, y),
                            xytext=(0, 2 if y >= 0 else -2), textcoords="offset points",
                            ha="center", va="bottom" if y >= 0 else "top", fontsize=8,
                            color=TEXT if won else MUTED, fontweight="bold" if won else "normal")
    if not detail:
        # One label per pair - who is better and by how much; exact values are on hover and in the table.
        for i, (k, _) in enumerate(keys):
            a, b, winner = values[k]
            if a is None or b is None or (a != b and not winner):
                continue
            text = "same" if a == b else f"{winner.upper()} {_margin(a, b, k, panel)}"
            ax.annotate(text, (x[i], tops[i]), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.5, color=MUTED if a == b else TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in keys], rotation=35, ha="right", rotation_mode="anchor")
    if panel == "pct":
        ax.set_ylim(0, 110)  # bars start at zero; the labels carry the precise differences
    else:
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.margins(y=0.18)


def comparison_figure(cmp: Comparison, curves: tuple[Curve | None, Curve | None],
                      labels: tuple[str, str] = ("A", "B"), dpi: float = 100,
                      figsize: tuple[float, float] = (9.0, 6.6), detail: bool = False) -> "Figure":
    """Learning-curve overlay on top, then one bar panel per unit (%, counts, ms).

    detail=False (the in-dialog view) labels each bar pair once with the winner and margin;
    detail=True (full screen) labels every bar with its value. attach_hover() adds tooltips.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    fig = Figure(figsize=figsize, dpi=dpi, facecolor=SURFACE)
    fig._mb_bars = []
    fig._mb_curve = None
    gs = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.1), width_ratios=(2.5, 1.2, 1.2),
                          hspace=0.62, wspace=0.3, left=0.07, right=0.97, top=0.9, bottom=0.14)
    curve_ax = fig.add_subplot(gs[0, :])
    _draw_curves(fig, curve_ax, curves, labels)
    values = _bar_values(cmp)
    for col, panel in enumerate(("pct", "count", "ms")):
        _draw_bars(fig, fig.add_subplot(gs[1, col]), panel, values, labels, detail)
    legend = [lab + (f"  ·  ends at {c.accuracy[-1]:.1f}%" if c is not None else "") for c, lab in zip(curves, labels)]
    fig.legend(handles=[Patch(color=c, label=lab) for c, lab in zip(SERIES, legend)], loc="upper left",
               ncol=2, frameon=False, fontsize=8.5, labelcolor=TEXT, bbox_to_anchor=(0.06, 0.995))
    hint = "Bold value = the better of the two" if detail else "Bar labels: which version is better, and by how much"
    curve_ax.text(1, 1.025, hint + "  ·  hover for exact values", transform=curve_ax.transAxes, ha="right",
                  va="bottom", color=MUTED, fontsize=7.5)
    return fig


def attach_hover(fig) -> None:
    """Tooltips: the value of the bar under the cursor; on the curve, both series at that timestep."""
    tip = fig.text(0, 0, "", fontsize=8.5, color=TEXT, visible=False, zorder=10,
                   bbox=dict(boxstyle="round,pad=0.45", facecolor="#282834", edgecolor=GRID))
    guide = fig._mb_curve[0].axvline(0, color=MUTED, linewidth=0.8, visible=False) if fig._mb_curve else None

    def show(event, text: str) -> None:
        w, h = fig.canvas.get_width_height()
        right, top = event.x > w * 0.7, event.y > h * 0.6  # flip near edges so the tip stays on screen
        tip.set_text(text)
        tip.set_ha("right" if right else "left")
        tip.set_va("top" if top else "bottom")
        tip.set_position(((event.x + (-12 if right else 12)) / w, (event.y + (-12 if top else 12)) / h))
        tip.set_visible(True)

    def curve_text(xdata: float) -> str | None:
        _, curves, labels = fig._mb_curve
        lines = []
        for c, lab in zip(curves, labels):
            if c is None or not c.timesteps[0] <= xdata <= c.timesteps[-1]:
                continue
            i = min(int(np.searchsorted(c.timesteps, xdata)), len(c.timesteps) - 1)
            if i and xdata - c.timesteps[i - 1] < c.timesteps[i] - xdata:
                i -= 1  # nearest play, not the next one
            lines.append(f"{lab}: {c.accuracy[i]:.2f}%")
        return "\n".join([f"{_steps(xdata)} timesteps", *lines]) if lines else None

    def on_move(event) -> None:
        was_shown = tip.get_visible()
        tip.set_visible(False)
        if guide is not None:
            guide.set_visible(False)
        ax = event.inaxes
        if ax is not None:
            hit = next((text for bar, bax, text in fig._mb_bars if bax is ax and bar.contains(event)[0]), None)
            if hit:
                show(event, hit)
            elif guide is not None and ax is fig._mb_curve[0] and event.xdata is not None:
                text = curve_text(event.xdata)
                if text:
                    guide.set_xdata([event.xdata, event.xdata])
                    guide.set_visible(True)
                    show(event, text)
        if was_shown or tip.get_visible():
            fig.canvas.draw_idle()

    fig._mb_hover = fig.canvas.mpl_connect("motion_notify_event", on_move)


def metrics_have_data(cmp: Comparison) -> bool:
    return bool(_bar_values(cmp))


def save_png(fig: Any, path: Path) -> Path:
    path = Path(path)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    return path
