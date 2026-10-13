"""Comparison charts: episode logs -> learning curves (across lineage) and the comparison figure."""
from __future__ import annotations

from maniabot import charts
from maniabot.evaluation import compare
from maniabot.training import run_training
from tests.conftest import tiny_spec

HEADER = "play,timesteps,accuracy,misses,broken_holds,stray_presses,map\n"


def write_log(path, steps, accs):
    rows = "".join(f"{i + 1},{t},{a:.5f},0,0,0,\"map, with comma\"\n" for i, (t, a) in enumerate(zip(steps, accs)))
    path.write_text(HEADER + rows, encoding="utf-8")


def test_read_and_smooth(tmp_path):
    p = tmp_path / "s_episodes.csv"
    write_log(p, [100, 200, 300, 400], [0.1, 0.2, 0.3, 0.4])
    with open(p, "a", encoding="utf-8") as f:
        f.write("5,50")  # half-written line from an interrupted session
    steps, accs = charts.read_episodes(p)
    assert steps == [100, 200, 300, 400] and accs == [0.1, 0.2, 0.3, 0.4]
    x, y, w = charts.smooth(steps, accs, window=2)
    assert w == 2 and list(x) == [200, 300, 400]
    assert [round(v, 6) for v in y] == [15.0, 25.0, 35.0]
    assert charts.smooth([1], [0.5])[2] == 0  # one play is not a curve


def test_version_curve_follows_lineage(store, trained, small_maps):
    session = run_training(store, tiny_spec(trained, small_maps, kind="continue", source_version="v001"))
    assert session["status"] == "completed", session.get("error")
    v1, v2 = store.get_version(trained, "v001"), store.get_version(trained, "v002")
    assert charts.lineage_sessions(store, trained, "v002") == [v1.meta["session_id"], v2.meta["session_id"]]
    sd = store.sessions_dir(trained)
    write_log(sd / f"{v1.meta['session_id']}_episodes.csv", [100, 200, 300, 400], [0.2, 0.4, 0.6, 0.8])
    # the second session overlaps the first by one play (e.g. branched from a checkpoint); it is dropped
    write_log(sd / f"{v2.meta['session_id']}_episodes.csv", [400, 500, 600], [0.0, 1.0, 1.0])

    c1 = charts.version_curve(store, trained, "v001", window=1)
    assert c1.sessions == 1 and list(c1.timesteps) == [100, 200, 300, 400]
    c2 = charts.version_curve(store, trained, "v002", window=1)
    assert c2.sessions == 2
    assert list(c2.timesteps) == [100, 200, 300, 400, 500, 600]
    assert list(c2.accuracy) == [20.0, 40.0, 60.0, 80.0, 100.0, 100.0]

    (sd / f"{v1.meta['session_id']}_episodes.csv").unlink()
    (sd / f"{v2.meta['session_id']}_episodes.csv").unlink()
    assert charts.version_curve(store, trained, "v002") is None


def test_comparison_figure_renders(tmp_path):
    a = {"accuracy": 0.91, "misses": 40, "mean_offset_ms": -3.0, "lanes": [{"accuracy": 0.9}] * 4}
    b = {"accuracy": 0.94, "misses": 25, "mean_offset_ms": 1.5, "lanes": [{"accuracy": 0.95}] * 4}
    x, y, w = charts.smooth(list(range(100, 1100, 100)), [i / 10 for i in range(10)], window=3)
    curve = charts.Curve(x, y, w, 1)

    fig = charts.comparison_figure(compare(a, b), (curve, None), ("A: bot v001", "B: bot v002"))
    assert len(fig.axes) == 4  # learning curve + %, counts, ms panels
    out = charts.save_png(fig, tmp_path / "compare.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    assert not charts.metrics_have_data(compare(None, None))
    empty = charts.comparison_figure(compare(None, None), (None, None))  # nothing at all still renders
    charts.save_png(empty, tmp_path / "empty.png")


def _texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def test_compact_labels_once_per_pair_detail_labels_every_bar():
    a = {"accuracy": 0.91, "misses": 40, "mean_offset_ms": -3.0}
    b = {"accuracy": 0.94, "misses": 25, "mean_offset_ms": 1.5}
    compact = _texts(charts.comparison_figure(compare(a, b), (None, None)))
    assert "B +3.0" in compact and "B −15" in compact and "B 1.5 closer" in compact
    assert "91.0" not in compact  # no raw per-bar numbers in the compact view
    detail = _texts(charts.comparison_figure(compare(a, b), (None, None), detail=True))
    assert {"91.0", "94.0", "40", "25", "-3.0", "1.5"} <= set(detail)


def test_hover_shows_bar_value_and_curve_values():
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    x, y, w = charts.smooth(list(range(100, 1100, 100)), [i / 10 for i in range(10)], window=1)
    fig = charts.comparison_figure(compare({"accuracy": 0.9}, {"accuracy": 0.8}), (charts.Curve(x, y, w, 1), None),
                                   ("A: v1", "B: v2"))
    FigureCanvasAgg(fig)
    charts.attach_hover(fig)
    fig.canvas.draw()
    tip = fig.texts[-1]

    def hover(ax, xy):
        px, py = ax.transData.transform(xy)
        fig.canvas.callbacks.process("motion_notify_event", MouseEvent("motion_notify_event", fig.canvas, px, py))
        return tip.get_text() if tip.get_visible() else None

    bar, bar_ax, _ = fig._mb_bars[0]
    assert hover(bar_ax, (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2)) == "Accuracy\nA: v1: 90.00%"
    text = hover(fig._mb_curve[0], (500, 50))
    assert text.startswith("500 timesteps") and "A: v1: 40.00%" in text
    assert hover(bar_ax, (5, 100)) is None  # empty space hides the tooltip
