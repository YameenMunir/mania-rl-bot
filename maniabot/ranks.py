"""osu!mania grade (rank) for a play, from the simulator's judgement counts.

Grades depend on accuracy only (no mods are simulated, so there are no silver ranks):

    SS  100% (every note MAX or 300)
    S   over 95%
    A   over 90%
    B   over 80%
    C   over 70%
    D   otherwise

osu! re-judges a replay with its own rules, so the rank on its results screen can differ slightly
from this one, mostly on the 50/100 edges and on hold-note releases.
"""
from __future__ import annotations

from typing import Sequence

RANKS = ("SS", "S", "A", "B", "C", "D")
RANK_COLOURS = {
    "SS": "#ffe14d", "S": "#ffc531", "A": "#7bdc4a", "B": "#4aa8ff", "C": "#b77bff", "D": "#ff5c6c",
}
_THRESHOLDS = (("S", 0.95), ("A", 0.90), ("B", 0.80), ("C", 0.70))


def mania_rank(accuracy: float | None, counts: Sequence[int] | None = None) -> str | None:
    """Grade for an accuracy in 0..1. `counts` is [MAX, 300, 200, 100, 50, miss] when known.

    Passing the counts makes SS exact: a single 200 in a very long map still isn't SS even though its
    accuracy rounds to 100%.
    """
    if accuracy is None:
        return None
    if counts is not None:
        if sum(counts) == 0:
            return None
        perfect = sum(counts[2:]) == 0
    else:
        perfect = accuracy >= 1.0 - 1e-12
    if perfect:
        return "SS"
    for rank, minimum in _THRESHOLDS:
        if accuracy > minimum:
            return rank
    return "D"
