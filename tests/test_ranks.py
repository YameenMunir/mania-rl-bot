"""osu!mania grades from accuracy / judgement counts."""
from __future__ import annotations

import pytest

from maniabot.ranks import RANK_COLOURS, RANKS, mania_rank


@pytest.mark.parametrize("acc, expected", [
    (1.0, "SS"), (0.9999, "S"), (0.9501, "S"), (0.95, "A"), (0.9001, "A"), (0.90, "B"),
    (0.8001, "B"), (0.80, "C"), (0.7001, "C"), (0.70, "D"), (0.0, "D"),
])
def test_rank_thresholds(acc, expected):
    assert mania_rank(acc) == expected


def test_ss_needs_every_note_max_or_300():
    assert mania_rank(1.0, [190, 70, 0, 0, 0, 0]) == "SS"
    # one 200 in 200,000 notes rounds to 100.00% accuracy but is not an SS
    assert mania_rank(0.9999983, [100_000, 99_999, 1, 0, 0, 0]) == "S"


def test_a_single_miss_costs_the_ss():
    assert mania_rank(0.99, [100, 99, 0, 0, 0, 1]) == "S"


def test_the_screenshot_play_is_an_s():
    # 190 MAX, 58x300, 10x200, 2x100: shown as 98.21% in the app
    counts = [190, 58, 10, 2, 0, 0]
    acc = (300 * 248 + 200 * 10 + 100 * 2) / (300 * 260)
    assert round(acc * 100, 2) == 98.21
    assert mania_rank(acc, counts) == "S"


def test_unknown_or_empty_plays_have_no_rank():
    assert mania_rank(None) is None
    assert mania_rank(1.0, [0, 0, 0, 0, 0, 0]) is None


def test_every_rank_has_a_colour():
    assert set(RANK_COLOURS) == set(RANKS)
