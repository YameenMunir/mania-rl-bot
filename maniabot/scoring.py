"""osu!mania stable score (ScoreV1), the number osu! shows for a play: 1,000,000 at most.

From the osu! wiki ("ScoreV1 (osu!mania)"), per judged note:
  unit       = 1,000,000 * ModMultiplier * 0.5 / TotalNotes
  base       = unit * HitValue / 320
  bonus part = unit * HitBonusValue * sqrt(Bonus) / 320
  Bonus      = Bonus + HitBonus - HitPunishment / ModDivider, kept in [0, 100], starting at 100

So half of the score is accuracy and half rewards keeping a run of good hits. Only an all-MAX play
(every note a rainbow 300) reaches 1,000,000; an all-300 play scores 968,750.
"""
from __future__ import annotations

import math

from maniabot.mods import DT, EZ, HR, HT

MAX_SCORE = 1_000_000
# Indexed like mania_env.JUDGEMENTS: MAX, 300, 200, 100, 50, miss.
HIT_VALUE = (320, 300, 200, 100, 50, 0)
HIT_BONUS_VALUE = (32, 32, 16, 8, 4, 0)
HIT_BONUS = (2, 1, 0, 0, 0, 0)
HIT_PUNISHMENT = (0, 0, 8, 24, 44, math.inf)


def mod_multiplier(mods: int) -> float:
    """Score multiplier: x0.5 for each of EZ and HT."""
    m = 1.0
    if mods & EZ:
        m *= 0.5
    if mods & HT:
        m *= 0.5
    return m


def mod_divider(mods: int) -> float:
    """Divides the bonus lost on a 200 or worse: HR 1.08, DT / NC 1.1."""
    d = 1.0
    if mods & HR:
        d *= 1.08
    if mods & DT:
        d *= 1.1
    return d


class ScoreV1:
    """Running osu!mania ScoreV1 for one play; call add() for every judgement, in order."""

    def __init__(self, total_notes: int, mods: int = 0):
        self.unit = MAX_SCORE * mod_multiplier(mods) * 0.5 / max(1, total_notes)
        self.divider = mod_divider(mods)
        self.bonus = 100.0
        self.raw = 0.0

    def add(self, judgement: int) -> None:
        self.bonus = min(100.0, max(0.0, self.bonus + HIT_BONUS[judgement] - HIT_PUNISHMENT[judgement] / self.divider))
        self.raw += self.unit * (HIT_VALUE[judgement] + HIT_BONUS_VALUE[judgement] * math.sqrt(self.bonus)) / 320.0

    @property
    def value(self) -> int:
        return int(round(self.raw))
