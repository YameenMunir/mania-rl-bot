"""osu!mania ScoreV1: 1,000,000 only for an all-MAX play, and the same score in the app and the .osr."""
from __future__ import annotations

import pytest

from evaluate import scripted_action
from mania_env import ManiaEnv
from maniabot import mods as M
from maniabot.replays import ReplayHistory, generate_replay
from maniabot.scoring import ScoreV1
from osu_parser import Beatmap, Note
from tests.conftest import DEMO_MAPS
from tests.test_replays import parse_osr

MAX, J300, J200, J100, J50, MISS = range(6)


def play(judgements, mods=0):
    s = ScoreV1(len(judgements), mods)
    for j in judgements:
        s.add(j)
    return s.value


def test_only_all_max_is_a_million():
    assert play([MAX] * 500) == 1_000_000
    assert play([J300] * 500) == 968_750  # 300s are worth 300/320 of a MAX, with the full run bonus
    assert play([MAX] * 499 + [J300]) < 1_000_000
    assert play([MAX] * 499 + [J200]) < play([MAX] * 499 + [J300])


def test_bonus_rewards_keeping_a_run():
    """A miss empties the bonus, so it costs more early than late."""
    early = play([MISS] + [MAX] * 99)
    late = play([MAX] * 99 + [MISS])
    assert early < late < 1_000_000
    assert play([MISS] * 50) == 0


def test_mod_multipliers_and_dividers():
    assert play([MAX] * 100, M.DT) == play([MAX] * 100, M.HR) == 1_000_000
    assert play([MAX] * 100, M.EZ) == play([MAX] * 100, M.HT) == 500_000
    assert play([MAX] * 100, M.EZ | M.HT) == 250_000
    # HR and DT soften the bonus lost on a 200 or worse.
    mixed = [MAX, J200, J100] * 30
    assert play(mixed, M.HR) > play(mixed) and play(mixed, M.DT | M.HR) > play(mixed, M.HR)


def test_env_tracks_score():
    bm = Beatmap("t", "v", 4, 8.0, [Note(2000 + 250 * i, i % 4, 2000 + 250 * i) for i in range(40)], md5="x")
    env = ManiaEnv([bm], clip_seconds=None, dt_ms=1.0)  # 1 ms ticks: the oracle lands every note dead on
    env.reset(seed=0)
    done, info = False, {}
    while not done:
        _o, _r, term, trunc, info = env.step(scripted_action(env))
        done = term or trunc
    assert info["counts"][MAX] == 40 and info["score"] == 1_000_000
    env.reset(seed=0)
    done = False
    while not done:  # never press: every note is a miss
        _o, _r, term, trunc, info = env.step([0, 0, 0, 0])
        done = term or trunc
    assert info["score"] == 0


def test_replay_header_has_the_app_score(store, storage, trained):
    rec = generate_replay(store, ReplayHistory(storage), trained, "v001", DEMO_MAPS / "demo_002.osu")
    osr = parse_osr(open(rec["path"], "rb").read())
    assert osr["score"] == rec["metrics"]["score"]
    assert 0 <= osr["score"] < 1_000_000  # a barely trained model is far from all-MAX


@pytest.mark.parametrize("name", ["demo_000.osu", "demo_003.osu"])
def test_oracle_score_matches_judgements(name):
    """Recomputing ScoreV1 from the play's judgements in order gives the env's score."""
    from osu_parser import parse_osu

    env = ManiaEnv([parse_osu(DEMO_MAPS / name)], clip_seconds=None)
    env.reset(seed=0)
    done, info = False, {}
    while not done:
        _o, _r, term, trunc, info = env.step(scripted_action(env))
        done = term or trunc
    assert info["counts"][MISS] == 0
    assert play([j for (_l, _d, j) in env.hit_log]) == info["score"]
    assert 950_000 < info["score"] <= 1_000_000
