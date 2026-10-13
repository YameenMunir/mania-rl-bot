"""Mods (DT / NC / HT / HR / EZ): parsing, how the simulator applies them, replays and training."""
from __future__ import annotations

import numpy as np
import pytest

from evaluate import scripted_action
from mania_env import ManiaEnv, hit_windows
from maniabot import mods as M
from maniabot.evaluation import evaluate
from maniabot.replays import ReplayHistory, generate_replay
from maniabot.training import TrainingSpec, compatibility_warnings, run_training, validate_spec
from osu_parser import Beatmap, Note, parse_osu
from tests.conftest import DEMO_MAPS, tiny_spec
from tests.test_replays import parse_osr


def test_parse_and_label():
    assert M.parse_mods("") == M.parse_mods("NM") == M.parse_mods(None) == 0
    assert M.parse_mods("DTHR") == M.parse_mods("hr+dt") == M.parse_mods("DT HR") == M.DT | M.HR
    assert M.parse_mods("NC") == M.NC | M.DT
    assert M.parse_mods(M.NC) == M.NC | M.DT  # osu! always sets DT with NC
    assert M.mods_label(M.DT | M.HR) == "DTHR"
    assert M.mods_label(M.NC | M.DT) == "NC"
    assert M.mods_label(M.EZ | M.HT) == "EZHT"
    assert M.mods_label(0) == "NM"
    for bad in ("EZHR", "DTHT", "HD", "D", "XX"):
        with pytest.raises(ValueError):
            M.parse_mods(bad)
    assert M.parse_mod_list("NM, DT, dt, DTHR") == [0, M.DT, M.DT | M.HR]
    assert M.parse_mod_list("") == [0]
    assert all(M.mods_label(M.parse_mods(c)) == c for c in M.CHOICES)


def test_speed_and_window_scale():
    assert (M.speed(M.DT), M.speed(M.NC | M.DT), M.speed(M.HT), M.speed(0)) == (1.5, 1.5, 0.75, 1.0)
    assert M.window_scale(M.HR) == pytest.approx(1 / 1.4)
    assert M.window_scale(M.EZ) == pytest.approx(1.4)
    assert M.window_scale(M.DT) == 1.0  # mania keeps DT/HT windows the same size in real time


def _map(times, od=8.0):
    return Beatmap("t", "v", 4, od, [Note(t, i % 4, t) for i, t in enumerate(times)], md5="x")


def test_env_applies_rate_and_windows():
    bm = _map([3000 + 300 * i for i in range(20)])
    env = ManiaEnv([bm], clip_seconds=None)
    env.reset(seed=0, options={"mods": M.DT})
    assert env.rate == 1.5
    assert env.times[0][0] == pytest.approx(3000 / 1.5)  # real time
    assert np.allclose(env.windows, hit_windows(8.0))
    env.reset(seed=0, options={"mods": M.HR | M.HT})
    assert env.times[0][0] == pytest.approx(3000 / 0.75)
    assert np.allclose(env.windows, hit_windows(8.0) / 1.4)
    env.reset(seed=0, options={"mods": M.EZ})
    assert np.allclose(env.windows, hit_windows(8.0) * 1.4)
    env.reset(seed=0)  # back to the env's default: no mod
    assert env.mods == 0 and env.times[0][0] == 3000


def test_env_samples_from_mod_list():
    env = ManiaEnv([parse_osu(DEMO_MAPS / "demo_000.osu")], clip_seconds=5.0, mods=[0, M.DT, M.HR])
    seen = set()
    for i in range(30):
        env.reset(seed=i)
        seen.add(env.mods)
    assert seen == {0, M.DT, M.HR}


@pytest.mark.parametrize("mods", [0, M.DT, M.HT, M.HR, M.EZ, M.DT | M.HR])
def test_oracle_is_near_perfect_under_mods(mods):
    """The simulator stays self-consistent: perfect timing still scores ~100% with any mod."""
    bm = parse_osu(DEMO_MAPS / "demo_003.osu")  # has hold notes
    env = ManiaEnv([bm], clip_seconds=None, record=True, mods=mods)
    env.reset(seed=0)
    done, info = False, {}
    while not done:
        _o, _r, term, trunc, info = env.step(scripted_action(env))
        done = term or trunc
    assert info["mods"] == M.mods_label(mods)
    assert info["accuracy"] > 0.97
    assert info["counts"][5] == 0


def test_recorded_events_are_in_song_time():
    """Replay frames must be in song time: a DT press on a note sits on the note's .osu time."""
    bm = _map([3000 + 400 * i for i in range(12)])
    env = ManiaEnv([bm], clip_seconds=None, record=True, mods=M.DT)
    env.reset(seed=0)
    done = False
    while not done:
        _o, _r, term, trunc, _i = env.step(scripted_action(env))
        done = term or trunc
    presses = [t for t, lane, down in env.events if down]
    assert len(presses) == 12
    for t, note in zip(presses, bm.notes):
        assert abs(t - note.time) <= env.dt * env.rate


def test_evaluate_cycles_mods(trained, store):
    from maniabot.evaluation import POLICIES

    model = POLICIES.load(store.get_version(trained, "v001").weights)
    maps = [parse_osu(DEMO_MAPS / f"demo_00{i}.osu") for i in range(3)]
    res = evaluate(model, maps, mods=[0, M.DT])
    assert [m["mods"] for m in res["maps"]] == ["NM", "DT", "NM"]
    assert res["mods"] == "NM, DT"


def test_replay_carries_mods(store, storage, trained):
    hist = ReplayHistory(storage)
    rec = generate_replay(store, hist, trained, "v001", DEMO_MAPS / "demo_002.osu", player="Tester", mods="DTHR")
    osr = parse_osr(open(rec["path"], "rb").read())
    assert osr["mods"] == M.DT | M.HR
    assert rec["mods"] == "DTHR" and rec["metrics"]["mods"] == "DTHR"
    assert "+DTHR" in rec["path"]
    # Frame times are song time: the last key event lands near the map's last note, not 1.5x earlier.
    t, last = 0, 0
    for f in osr["frames"][2:-1]:
        t += int(f.split("|")[0])
        last = t
    end = max(n.end_time for n in parse_osu(DEMO_MAPS / "demo_002.osu").notes)
    assert last > end * 0.8
    with pytest.raises(Exception, match="EZ and HR"):
        generate_replay(store, hist, trained, "v001", DEMO_MAPS / "demo_002.osu", mods="EZHR")


def test_spec_mods_validation_and_warning(small_maps):
    spec = tiny_spec("m", small_maps, mods=["NM", "DT"])
    assert spec.mod_bits == [0, M.DT]
    assert not validate_spec(spec)
    assert any("Mods" in e for e in validate_spec(tiny_spec("m", small_maps, mods=["EZHR"])))
    warns = compatibility_warnings({"mods": ["NM"]}, spec)
    assert any("Mods change from NM to NM, DT" in w for w in warns)
    assert not any("Mods" in w for w in compatibility_warnings({}, tiny_spec("m", small_maps)))
    assert TrainingSpec.from_dict(spec.to_dict()).mods == ["NM", "DT"]


def test_training_with_mods_records_them(store, small_maps):
    rec = store.create_model("Mod Bot", "")
    session = run_training(store, tiny_spec(rec.id, small_maps, mods=["DT", "HR"]))
    assert session["status"] == "completed", session.get("error")
    assert session["mods"] == "DT, HR"
    v = store.get_version(rec.id, session["result_version"])
    assert v.meta["mods"] == ["DT", "HR"]
    assert v.meta["validation"]["mods"] == "DT, HR"
