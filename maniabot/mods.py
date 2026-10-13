"""osu! mods that change how a map plays: DT / NC / HT (song speed) and EZ / HR (hit windows).

Mods are kept as osu!'s bit flags, the value an .osr replay stores, and shown as acronyms
such as "DTHR" ("NM" for no mod).

How osu!mania stable applies them, and so how the simulator does:
  DT / NC  the song plays 1.5x faster, HT 0.75x. Notes arrive faster, but mania scales the hit
           windows with the rate, so in real (wall-clock) time the windows keep their size.
  HR       every hit window is divided by 1.4.
  EZ       every hit window is multiplied by 1.4.

The simulator runs in real time: note times are divided by the rate and the windows are scaled
for EZ / HR. Replay frames are written in song time, which is what osu! expects.
"""
from __future__ import annotations

import re
from typing import Iterable

EZ = 1 << 1
HR = 1 << 4
DT = 1 << 6
HT = 1 << 8
NC = 1 << 9  # osu! always sets DT alongside NC
SUPPORTED = EZ | HR | DT | HT | NC

# Acronyms in display order (osu! writes DT before HR, e.g. "HDDTHR").
_ACRONYMS: tuple[tuple[str, int], ...] = (("EZ", EZ), ("HT", HT), ("NC", NC | DT), ("DT", DT), ("HR", HR))
_BY_ACRONYM = dict(_ACRONYMS)
_NO_MOD = {"", "NM", "NOMOD", "NONE"}

# The combinations offered in the app's mod pickers.
CHOICES = ("NM", "DT", "HT", "HR", "EZ", "DTHR", "HTHR", "EZDT", "EZHT")


def check_mods(bits: int) -> int:
    """Return `bits` if it is a playable combination, else raise ValueError."""
    bits = int(bits)
    if bits & ~SUPPORTED:
        raise ValueError("Only the DT, NC, HT, HR and EZ mods are supported.")
    if bits & NC and not bits & DT:
        bits |= DT
    if bits & EZ and bits & HR:
        raise ValueError("EZ and HR can't be combined.")
    if bits & DT and bits & HT:
        raise ValueError("DT and HT can't be combined.")
    return bits


def parse_mods(value: str | int | None) -> int:
    """'DTHR', 'DT+HR', 'hr dt', 'NM', '' or a bit mask -> validated bit mask."""
    if value is None:
        return 0
    if isinstance(value, int):
        return check_mods(value)
    text = re.sub(r"[\s+]", "", str(value)).upper()
    if text in _NO_MOD:
        return 0
    if len(text) % 2:
        raise ValueError(f"'{value}' isn't a mod combination. Write mods as two-letter acronyms, e.g. DTHR.")
    bits = 0
    for i in range(0, len(text), 2):
        flag = _BY_ACRONYM.get(text[i:i + 2])
        if flag is None:
            raise ValueError(f"Unknown mod '{text[i:i + 2]}'. Supported: DT, NC, HT, HR, EZ (or NM for no mod).")
        bits |= flag
    return check_mods(bits)


def parse_mod_list(text: str | Iterable[str | int] | None) -> list[int]:
    """'NM, DT, DTHR' (or a list) -> distinct bit masks in order. Empty means no mod only."""
    if text is None:
        return [0]
    items = re.split(r"[,;/]", text) if isinstance(text, str) else list(text)
    out: list[int] = []
    for item in items:
        if isinstance(item, str) and not item.strip():
            continue
        bits = parse_mods(item)
        if bits not in out:
            out.append(bits)
    return out or [0]


def mods_label(bits: int | None) -> str:
    bits = int(bits or 0)
    if not bits:
        return "NM"
    parts = []
    for name, flag in _ACRONYMS:
        if name == "DT" and bits & NC:
            continue
        if bits & flag == flag:
            parts.append(name)
    return "".join(parts)


def mod_list_label(mods: Iterable[int]) -> str:
    return ", ".join(mods_label(m) for m in mods)


def speed(bits: int | None) -> float:
    """Song speed: 1.5 with DT / NC, 0.75 with HT, else 1."""
    bits = int(bits or 0)
    if bits & DT:
        return 1.5
    if bits & HT:
        return 0.75
    return 1.0


def window_scale(bits: int | None) -> float:
    """Factor applied to every hit window, in real time: 1.4 with EZ, 1/1.4 with HR, else 1."""
    bits = int(bits or 0)
    if bits & EZ:
        return 1.4
    if bits & HR:
        return 1.0 / 1.4
    return 1.0
