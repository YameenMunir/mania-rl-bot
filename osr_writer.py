"""Write osu! replay (.osr) files for osu!mania.

Format reference (osu! wiki, "osu! file formats / .osr"):
  byte   game mode (3 = mania)
  int    game version
  string beatmap MD5 (must match the .osu file in your osu! library)
  string player name
  string replay MD5
  short  n300, n100, n50, geki (MAX), katu (200), miss
  int    total score
  short  max combo
  byte   perfect combo
  int    mods
  string life bar graph (empty here)
  long   timestamp (Windows ticks)
  int    length of compressed replay data
  bytes  LZMA-compressed replay frames
  long   online score id (0 for local)

Mania replay frames are "delta_ms|keys_bitmask|0|0" separated by commas, where bit N of
the bitmask is column N (1 = first column). The stream starts with two dummy frames and
ends with a "-12345|0|0|seed" frame, like real osu! replays.
"""
from __future__ import annotations

import hashlib
import lzma
import struct
import time

GAME_MODE_MANIA = 3
GAME_VERSION = 20210520
EPOCH_TICKS = 621_355_968_000_000_000  # .NET ticks at the Unix epoch


def _uleb128(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def osu_string(s: str) -> bytes:
    if not s:
        return b"\x00"
    raw = s.encode("utf-8")
    return b"\x0b" + _uleb128(len(raw)) + raw


def events_to_frames(events: list[tuple[float, int, int]]) -> list[tuple[int, int]]:
    """Turn (time_ms, lane, down) key events into (time_ms, keys_bitmask) frames.

    Times are rounded to whole ms, clamped to >= 0, and kept strictly increasing within
    each lane so a press can never be reordered after its own release.
    """
    last: dict[int, int] = {}
    fixed: list[tuple[int, int, int]] = []
    for t, lane, down in events:  # emission order is already chronological per lane
        ti = max(0, int(round(t)))
        ti = max(ti, last.get(lane, -1) + 1)
        last[lane] = ti
        fixed.append((ti, lane, down))
    fixed.sort(key=lambda e: e[0])  # stable sort keeps per-lane order

    mask = 0
    frames: list[tuple[int, int]] = [(0, 0)]
    for ti, lane, down in fixed:
        mask = (mask | (1 << lane)) if down else (mask & ~(1 << lane))
        if frames[-1][0] == ti:
            frames[-1] = (ti, mask)
        else:
            frames.append((ti, mask))
    return frames


def frames_to_text(frames: list[tuple[int, int]], seed: int = 0) -> str:
    parts = ["0|256|-500|0", "-1|256|-500|0"]
    prev = -1
    for t, mask in frames:
        parts.append(f"{t - prev}|{mask}|0|0")
        prev = t
    parts.append(f"-12345|0|0|{seed}")
    return ",".join(parts) + ","


def lzma_alone(data: bytes) -> bytes:
    """Legacy .lzma ("alone") stream with the real uncompressed size in the header."""
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 21, "lc": 3, "lp": 0, "pb": 2}]
    blob = lzma.compress(data, format=lzma.FORMAT_ALONE, filters=filters)
    return blob[:5] + struct.pack("<Q", len(data)) + blob[13:]


def build_osr(
    *,
    beatmap_md5: str,
    player: str,
    counts: list[int],  # [MAX, 300, 200, 100, 50, miss]
    score: int,
    max_combo: int,
    events: list[tuple[float, int, int]],
    mods: int = 0,
    seed: int = 0,
    timestamp: float | None = None,
) -> bytes:
    geki, n300, katu, n100, n50, miss = (min(int(c), 65535) for c in counts)
    compressed = lzma_alone(frames_to_text(events_to_frames(events), seed).encode("ascii"))
    replay_md5 = hashlib.md5(compressed).hexdigest()
    ticks = int((time.time() if timestamp is None else timestamp) * 10_000_000) + EPOCH_TICKS

    out = bytearray()
    out += struct.pack("<B", GAME_MODE_MANIA)
    out += struct.pack("<i", GAME_VERSION)
    out += osu_string(beatmap_md5)
    out += osu_string(player)
    out += osu_string(replay_md5)
    out += struct.pack("<6H", n300, n100, n50, geki, katu, miss)
    out += struct.pack("<i", int(score))
    out += struct.pack("<H", min(int(max_combo), 65535))
    out += struct.pack("<B", 1 if miss == 0 else 0)
    out += struct.pack("<i", int(mods))
    out += osu_string("")  # life bar graph
    out += struct.pack("<q", ticks)
    out += struct.pack("<i", len(compressed))
    out += compressed
    out += struct.pack("<q", 0)  # online score id
    return bytes(out)
