"""osu!mania 4K beatmap validation and scanning.

Unlike `osu_parser.load_maps` (which silently skips anything unusable), these helpers report
*why* a map was rejected so the UI can show it. Nothing is ever converted: a 7K map stays
rejected rather than being squeezed into 4 lanes.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from osu_parser import Beatmap, NotManiaError, parse_osu

KEYS = 4
MIN_NOTES = 50
LANE_NAMES = ("Lane 1", "Lane 2", "Lane 3", "Lane 4")
log = logging.getLogger("maniabot.beatmaps")


@dataclass
class MapCheck:
    path: str
    ok: bool
    reason: str = ""  # why it was rejected (empty when ok)
    beatmap: Beatmap | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.beatmap:
            bm = self.beatmap
            return f"{bm.artist} - {bm.title} [{bm.version}]" if bm.artist else bm.name
        return Path(self.path).name

    @property
    def audio_path(self) -> Path | None:
        if not self.beatmap or not self.beatmap.audio_filename:
            return None
        return Path(self.path).parent / self.beatmap.audio_filename


def check_beatmap(path: str | Path, min_notes: int = MIN_NOTES) -> MapCheck:
    path = Path(path)
    if not path.exists():
        return MapCheck(str(path), False, "File not found.")
    if path.suffix.lower() != ".osu":
        return MapCheck(str(path), False, "Not a .osu beatmap file.")
    try:
        bm = parse_osu(path)
    except NotManiaError:
        return MapCheck(str(path), False, "Not an osu!mania beatmap (it's for another game mode).")
    except OSError as exc:
        return MapCheck(str(path), False, f"Can't read the file: {exc.strerror or exc}")
    except (ValueError, IndexError) as exc:
        return MapCheck(str(path), False, f"The beatmap file looks corrupt ({exc}).")
    if bm.keys != KEYS:
        return MapCheck(str(path), False, f"This is a {bm.keys}K map. Only 4K maps are supported.", bm)
    if len(bm.notes) < min_notes:
        return MapCheck(str(path), False, f"Only {len(bm.notes)} notes (needs at least {min_notes}).", bm)
    lanes = {n.lane for n in bm.notes}
    warnings = []
    if len(lanes) < KEYS:
        used = ", ".join(LANE_NAMES[i] for i in sorted(lanes))
        warnings.append(f"Only uses {used}.")
    check = MapCheck(str(path), True, "", bm, warnings)
    audio = check.audio_path
    if audio is None or not audio.exists():
        warnings.append(
            "Audio file not found next to the map. Training and replays still work, "
            "but osu! needs the audio to play the map."
        )
    return check


@dataclass
class ScanResult:
    valid: list[MapCheck] = field(default_factory=list)
    rejected: list[MapCheck] = field(default_factory=list)

    @property
    def beatmaps(self) -> list[Beatmap]:
        return [c.beatmap for c in self.valid if c.beatmap]


def iter_osu_files(path: str | Path) -> Iterable[Path]:
    path = Path(path)
    if path.is_file():
        yield path
        return
    if path.is_dir():
        yield from sorted(path.rglob("*.osu"))


def scan_maps(
    path: str | Path,
    min_notes: int = MIN_NOTES,
    progress: Callable[[int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ScanResult:
    """Validate every .osu file under `path` (a file or folder, searched recursively)."""
    result = ScanResult()
    for i, file in enumerate(iter_osu_files(path)):
        if should_stop and should_stop():
            break
        try:
            check = check_beatmap(file, min_notes)
        except Exception as exc:  # never let one odd file abort a scan
            log.exception("unexpected error checking %s", file)
            check = MapCheck(str(file), False, f"Unexpected error: {exc}")
        (result.valid if check.ok else result.rejected).append(check)
        if progress and i % 25 == 0:
            progress(i + 1)
    return result


def split_maps(maps: list[Beatmap], n_val: int, seed: int = 0) -> tuple[list[Beatmap], list[Beatmap]]:
    """Deterministically hold out `n_val` maps for validation. Returns (train, val)."""
    if n_val <= 0 or len(maps) <= n_val:
        return list(maps), []
    order = list(range(len(maps)))
    random.Random(seed).shuffle(order)
    val_idx = set(order[:n_val])
    train = [m for i, m in enumerate(maps) if i not in val_idx]
    val = [m for i, m in enumerate(maps) if i in val_idx]
    return train, val


def default_val_count(n_maps: int) -> int:
    if n_maps >= 10:
        return max(2, n_maps // 6)
    if n_maps >= 6:
        return 2
    return 0


def map_summary(bm: Beatmap) -> dict:
    """Small JSON-safe description of a map, stored in version / replay metadata."""
    return {
        "title": bm.title,
        "artist": bm.artist,
        "version": bm.version,
        "creator": bm.creator,
        "md5": bm.md5,
        "path": bm.path,
        "od": bm.od,
        "notes": len(bm.notes),
        "holds": sum(1 for n in bm.notes if n.is_hold),
    }
