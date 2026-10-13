"""osu!mania 4K beatmap validation and scanning.

Unlike `osu_parser.load_maps` (which silently skips anything unusable), these helpers report
*why* a map was rejected so the UI can show it. Nothing is ever converted: a 7K map stays
rejected rather than being squeezed into 4 lanes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from maniabot.storage import base_root
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
    stars: float | None = None  # osu! star rating (None for rejected maps)
    stars_exact: bool = True  # False when `stars` is a density-based estimate

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
    if path.suffix.lower() in (".osz", ".zip"):
        return MapCheck(str(path), False, f"Couldn't open this {path.suffix.lower()} archive (corrupt or not a zip).")
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
    check.stars, check.stars_exact = rate_map(path, bm)
    audio = check.audio_path
    if audio is None or not audio.exists():
        warnings.append(
            "Audio file not found next to the map. Training and replays still work, "
            "but osu! needs the audio to play the map."
        )
    return check


# ------------------------------------------------------------- star ratings
# Fallback when the rating library is missing or can't parse a map: stars ~= A * nps ** B, where nps
# is the 90th-percentile notes/second over 4 s windows. Fitted against rosu-pp on 176 real 4K maps
# (correlation 0.97, mean error 0.22 stars).
STAR_FIT = (0.436, 0.808)


def _engine() -> str:
    try:
        from importlib.metadata import version

        return f"rosu-pp-py {version('rosu-pp-py')}"
    except Exception:  # noqa: BLE001
        return "estimate"


def _rosu_stars(path: Path) -> float | None:
    try:
        import rosu_pp_py as rosu

        stars = float(rosu.Difficulty().calculate(rosu.Beatmap(path=str(path))).stars)
    except Exception:  # noqa: BLE001  (library missing, or it can't rate this particular map)
        return None
    return stars if math.isfinite(stars) and stars >= 0 else None


def estimate_stars(bm: Beatmap) -> float:
    times = np.sort(np.array([n.time for n in bm.notes], dtype=float))
    if len(times) < 2:
        return 0.0
    in_window = np.searchsorted(times, times + 4000.0) - np.arange(len(times))
    nps = float(np.percentile(in_window, 90)) / 4.0
    return round(STAR_FIT[0] * nps ** STAR_FIT[1], 2)


class StarCache:
    """Star ratings remembered by beatmap MD5, so rescanning a big folder stays fast."""

    def __init__(self, path: Path | None = None):
        self._path = path
        self._loaded_from: Path | None = None
        self._data: dict[str, float] = {}
        self._dirty = False
        self._lock = threading.Lock()

    def _file(self) -> Path:
        return self._path or base_root() / "cache" / "stars.json"

    def _ensure(self) -> None:
        f = self._file()
        if self._loaded_from == f:
            return
        self._data, self._dirty, self._loaded_from = {}, False, f
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            if raw.get("engine") == _engine():
                self._data = {str(k): float(v) for k, v in raw.get("stars", {}).items()}
        except (OSError, ValueError, AttributeError):
            pass

    def get(self, md5: str) -> float | None:
        with self._lock:
            self._ensure()
            return self._data.get(md5)

    def put(self, md5: str, stars: float) -> None:
        with self._lock:
            self._ensure()
            if self._data.get(md5) != stars:
                self._data[md5] = stars
                self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            f = self._file()
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps({"engine": _engine(), "stars": self._data}), encoding="utf-8")
                os.replace(tmp, f)
                self._dirty = False
            except OSError:
                log.warning("couldn't save the star-rating cache", exc_info=True)


_STARS = StarCache()


def rate_map(path: str | Path, bm: Beatmap, cache: StarCache | None = None) -> tuple[float, bool]:
    """(stars, exact). `exact` is False when the number is the density-based estimate."""
    cache = cache or _STARS
    if bm.md5:
        hit = cache.get(bm.md5)
        if hit is not None:
            return hit, True
    stars = _rosu_stars(Path(path))
    if stars is not None:
        if bm.md5:
            cache.put(bm.md5, stars)
        return stars, True
    return estimate_stars(bm), False


def star_range_label(lo: float | None, hi: float | None) -> str:
    if lo is None and hi is None:
        return "any star rating"
    if hi is None:
        return f"{lo:g}★ and up"
    if lo is None:
        return f"up to {hi:g}★"
    return f"{lo:g}–{hi:g}★"


@dataclass
class ScanResult:
    valid: list[MapCheck] = field(default_factory=list)
    rejected: list[MapCheck] = field(default_factory=list)

    @property
    def beatmaps(self) -> list[Beatmap]:
        return [c.beatmap for c in self.valid if c.beatmap]

    def in_star_range(self, lo: float | None = None, hi: float | None = None) -> list[MapCheck]:
        """Valid maps rated between `lo` and `hi` stars (inclusive; None = no limit)."""
        return [c for c in self.valid
                if (lo is None or (c.stars or 0.0) >= lo) and (hi is None or (c.stars or 0.0) <= hi)]

    def star_bounds(self) -> tuple[float, float] | None:
        rated = [c.stars for c in self.valid if c.stars is not None]
        return (min(rated), max(rated)) if rated else None


_AUDIO_EXTS = {".mp3", ".ogg", ".wav"}


def osz_cache_root() -> Path:
    return base_root() / "cache" / "mapsets"


_LONG_PATH = 240  # Windows' classic limit is 260 characters; keep headroom


def _verbatim(p: Path) -> Path:
    r"""The Windows long-path form (\\?\C:\...) of `p`; unchanged elsewhere or if already in that form."""
    s = str(p)
    if sys.platform != "win32" or s.startswith("\\\\?\\"):
        return p
    return Path("\\\\?\\" + os.path.abspath(s))


def extract_osz(archive: str | Path, cache_root: Path | None = None, _depth: int = 0) -> list[Path]:
    """Unpack the .osu files (and audio) of an .osz / .zip into the cache; returns the .osu paths.

    Works for a single mapset and for a map pack (a zip of .osz files, unpacked one level deep).
    The source archive is never modified. Extraction is skipped when the archive (same path,
    size and mtime) was already unpacked. Raises zipfile.BadZipFile / OSError on a bad archive.
    """
    archive = Path(archive)
    st = archive.stat()
    key = hashlib.md5(f"{archive.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:10]
    safe = re.sub(r"[^A-Za-z0-9 ._-]", "_", archive.stem)[:60].strip() or "mapset"
    dest = (cache_root or osz_cache_root()) / f"{safe}-{key}"
    if len(str(dest)) >= _LONG_PATH:
        dest = _verbatim(dest)
    done = dest / ".extracted"
    if not done.exists():
        wanted = {".osu"} | _AUDIO_EXTS | ({".osz"} if _depth == 0 else set())
        with zipfile.ZipFile(archive) as z:
            members = []
            for info in z.infolist():
                name = Path(info.filename.replace("\\", "/"))
                if not info.is_dir() and name.suffix.lower() in wanted:  # skip backgrounds / videos / storyboards
                    members.append((info, name))
            # Mapset file names can be very long; past Windows' 260-character limit the file can't be
            # created at all, so switch this whole mapset to the long-path form (and remember that).
            longest = max((len(str(dest / n)) for _i, n in members), default=0)
            long = sys.platform == "win32" and longest >= _LONG_PATH
            if long:
                dest = _verbatim(dest)
            dest.mkdir(parents=True, exist_ok=True)
            root = os.path.normpath(str(dest))
            for info, name in members:
                target = os.path.normpath(os.path.join(root, *name.parts))
                if not target.startswith(root + os.sep):
                    continue  # never write outside the cache folder (zip-slip)
                try:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with z.open(info) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                except OSError as exc:
                    log.warning("can't extract %s from %s: %s", name.name, archive.name, exc)
        (dest / ".extracted").write_text("long" if long else "ok", encoding="utf-8")
    elif done.read_text(encoding="utf-8").strip() == "long":
        dest = _verbatim(dest)
    found = sorted(dest.rglob("*.osu"))
    if _depth == 0:
        for inner in sorted(dest.rglob("*.osz")):
            try:
                found += extract_osz(inner, cache_root, _depth=1)
            except (zipfile.BadZipFile, OSError) as exc:
                log.warning("can't unpack %s inside %s: %s", inner.name, archive, exc)
    return found


_ARCHIVE_EXTS = (".osz", ".zip")


def iter_osu_files(path: str | Path) -> Iterable[Path]:
    """Yield .osu files under `path` (file or folder, recursive). .osz / .zip archives are
    unpacked into the cache and their .osu files yielded; an unreadable archive is yielded
    as-is so the scan can report it."""
    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(f for f in path.rglob("*")
                       if f.suffix.lower() in (".osu",) + _ARCHIVE_EXTS and f.is_file())
    else:
        return
    for f in files:
        if f.suffix.lower() not in _ARCHIVE_EXTS:
            yield f
            continue
        try:
            yield from extract_osz(f)
        except (zipfile.BadZipFile, OSError) as exc:
            log.warning("can't unpack %s: %s", f, exc)
            yield f


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
    _STARS.flush()
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
