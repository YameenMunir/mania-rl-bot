"""osu!mania 4K beatmap validation and scanning.

Unlike `osu_parser.load_maps` (which silently skips anything unusable), these helpers report
*why* a map was rejected so the UI can show it. Nothing is ever converted: a 7K map stays
rejected rather than being squeezed into 4 lanes.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

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


_AUDIO_EXTS = {".mp3", ".ogg", ".wav"}


def osz_cache_root() -> Path:
    return base_root() / "cache" / "mapsets"


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
    done = dest / ".extracted"
    if not done.exists():
        dest.mkdir(parents=True, exist_ok=True)
        root = dest.resolve()
        wanted = {".osu"} | _AUDIO_EXTS | ({".osz"} if _depth == 0 else set())
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename.replace("\\", "/"))
                if name.suffix.lower() not in wanted:
                    continue  # skip backgrounds / videos / storyboards
                target = (dest / name).resolve()
                if root != target and root not in target.parents:
                    continue  # never write outside the cache folder (zip-slip)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        done.write_text("ok")
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
