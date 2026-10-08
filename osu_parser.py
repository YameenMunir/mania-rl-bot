"""Minimal osu!mania .osu beatmap parser (built for 4K, works for any key count)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Note:
    time: int  # ms from audio start
    lane: int  # 0-indexed column
    end_time: int  # ms; equals time for tap notes

    @property
    def is_hold(self) -> bool:
        return self.end_time > self.time


@dataclass
class Beatmap:
    title: str
    version: str
    keys: int
    od: float
    notes: list[Note] = field(default_factory=list)
    path: str = ""
    artist: str = ""
    md5: str = ""  # MD5 of the raw .osu file bytes; osu! uses it to match replays to maps
    audio_filename: str = ""  # [General] AudioFilename, relative to the .osu file
    creator: str = ""

    @property
    def name(self) -> str:
        return f"{self.title} [{self.version}]"


class NotManiaError(ValueError):
    """Raised when a .osu file is not an osu!mania beatmap."""


def parse_osu(path: str | Path) -> Beatmap:
    raw = Path(path).read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    text = raw.decode("utf-8-sig", errors="replace")

    sections: dict[str, dict[str, str]] = {"General": {}, "Metadata": {}, "Difficulty": {}}
    hit_lines: list[str] = []
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            continue
        if current in sections and ":" in line:
            key, value = line.split(":", 1)
            sections[current][key.strip()] = value.strip()
        elif current == "HitObjects":
            hit_lines.append(line)

    if sections["General"].get("Mode") != "3":
        raise NotManiaError(f"{path} is not an osu!mania map")

    keys = int(float(sections["Difficulty"].get("CircleSize", 4)))
    od = float(sections["Difficulty"].get("OverallDifficulty", 5))

    notes: list[Note] = []
    for line in hit_lines:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        x = int(float(parts[0]))
        time = int(float(parts[2]))
        kind = int(parts[3])
        lane = min(keys - 1, max(0, int(x * keys / 512)))
        end = time
        if kind & 128 and len(parts) > 5:  # hold note: endTime:hitSample
            end = int(float(parts[5].split(":")[0]))
        notes.append(Note(time, lane, max(end, time)))
    notes.sort(key=lambda n: (n.time, n.lane))

    meta = sections["Metadata"]
    return Beatmap(
        title=meta.get("Title", "unknown"),
        version=meta.get("Version", "unknown"),
        keys=keys,
        od=od,
        notes=notes,
        path=str(path),
        artist=meta.get("Artist", ""),
        md5=md5,
        audio_filename=sections["General"].get("AudioFilename", ""),
        creator=meta.get("Creator", ""),
    )


def load_maps(path: str | Path, keys: int = 4, min_notes: int = 50) -> list[Beatmap]:
    """Load every usable .osu file under `path` (a file or a folder, searched recursively)."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.osu"))
    maps: list[Beatmap] = []
    for file in files:
        try:
            bm = parse_osu(file)
        except (NotManiaError, ValueError, OSError):
            continue
        if bm.keys == keys and len(bm.notes) >= min_notes:
            maps.append(bm)
    if not maps:
        raise FileNotFoundError(f"No usable {keys}K osu!mania maps found under {path}")
    return maps
