"""osu! stable integration: find the install, check if it's running, hand it replays and maps.

This only uses what osu! stable supports for any user: opening an .osr makes osu! play the
replay, and opening an .osz imports the mapset. There is no memory reading or input
injection. Everything degrades gracefully when osu! isn't installed.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from maniabot import REPO_ROOT
from maniabot.logs import event

log = logging.getLogger("maniabot.osu")
EXE_NAME = "osu!.exe"


class OsuError(Exception):
    pass


def _registry_candidates() -> list[Path]:
    if sys.platform != "win32":
        return []
    import winreg

    out = []
    keys = [
        (winreg.HKEY_CLASSES_ROOT, r"osu\shell\open\command"),
        (winreg.HKEY_CLASSES_ROOT, r"osu!\shell\open\command"),
        (winreg.HKEY_CURRENT_USER, r"Software\Classes\osu\shell\open\command"),
        (winreg.HKEY_CURRENT_USER, r"Software\Classes\osu!\shell\open\command"),
    ]
    for hive, sub in keys:
        try:
            with winreg.OpenKey(hive, sub) as k:
                value, _ = winreg.QueryValueEx(k, "")
        except OSError:
            continue
        m = re.match(r'\s*"([^"]+)"', value) or re.match(r"\s*(\S+)", value)
        if m:
            out.append(Path(m.group(1)))
    return out


def find_osu_exe(configured: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if configured:
        p = Path(configured)
        candidates.append(p if p.suffix.lower() == ".exe" else p / EXE_NAME)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(Path(local) / "osu!" / EXE_NAME)
        candidates += _registry_candidates()
    for c in candidates:
        if c.name.lower() == EXE_NAME and c.exists():
            return c
    return None


def songs_dir(exe: Path | None) -> Path | None:
    """The Songs folder, honouring a custom BeatmapDirectory in osu!.<user>.cfg."""
    if exe is None:
        return None
    base = exe.parent
    for cfg in base.glob("osu!.*.cfg"):
        try:
            for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("BeatmapDirectory"):
                    value = line.split("=", 1)[1].strip()
                    p = Path(value)
                    p = p if p.is_absolute() else base / p
                    if p.exists():
                        return p
        except OSError:
            continue
    default = base / "Songs"
    return default if default.exists() else None


def is_running() -> bool:
    if sys.platform != "win32":
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {EXE_NAME}", "/NH"],
            capture_output=True, text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return EXE_NAME.lower() in out.lower()


def status(configured: str | None = None) -> dict:
    exe = find_osu_exe(configured)
    songs = songs_dir(exe)
    return {
        "installed": exe is not None,
        "exe": str(exe) if exe else None,
        "songs_dir": str(songs) if songs else None,
        "running": is_running() if exe else False,
    }


def open_file(path: str | Path) -> None:
    """Open a file with its default application."""
    _open_with_default(Path(path))


def _open_with_default(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 - opening a user file with its associated app
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def open_in_osu(file: str | Path, configured: str | None = None) -> None:
    """Open an .osr (watch replay) or .osz (import mapset) in osu! stable."""
    file = Path(file)
    if not file.exists():
        raise OsuError(f"File not found: {file}")
    exe = find_osu_exe(configured)
    try:
        if exe:
            subprocess.Popen([str(exe), str(file)], cwd=str(exe.parent))
        elif sys.platform == "win32":
            _open_with_default(file)  # uses the .osr/.osz file association if osu! registered one
        else:
            raise OsuError("osu! stable was not found. Set its folder in Settings.")
    except OSError as exc:
        raise OsuError(
            "osu! stable was not found. Install it or set its folder in Settings, "
            "then open the replay again."
        ) from exc
    event(log, "osu.opened", file=file, exe=exe or "file association")


def open_folder(path: str | Path, select: str | Path | None = None) -> None:
    path = Path(path)
    if sys.platform == "win32" and select and Path(select).exists():
        subprocess.Popen(["explorer", "/select,", str(Path(select))])
        return
    if not path.exists():
        raise OsuError(f"Folder not found: {path}")
    _open_with_default(path)


def map_in_library(osu_path: str | Path, md5: str, songs: Path | None) -> bool | None:
    """True if an identical .osu (same MD5) is in the Songs folder, False if not, None if unknown."""
    if songs is None:
        return None
    try:
        Path(osu_path).resolve().relative_to(songs.resolve())
        return True
    except ValueError:
        pass
    name = Path(osu_path).name
    for root, _dirs, files in os.walk(songs):
        if name in files:
            try:
                if hashlib.md5(Path(root, name).read_bytes()).hexdigest() == md5:
                    return True
            except OSError:
                continue
    return False


def importable_mapset(osu_path: str | Path, cache_dir: Path) -> Path:
    """An .osz osu! can import for this map: the bundled demo .osz, or one built from the map's folder."""
    osu_path = Path(osu_path)
    bundled = REPO_ROOT / "maps" / "osz" / f"{osu_path.stem}.osz"
    if bundled.exists():
        with zipfile.ZipFile(bundled) as z:
            if osu_path.name in z.namelist() and z.read(osu_path.name) == osu_path.read_bytes():
                return bundled
    folder = osu_path.parent
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() != ".osz"]
    if not any(f.suffix.lower() in (".mp3", ".ogg", ".wav") for f in files):
        raise OsuError("The map's folder has no audio file, so osu! can't import it as a mapset.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{re.sub(r'[^A-Za-z0-9 ._-]', '_', folder.name)[:80] or 'mapset'}.osz"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.suffix.lower() == ".osu" and f != osu_path:
                continue  # only the selected difficulty
            z.write(f, f.name)
    return target
