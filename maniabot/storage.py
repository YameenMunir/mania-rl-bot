"""Where user data lives, settings persistence, atomic file writes and storage migration.

User data (models, versions, checkpoints, replays, logs, evaluation output) is kept in a
per-user application-data folder, never in the source repository:

  Windows: %LOCALAPPDATA%\\OsuManiaAI
  macOS:   ~/Library/Application Support/OsuManiaAI
  Linux:   $XDG_DATA_HOME/OsuManiaAI  (default ~/.local/share/OsuManiaAI)

The OSUMANIAAI_HOME environment variable overrides the base folder (used by the tests).
The settings file always lives in the base folder; the data root itself can be moved to a
custom location from Settings -> Storage, which copies (never moves) existing data.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from maniabot import APP_NAME, REPO_ROOT

ENV_VAR = "OSUMANIAAI_HOME"
log = logging.getLogger("maniabot.storage")

# Folders inside the repository that are git-ignored and may hold data during development.
IGNORED_REPO_DIRS = ("user_data", "runtime", "local_data")


def platform_default_root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def base_root() -> Path:
    override = os.environ.get(ENV_VAR)
    return Path(override).expanduser() if override else platform_default_root()


def settings_path() -> Path:
    return base_root() / "settings" / "settings.json"


# ------------------------------------------------------------------ file helpers
def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temp file in the same folder, fsync, then rename over the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=False, default=str).encode("utf-8"))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return default


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def remove_tree(path: Path) -> None:
    """shutil.rmtree that also clears read-only flags (common on Windows)."""

    def _retry(func, p, _exc):
        os.chmod(p, 0o700)
        func(p)

    if Path(path).exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_retry)
        else:
            shutil.rmtree(path, onerror=_retry)


# ---------------------------------------------------------------------- settings
DEFAULT_SETTINGS: dict[str, Any] = {
    "storage_root": None,  # None = base_root()
    "osu_path": None,  # folder containing osu!.exe; None = auto-detect
    "player_name": "ManiaBot",
    "default_maps": str(REPO_ROOT / "maps" / "demo"),
    "default_device": "auto",
    "legacy_import_dismissed": False,
    "recent_maps": [],
}


@dataclass
class Settings:
    values: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SETTINGS))

    def __getitem__(self, key: str) -> Any:
        return self.values.get(key, DEFAULT_SETTINGS.get(key))

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    @classmethod
    def load(cls) -> "Settings":
        data = read_json(settings_path(), {}) or {}
        merged = dict(DEFAULT_SETTINGS)
        merged.update({k: v for k, v in data.items()})
        return cls(merged)

    def save(self) -> None:
        atomic_write_json(settings_path(), self.values)

    def add_recent_map(self, path: str, limit: int = 10) -> None:
        recent = [p for p in self["recent_maps"] if p != path]
        self["recent_maps"] = [path, *recent][:limit]


# ----------------------------------------------------------------------- storage
class Storage:
    """Resolved folder layout of the data root."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Storage":
        settings = settings or Settings.load()
        root = settings["storage_root"]
        return cls(Path(root) if root else base_root())

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def replays_dir(self) -> Path:
        return self.root / "replays"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    def ensure(self) -> "Storage":
        for d in (self.models_dir, self.replays_dir, self.logs_dir, self.cache_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def usage(self) -> dict[str, int]:
        return {
            "models": dir_size(self.models_dir),
            "replays": dir_size(self.replays_dir),
            "logs": dir_size(self.logs_dir),
            "cache": dir_size(self.cache_dir) + dir_size(self.tmp_dir),
        }

    def clear_tmp(self) -> None:
        """Remove leftovers of interrupted atomic saves (only ever temp data)."""
        if self.tmp_dir.exists():
            for child in self.tmp_dir.iterdir():
                remove_tree(child) if child.is_dir() else child.unlink(missing_ok=True)


def is_inside(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def validate_storage_dir(path: str | Path) -> tuple[bool, str]:
    """Check that `path` can be used as a data root. Returns (ok, human-readable message)."""
    if not str(path).strip():
        return False, "Choose a folder."
    p = Path(path).expanduser()
    if is_inside(p, REPO_ROOT):
        allowed = any(is_inside(p, REPO_ROOT / d) for d in IGNORED_REPO_DIRS)
        if not allowed:
            return False, (
                "That folder is inside the source-code repository, so models would end up in Git. "
                f"Pick a folder outside it (or one of the ignored dev folders: {', '.join(IGNORED_REPO_DIRS)})."
            )
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"Can't create the folder: {exc.strerror or exc}"
    probe = p / f".write-test-{uuid.uuid4().hex}"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        return False, f"The folder is not writable: {exc.strerror or exc}"
    return True, "Folder is usable."


@dataclass
class MigrationReport:
    copied_models: list[str] = field(default_factory=list)
    skipped_models: list[str] = field(default_factory=list)  # id already exists at destination
    copied_replays: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _verify_copy(src: Path, dst: Path) -> bool:
    for root, _dirs, files in os.walk(src):
        for name in files:
            s = Path(root) / name
            d = dst / s.relative_to(src)
            if not d.exists() or d.stat().st_size != s.stat().st_size:
                return False
    return True


def migrate_storage(src: Storage, dst: Storage, progress: Callable[[str], None] | None = None) -> MigrationReport:
    """Copy models and replays from `src` to `dst`. The source is never modified or deleted."""
    progress = progress or (lambda _m: None)
    report = MigrationReport()
    dst.ensure()
    if src.models_dir.exists():
        for model_dir in sorted(p for p in src.models_dir.iterdir() if p.is_dir()):
            target = dst.models_dir / model_dir.name
            if target.exists():
                report.skipped_models.append(model_dir.name)
                continue
            progress(f"Copying model {model_dir.name}…")
            try:
                shutil.copytree(model_dir, target)
                if not _verify_copy(model_dir, target):
                    raise OSError("copied files do not match the originals")
                report.copied_models.append(model_dir.name)
            except OSError as exc:
                report.errors.append(f"{model_dir.name}: {exc}")
                remove_tree(target)  # remove the partial copy only; the original is untouched
    if src.replays_dir.exists():
        progress("Copying replays…")
        for item in src.replays_dir.iterdir():
            if not item.is_file() or item.name == "history.json":
                continue
            target = dst.replays_dir / item.name
            if not target.exists():
                try:
                    shutil.copy2(item, target)
                    report.copied_replays += 1
                except OSError as exc:
                    report.errors.append(f"{item.name}: {exc}")
        _merge_replay_history(src, dst)
    log.info("storage migration %s -> %s: %s", src.root, dst.root, report)
    return report


def _merge_replay_history(src: Storage, dst: Storage) -> None:
    src_hist = read_json(src.replays_dir / "history.json", {"replays": []}) or {"replays": []}
    dst_path = dst.replays_dir / "history.json"
    dst_hist = read_json(dst_path, {"replays": []}) or {"replays": []}
    known = {r["id"] for r in dst_hist["replays"]}
    for rec in src_hist["replays"]:
        if rec["id"] in known:
            continue
        rec = dict(rec)
        rec["path"] = str(dst.replays_dir / Path(rec["path"]).name)
        dst_hist["replays"].append(rec)
    atomic_write_json(dst_path, dst_hist)
