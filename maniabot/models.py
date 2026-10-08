"""Model library: models, immutable versions, lineage, flags, checkpoints, sessions, deletion.

On-disk layout (inside <data root>/models):

  <model_id>/
    model.json                 name, description, architecture, active/best/latest pointers
    versions/
      v001/ model.zip          stable-baselines3 PPO zip (weights + optimiser + schedule)
            version.json       lineage, training settings, maps, validation metrics, flags
      v002/ ...
    checkpoints/<session_id>/ppo_<steps>_steps.zip   intermediate checkpoints of a session
    sessions/<session_id>.json (+ <session_id>_episodes.csv, _curve.png)
    evaluation/<version>_<timestamp>.json

Versions are never overwritten. Training always writes a *new* version: the weights go to
a temp folder first, are verified, get their metadata, and only then is the folder renamed
into place and model.json updated. Version numbers are never reused, even after deletion.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from maniabot.logs import event
from maniabot.storage import Storage, atomic_write_json, dir_size, read_json, remove_tree

FORMAT = 1
MODE = "osu!mania 4K"
log = logging.getLogger("maniabot.models")

REQUIRED_ZIP_MEMBERS = ("data", "policy.pth")


class ModelError(Exception):
    """Base class; str(exc) is a human-readable message suitable for the UI."""


class ModelNotFound(ModelError):
    pass


class VersionNotFound(ModelError):
    pass


class InvalidCheckpoint(ModelError):
    pass


class InUseError(ModelError):
    pass


class ProtectedError(ModelError):
    pass


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def version_name(n: int) -> str:
    return f"v{n:03d}"


def version_number(v: str) -> int:
    m = re.fullmatch(r"v(\d+)", v or "")
    if not m:
        raise VersionNotFound(f"'{v}' is not a version name")
    return int(m.group(1))


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return slug[:60] or "model"


def verify_weights(path: Path) -> None:
    """Raise InvalidCheckpoint unless `path` is a readable stable-baselines3 model zip."""
    path = Path(path)
    if not path.exists():
        raise InvalidCheckpoint(f"Model file is missing: {path}")
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            missing = [m for m in REQUIRED_ZIP_MEMBERS if m not in names]
            if missing:
                raise InvalidCheckpoint(f"{path.name} is not a ManiaBot model (missing {', '.join(missing)}).")
            bad = z.testzip()
            if bad:
                raise InvalidCheckpoint(f"{path.name} is corrupt (damaged entry '{bad}').")
    except zipfile.BadZipFile as exc:
        raise InvalidCheckpoint(f"{path.name} is corrupt or not a model file ({exc}).") from exc
    except OSError as exc:
        raise InvalidCheckpoint(f"Can't read {path.name}: {exc.strerror or exc}") from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- in-use registry
class _InUse:
    """Process-wide record of models/versions that a running operation depends on."""

    def __init__(self):
        self._lock = threading.Lock()
        self._uses: dict[int, tuple[str, str | None, str]] = {}
        self._next = 0

    def acquire(self, model_id: str, version: str | None, reason: str) -> int:
        with self._lock:
            self._next += 1
            self._uses[self._next] = (model_id, version, reason)
            return self._next

    def release(self, token: int) -> None:
        with self._lock:
            self._uses.pop(token, None)

    def reasons(self, model_id: str, version: str | None = None) -> list[str]:
        """Reasons the model (version=None) or a specific version is in use."""
        with self._lock:
            out = []
            for mid, ver, reason in self._uses.values():
                if mid != model_id:
                    continue
                if version is None or ver is None or ver == version:
                    out.append(reason)
            return out

    @contextlib.contextmanager
    def use(self, model_id: str, version: str | None, reason: str) -> Iterator[None]:
        token = self.acquire(model_id, version, reason)
        try:
            yield
        finally:
            self.release(token)


IN_USE = _InUse()


# ------------------------------------------------------------------- records
@dataclass
class VersionRecord:
    model_id: str
    meta: dict[str, Any]
    path: Path
    model_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return self.meta["version"]

    @property
    def number(self) -> int:
        return version_number(self.version)

    @property
    def weights(self) -> Path:
        return self.path / "model.zip"

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent_version")

    @property
    def is_active(self) -> bool:
        return self.model_meta.get("active_version") == self.version

    @property
    def is_best(self) -> bool:
        return self.model_meta.get("best_version") == self.version

    @property
    def is_latest(self) -> bool:
        return self.model_meta.get("latest_version") == self.version

    @property
    def protected(self) -> bool:
        return bool(self.meta.get("protected"))

    @property
    def favourite(self) -> bool:
        return bool(self.meta.get("favourite"))

    @property
    def archived(self) -> bool:
        return bool(self.meta.get("archived"))

    @property
    def val_accuracy(self) -> float | None:
        val = self.meta.get("validation") or {}
        acc = val.get("accuracy")
        return float(acc) if acc is not None else None

    def badges(self) -> list[str]:
        out = []
        if self.is_active:
            out.append("Active")
        if self.is_best:
            out.append("Best")
        if self.is_latest:
            out.append("Latest")
        if self.protected:
            out.append("Protected")
        if self.favourite:
            out.append("Favourite")
        if self.archived:
            out.append("Archived")
        return out


@dataclass
class ModelRecord:
    meta: dict[str, Any]
    path: Path

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.id

    @property
    def archived(self) -> bool:
        return bool(self.meta.get("archived"))


@dataclass
class DeletionPlan:
    title: str
    items: list[str]
    size_bytes: int
    warnings: list[str] = field(default_factory=list)
    needs_protected_confirmation: bool = False
    blocked_reason: str | None = None  # set when deletion is not allowed right now


@dataclass
class PendingVersion:
    model_id: str
    version: str
    tmp_dir: Path

    @property
    def weights(self) -> Path:
        return self.tmp_dir / "model.zip"


# --------------------------------------------------------------------- store
class ModelStore:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.root = storage.models_dir
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ paths
    def model_dir(self, model_id: str) -> Path:
        return self.root / model_id

    def versions_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "versions"

    def version_dir(self, model_id: str, version: str) -> Path:
        version_number(version)  # validates the name, so it can't escape the folder
        return self.versions_dir(model_id) / version

    def checkpoints_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "checkpoints"

    def sessions_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "sessions"

    def evaluation_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "evaluation"

    # ----------------------------------------------------------- locking
    @contextlib.contextmanager
    def _lock(self, model_id: str, timeout: float = 15.0) -> Iterator[None]:
        """Cross-process lock for read-modify-write of model.json (training runs in a child process)."""
        lock = self.model_dir(model_id) / ".lock"
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > 60:  # stale lock from a crashed process
                        lock.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise ModelError("The model is busy (another operation is saving it). Try again.")
                time.sleep(0.05)
        try:
            yield
        finally:
            lock.unlink(missing_ok=True)

    def _read_model_meta(self, model_id: str) -> dict[str, Any]:
        meta = read_json(self.model_dir(model_id) / "model.json")
        if not meta:
            raise ModelNotFound(f"Model '{model_id}' was not found (it may have been deleted).")
        return meta

    def _write_model_meta(self, meta: dict[str, Any]) -> None:
        meta["updated_at"] = now_iso()
        atomic_write_json(self.model_dir(meta["id"]) / "model.json", meta)

    @contextlib.contextmanager
    def _edit_model(self, model_id: str) -> Iterator[dict[str, Any]]:
        if not (self.model_dir(model_id) / "model.json").exists():
            raise ModelNotFound(f"Model '{model_id}' was not found (it may have been deleted).")
        with self._lock(model_id):
            meta = self._read_model_meta(model_id)
            yield meta
            self._write_model_meta(meta)

    # ------------------------------------------------------------ models
    def list_models(self, include_archived: bool = True) -> list[ModelRecord]:
        out = []
        if not self.root.exists():
            return out
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            meta = read_json(d / "model.json")
            if not meta or "id" not in meta:
                continue
            rec = ModelRecord(meta, d)
            if include_archived or not rec.archived:
                out.append(rec)
        out.sort(key=lambda r: r.meta.get("last_trained_at") or r.meta.get("created_at") or "", reverse=True)
        return out

    def get_model(self, model_id: str) -> ModelRecord:
        return ModelRecord(self._read_model_meta(model_id), self.model_dir(model_id))

    def exists(self, model_id: str) -> bool:
        return (self.model_dir(model_id) / "model.json").exists()

    def name_taken(self, name: str, exclude_id: str | None = None) -> bool:
        lname = name.strip().lower()
        return any(m.name.lower() == lname and m.id != exclude_id for m in self.list_models())

    def create_model(
        self,
        name: str,
        description: str = "",
        architecture: dict[str, Any] | None = None,
        env_config: dict[str, Any] | None = None,
    ) -> ModelRecord:
        name = name.strip()
        if not name:
            raise ModelError("Give the model a name.")
        if self.name_taken(name):
            raise ModelError(f"A model called '{name}' already exists.")
        base = slugify(name)
        model_id, n = base, 2
        while self.model_dir(model_id).exists():
            model_id, n = f"{base}-{n}", n + 1
        d = self.model_dir(model_id)
        for sub in ("versions", "checkpoints", "sessions", "evaluation"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        meta = {
            "format": FORMAT,
            "id": model_id,
            "name": name,
            "description": description.strip(),
            "mode": MODE,
            "keys": 4,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "last_trained_at": None,
            "architecture": architecture or {},
            "env_config": env_config or {},
            "active_version": None,
            "best_version": None,
            "best_source": None,
            "latest_version": None,
            "next_version": 1,
            "archived": False,
        }
        self._write_model_meta(meta)
        event(log, "model.created", model=model_id, name=name)
        return ModelRecord(meta, d)

    def update_model(self, model_id: str, **fields: Any) -> ModelRecord:
        allowed = {"description", "archived"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"can't update {bad}")
        with self._edit_model(model_id) as meta:
            meta.update(fields)
        event(log, "model.updated", model=model_id, **fields)
        return self.get_model(model_id)

    def rename_model(self, model_id: str, new_name: str) -> ModelRecord:
        """Changes the display name only; the folder id (and every reference to it) stays."""
        new_name = new_name.strip()
        if not new_name:
            raise ModelError("The name can't be empty.")
        if self.name_taken(new_name, exclude_id=model_id):
            raise ModelError(f"A model called '{new_name}' already exists.")
        with self._edit_model(model_id) as meta:
            old = meta.get("name")
            meta["name"] = new_name
        event(log, "model.renamed", model=model_id, old=old, new=new_name)
        return self.get_model(model_id)

    def model_size(self, model_id: str) -> int:
        return dir_size(self.model_dir(model_id))

    def plan_delete_model(self, model_id: str) -> DeletionPlan:
        rec = self.get_model(model_id)
        versions = self.list_versions(model_id)
        ckpts = self.list_checkpoints(model_id)
        protected = [v.version for v in versions if v.protected]
        plan = DeletionPlan(
            title=f"Delete {rec.name}?",
            items=[
                f"{len(versions)} model version(s)",
                f"{len(ckpts)} checkpoint(s)",
                "Training sessions, logs and metadata",
                "Evaluation results",
                "All other files in the model folder",
            ],
            size_bytes=self.model_size(model_id),
        )
        if protected:
            plan.warnings.append(f"Protected versions will also be deleted: {', '.join(protected)}.")
            plan.needs_protected_confirmation = True
        reasons = IN_USE.reasons(model_id)
        if reasons:
            plan.blocked_reason = f"This model is in use ({'; '.join(sorted(set(reasons)))}). Stop that first."
        plan.warnings.append("Replays you generated are kept in Replay History.")
        return plan

    def delete_model(self, model_id: str, confirm_protected: bool = False) -> None:
        plan = self.plan_delete_model(model_id)
        if plan.blocked_reason:
            raise InUseError(plan.blocked_reason)
        if plan.needs_protected_confirmation and not confirm_protected:
            raise ProtectedError("This model has protected versions; confirm deleting them too.")
        d = self.model_dir(model_id)
        # Rename first so a half-finished delete never looks like a valid model.
        trash = self.root / f".deleting-{model_id}-{uuid.uuid4().hex[:8]}"
        os.replace(d, trash)
        remove_tree(trash)
        event(log, "model.deleted", model=model_id, bytes=plan.size_bytes)

    # ---------------------------------------------------------- versions
    def list_versions(self, model_id: str) -> list[VersionRecord]:
        model_meta = self._read_model_meta(model_id)
        out = []
        vdir = self.versions_dir(model_id)
        if vdir.exists():
            for d in vdir.iterdir():
                if not d.is_dir() or not re.fullmatch(r"v\d+", d.name):
                    continue
                meta = read_json(d / "version.json")
                if not meta:
                    meta = {"version": d.name, "missing_metadata": True}
                out.append(VersionRecord(model_id, meta, d, model_meta))
        out.sort(key=lambda v: v.number)
        return out

    def get_version(self, model_id: str, version: str) -> VersionRecord:
        model_meta = self._read_model_meta(model_id)
        d = self.version_dir(model_id, version)
        meta = read_json(d / "version.json")
        if not meta:
            if d.exists():
                meta = {"version": version, "missing_metadata": True}
            else:
                raise VersionNotFound(f"{model_meta.get('name', model_id)} {version} was not found (it may have been deleted).")
        return VersionRecord(model_id, meta, d, model_meta)

    def resolve_version(self, model_id: str, which: str = "active") -> VersionRecord:
        """`which` is a version name or one of 'active', 'best', 'latest'."""
        meta = self._read_model_meta(model_id)
        if which in ("active", "best", "latest"):
            v = meta.get(f"{which}_version")
            if not v:
                if which == "active" and meta.get("latest_version"):
                    v = meta["latest_version"]
                else:
                    raise VersionNotFound(f"{meta['name']} has no {which} version yet.")
            which = v
        return self.get_version(model_id, which)

    def reserve_version(self, model_id: str) -> str:
        """Allocate the next version number. Numbers are never reused, even after deletion."""
        with self._edit_model(model_id) as meta:
            n = int(meta.get("next_version") or 1)
            existing = [version_number(p.name) for p in self.versions_dir(model_id).glob("v*") if re.fullmatch(r"v\d+", p.name)]
            n = max([n, *(e + 1 for e in existing)])
            meta["next_version"] = n + 1
        return version_name(n)

    def begin_version(self, model_id: str, version: str | None = None) -> PendingVersion:
        """Create a temp folder to save a new version into (reserving a number if not given)."""
        version = version or self.reserve_version(model_id)
        version_number(version)
        tmp = self.versions_dir(model_id) / f".tmp-{version}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        return PendingVersion(model_id, version, tmp)

    def abort_version(self, pending: PendingVersion) -> None:
        remove_tree(pending.tmp_dir)

    def commit_version(
        self, pending: PendingVersion, metadata: dict[str, Any], set_active: bool = True
    ) -> VersionRecord:
        """Verify the saved weights, write metadata, then atomically publish the version."""
        try:
            verify_weights(pending.weights)
            meta = {
                "format": FORMAT,
                "version": pending.version,
                "model_id": pending.model_id,
                "parent_version": None,
                "created_at": now_iso(),
                "protected": False,
                "favourite": False,
                "archived": False,
                "notes": "",
                **metadata,
            }
            meta["version"] = pending.version
            meta["file_size"] = pending.weights.stat().st_size
            meta["weights_sha256"] = sha256_file(pending.weights)
            atomic_write_json(pending.tmp_dir / "version.json", meta)
            final = self.version_dir(pending.model_id, pending.version)
            if final.exists():
                raise ModelError(f"{pending.version} already exists; refusing to overwrite it.")
            os.replace(pending.tmp_dir, final)
        except BaseException:
            self.abort_version(pending)
            raise
        with self._edit_model(pending.model_id) as mm:
            mm["latest_version"] = pending.version
            mm["last_trained_at"] = now_iso()
            if set_active or not mm.get("active_version"):
                mm["active_version"] = pending.version
            self._maybe_auto_best(mm, pending.model_id)
        event(log, "version.created", model=pending.model_id, version=pending.version,
              parent=meta.get("parent_version"), val_acc=(meta.get("validation") or {}).get("accuracy"))
        return self.get_version(pending.model_id, pending.version)

    def _maybe_auto_best(self, mm: dict[str, Any], model_id: str) -> None:
        """Best = highest validation accuracy, unless the user picked one by hand."""
        if mm.get("best_source") == "manual" and mm.get("best_version"):
            if (self.versions_dir(model_id) / mm["best_version"]).exists():
                return
        best, best_acc = None, -1.0
        for d in self.versions_dir(model_id).glob("v*"):
            meta = read_json(d / "version.json") or {}
            acc = (meta.get("validation") or {}).get("accuracy")
            if acc is not None and float(acc) > best_acc:
                best, best_acc = meta["version"], float(acc)
        mm["best_version"] = best
        mm["best_source"] = "auto" if best else None

    def set_active(self, model_id: str, version: str) -> None:
        self.get_version(model_id, version)
        with self._edit_model(model_id) as mm:
            old = mm.get("active_version")
            mm["active_version"] = version
        event(log, "version.restored", model=model_id, version=version, previous_active=old)

    def set_best(self, model_id: str, version: str | None) -> None:
        """Mark a version as best by hand (None = go back to automatic best by validation)."""
        if version:
            self.get_version(model_id, version)
        with self._edit_model(model_id) as mm:
            if version:
                mm["best_version"], mm["best_source"] = version, "manual"
            else:
                mm["best_source"] = None
                self._maybe_auto_best(mm, model_id)
        event(log, "version.best", model=model_id, version=version or "auto")

    def update_version(self, model_id: str, version: str, **fields: Any) -> VersionRecord:
        allowed = {"protected", "favourite", "archived", "notes", "validation", "evaluations"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"can't update {bad}")
        with self._lock(model_id):
            rec = self.get_version(model_id, version)
            meta = dict(rec.meta)
            meta.update(fields)
            atomic_write_json(rec.path / "version.json", meta)
        if "validation" in fields:
            with self._edit_model(model_id) as mm:
                self._maybe_auto_best(mm, model_id)
        event(log, "version.updated", model=model_id, version=version, **{k: v for k, v in fields.items() if k != "validation"})
        return self.get_version(model_id, version)

    def children(self, model_id: str, version: str) -> list[str]:
        return [v.version for v in self.list_versions(model_id) if v.parent == version]

    def lineage(self, model_id: str, version: str) -> list[tuple[str, bool]]:
        """Ancestry from the root to `version` as (version, still_exists)."""
        chain: list[tuple[str, bool]] = []
        seen = set()
        cur: str | None = version
        while cur and cur not in seen:
            seen.add(cur)
            meta = read_json(self.version_dir(model_id, cur) / "version.json")
            chain.append((cur, meta is not None))
            cur = (meta or {}).get("parent_version")
        return list(reversed(chain))

    def plan_delete_version(self, model_id: str, version: str) -> DeletionPlan:
        rec = self.get_version(model_id, version)
        name = rec.model_meta.get("name", model_id)
        plan = DeletionPlan(
            title=f"Delete {name} {version}?",
            items=[f"Model weights and metadata of {version}"],
            size_bytes=dir_size(rec.path),
        )
        remaining = [v for v in self.list_versions(model_id) if v.version != version]
        if rec.is_active:
            fallback = remaining[-1].version if remaining else None
            plan.warnings.append(
                f"{version} is the active version; {fallback} will become active." if fallback
                else f"{version} is the only version; the model will have no versions left."
            )
        if rec.is_best:
            plan.warnings.append(f"{version} is marked Best; the best version will be re-chosen from validation results.")
        kids = self.children(model_id, version)
        if kids:
            plan.warnings.append(
                f"{', '.join(kids)} were trained from {version}. They keep working; their lineage will show {version} as deleted."
            )
        if rec.protected:
            plan.warnings.append(f"{version} is PROTECTED.")
            plan.needs_protected_confirmation = True
        reasons = IN_USE.reasons(model_id, version)
        if reasons:
            plan.blocked_reason = f"{version} is in use ({'; '.join(sorted(set(reasons)))}). Stop that first."
        return plan

    def delete_version(self, model_id: str, version: str, confirm_protected: bool = False) -> None:
        plan = self.plan_delete_version(model_id, version)
        if plan.blocked_reason:
            raise InUseError(plan.blocked_reason)
        if plan.needs_protected_confirmation and not confirm_protected:
            raise ProtectedError(f"{version} is protected; confirm deleting a protected version.")
        d = self.version_dir(model_id, version)
        with self._lock(model_id):
            trash = self.versions_dir(model_id) / f".deleting-{version}-{uuid.uuid4().hex[:8]}"
            os.replace(d, trash)
        remove_tree(trash)
        with self._edit_model(model_id) as mm:
            remaining = sorted(
                (p.name for p in self.versions_dir(model_id).glob("v*") if re.fullmatch(r"v\d+", p.name)),
                key=version_number,
            )
            if mm.get("latest_version") == version:
                mm["latest_version"] = remaining[-1] if remaining else None
            if mm.get("active_version") == version:
                mm["active_version"] = remaining[-1] if remaining else None
            if mm.get("best_version") == version:
                mm["best_source"] = None
                self._maybe_auto_best(mm, model_id)
        for ev in self.evaluation_dir(model_id).glob(f"{version}_*.json"):
            ev.unlink(missing_ok=True)
        event(log, "version.deleted", model=model_id, version=version, protected=plan.needs_protected_confirmation)

    def version_from_file(
        self, model_id: str, weights: Path, metadata: dict[str, Any], set_active: bool = False
    ) -> VersionRecord:
        """Create a new version by copying an existing model zip (checkpoint promotion, import)."""
        verify_weights(weights)
        pending = self.begin_version(model_id)
        try:
            shutil.copy2(weights, pending.weights)
        except BaseException:
            self.abort_version(pending)
            raise
        return self.commit_version(pending, metadata, set_active=set_active)

    # ------------------------------------------------------- checkpoints
    def list_checkpoints(self, model_id: str) -> list[dict[str, Any]]:
        out = []
        cdir = self.checkpoints_dir(model_id)
        if not cdir.exists():
            return out
        session_meta = {s["session_id"]: s for s in self.list_sessions(model_id)}
        for f in cdir.rglob("*.zip"):
            m = re.search(r"_(\d+)_steps\.zip$", f.name)
            session_id = f.parent.name
            sess = session_meta.get(session_id, {})
            val = next((c.get("val_accuracy") for c in sess.get("checkpoints", []) if Path(c["path"]).name == f.name), None)
            st = f.stat()
            out.append({
                "path": str(f),
                "session_id": session_id,
                "timesteps": int(m.group(1)) if m else None,
                "size": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "val_accuracy": val,
                "is_best": sess.get("best_checkpoint") == str(f),
                "is_latest": sess.get("latest_checkpoint") == str(f),
                "parent_version": sess.get("start_version"),
            })
        out.sort(key=lambda c: (c["modified"], c["timesteps"] or 0))
        return out

    def delete_checkpoints(self, model_id: str, paths: list[str]) -> int:
        cdir = self.checkpoints_dir(model_id).resolve()
        n = 0
        for p in paths:
            f = Path(p).resolve()
            if cdir not in f.parents:  # never delete outside the model's checkpoint folder
                raise ModelError(f"{p} is not a checkpoint of this model.")
            f.unlink(missing_ok=True)
            n += 1
            if f.parent != cdir and not any(f.parent.iterdir()):
                f.parent.rmdir()
        event(log, "checkpoints.deleted", model=model_id, count=n)
        return n

    # ---------------------------------------------------------- sessions
    def save_session(self, model_id: str, session: dict[str, Any]) -> Path:
        path = self.sessions_dir(model_id) / f"{session['session_id']}.json"
        atomic_write_json(path, session)
        return path

    def list_sessions(self, model_id: str) -> list[dict[str, Any]]:
        out = []
        sdir = self.sessions_dir(model_id)
        if sdir.exists():
            for f in sdir.glob("*.json"):
                s = read_json(f)
                if s and "session_id" in s:
                    out.append(s)
        out.sort(key=lambda s: s.get("started_at") or "")
        return out

    def mark_interrupted_sessions(self, live_session_ids: set[str] = frozenset()) -> int:
        """Sessions still marked 'running' with no live process were interrupted (crash/power loss)."""
        n = 0
        for rec in self.list_models():
            for s in self.list_sessions(rec.id):
                if s.get("status") in ("running", "paused") and s["session_id"] not in live_session_ids:
                    s["status"] = "interrupted"
                    s["ended_at"] = s.get("ended_at") or now_iso()
                    self.save_session(rec.id, s)
                    n += 1
        if n:
            event(log, "sessions.interrupted", count=n)
        return n

    def delete_logs(self, model_id: str) -> int:
        """Delete per-session episode logs and curves (session summaries are kept for lineage)."""
        n = 0
        for f in list(self.sessions_dir(model_id).glob("*_episodes.csv")) + list(self.sessions_dir(model_id).glob("*_curve.png")):
            f.unlink(missing_ok=True)
            n += 1
        event(log, "logs.deleted", model=model_id, count=n)
        return n

    # -------------------------------------------------------- evaluation
    def save_evaluation(self, model_id: str, version: str, result: dict[str, Any]) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self.evaluation_dir(model_id) / f"{version}_{stamp}.json"
        atomic_write_json(path, {"model_id": model_id, "version": version, "created_at": now_iso(), **result})
        event(log, "evaluation.saved", model=model_id, version=version, acc=result.get("accuracy"))
        return path

    def list_evaluations(self, model_id: str, version: str | None = None) -> list[dict[str, Any]]:
        out = []
        edir = self.evaluation_dir(model_id)
        if edir.exists():
            for f in edir.glob(f"{version or 'v*'}_*.json"):
                e = read_json(f)
                if e:
                    e["_path"] = str(f)
                    out.append(e)
        out.sort(key=lambda e: e.get("created_at") or "")
        return out

    def delete_evaluations(self, model_id: str, version: str | None = None) -> int:
        n = 0
        for e in self.list_evaluations(model_id, version):
            Path(e["_path"]).unlink(missing_ok=True)
            n += 1
        event(log, "evaluations.deleted", model=model_id, version=version or "all", count=n)
        return n

    # -------------------------------------------------------------- misc
    def cleanup_temp(self) -> int:
        """Remove leftovers of interrupted saves/deletes. Never touches committed versions."""
        n = 0
        for rec in self.list_models():
            for d in self.versions_dir(rec.id).glob(".tmp-*"):
                remove_tree(d)
                n += 1
            for d in self.versions_dir(rec.id).glob(".deleting-*"):
                remove_tree(d)
                n += 1
        for d in self.root.glob(".deleting-*"):
            remove_tree(d)
            n += 1
        return n

    def totals(self) -> dict[str, Any]:
        models = self.list_models()
        n_versions = 0
        for m in models:
            try:
                n_versions += len(self.list_versions(m.id))
            except ModelError:
                pass
        return {"models": len(models), "versions": n_versions, "archived": sum(1 for m in models if m.archived)}
