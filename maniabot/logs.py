"""Structured application logging.

Every log line is written to <data root>/logs/app.log (rotating) as
`timestamp level logger event key=value ...`. Use `event()` for domain events so they are
easy to grep; full stack traces of errors go to the log file, not the UI.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_configured: Path | None = None


def setup_logging(logs_dir: Path, level: int = logging.INFO, filename: str = "app.log") -> Path:
    global _configured
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / filename
    if _configured == log_file:
        return log_file
    root = logging.getLogger("maniabot")
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _configured = log_file
    return log_file


def _fmt(value) -> str:
    text = str(value)
    return f'"{text}"' if (" " in text or not text) else text


def event(logger: logging.Logger | str, name: str, /, level: int = logging.INFO, **fields) -> None:
    """Log a structured domain event, e.g. event(log, "version.created", model="x", version="v002")."""
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    parts = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
    logger.log(level, "%s %s", name, parts)
