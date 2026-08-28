"""Structured local diagnostics with bounded retention."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType

ACTIVE_RETENTION = timedelta(days=21)
ARCHIVE_RETENTION = timedelta(days=90)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def log_root() -> Path:
    return Path(os.environ.get("REGIME_LOG_DIR", ".logs"))


def apply_retention(root: Path | None = None, *, now: datetime | None = None) -> None:
    """Move active logs after 21 days and delete archives after 90 days."""

    root = root or log_root()
    active, archive = root / "active", root / "archive"
    active.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(UTC)
    for file in active.glob("*.jsonl"):
        age = now - datetime.fromtimestamp(file.stat().st_mtime, UTC)
        if age >= ACTIVE_RETENTION:
            shutil.move(str(file), archive / file.name)
    for file in archive.glob("*.jsonl"):
        age = now - datetime.fromtimestamp(file.stat().st_mtime, UTC)
        if age >= ARCHIVE_RETENTION:
            file.unlink()


def configure_debug_logging(component: str) -> logging.Logger:
    """Configure daily JSONL DEBUG logging, warning capture and exception evidence."""

    root = log_root()
    apply_retention(root)
    logger = logging.getLogger("market_regime_engine")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    file_path = root / "active" / f"{component}-{datetime.now(UTC):%Y-%m-%d}.jsonl"
    has_file_handler = any(
        getattr(handler, "baseFilename", None) == str(file_path) for handler in logger.handlers
    )
    if not has_file_handler:
        handler = logging.FileHandler(file_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logging.captureWarnings(True)
    warning_logger = logging.getLogger("py.warnings")
    warning_logger.setLevel(logging.DEBUG)
    warning_logger.handlers = logger.handlers
    warning_logger.propagate = False
    warnings.simplefilter("default")

    def excepthook(
        exc_type: type[BaseException], value: BaseException, trace: TracebackType | None
    ) -> None:
        logger.critical("unhandled_exception", exc_info=(exc_type, value, trace))
        sys.__excepthook__(exc_type, value, trace)

    sys.excepthook = excepthook
    logger.debug("logging_configured component=%s log_dir=%s", component, root)
    return logger
