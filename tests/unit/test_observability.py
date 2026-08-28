from __future__ import annotations

import os
from datetime import UTC, datetime

from market_regime_engine.observability import apply_retention, configure_debug_logging


def test_retention_archives_after_three_weeks_then_deletes_after_three_months(tmp_path) -> None:
    active = tmp_path / "active"
    archive = tmp_path / "archive"
    active.mkdir()
    archive.mkdir()
    old_active = active / "old.jsonl"
    old_active.write_text("active", encoding="utf-8")
    old_archive = archive / "expired.jsonl"
    old_archive.write_text("archive", encoding="utf-8")
    now = datetime(2026, 8, 28, tzinfo=UTC)
    os.utime(old_active, (now.timestamp() - 22 * 86400,) * 2)
    os.utime(old_archive, (now.timestamp() - 91 * 86400,) * 2)

    apply_retention(tmp_path, now=now)

    assert (archive / "old.jsonl").exists()
    assert not old_archive.exists()


def test_debug_logging_writes_structured_event(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REGIME_LOG_DIR", str(tmp_path))
    logger = configure_debug_logging("unit")
    logger.debug("calculation_completed fold=3")
    for handler in logger.handlers:
        handler.flush()

    files = list((tmp_path / "active").glob("unit-*.jsonl"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert '"level": "DEBUG"' in content
    assert "calculation_completed fold=3" in content
