"""Persistent, atomic evaluation-suite deduplication."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class EvaluationClaim(StrEnum):
    CLAIMED = "claimed"
    COMPLETED = "completed"
    RUNNING = "running"


@dataclass(slots=True)
class EvaluationDeduplicator:
    root: Path
    code_sha: str
    dataset_sha256: str

    def __post_init__(self) -> None:
        if len(self.code_sha) != 40 or any(
            char not in "0123456789abcdef" for char in self.code_sha
        ):
            raise ValueError("code_sha must be a lowercase 40-character Git SHA")
        if len(self.dataset_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.dataset_sha256
        ):
            raise ValueError("dataset_sha256 must be a lowercase SHA-256 digest")

    @property
    def _key(self) -> str:
        return f"xetra-v3-{self.code_sha}-{self.dataset_sha256}"

    @property
    def _state_path(self) -> Path:
        return self.root / f"{self._key}.json"

    @property
    def _lock_path(self) -> Path:
        return self.root / f"{self._key}.lock"

    def claim(self) -> EvaluationClaim:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self._state_path.exists():
                state = json.loads(self._state_path.read_text(encoding="utf-8"))
                status = state.get("status")
                if status == "FINISHED":
                    return EvaluationClaim.COMPLETED
                if status == "RUNNING":
                    return EvaluationClaim.RUNNING
            self._write("RUNNING")
            return EvaluationClaim.CLAIMED

    def complete(self) -> None:
        self._write("FINISHED")

    def abort(self) -> None:
        self._state_path.unlink(missing_ok=True)

    def _write(self, status: str) -> None:
        payload = {
            "code_sha": self.code_sha,
            "dataset_sha256": self.dataset_sha256,
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        temporary = self._state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self._state_path)
