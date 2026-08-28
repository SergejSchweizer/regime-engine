"""Atomic immutable filesystem writer for evaluation statistics dossiers."""

from __future__ import annotations

import os
import tempfile
from hashlib import sha256
from pathlib import Path

from market_regime_engine.evaluation_statistics.contracts import RunStatistics, Status
from market_regime_engine.evaluation_statistics.render import render_statistics


class StatisticsWriter:
    def __init__(self, checkout_root: str | Path) -> None:
        self._root = Path(checkout_root) / "evaluations"

    def preflight(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        probe = self._root / ".write-probe"
        try:
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as exc:
            raise OSError("evaluation statistics root is not writable") from exc
        return self._root

    def _directory(self, statistics: RunStatistics) -> Path:
        return self._root / statistics.evaluation_id.value / statistics.mlflow_run_id

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def start(self, statistics: RunStatistics) -> Path:
        if statistics.status is not Status.RUNNING:
            raise ValueError("initial statistics must have RUNNING status")
        self.preflight()
        directory = self._directory(statistics)
        try:
            directory.mkdir(parents=True)
        except FileExistsError as exc:
            raise FileExistsError("evaluation statistics run directory is immutable") from exc
        content = statistics.canonical_json()
        self._atomic_write(directory / "statistics.json", content)
        self._atomic_write(
            directory / "statistics.md", render_statistics(statistics).encode("utf-8")
        )
        return directory

    def finalize(self, statistics: RunStatistics) -> str:
        if statistics.status is Status.RUNNING:
            raise ValueError("final statistics must be FINISHED or FAILED")
        directory = self._directory(statistics)
        data_path = directory / "statistics.json"
        if not data_path.is_file() or not (directory / "statistics.md").is_file():
            raise FileNotFoundError("statistics run must be initialized before finalization")
        initial = data_path.read_bytes()
        if b'"status":"RUNNING"' not in initial:
            raise FileExistsError("finalized evaluation statistics are immutable")
        content = statistics.canonical_json()
        digest = sha256(content).hexdigest()
        self._atomic_write(data_path, content)
        self._atomic_write(
            directory / "statistics.md", render_statistics(statistics, digest).encode("utf-8")
        )
        return digest
