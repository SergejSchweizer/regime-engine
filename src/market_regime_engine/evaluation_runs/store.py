"""Transactional SQLite ledger for resumable evaluation work units."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from string import hexdigits
from typing import cast
from uuid import uuid4

from market_regime_engine.evaluation_runs.contracts import (
    EvaluationRunIdentity,
    WorkUnitIdentity,
    canonical_json,
)


class WorkUnitStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    DOMAIN_INVALID = "DOMAIN_INVALID"


@dataclass(frozen=True, slots=True)
class EvaluationRunState:
    run_key: str
    dataset_snapshot_key: str
    status: str
    root_identity_hash: str | None
    work_unit_count: int


@dataclass(frozen=True, slots=True)
class WorkUnitState:
    work_unit_key: str
    input_hash: str
    status: WorkUnitStatus
    payload_hash: str | None
    attempt_count: int
    lease_expires_at: datetime | None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_utc(value: str | None, field: str) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return parsed


def _digest(payload: bytes, field: str) -> str:
    value = sha256(payload).hexdigest()
    if len(value) != 64:
        raise ValueError(f"{field} is not a SHA-256 digest")
    return value


def _require_hash(value: str, field: str) -> None:
    if len(value) != 64 or value != value.lower() or any(c not in hexdigits for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


class SQLiteEvaluationRunStore:
    """A single SQLite database with atomic claim/complete transitions."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(mode=0o750, parents=True, exist_ok=True)
        self._database = self._root / "evaluation-runs.sqlite3"
        self._initialize()

    @property
    def root(self) -> Path:
        """Return the durable directory used by this ledger."""

        return self._root

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        # Journal mode is a database-wide setting.  Re-applying WAL on every
        # connection is itself a schema-level write and races when spawned
        # workers initialise stores concurrently.  Keep the database's
        # existing mode and let busy_timeout cover concurrent ledger writes.
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_key TEXT PRIMARY KEY,
                    identity_json BLOB NOT NULL,
                    dataset_snapshot_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'COMPLETE')),
                    root_identity_hash TEXT,
                    final_payload BLOB,
                    final_payload_hash TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS work_units (
                    run_key TEXT NOT NULL REFERENCES runs(run_key),
                    work_unit_key TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('PENDING', 'RUNNING', 'COMPLETE', 'DOMAIN_INVALID')
                    ),
                    payload BLOB,
                    payload_hash TEXT,
                    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    failure_json BLOB,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_key, work_unit_key)
                );
                CREATE INDEX IF NOT EXISTS work_units_status_idx
                    ON work_units(run_key, status);
                """
            )

    @staticmethod
    def _identity_json(identity: EvaluationRunIdentity) -> bytes:
        return canonical_json(identity.as_dict())

    @staticmethod
    def _check_identity(row: sqlite3.Row, identity: EvaluationRunIdentity) -> None:
        if row["identity_json"] != SQLiteEvaluationRunStore._identity_json(identity):
            raise ValueError("evaluation run identity changed for an existing run key")
        if row["dataset_snapshot_key"] != identity.dataset_snapshot_key:
            raise ValueError("evaluation run dataset snapshot identity changed")

    def open_run(self, identity: EvaluationRunIdentity) -> EvaluationRunState:
        now = _utc_now().isoformat()
        encoded = self._identity_json(identity)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runs WHERE run_key = ?", (identity.key,)
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO runs(
                        run_key, identity_json, dataset_snapshot_key, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, 'RUNNING', ?, ?)
                    """,
                    (identity.key, encoded, identity.dataset_snapshot_key, now, now),
                )
                row = connection.execute(
                    "SELECT * FROM runs WHERE run_key = ?", (identity.key,)
                ).fetchone()
            assert row is not None
            self._check_identity(row, identity)
            count = connection.execute(
                "SELECT COUNT(*) FROM work_units WHERE run_key = ?", (identity.key,)
            ).fetchone()[0]
            connection.commit()
        return EvaluationRunState(
            run_key=identity.key,
            dataset_snapshot_key=identity.dataset_snapshot_key,
            status=str(row["status"]),
            root_identity_hash=row["root_identity_hash"],
            work_unit_count=int(count),
        )

    def load_identity(self, run_key: str) -> EvaluationRunIdentity:
        """Load and revalidate the immutable identity stored for ``run_key``."""

        _require_hash(run_key, "run_key")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT identity_json FROM runs WHERE run_key = ?", (run_key,)
            ).fetchone()
        if row is None:
            raise ValueError("evaluation run ledger is missing")
        raw = row["identity_json"]
        if not isinstance(raw, bytes):
            raise ValueError("evaluation run identity payload is invalid")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("evaluation run identity payload is invalid") from exc
        if not isinstance(payload, dict):
            raise ValueError("evaluation run identity payload must be an object")
        identity = EvaluationRunIdentity.from_dict(payload)
        if identity.key != run_key:
            raise ValueError("evaluation run identity hash does not match requested run key")
        return identity

    @staticmethod
    def _unit_input(unit: WorkUnitIdentity) -> str:
        return unit.work_unit_input_hash

    def _get_unit(
        self,
        connection: sqlite3.Connection,
        unit: WorkUnitIdentity,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                "SELECT * FROM work_units WHERE run_key = ? AND work_unit_key = ?",
                (unit.evaluation_run_key, unit.key),
            ).fetchone(),
        )

    def claim_work_unit(
        self,
        identity: EvaluationRunIdentity,
        unit: WorkUnitIdentity,
        *,
        lease_seconds: int = 3_600,
        owner: str | None = None,
    ) -> bool:
        if unit.evaluation_run_key != identity.key:
            raise ValueError("work unit does not belong to evaluation run")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        owner = owner or f"{os.getpid()}-{uuid4().hex}"
        now = _utc_now()
        expires = now + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT * FROM runs WHERE run_key = ?", (identity.key,)
            ).fetchone()
            if run is None:
                raise ValueError("evaluation run must be opened before claiming work")
            self._check_identity(run, identity)
            if run["status"] == "COMPLETE":
                connection.commit()
                return False
            existing = self._get_unit(connection, unit)
            if existing is not None:
                if existing["input_hash"] != self._unit_input(unit):
                    raise ValueError("work unit input identity changed")
                status = WorkUnitStatus(existing["status"])
                if status in {WorkUnitStatus.COMPLETE, WorkUnitStatus.DOMAIN_INVALID}:
                    connection.commit()
                    return False
                if status is WorkUnitStatus.RUNNING:
                    lease = _parse_utc(existing["lease_expires_at"], "lease_expires_at")
                    if lease is not None and lease > now:
                        connection.commit()
                        return False
                connection.execute(
                    """
                    UPDATE work_units SET status='RUNNING', attempt_count=attempt_count + 1,
                        lease_owner=?, lease_expires_at=?, updated_at=?, failure_json=NULL
                    WHERE run_key=? AND work_unit_key=? AND input_hash=?
                    """,
                    (
                        owner,
                        expires.isoformat(),
                        now.isoformat(),
                        identity.key,
                        unit.key,
                        self._unit_input(unit),
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO work_units(
                        run_key, work_unit_key, input_hash, status, attempt_count,
                        lease_owner, lease_expires_at, created_at, updated_at
                    ) VALUES (?, ?, ?, 'RUNNING', 1, ?, ?, ?, ?)
                    """,
                    (
                        identity.key,
                        unit.key,
                        self._unit_input(unit),
                        owner,
                        expires.isoformat(),
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
            connection.commit()
        return True

    def _load_terminal(
        self,
        connection: sqlite3.Connection,
        identity: EvaluationRunIdentity,
        unit: WorkUnitIdentity,
    ) -> bytes | None:
        row = self._get_unit(connection, unit)
        if row is None:
            return None
        if row["input_hash"] != self._unit_input(unit):
            raise ValueError("work unit input identity changed")
        status = WorkUnitStatus(row["status"])
        if status not in {WorkUnitStatus.COMPLETE, WorkUnitStatus.DOMAIN_INVALID}:
            return None
        payload = row["payload"]
        payload_hash = row["payload_hash"]
        if not isinstance(payload, bytes) or not isinstance(payload_hash, str):
            raise ValueError("terminal work unit payload metadata is missing")
        if _digest(payload, "work unit payload") != payload_hash:
            raise ValueError("terminal work unit payload hash does not match")
        return payload

    def load_completed_work_unit(
        self,
        identity: EvaluationRunIdentity,
        unit: WorkUnitIdentity,
    ) -> bytes | None:
        if unit.evaluation_run_key != identity.key:
            raise ValueError("work unit does not belong to evaluation run")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_key = ?", (identity.key,)
            ).fetchone()
            if row is None:
                raise ValueError("evaluation run ledger is missing")
            self._check_identity(row, identity)
            return self._load_terminal(connection, identity, unit)

    def work_unit_state(
        self,
        identity: EvaluationRunIdentity,
        unit: WorkUnitIdentity,
    ) -> WorkUnitState | None:
        """Return the persisted operational state for duplicate executors."""

        if unit.evaluation_run_key != identity.key:
            raise ValueError("work unit does not belong to evaluation run")
        with self._connect() as connection:
            run = connection.execute(
                "SELECT * FROM runs WHERE run_key = ?", (identity.key,)
            ).fetchone()
            if run is None:
                raise ValueError("evaluation run ledger is missing")
            self._check_identity(run, identity)
            row = self._get_unit(connection, unit)
            if row is None:
                return None
            if row["input_hash"] != self._unit_input(unit):
                raise ValueError("work unit input identity changed")
            return WorkUnitState(
                work_unit_key=str(row["work_unit_key"]),
                input_hash=str(row["input_hash"]),
                status=WorkUnitStatus(row["status"]),
                payload_hash=(None if row["payload_hash"] is None else str(row["payload_hash"])),
                attempt_count=int(row["attempt_count"]),
                lease_expires_at=_parse_utc(row["lease_expires_at"], "lease_expires_at"),
            )

    def complete_work_unit(
        self,
        identity: EvaluationRunIdentity,
        unit: WorkUnitIdentity,
        payload: bytes,
        *,
        domain_invalid: bool = False,
    ) -> None:
        if unit.evaluation_run_key != identity.key:
            raise ValueError("work unit does not belong to evaluation run")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("terminal work unit payload must be non-empty bytes")
        payload_hash = _digest(payload, "work unit payload")
        status = WorkUnitStatus.DOMAIN_INVALID if domain_invalid else WorkUnitStatus.COMPLETE
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT * FROM runs WHERE run_key = ?", (identity.key,)
            ).fetchone()
            if run is None:
                raise ValueError("evaluation run ledger is missing")
            self._check_identity(run, identity)
            row = self._get_unit(connection, unit)
            if row is None or row["input_hash"] != self._unit_input(unit):
                raise ValueError("work unit was not claimed with the requested input")
            existing_status = WorkUnitStatus(row["status"])
            if existing_status in {WorkUnitStatus.COMPLETE, WorkUnitStatus.DOMAIN_INVALID}:
                if row["payload_hash"] != payload_hash or row["payload"] != payload:
                    raise ValueError("terminal work unit is immutable")
                connection.commit()
                return
            if existing_status is not WorkUnitStatus.RUNNING:
                raise ValueError("only a running work unit can be completed")
            now = _utc_now().isoformat()
            updated = connection.execute(
                """
                UPDATE work_units SET status=?, payload=?, payload_hash=?,
                    lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                WHERE run_key=? AND work_unit_key=? AND status='RUNNING'
                """,
                (status.value, payload, payload_hash, now, identity.key, unit.key),
            ).rowcount
            if updated != 1:
                raise ValueError("work unit completion lost its compare-and-set race")
            connection.commit()

    def fail_work_unit(
        self,
        identity: EvaluationRunIdentity,
        unit: WorkUnitIdentity,
        failure: str,
    ) -> None:
        if not failure or failure.strip() != failure:
            raise ValueError("technical failure metadata must be non-empty and trimmed")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._get_unit(connection, unit)
            if row is None or row["input_hash"] != self._unit_input(unit):
                raise ValueError("work unit was not claimed with the requested input")
            if WorkUnitStatus(row["status"]) is not WorkUnitStatus.RUNNING:
                raise ValueError("only a running work unit can record a technical failure")
            connection.execute(
                """
                UPDATE work_units SET status='PENDING', lease_owner=NULL,
                    lease_expires_at=NULL, failure_json=?, updated_at=?
                WHERE run_key=? AND work_unit_key=? AND status='RUNNING'
                """,
                (
                    canonical_json({"error": failure}),
                    _utc_now().isoformat(),
                    identity.key,
                    unit.key,
                ),
            )
            connection.commit()

    def terminalize_pending_stage_units_for_invalid_outer_fold(
        self,
        identity: EvaluationRunIdentity,
        fold_id: str,
        reason: str,
    ) -> int:
        """Safely repair old stage ledgers after an invalid outer fold.

        Early resumable-run versions recorded an invalid outer-fold payload but
        left the stage that caused the domain rejection pending.  This method
        only transitions pending ``v4_stage`` units in the named fold scope,
        never running or completed units, and records the same deterministic
        domain-invalid reason for each repaired unit.
        """

        if not fold_id or fold_id.strip() != fold_id or "/" in fold_id:
            raise ValueError("fold_id must be a non-empty path-safe identifier")
        if not reason or reason.strip() != reason:
            raise ValueError("domain-invalid reason must be non-empty and trimmed")
        payload = canonical_json({"status": WorkUnitStatus.DOMAIN_INVALID.value, "reason": reason})
        payload_hash = _digest(payload, "domain-invalid payload")
        prefix = f"v4_stage/scope={fold_id}/stage="
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT * FROM runs WHERE run_key = ?", (identity.key,)
            ).fetchone()
            if run is None:
                raise ValueError("evaluation run ledger is missing")
            self._check_identity(run, identity)
            rows = connection.execute(
                """
                SELECT work_unit_key, status FROM work_units
                WHERE run_key=? AND work_unit_key LIKE ?
                """,
                (identity.key, prefix + "%"),
            ).fetchall()
            if any(WorkUnitStatus(row["status"]) is WorkUnitStatus.RUNNING for row in rows):
                connection.rollback()
                raise RuntimeError(f"stage work unit remains claimed for outer fold {fold_id}")
            updated = connection.execute(
                """
                UPDATE work_units
                SET status='DOMAIN_INVALID', payload=?, payload_hash=?,
                    lease_owner=NULL, lease_expires_at=NULL, failure_json=NULL,
                    updated_at=?
                WHERE run_key=? AND work_unit_key LIKE ? AND status='PENDING'
                """,
                (payload, payload_hash, _utc_now().isoformat(), identity.key, prefix + "%"),
            ).rowcount
            connection.commit()
        return int(updated)

    def complete_run(
        self,
        identity: EvaluationRunIdentity,
        root_identity_hash: str,
        payload: bytes,
    ) -> None:
        _require_hash(root_identity_hash, "root_identity_hash")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("completed run payload must be non-empty bytes")
        payload_hash = _digest(payload, "final payload")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runs WHERE run_key = ?", (identity.key,)
            ).fetchone()
            if row is None:
                raise ValueError("evaluation run ledger is missing")
            self._check_identity(row, identity)
            if row["status"] == "COMPLETE":
                if (
                    row["root_identity_hash"] != root_identity_hash
                    or row["final_payload"] != payload
                ):
                    raise ValueError("completed evaluation root/payload is immutable")
                connection.commit()
                return
            incomplete = connection.execute(
                """
                SELECT COUNT(*) FROM work_units
                WHERE run_key=? AND status NOT IN ('COMPLETE', 'DOMAIN_INVALID')
                """,
                (identity.key,),
            ).fetchone()[0]
            if incomplete != 0:
                raise ValueError("all evaluation work units must be terminal before completion")
            if (
                connection.execute(
                    "SELECT COUNT(*) FROM work_units WHERE run_key=?", (identity.key,)
                ).fetchone()[0]
                == 0
            ):
                raise ValueError("completed evaluation requires at least one work unit")
            connection.execute(
                """
                UPDATE runs SET status='COMPLETE', root_identity_hash=?,
                    final_payload=?, final_payload_hash=?, updated_at=?
                WHERE run_key=? AND status='RUNNING'
                """,
                (root_identity_hash, payload, payload_hash, _utc_now().isoformat(), identity.key),
            )
            connection.commit()

    def load_completed_run(self, identity: EvaluationRunIdentity) -> bytes | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_key = ?", (identity.key,)
            ).fetchone()
            if row is None:
                raise ValueError("evaluation run ledger is missing")
            self._check_identity(row, identity)
            if row["status"] != "COMPLETE":
                return None
            payload = row["final_payload"]
            if not isinstance(payload, bytes) or (
                _digest(payload, "final payload") != row["final_payload_hash"]
            ):
                raise ValueError("completed evaluation payload is missing or corrupt")
            return payload

    def state(self, identity: EvaluationRunIdentity) -> EvaluationRunState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_key = ?", (identity.key,)
            ).fetchone()
            if row is None:
                raise ValueError("evaluation run ledger is missing")
            self._check_identity(row, identity)
            count = connection.execute(
                "SELECT COUNT(*) FROM work_units WHERE run_key = ?", (identity.key,)
            ).fetchone()[0]
            return EvaluationRunState(
                identity.key,
                identity.dataset_snapshot_key,
                str(row["status"]),
                row["root_identity_hash"],
                int(count),
            )


__all__ = [
    "EvaluationRunState",
    "SQLiteEvaluationRunStore",
    "WorkUnitState",
    "WorkUnitStatus",
]
