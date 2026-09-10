"""Durable dataset snapshots and crash-safe evaluation run state."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from market_regime_engine.contracts import DATA_TIME_SEMANTICS, SourceLineage
from market_regime_engine.features.ports import (
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)

_SNAPSHOT_FORMAT_VERSION = 1
_RUN_FORMAT_VERSION = 1
_SHA256_LENGTH = 64
_GIT_SHA_LENGTH = 40


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _require_sha256(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _require_git_sha(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _GIT_SHA_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase Git commit SHA")


def _require_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must be timezone-aware UTC")


def _timestamp(value: datetime | None, field: str) -> str | None:
    if value is not None:
        _require_utc(value, field)
        return value.isoformat()
    return None


def _parse_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO timestamp") from exc
    _require_utc(result, field)
    return result


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class DatasetSnapshotKey:
    """Identity of one exact, materialized source matrix."""

    source_dataset: str
    source_table: str
    source_build_id: str
    data_sha256: str
    schema_version: int
    feature_version: int
    data_time_semantics: str
    row_count: int
    min_timestamp: datetime
    max_timestamp: datetime
    source_catalog_hash: str
    materialized_feature_data_sha256: str
    materialized_row_count: int
    materialized_min_timestamp: datetime | None
    materialized_max_timestamp: datetime | None

    def __post_init__(self) -> None:
        _require_text(self.source_dataset, "source_dataset")
        _require_text(self.source_table, "source_table")
        _require_text(self.source_build_id, "source_build_id")
        _require_sha256(self.data_sha256, "data_sha256")
        _require_sha256(self.source_catalog_hash, "source_catalog_hash")
        _require_sha256(
            self.materialized_feature_data_sha256,
            "materialized_feature_data_sha256",
        )
        if self.schema_version < 1 or self.feature_version < 1:
            raise ValueError("source schema and feature versions must be positive")
        if self.data_time_semantics != DATA_TIME_SEMANTICS:
            raise ValueError("unsupported data_time_semantics")
        if self.row_count < 0 or self.materialized_row_count < 0:
            raise ValueError("snapshot row counts must be non-negative")
        _require_utc(self.min_timestamp, "min_timestamp")
        _require_utc(self.max_timestamp, "max_timestamp")
        if self.min_timestamp > self.max_timestamp:
            raise ValueError("source timestamp bounds are inverted")
        if self.materialized_row_count == 0:
            if (
                self.materialized_min_timestamp is not None
                or self.materialized_max_timestamp is not None
            ):
                raise ValueError("empty materialization cannot have timestamp bounds")
        else:
            if self.materialized_min_timestamp is None or self.materialized_max_timestamp is None:
                raise ValueError("non-empty materialization requires timestamp bounds")
            _require_utc(self.materialized_min_timestamp, "materialized_min_timestamp")
            _require_utc(self.materialized_max_timestamp, "materialized_max_timestamp")
            if self.materialized_min_timestamp > self.materialized_max_timestamp:
                raise ValueError("materialized timestamp bounds are inverted")

    @classmethod
    def from_catalog(cls, catalog: FeatureCatalogSnapshot) -> DatasetSnapshotKey:
        lineage = catalog.lineage
        if (
            lineage.row_count is None
            or lineage.min_timestamp is None
            or lineage.max_timestamp is None
        ):
            raise ValueError("source lineage must contain row count and timestamp bounds")
        if (
            catalog.materialized_feature_data_sha256 is None
            or catalog.materialized_row_count is None
        ):
            raise ValueError("catalog must be bound to a materialized feature snapshot")
        return cls(
            source_dataset=lineage.source_dataset,
            source_table=lineage.source_table,
            source_build_id=lineage.source_build_id,
            data_sha256=lineage.data_sha256,
            schema_version=lineage.schema_version,
            feature_version=lineage.feature_version,
            data_time_semantics=lineage.data_time_semantics,
            row_count=lineage.row_count,
            min_timestamp=lineage.min_timestamp,
            max_timestamp=lineage.max_timestamp,
            source_catalog_hash=catalog.catalog_hash,
            materialized_feature_data_sha256=catalog.materialized_feature_data_sha256,
            materialized_row_count=catalog.materialized_row_count,
            materialized_min_timestamp=catalog.materialized_min_timestamp,
            materialized_max_timestamp=catalog.materialized_max_timestamp,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_dataset": self.source_dataset,
            "source_table": self.source_table,
            "source_build_id": self.source_build_id,
            "data_sha256": self.data_sha256,
            "schema_version": self.schema_version,
            "feature_version": self.feature_version,
            "data_time_semantics": self.data_time_semantics,
            "row_count": self.row_count,
            "min_timestamp": self.min_timestamp.isoformat(),
            "max_timestamp": self.max_timestamp.isoformat(),
            "source_catalog_hash": self.source_catalog_hash,
            "materialized_feature_data_sha256": self.materialized_feature_data_sha256,
            "materialized_row_count": self.materialized_row_count,
            "materialized_min_timestamp": _timestamp(
                self.materialized_min_timestamp, "materialized_min_timestamp"
            ),
            "materialized_max_timestamp": _timestamp(
                self.materialized_max_timestamp, "materialized_max_timestamp"
            ),
        }

    @property
    def key(self) -> str:
        return sha256(_canonical_bytes(self.as_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationRunKey:
    """Identity of one statistical evaluation under one immutable snapshot."""

    evaluation_id: str
    profile_id: str
    profile_config_version: int
    profile_hash: str
    evaluation_contract_version: int
    evaluation_plan_hash: str
    dataset_snapshot_key: str
    evaluation_cutoff: datetime | None
    repository_commit_sha: str
    uv_lock_sha256: str
    python_version: str

    def __post_init__(self) -> None:
        _require_text(self.evaluation_id, "evaluation_id")
        _require_text(self.profile_id, "profile_id")
        if self.profile_config_version < 1 or self.evaluation_contract_version < 1:
            raise ValueError("evaluation/profile contract versions must be positive")
        for value, field in (
            (self.profile_hash, "profile_hash"),
            (self.evaluation_plan_hash, "evaluation_plan_hash"),
            (self.dataset_snapshot_key, "dataset_snapshot_key"),
            (self.uv_lock_sha256, "uv_lock_sha256"),
        ):
            _require_sha256(value, field)
        _require_git_sha(self.repository_commit_sha, "repository_commit_sha")
        _require_text(self.python_version, "python_version")
        if self.evaluation_cutoff is not None:
            _require_utc(self.evaluation_cutoff, "evaluation_cutoff")

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluation_id": self.evaluation_id,
            "profile_id": self.profile_id,
            "profile_config_version": self.profile_config_version,
            "profile_hash": self.profile_hash,
            "evaluation_contract_version": self.evaluation_contract_version,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "dataset_snapshot_key": self.dataset_snapshot_key,
            "evaluation_cutoff": _timestamp(self.evaluation_cutoff, "evaluation_cutoff"),
            "repository_commit_sha": self.repository_commit_sha,
            "uv_lock_sha256": self.uv_lock_sha256,
            "python_version": self.python_version,
        }

    @property
    def key(self) -> str:
        return sha256(_canonical_bytes(self.as_dict())).hexdigest()


class DatasetSnapshotStore(Protocol):
    def finalize(self, key: DatasetSnapshotKey, snapshot: FeatureSnapshot) -> None: ...

    def load(self, key: DatasetSnapshotKey) -> FeatureSnapshot: ...


class EvaluationRunStore(Protocol):
    def open_run(self, key: EvaluationRunKey) -> EvaluationRunState: ...

    def claim_work_unit(
        self,
        key: EvaluationRunKey,
        work_unit_key: str,
        input_hash: str,
        *,
        lease_seconds: int = 3_600,
    ) -> bool: ...

    def load_completed_work_unit(
        self, key: EvaluationRunKey, work_unit_key: str, input_hash: str
    ) -> bytes | None: ...

    def complete_work_unit(
        self,
        key: EvaluationRunKey,
        work_unit_key: str,
        input_hash: str,
        payload: bytes,
    ) -> None: ...

    def complete_run(
        self, key: EvaluationRunKey, root_identity_hash: str, payload: bytes
    ) -> None: ...

    def load_completed_run(self, key: EvaluationRunKey) -> bytes | None: ...


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


class FileDatasetSnapshotStore:
    """Persist immutable snapshot payload and manifest with atomic replacement."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _directory(self, key: DatasetSnapshotKey) -> Path:
        return self._root / key.key

    @staticmethod
    def _payload(key: DatasetSnapshotKey, snapshot: FeatureSnapshot) -> bytes:
        return _canonical_bytes(
            {
                "format_version": _SNAPSHOT_FORMAT_VERSION,
                "dataset_snapshot_key": key.key,
                "identity": key.as_dict(),
                "lineage": {
                    "source_dataset": snapshot.lineage.source_dataset,
                    "source_build_id": snapshot.lineage.source_build_id,
                    "data_sha256": snapshot.lineage.data_sha256,
                    "schema_version": snapshot.lineage.schema_version,
                    "feature_version": snapshot.lineage.feature_version,
                    "source_table": snapshot.lineage.source_table,
                    "synced_at_utc": snapshot.lineage.synced_at_utc.isoformat(),
                    "data_time_semantics": snapshot.lineage.data_time_semantics,
                    "row_count": snapshot.lineage.row_count,
                    "min_timestamp": _timestamp(
                        snapshot.lineage.min_timestamp, "lineage.min_timestamp"
                    ),
                    "max_timestamp": _timestamp(
                        snapshot.lineage.max_timestamp, "lineage.max_timestamp"
                    ),
                },
                "feature_names": list(snapshot.feature_names),
                "rows": [
                    {
                        "timestamp": row.timestamp.isoformat(),
                        "values": [
                            None if value is None else float(value).hex() for value in row.values
                        ],
                    }
                    for row in snapshot.rows
                ],
                "skipped_incomplete_row_count": snapshot.skipped_incomplete_row_count,
                "materialized_feature_data_sha256": snapshot.materialized_feature_data_sha256,
            }
        )

    @staticmethod
    def _decode(key: DatasetSnapshotKey, payload: bytes) -> FeatureSnapshot:
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("durable snapshot payload is not valid JSON") from exc
        if not isinstance(document, dict):
            raise ValueError("durable snapshot payload must be an object")
        if document.get("format_version") != _SNAPSHOT_FORMAT_VERSION:
            raise ValueError("unsupported durable snapshot format version")
        if document.get("dataset_snapshot_key") != key.key:
            raise ValueError("durable snapshot key does not match requested key")
        if document.get("identity") != key.as_dict():
            raise ValueError("durable snapshot identity does not match requested key")
        lineage_document = document.get("lineage")
        if not isinstance(lineage_document, dict):
            raise ValueError("durable snapshot lineage is missing")
        min_timestamp = lineage_document.get("min_timestamp")
        max_timestamp = lineage_document.get("max_timestamp")
        row_count = lineage_document.get("row_count")
        if not isinstance(row_count, int) or isinstance(row_count, bool):
            raise ValueError("durable snapshot lineage row_count is invalid")
        schema_version = lineage_document.get("schema_version")
        feature_version = lineage_document.get("feature_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version < 1
            or not isinstance(feature_version, int)
            or isinstance(feature_version, bool)
            or feature_version < 1
        ):
            raise ValueError("durable snapshot lineage versions are invalid")
        lineage = SourceLineage(
            source_dataset=_require_text(lineage_document.get("source_dataset"), "source_dataset"),
            source_build_id=_require_text(
                lineage_document.get("source_build_id"), "source_build_id"
            ),
            data_sha256=_require_text(lineage_document.get("data_sha256"), "data_sha256"),
            schema_version=schema_version,
            feature_version=feature_version,
            source_table=_require_text(lineage_document.get("source_table"), "source_table"),
            synced_at_utc=_parse_timestamp(
                lineage_document.get("synced_at_utc"), "lineage.synced_at_utc"
            ),
            data_time_semantics=_require_text(
                lineage_document.get("data_time_semantics"), "data_time_semantics"
            ),
            row_count=row_count,
            min_timestamp=None
            if min_timestamp is None
            else _parse_timestamp(min_timestamp, "lineage.min_timestamp"),
            max_timestamp=None
            if max_timestamp is None
            else _parse_timestamp(max_timestamp, "lineage.max_timestamp"),
        )
        feature_names_document = document.get("feature_names")
        rows_document = document.get("rows")
        if not isinstance(feature_names_document, list) or not isinstance(rows_document, list):
            raise ValueError("durable snapshot features or rows are malformed")
        if any(not isinstance(name, str) for name in feature_names_document):
            raise ValueError("durable snapshot feature names are malformed")
        feature_names = tuple(feature_names_document)
        rows: list[FeatureRow] = []
        for item in rows_document:
            if not isinstance(item, dict) or not isinstance(item.get("values"), list):
                raise ValueError("durable snapshot row is malformed")
            values: list[float | None] = []
            for value in item["values"]:
                if value is None:
                    values.append(None)
                elif isinstance(value, str):
                    try:
                        values.append(float.fromhex(value))
                    except ValueError as exc:
                        raise ValueError("durable snapshot row contains invalid float") from exc
                else:
                    raise ValueError("durable snapshot row value is not a float encoding")
            rows.append(
                FeatureRow(
                    _parse_timestamp(item.get("timestamp"), "snapshot row timestamp"),
                    tuple(values),
                )
            )
        skipped_count = document.get("skipped_incomplete_row_count", 0)
        if (
            not isinstance(skipped_count, int)
            or isinstance(skipped_count, bool)
            or skipped_count < 0
        ):
            raise ValueError("durable snapshot skipped row count is invalid")
        snapshot = FeatureSnapshot(
            lineage=lineage,
            feature_names=feature_names,
            rows=tuple(rows),
            skipped_incomplete_row_count=skipped_count,
            materialized_feature_data_sha256=cast(
                str | None, document.get("materialized_feature_data_sha256")
            ),
        )
        if (
            snapshot.lineage.source_dataset != key.source_dataset
            or snapshot.lineage.source_table != key.source_table
            or snapshot.lineage.source_build_id != key.source_build_id
            or snapshot.lineage.data_sha256 != key.data_sha256
            or snapshot.lineage.schema_version != key.schema_version
            or snapshot.lineage.feature_version != key.feature_version
            or snapshot.lineage.data_time_semantics != key.data_time_semantics
            or snapshot.lineage.row_count != key.row_count
            or snapshot.lineage.min_timestamp != key.min_timestamp
            or snapshot.lineage.max_timestamp != key.max_timestamp
        ):
            raise ValueError("durable snapshot lineage does not match dataset key")
        if snapshot.materialized_feature_data_sha256 != key.materialized_feature_data_sha256:
            raise ValueError("durable snapshot matrix hash does not match dataset key")
        if snapshot.row_count != key.materialized_row_count:
            raise ValueError("durable snapshot row count does not match dataset key")
        if snapshot.min_timestamp != key.materialized_min_timestamp:
            raise ValueError("durable snapshot minimum timestamp does not match dataset key")
        if snapshot.max_timestamp != key.materialized_max_timestamp:
            raise ValueError("durable snapshot maximum timestamp does not match dataset key")
        return snapshot

    def finalize(self, key: DatasetSnapshotKey, snapshot: FeatureSnapshot) -> None:
        if (
            snapshot.lineage.source_dataset != key.source_dataset
            or snapshot.lineage.source_table != key.source_table
            or snapshot.lineage.source_build_id != key.source_build_id
            or snapshot.lineage.data_sha256 != key.data_sha256
            or snapshot.lineage.schema_version != key.schema_version
            or snapshot.lineage.feature_version != key.feature_version
            or snapshot.lineage.data_time_semantics != key.data_time_semantics
            or snapshot.lineage.row_count != key.row_count
            or snapshot.lineage.min_timestamp != key.min_timestamp
            or snapshot.lineage.max_timestamp != key.max_timestamp
        ):
            raise ValueError("snapshot lineage does not match dataset snapshot key")
        if snapshot.materialized_feature_data_sha256 != key.materialized_feature_data_sha256:
            raise ValueError("snapshot matrix hash does not match dataset snapshot key")
        if snapshot.row_count != key.materialized_row_count:
            raise ValueError("snapshot row count does not match dataset snapshot key")
        if snapshot.min_timestamp != key.materialized_min_timestamp:
            raise ValueError("snapshot minimum timestamp does not match dataset snapshot key")
        if snapshot.max_timestamp != key.materialized_max_timestamp:
            raise ValueError("snapshot maximum timestamp does not match dataset snapshot key")
        directory = self._directory(key)
        payload_path = directory / "snapshot.json"
        manifest_path = directory / "manifest.json"
        payload = self._payload(key, snapshot)
        manifest = _canonical_bytes(
            {
                "format_version": _SNAPSHOT_FORMAT_VERSION,
                "dataset_snapshot_key": key.key,
                "payload_sha256": sha256(payload).hexdigest(),
            }
        )
        with _exclusive_lock(directory / ".lock"):
            payload_exists = payload_path.is_file()
            manifest_exists = manifest_path.is_file()
            if payload_exists != manifest_exists:
                raise ValueError("durable snapshot is incomplete and cannot be resumed")
            if payload_exists:
                existing_payload = payload_path.read_bytes()
                existing_manifest = manifest_path.read_bytes()
                if existing_payload != payload or existing_manifest != manifest:
                    raise ValueError("durable dataset snapshot is immutable")
                self._decode(key, existing_payload)
                return
            _atomic_write(payload_path, payload)
            _atomic_write(manifest_path, manifest)

    def load(self, key: DatasetSnapshotKey) -> FeatureSnapshot:
        directory = self._directory(key)
        payload_path = directory / "snapshot.json"
        manifest_path = directory / "manifest.json"
        if not payload_path.is_file() or not manifest_path.is_file():
            raise ValueError("durable dataset snapshot is missing or incomplete")
        payload = payload_path.read_bytes()
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("durable snapshot manifest is not valid JSON") from exc
        if not isinstance(manifest, dict):
            raise ValueError("durable snapshot manifest must be an object")
        if manifest.get("format_version") != _SNAPSHOT_FORMAT_VERSION:
            raise ValueError("unsupported durable snapshot manifest version")
        if manifest.get("dataset_snapshot_key") != key.key:
            raise ValueError("durable snapshot manifest key does not match")
        payload_hash = manifest.get("payload_sha256")
        if not isinstance(payload_hash, str):
            raise ValueError("durable snapshot manifest payload hash is missing")
        _require_sha256(payload_hash, "payload_sha256")
        if sha256(payload).hexdigest() != payload_hash:
            raise ValueError("durable snapshot payload hash does not match manifest")
        return self._decode(key, payload)


class FileEvaluationRunStore:
    """A small JSON ledger with file locks and immutable unit payloads."""

    def __init__(self, root: Path, *, lease_seconds: int = 3_600) -> None:
        if lease_seconds < 1:
            raise ValueError("default lease_seconds must be positive")
        self._root = root
        self._lease_seconds = lease_seconds

    def _directory(self, key: EvaluationRunKey) -> Path:
        return self._root / key.key

    def _ledger_path(self, key: EvaluationRunKey) -> Path:
        return self._directory(key) / "run.json"

    def _unit_payload_path(self, key: EvaluationRunKey, work_unit_key: str) -> Path:
        unit_hash = sha256(work_unit_key.encode("utf-8")).hexdigest()
        return self._directory(key) / "units" / f"{unit_hash}.payload"

    @staticmethod
    def _read_json(path: Path, description: str) -> dict[str, Any]:
        try:
            document = json.loads(path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{description} is not valid JSON") from exc
        if not isinstance(document, dict):
            raise ValueError(f"{description} must be a JSON object")
        return document

    @staticmethod
    def _new_ledger(key: EvaluationRunKey) -> dict[str, Any]:
        return {
            "format_version": _RUN_FORMAT_VERSION,
            "run_key": key.key,
            "identity": key.as_dict(),
            "dataset_snapshot_key": key.dataset_snapshot_key,
            "status": "RUNNING",
            "root_identity_hash": None,
            "final_payload_hash": None,
            "work_units": {},
        }

    @staticmethod
    def _validate_ledger(key: EvaluationRunKey, ledger: dict[str, Any]) -> None:
        if ledger.get("format_version") != _RUN_FORMAT_VERSION:
            raise ValueError("unsupported evaluation run ledger version")
        if ledger.get("run_key") != key.key or ledger.get("identity") != key.as_dict():
            raise ValueError("evaluation run ledger identity does not match requested run")
        if ledger.get("dataset_snapshot_key") != key.dataset_snapshot_key:
            raise ValueError("evaluation run ledger dataset key does not match")
        if ledger.get("status") not in {"RUNNING", "COMPLETE"}:
            raise ValueError("evaluation run ledger status is invalid")
        if not isinstance(ledger.get("work_units"), dict):
            raise ValueError("evaluation run ledger work_units is invalid")

    def _state(self, key: EvaluationRunKey, ledger: dict[str, Any]) -> EvaluationRunState:
        root_hash = ledger.get("root_identity_hash")
        if root_hash is not None:
            if not isinstance(root_hash, str):
                raise ValueError("evaluation run root identity hash is invalid")
            _require_sha256(root_hash, "root_identity_hash")
        return EvaluationRunState(
            run_key=key.key,
            dataset_snapshot_key=key.dataset_snapshot_key,
            status=cast(str, ledger["status"]),
            root_identity_hash=root_hash,
            work_unit_count=len(cast(dict[str, Any], ledger["work_units"])),
        )

    def open_run(self, key: EvaluationRunKey) -> EvaluationRunState:
        directory = self._directory(key)
        with _exclusive_lock(directory / ".lock"):
            path = self._ledger_path(key)
            if path.is_file():
                ledger = self._read_json(path, "evaluation run ledger")
                self._validate_ledger(key, ledger)
            else:
                ledger = self._new_ledger(key)
                _atomic_write(path, _canonical_bytes(ledger))
            return self._state(key, ledger)

    @staticmethod
    def _require_work_unit(work_unit_key: str, input_hash: str) -> None:
        _require_text(work_unit_key, "work_unit_key")
        _require_sha256(input_hash, "work_unit_input_hash")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def claim_work_unit(
        self,
        key: EvaluationRunKey,
        work_unit_key: str,
        input_hash: str,
        *,
        lease_seconds: int | None = None,
    ) -> bool:
        self._require_work_unit(work_unit_key, input_hash)
        duration = self._lease_seconds if lease_seconds is None else lease_seconds
        if duration < 1:
            raise ValueError("lease_seconds must be positive")
        directory = self._directory(key)
        with _exclusive_lock(directory / ".lock"):
            path = self._ledger_path(key)
            if not path.is_file():
                raise ValueError("evaluation run must be opened before claiming work")
            ledger = self._read_json(path, "evaluation run ledger")
            self._validate_ledger(key, ledger)
            if ledger["status"] == "COMPLETE":
                return False
            units = cast(dict[str, Any], ledger["work_units"])
            existing = units.get(work_unit_key)
            if existing is not None:
                if existing.get("input_hash") != input_hash:
                    raise ValueError("work unit input identity changed")
                if existing.get("status") in {
                    WorkUnitStatus.COMPLETE,
                    WorkUnitStatus.DOMAIN_INVALID,
                }:
                    return False
                if existing.get("status") == WorkUnitStatus.RUNNING:
                    expires = existing.get("lease_expires_at")
                    if (
                        isinstance(expires, str)
                        and _parse_timestamp(expires, "work unit lease") > self._now()
                    ):
                        return False
            now = self._now()
            units[work_unit_key] = {
                "input_hash": input_hash,
                "status": WorkUnitStatus.RUNNING,
                "lease_expires_at": (now + timedelta(seconds=duration)).isoformat(),
                "updated_at": now.isoformat(),
                "owner": f"{os.getpid()}-{uuid4().hex}",
            }
            _atomic_write(path, _canonical_bytes(ledger))
            return True

    def _read_completed_work_unit(
        self,
        key: EvaluationRunKey,
        ledger: dict[str, Any],
        work_unit_key: str,
        input_hash: str,
    ) -> bytes | None:
        record = cast(dict[str, Any], ledger["work_units"]).get(work_unit_key)
        if record is None or record.get("status") != WorkUnitStatus.COMPLETE:
            return None
        if record.get("input_hash") != input_hash:
            raise ValueError("completed work unit input identity changed")
        location = record.get("payload_location")
        payload_hash = record.get("payload_hash")
        if not isinstance(location, str) or not isinstance(payload_hash, str):
            raise ValueError("completed work unit payload metadata is missing")
        _require_sha256(payload_hash, "work_unit_payload_hash")
        payload_path = self._directory(key) / location
        if (
            payload_path != self._unit_payload_path(key, work_unit_key)
            or not payload_path.is_file()
        ):
            raise ValueError("completed work unit payload is missing")
        payload = payload_path.read_bytes()
        if sha256(payload).hexdigest() != payload_hash:
            raise ValueError("completed work unit payload hash does not match")
        return payload

    def load_completed_work_unit(
        self, key: EvaluationRunKey, work_unit_key: str, input_hash: str
    ) -> bytes | None:
        self._require_work_unit(work_unit_key, input_hash)
        directory = self._directory(key)
        with _exclusive_lock(directory / ".lock"):
            path = self._ledger_path(key)
            if not path.is_file():
                raise ValueError("evaluation run ledger is missing")
            ledger = self._read_json(path, "evaluation run ledger")
            self._validate_ledger(key, ledger)
            return self._read_completed_work_unit(key, ledger, work_unit_key, input_hash)

    def complete_work_unit(
        self,
        key: EvaluationRunKey,
        work_unit_key: str,
        input_hash: str,
        payload: bytes,
    ) -> None:
        self._require_work_unit(work_unit_key, input_hash)
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("completed work unit payload must be non-empty bytes")
        directory = self._directory(key)
        ledger_path = self._ledger_path(key)
        payload_path = self._unit_payload_path(key, work_unit_key)
        payload_hash = sha256(payload).hexdigest()
        relative_location = str(payload_path.relative_to(directory))
        with _exclusive_lock(directory / ".lock"):
            ledger = self._read_json(ledger_path, "evaluation run ledger")
            self._validate_ledger(key, ledger)
            record = cast(dict[str, Any], ledger["work_units"]).get(work_unit_key)
            if record is None or record.get("input_hash") != input_hash:
                raise ValueError("work unit was not claimed with the requested input")
            if record.get("status") == WorkUnitStatus.COMPLETE:
                existing = self._read_completed_work_unit(key, ledger, work_unit_key, input_hash)
                if existing != payload:
                    raise ValueError("completed work unit is immutable")
                return
            if record.get("status") != WorkUnitStatus.RUNNING:
                raise ValueError("only a running work unit can be completed")
            _atomic_write(payload_path, payload)
            record.update(
                {
                    "status": WorkUnitStatus.COMPLETE,
                    "payload_hash": payload_hash,
                    "payload_location": relative_location,
                    "lease_expires_at": None,
                    "updated_at": self._now().isoformat(),
                }
            )
            _atomic_write(ledger_path, _canonical_bytes(ledger))

    def complete_run(self, key: EvaluationRunKey, root_identity_hash: str, payload: bytes) -> None:
        _require_sha256(root_identity_hash, "root_identity_hash")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("completed evaluation payload must be non-empty bytes")
        directory = self._directory(key)
        ledger_path = self._ledger_path(key)
        final_path = directory / "final.payload"
        with _exclusive_lock(directory / ".lock"):
            ledger = self._read_json(ledger_path, "evaluation run ledger")
            self._validate_ledger(key, ledger)
            if ledger["status"] == "COMPLETE":
                if ledger.get("root_identity_hash") != root_identity_hash:
                    raise ValueError("completed evaluation root identity is immutable")
                if not final_path.is_file() or final_path.read_bytes() != payload:
                    raise ValueError("completed evaluation payload is immutable")
                return
            units = cast(dict[str, Any], ledger["work_units"])
            if not units or any(
                record.get("status") not in {WorkUnitStatus.COMPLETE, WorkUnitStatus.DOMAIN_INVALID}
                for record in units.values()
            ):
                raise ValueError("all evaluation work units must be terminal before completion")
            _atomic_write(final_path, payload)
            ledger["root_identity_hash"] = root_identity_hash
            ledger["final_payload_hash"] = sha256(payload).hexdigest()
            ledger["status"] = "COMPLETE"
            _atomic_write(ledger_path, _canonical_bytes(ledger))

    def load_completed_run(self, key: EvaluationRunKey) -> bytes | None:
        directory = self._directory(key)
        with _exclusive_lock(directory / ".lock"):
            path = self._ledger_path(key)
            if not path.is_file():
                raise ValueError("evaluation run ledger is missing")
            ledger = self._read_json(path, "evaluation run ledger")
            self._validate_ledger(key, ledger)
            if ledger["status"] != "COMPLETE":
                return None
            payload_path = directory / "final.payload"
            if not payload_path.is_file():
                raise ValueError("completed evaluation payload is missing")
            payload = payload_path.read_bytes()
            root_hash = ledger.get("root_identity_hash")
            if not isinstance(root_hash, str):
                raise ValueError("completed evaluation root identity is missing")
            _require_sha256(root_hash, "root_identity_hash")
            payload_hash = ledger.get("final_payload_hash")
            if not isinstance(payload_hash, str):
                raise ValueError("completed evaluation payload hash is missing")
            _require_sha256(payload_hash, "final_payload_hash")
            if sha256(payload).hexdigest() != payload_hash:
                raise ValueError("completed evaluation payload hash does not match")
            return payload


__all__ = [
    "DatasetSnapshotKey",
    "DatasetSnapshotStore",
    "EvaluationRunKey",
    "EvaluationRunState",
    "EvaluationRunStore",
    "FileDatasetSnapshotStore",
    "FileEvaluationRunStore",
    "WorkUnitStatus",
]
