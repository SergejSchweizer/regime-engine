"""Canonical identities for immutable evaluation execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from string import hexdigits

from market_regime_engine.contracts import DATA_TIME_SEMANTICS
from market_regime_engine.features.ports import FeatureCatalogSnapshot


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in hexdigits for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _git_sha(value: object, field: str) -> str:
    value = _text(value, field)
    if len(value) != 40 or value != value.lower() or any(c not in hexdigits for c in value):
        raise ValueError(f"{field} must be a lowercase Git commit SHA")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware UTC")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


def _canonical(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def canonical_json(value: object) -> bytes:
    import json

    return (
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def content_hash(value: object) -> str:
    return sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetSnapshotIdentity:
    """Identity of one exact ordered schema-wide source snapshot."""

    source_dataset: str
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
    source_table: str

    def __post_init__(self) -> None:
        for value, field in (
            (self.source_dataset, "source_dataset"),
            (self.source_build_id, "source_build_id"),
            (self.source_table, "source_table"),
        ):
            _text(value, field)
        for value, field in (
            (self.data_sha256, "data_sha256"),
            (self.source_catalog_hash, "source_catalog_hash"),
            (self.materialized_feature_data_sha256, "materialized_feature_data_sha256"),
        ):
            _sha256(value, field)
        if self.schema_version < 1 or self.feature_version < 1:
            raise ValueError("schema and feature versions must be positive")
        if self.data_time_semantics != DATA_TIME_SEMANTICS:
            raise ValueError("unsupported data_time_semantics")
        if self.row_count < 0 or self.materialized_row_count < 0:
            raise ValueError("snapshot row counts cannot be negative")
        _utc(self.min_timestamp, "min_timestamp")
        _utc(self.max_timestamp, "max_timestamp")
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
            _utc(self.materialized_min_timestamp, "materialized_min_timestamp")
            _utc(self.materialized_max_timestamp, "materialized_max_timestamp")
            if self.materialized_min_timestamp > self.materialized_max_timestamp:
                raise ValueError("materialized timestamp bounds are inverted")

    @classmethod
    def from_catalog(cls, catalog: FeatureCatalogSnapshot) -> DatasetSnapshotIdentity:
        lineage = catalog.lineage
        if (
            lineage.row_count is None
            or lineage.min_timestamp is None
            or lineage.max_timestamp is None
            or catalog.materialized_feature_data_sha256 is None
            or catalog.materialized_row_count is None
        ):
            raise ValueError("catalog must contain complete lineage and materialization bounds")
        return cls(
            source_dataset=lineage.source_dataset,
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
            source_table=lineage.source_table,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_dataset": self.source_dataset,
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
            "materialized_min_timestamp": (
                None
                if self.materialized_min_timestamp is None
                else self.materialized_min_timestamp.isoformat()
            ),
            "materialized_max_timestamp": (
                None
                if self.materialized_max_timestamp is None
                else self.materialized_max_timestamp.isoformat()
            ),
            "source_table": self.source_table,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DatasetSnapshotIdentity:
        """Reconstruct an identity from its canonical durable manifest form."""

        expected = {
            "source_dataset",
            "source_build_id",
            "data_sha256",
            "schema_version",
            "feature_version",
            "data_time_semantics",
            "row_count",
            "min_timestamp",
            "max_timestamp",
            "source_catalog_hash",
            "materialized_feature_data_sha256",
            "materialized_row_count",
            "materialized_min_timestamp",
            "materialized_max_timestamp",
            "source_table",
        }
        if set(payload) != expected:
            raise ValueError("dataset snapshot identity fields are incomplete or unknown")

        def timestamp(value: object, field: str) -> datetime:
            if not isinstance(value, str):
                raise ValueError(f"{field} must be an ISO timestamp")
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{field} must be an ISO timestamp") from exc
            return parsed

        return cls(
            source_dataset=str(payload["source_dataset"]),
            source_build_id=str(payload["source_build_id"]),
            data_sha256=str(payload["data_sha256"]),
            schema_version=_integer(payload["schema_version"], "schema_version"),
            feature_version=_integer(payload["feature_version"], "feature_version"),
            data_time_semantics=str(payload["data_time_semantics"]),
            row_count=_integer(payload["row_count"], "row_count"),
            min_timestamp=timestamp(payload["min_timestamp"], "min_timestamp"),
            max_timestamp=timestamp(payload["max_timestamp"], "max_timestamp"),
            source_catalog_hash=str(payload["source_catalog_hash"]),
            materialized_feature_data_sha256=str(payload["materialized_feature_data_sha256"]),
            materialized_row_count=_integer(
                payload["materialized_row_count"], "materialized_row_count"
            ),
            materialized_min_timestamp=(
                None
                if payload["materialized_min_timestamp"] is None
                else timestamp(payload["materialized_min_timestamp"], "materialized_min_timestamp")
            ),
            materialized_max_timestamp=(
                None
                if payload["materialized_max_timestamp"] is None
                else timestamp(payload["materialized_max_timestamp"], "materialized_max_timestamp")
            ),
            source_table=str(payload["source_table"]),
        )

    @property
    def key(self) -> str:
        return content_hash(self.as_dict())


@dataclass(frozen=True, slots=True)
class EvaluationRunIdentity:
    """Statistical identity of one evaluation invocation."""

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
        _text(self.evaluation_id, "evaluation_id")
        _text(self.profile_id, "profile_id")
        if self.profile_config_version < 1 or self.evaluation_contract_version < 1:
            raise ValueError("profile and evaluation contract versions must be positive")
        for value, field in (
            (self.profile_hash, "profile_hash"),
            (self.evaluation_plan_hash, "evaluation_plan_hash"),
            (self.dataset_snapshot_key, "dataset_snapshot_key"),
            (self.uv_lock_sha256, "uv_lock_sha256"),
        ):
            _sha256(value, field)
        _git_sha(self.repository_commit_sha, "repository_commit_sha")
        _text(self.python_version, "python_version")
        if self.evaluation_cutoff is not None:
            _utc(self.evaluation_cutoff, "evaluation_cutoff")

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluation_id": self.evaluation_id,
            "profile_id": self.profile_id,
            "profile_config_version": self.profile_config_version,
            "profile_hash": self.profile_hash,
            "evaluation_contract_version": self.evaluation_contract_version,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "dataset_snapshot_key": self.dataset_snapshot_key,
            "evaluation_cutoff": (
                None if self.evaluation_cutoff is None else self.evaluation_cutoff.isoformat()
            ),
            "repository_commit_sha": self.repository_commit_sha,
            "uv_lock_sha256": self.uv_lock_sha256,
            "python_version": self.python_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> EvaluationRunIdentity:
        """Reconstruct an evaluation identity persisted by the run ledger."""

        expected = {
            "evaluation_id",
            "profile_id",
            "profile_config_version",
            "profile_hash",
            "evaluation_contract_version",
            "evaluation_plan_hash",
            "dataset_snapshot_key",
            "evaluation_cutoff",
            "repository_commit_sha",
            "uv_lock_sha256",
            "python_version",
        }
        if set(payload) != expected:
            raise ValueError("evaluation run identity fields are incomplete or unknown")
        cutoff = payload["evaluation_cutoff"]
        parsed_cutoff: datetime | None
        if cutoff is None:
            parsed_cutoff = None
        elif isinstance(cutoff, str):
            try:
                parsed_cutoff = datetime.fromisoformat(cutoff)
            except ValueError as exc:
                raise ValueError("evaluation_cutoff must be an ISO timestamp") from exc
        else:
            raise ValueError("evaluation_cutoff must be null or an ISO timestamp")
        return cls(
            evaluation_id=str(payload["evaluation_id"]),
            profile_id=str(payload["profile_id"]),
            profile_config_version=_integer(
                payload["profile_config_version"], "profile_config_version"
            ),
            profile_hash=str(payload["profile_hash"]),
            evaluation_contract_version=_integer(
                payload["evaluation_contract_version"], "evaluation_contract_version"
            ),
            evaluation_plan_hash=str(payload["evaluation_plan_hash"]),
            dataset_snapshot_key=str(payload["dataset_snapshot_key"]),
            evaluation_cutoff=parsed_cutoff,
            repository_commit_sha=str(payload["repository_commit_sha"]),
            uv_lock_sha256=str(payload["uv_lock_sha256"]),
            python_version=str(payload["python_version"]),
        )

    @property
    def key(self) -> str:
        return content_hash(self.as_dict())


@dataclass(frozen=True, slots=True)
class WorkUnitIdentity:
    """Structural unit identity and deterministic dependency input hash."""

    evaluation_run_key: str
    unit_type: str
    coordinates: tuple[tuple[str, str], ...] = ()
    parent_payload_hashes: tuple[str, ...] = ()
    unit_parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _sha256(self.evaluation_run_key, "evaluation_run_key")
        _text(self.unit_type, "unit_type")
        for collection, field in (
            (self.coordinates, "coordinates"),
            (self.unit_parameters, "unit_parameters"),
        ):
            if tuple(sorted(collection)) != collection:
                raise ValueError(f"{field} must be canonically sorted")
            if any(
                not isinstance(key, str) or not isinstance(value, str) or not key or not value
                for key, value in collection
            ):
                raise ValueError(f"{field} must contain non-empty string pairs")
        for value in self.parent_payload_hashes:
            _sha256(value, "parent_payload_hash")

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluation_run_key": self.evaluation_run_key,
            "unit_type": self.unit_type,
            "coordinates": list(self.coordinates),
            "parent_payload_hashes": list(self.parent_payload_hashes),
            "unit_parameters": list(self.unit_parameters),
        }

    @property
    def key(self) -> str:
        coordinate_path = "/".join(f"{key}={value}" for key, value in self.coordinates)
        return self.unit_type if not coordinate_path else f"{self.unit_type}/{coordinate_path}"

    @property
    def work_unit_input_hash(self) -> str:
        return content_hash(
            {
                "evaluation_run_key": self.evaluation_run_key,
                "unit_type": self.unit_type,
                "unit_coordinates": list(self.coordinates),
                "parent_payload_hashes": list(self.parent_payload_hashes),
                "unit_parameters": list(self.unit_parameters),
            }
        )


__all__ = [
    "DatasetSnapshotIdentity",
    "EvaluationRunIdentity",
    "WorkUnitIdentity",
    "canonical_json",
    "content_hash",
]
