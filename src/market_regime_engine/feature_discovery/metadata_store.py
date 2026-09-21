"""Transactional local DuckDB metadata store for feature selection."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import duckdb

_TABLES = (
    "feature_registry",
    "fold_feature_stats",
    "pca_loadings",
    "correlation_mapping",
    "sffs_steps",
    "fold_model_stats",
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def _hash(value: object) -> str:
    return sha256(_json(value).encode("utf-8")).hexdigest()


def _text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")


def _sha256(value: str, field: str) -> None:
    _text(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class FeatureRegistryRow:
    feature_identity: str
    source_dataset: str
    source_build_id: str
    source_catalog_hash: str
    feature_name: str
    role: str
    family: str | None
    transformation_name: str | None
    transformation_parameters: Mapping[str, object]
    first_seen_utc: datetime
    lifecycle_status: str

    def __post_init__(self) -> None:
        _text(self.feature_identity, "feature_identity")
        _text(self.source_dataset, "source_dataset")
        _text(self.source_build_id, "source_build_id")
        _sha256(self.source_catalog_hash, "source_catalog_hash")
        _text(self.feature_name, "feature_name")
        _text(self.role, "role")
        if self.family is not None:
            _text(self.family, "family")
        if self.transformation_name is not None:
            _text(self.transformation_name, "transformation_name")
        if self.first_seen_utc.tzinfo is None or self.first_seen_utc.utcoffset() != UTC.utcoffset(
            self.first_seen_utc
        ):
            raise ValueError("first_seen_utc must be timezone-aware UTC")
        _text(self.lifecycle_status, "lifecycle_status")


@dataclass(frozen=True, slots=True)
class FoldFeatureStat:
    fold_id: str
    profile_hash: str
    source_build_id: str
    feature_name: str
    eligible: bool
    quality_reason: str | None
    direct_participation: bool
    pc_participation: bool
    pca_credit: float | None
    representative: bool
    sffs_participation: bool
    final_selection: bool
    ablation_loss: float | None


@dataclass(frozen=True, slots=True)
class PcaLoading:
    fold_id: str
    profile_hash: str
    source_build_id: str
    family: str
    pc_ordinal: int
    source_feature: str
    loading: float
    squared_loading: float
    explained_variance: float


@dataclass(frozen=True, slots=True)
class CorrelationMapping:
    fold_id: str
    profile_hash: str
    source_build_id: str
    candidate: str
    chosen_representative: str
    full_abs_correlation: float | None
    subwindow_1_abs_correlation: float | None
    subwindow_2_abs_correlation: float | None
    subwindow_3_abs_correlation: float | None
    full_support_count: int
    subwindow_1_support_count: int
    subwindow_2_support_count: int
    subwindow_3_support_count: int
    retained: bool
    reason: str


@dataclass(frozen=True, slots=True)
class SFFSStepRecord:
    fold_id: str
    profile_hash: str
    source_build_id: str
    state_count: int
    step_number: int
    action: str
    candidate: str
    selected_tuple_hash: str
    forecast_score: float | None
    calibration_score: float | None
    stability_score: float | None
    robustness_score: float | None
    total_score: float | None
    eligible: bool
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class FoldModelStat:
    fold_id: str
    profile_hash: str
    source_build_id: str
    selected_tuple_hash: str
    state_count: int
    model_family: str
    valid: bool
    diagnostics: Mapping[str, object]
    mlflow_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class FoldMetadataBundle:
    fold_id: str
    profile_hash: str
    source_build_id: str
    feature_registry: tuple[FeatureRegistryRow, ...] = ()
    fold_feature_stats: tuple[FoldFeatureStat, ...] = ()
    pca_loadings: tuple[PcaLoading, ...] = ()
    correlation_mapping: tuple[CorrelationMapping, ...] = ()
    sffs_steps: tuple[SFFSStepRecord, ...] = ()
    fold_model_stats: tuple[FoldModelStat, ...] = ()

    def __post_init__(self) -> None:
        _text(self.fold_id, "fold_id")
        _sha256(self.profile_hash, "profile_hash")
        _text(self.source_build_id, "source_build_id")
        if not self.fold_model_stats:
            raise ValueError("a fold metadata bundle requires model statistics")


class FeatureSelectionMetadataStore:
    """One local DuckDB database containing only feature-selection metadata."""

    def __init__(self, state_root: str | Path) -> None:
        self._root = Path(state_root)
        self._root.mkdir(mode=0o750, parents=True, exist_ok=True)
        self._database = self._root / "feature_selection.duckdb"
        self._lock = threading.RLock()
        with self._connect() as connection:
            self._initialize(connection)

    @property
    def database(self) -> Path:
        return self._database

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._database))

    @staticmethod
    def _initialize(connection: duckdb.DuckDBPyConnection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_registry (
                feature_identity VARCHAR PRIMARY KEY,
                source_dataset VARCHAR NOT NULL,
                source_build_id VARCHAR NOT NULL,
                source_catalog_hash VARCHAR NOT NULL,
                feature_name VARCHAR NOT NULL,
                role VARCHAR NOT NULL,
                family VARCHAR,
                transformation_name VARCHAR,
                transformation_parameters_json VARCHAR NOT NULL,
                first_seen_utc TIMESTAMPTZ NOT NULL,
                lifecycle_status VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fold_feature_stats (
                fold_id VARCHAR NOT NULL,
                profile_hash VARCHAR NOT NULL,
                source_build_id VARCHAR NOT NULL,
                feature_name VARCHAR NOT NULL,
                eligible BOOLEAN NOT NULL,
                quality_reason VARCHAR,
                direct_participation BOOLEAN NOT NULL,
                pc_participation BOOLEAN NOT NULL,
                pca_credit DOUBLE,
                representative BOOLEAN NOT NULL,
                sffs_participation BOOLEAN NOT NULL,
                final_selection BOOLEAN NOT NULL,
                ablation_loss DOUBLE,
                PRIMARY KEY (fold_id, profile_hash, source_build_id, feature_name)
            );
            CREATE TABLE IF NOT EXISTS pca_loadings (
                fold_id VARCHAR NOT NULL,
                profile_hash VARCHAR NOT NULL,
                source_build_id VARCHAR NOT NULL,
                family VARCHAR NOT NULL,
                pc_ordinal INTEGER NOT NULL,
                source_feature VARCHAR NOT NULL,
                loading DOUBLE NOT NULL,
                squared_loading DOUBLE NOT NULL,
                explained_variance DOUBLE NOT NULL,
                PRIMARY KEY (
                    fold_id, profile_hash, source_build_id, family, pc_ordinal, source_feature
                )
            );
            CREATE TABLE IF NOT EXISTS correlation_mapping (
                fold_id VARCHAR NOT NULL,
                profile_hash VARCHAR NOT NULL,
                source_build_id VARCHAR NOT NULL,
                candidate VARCHAR NOT NULL,
                chosen_representative VARCHAR NOT NULL,
                full_abs_correlation DOUBLE,
                subwindow_1_abs_correlation DOUBLE,
                subwindow_2_abs_correlation DOUBLE,
                subwindow_3_abs_correlation DOUBLE,
                full_support_count BIGINT NOT NULL,
                subwindow_1_support_count BIGINT NOT NULL,
                subwindow_2_support_count BIGINT NOT NULL,
                subwindow_3_support_count BIGINT NOT NULL,
                retained BOOLEAN NOT NULL,
                reason VARCHAR NOT NULL,
                PRIMARY KEY (fold_id, profile_hash, source_build_id, candidate)
            );
            CREATE TABLE IF NOT EXISTS sffs_steps (
                fold_id VARCHAR NOT NULL,
                profile_hash VARCHAR NOT NULL,
                source_build_id VARCHAR NOT NULL,
                state_count INTEGER NOT NULL,
                step_number INTEGER NOT NULL,
                action VARCHAR NOT NULL,
                candidate VARCHAR NOT NULL,
                selected_tuple_hash VARCHAR NOT NULL,
                forecast_score DOUBLE,
                calibration_score DOUBLE,
                stability_score DOUBLE,
                robustness_score DOUBLE,
                total_score DOUBLE,
                eligible BOOLEAN NOT NULL,
                rejection_reason VARCHAR,
                PRIMARY KEY (
                    fold_id,
                    profile_hash,
                    source_build_id,
                    state_count,
                    step_number,
                    action,
                    candidate
                )
            );
            CREATE TABLE IF NOT EXISTS fold_model_stats (
                fold_id VARCHAR NOT NULL,
                profile_hash VARCHAR NOT NULL,
                source_build_id VARCHAR NOT NULL,
                selected_tuple_hash VARCHAR NOT NULL,
                state_count INTEGER NOT NULL,
                model_family VARCHAR NOT NULL,
                valid BOOLEAN NOT NULL,
                diagnostics_json VARCHAR NOT NULL,
                mlflow_run_id VARCHAR,
                bundle_hash VARCHAR NOT NULL,
                PRIMARY KEY (fold_id, profile_hash, source_build_id, state_count, model_family)
            );
            CREATE OR REPLACE VIEW feature_global_stats AS
            SELECT
                feature_name,
                count(*) AS fold_count,
                count(*) FILTER (WHERE eligible) AS eligible_fold_count,
                count(*) FILTER (WHERE final_selection) AS final_selection_count,
                count(*) FILTER (WHERE representative) AS representative_fold_count,
                avg(ablation_loss) FILTER (WHERE ablation_loss IS NOT NULL) AS mean_ablation_loss
            FROM fold_feature_stats
            GROUP BY feature_name;
            """
        )

    @staticmethod
    def _bundle_payload(bundle: FoldMetadataBundle) -> dict[str, object]:
        return asdict(bundle)

    def commit_fold(self, bundle: FoldMetadataBundle) -> bool:
        """Atomically commit one fold, returning ``False`` for an identical replay."""

        if any(
            row.fold_id != bundle.fold_id
            or row.profile_hash != bundle.profile_hash
            or row.source_build_id != bundle.source_build_id
            for rows in (
                bundle.fold_feature_stats,
                bundle.pca_loadings,
                bundle.correlation_mapping,
                bundle.sffs_steps,
                bundle.fold_model_stats,
            )
            for row in rows
        ):
            raise ValueError("all fold metadata rows must share the bundle identity")
        bundle_hash = _hash(self._bundle_payload(bundle))
        with self._lock, self._connect() as connection:
            try:
                connection.execute("BEGIN TRANSACTION")
                existing = connection.execute(
                    """
                    SELECT DISTINCT bundle_hash FROM fold_model_stats
                    WHERE fold_id = ? AND profile_hash = ? AND source_build_id = ?
                    """,
                    [bundle.fold_id, bundle.profile_hash, bundle.source_build_id],
                ).fetchall()
                if existing:
                    if {row[0] for row in existing} != {bundle_hash}:
                        raise ValueError("conflicting immutable fold metadata identity")
                    connection.execute("COMMIT")
                    return False
                for row in bundle.feature_registry:
                    connection.execute(
                        """
                        INSERT INTO feature_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (feature_identity) DO NOTHING
                        """,
                        [
                            row.feature_identity,
                            row.source_dataset,
                            row.source_build_id,
                            row.source_catalog_hash,
                            row.feature_name,
                            row.role,
                            row.family,
                            row.transformation_name,
                            _json(row.transformation_parameters),
                            row.first_seen_utc,
                            row.lifecycle_status,
                        ],
                    )
                self._insert_fold_rows(connection, bundle, bundle_hash)
                connection.execute("COMMIT")
                return True
            except Exception:
                connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _insert_fold_rows(
        connection: duckdb.DuckDBPyConnection,
        bundle: FoldMetadataBundle,
        bundle_hash: str,
    ) -> None:
        for feature_stat in bundle.fold_feature_stats:
            connection.execute(
                """
                INSERT INTO fold_feature_stats
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                list(asdict(feature_stat).values()),
            )
        for loading in bundle.pca_loadings:
            connection.execute(
                """
                INSERT INTO pca_loadings
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                list(asdict(loading).values()),
            )
        for mapping in bundle.correlation_mapping:
            connection.execute(
                """
                INSERT INTO correlation_mapping
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                list(asdict(mapping).values()),
            )
        for sffs_step in bundle.sffs_steps:
            connection.execute(
                """
                INSERT INTO sffs_steps
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                list(asdict(sffs_step).values()),
            )
        for model_stat in bundle.fold_model_stats:
            connection.execute(
                """
                INSERT INTO fold_model_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                [
                    model_stat.fold_id,
                    model_stat.profile_hash,
                    model_stat.source_build_id,
                    model_stat.selected_tuple_hash,
                    model_stat.state_count,
                    model_stat.model_family,
                    model_stat.valid,
                    _json(model_stat.diagnostics),
                    model_stat.mlflow_run_id,
                    bundle_hash,
                ],
            )

    def table_names(self) -> tuple[str, ...]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ).fetchall()
        return tuple(row[0] for row in rows)

    def view_names(self) -> tuple[str, ...]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND table_type = 'VIEW'
                ORDER BY table_name
                """
            ).fetchall()
        return tuple(row[0] for row in rows)

    def close(self) -> None:
        """The store opens short-lived connections per operation; kept for API symmetry."""


__all__ = [
    "CorrelationMapping",
    "FeatureRegistryRow",
    "FeatureSelectionMetadataStore",
    "FoldFeatureStat",
    "FoldMetadataBundle",
    "FoldModelStat",
    "PcaLoading",
    "SFFSStepRecord",
]
