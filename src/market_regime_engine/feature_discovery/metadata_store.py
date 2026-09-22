"""Transactional local DuckDB metadata store for feature selection."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path

import duckdb

from market_regime_engine.feature_discovery.ablation import AblationResult
from market_regime_engine.feature_discovery.feature_roles import family_pc_name
from market_regime_engine.feature_discovery.sffs import SFFSResult

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


def apply_ablation_to_feature_stats(
    rows: tuple[FoldFeatureStat, ...], result: AblationResult
) -> tuple[FoldFeatureStat, ...]:
    """Attach unclipped marginal losses to the committed fold feature rows."""

    by_feature = {item.removed_feature: item for item in result.one_feature_results}
    if set(by_feature) != set(result.selected_features):
        raise ValueError("ablation rows must cover every selected feature exactly once")
    row_names = {row.feature_name for row in rows}
    if not set(result.selected_features) <= row_names:
        raise ValueError("feature-stat rows are missing an ablated selected feature")
    output: list[FoldFeatureStat] = []
    for row in rows:
        observation = by_feature.get(row.feature_name)
        if observation is None:
            output.append(row)
            continue
        output.append(
            replace(
                row,
                final_selection=True,
                ablation_loss=observation.ablation_loss,
            )
        )
    return tuple(output)


def apply_pca_credit_to_feature_stats(
    rows: tuple[FoldFeatureStat, ...],
    loadings: tuple[PcaLoading, ...],
    result: AblationResult,
) -> tuple[FoldFeatureStat, ...]:
    """Attribute selected PCA-component ablation loss to source features.

    Credits are intentionally based on the absolute component ablation loss and
    squared loading.  Direct features keep their own ablation loss and never
    receive synthetic PCA credit.
    """

    identities = {(row.fold_id, row.profile_hash, row.source_build_id) for row in rows} | {
        (item.fold_id, item.profile_hash, item.source_build_id) for item in loadings
    }
    if len(identities) > 1:
        raise ValueError("feature stats and PCA loadings must share one fold identity")
    losses = {
        observation.removed_feature: observation.ablation_loss
        for observation in result.one_feature_results
        if observation.ablation_loss is not None
    }
    credits: dict[str, float] = {}
    for loading in loadings:
        if not isfinite(loading.squared_loading) or loading.squared_loading < 0.0:
            raise ValueError("PCA squared loading must be finite and non-negative")
        component = family_pc_name(loading.family, loading.pc_ordinal)
        loss = losses.get(component)
        if loss is not None:
            credits[loading.source_feature] = credits.get(loading.source_feature, 0.0) + (
                abs(loss) * loading.squared_loading
            )
    return tuple(
        replace(
            row,
            pca_credit=(credits.get(row.feature_name, 0.0) if row.pc_participation else None),
        )
        for row in rows
    )


def sffs_step_records(
    result: SFFSResult,
    *,
    fold_id: str,
    profile_hash: str,
    source_build_id: str,
    state_count: int,
) -> tuple[SFFSStepRecord, ...]:
    """Convert every SFFS candidate evaluation into immutable store rows."""

    _text(fold_id, "fold_id")
    _sha256(profile_hash, "profile_hash")
    _text(source_build_id, "source_build_id")
    if state_count not in (2, 3, 4, 5):
        raise ValueError("state_count must be 2, 3, 4, or 5")
    rows: list[SFFSStepRecord] = []
    for step_number, evaluation in enumerate(result.evaluations, start=1):
        score = evaluation.score
        candidate = "|".join(evaluation.candidate)
        selected_hash = _hash(evaluation.candidate)
        rows.append(
            SFFSStepRecord(
                fold_id,
                profile_hash,
                source_build_id,
                state_count,
                step_number,
                evaluation.action,
                candidate,
                selected_hash,
                None if score is None else score.forecast_score,
                None if score is None else score.calibration_score,
                None if score is None else score.stability_score,
                None if score is None else score.robustness_score,
                None if score is None else score.value,
                score is not None,
                None if score is not None else "ineligible candidate",
            )
        )
    return tuple(rows)


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
            WITH ordered AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY feature_name ORDER BY fold_id DESC
                    ) AS reverse_fold_number
                FROM fold_feature_stats
            )
            SELECT
                feature_name,
                count(*) AS fold_count,
                count(*) FILTER (WHERE eligible) AS eligible_fold_count,
                count(*) FILTER (WHERE eligible) AS eligible_folds,
                count(*) FILTER (WHERE eligible) AS quality_pass_folds,
                count(*) FILTER (WHERE final_selection) AS final_selection_count,
                count(*) FILTER (WHERE final_selection) AS selected_folds,
                count(*) FILTER (WHERE representative) AS representative_fold_count,
                count(*) FILTER (WHERE representative) AS representative_folds,
                count(*) FILTER (WHERE direct_participation AND final_selection)
                    AS direct_selection_count,
                CASE
                    WHEN count(*) FILTER (WHERE eligible) = 0 THEN NULL
                    ELSE CAST(count(*) FILTER (WHERE final_selection) AS DOUBLE)
                         / count(*) FILTER (WHERE eligible)
                END AS selection_rate,
                avg(ablation_loss) FILTER (WHERE ablation_loss IS NOT NULL)
                    AS mean_ablation_loss,
                median(ablation_loss) FILTER (WHERE ablation_loss IS NOT NULL)
                    AS median_ablation_loss,
                sum(pca_credit) FILTER (WHERE pca_credit IS NOT NULL) AS total_pca_credit,
                avg(pca_credit) FILTER (WHERE pca_credit IS NOT NULL) AS mean_pca_credit,
                max(fold_id) FILTER (WHERE final_selection) AS last_selected_fold,
                coalesce(
                    min(reverse_fold_number) FILTER (WHERE final_selection) - 1,
                    count(*)
                ) AS consecutive_unused_folds
            FROM ordered
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
                    existing_feature = connection.execute(
                        """
                        SELECT source_dataset, source_build_id, source_catalog_hash,
                               feature_name, role, family, transformation_name,
                               transformation_parameters_json, first_seen_utc, lifecycle_status
                        FROM feature_registry
                        WHERE feature_identity = ?
                        """,
                        [row.feature_identity],
                    ).fetchone()
                    expected_feature = (
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
                    )
                    if existing_feature is not None and existing_feature != expected_feature:
                        raise ValueError("conflicting immutable feature identity")
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

    def commit_sffs_steps(self, rows: tuple[SFFSStepRecord, ...]) -> bool:
        """Atomically persist SFFS evaluations without requiring model rows yet."""

        if not rows:
            return False
        identities = {
            (row.fold_id, row.profile_hash, row.source_build_id, row.state_count) for row in rows
        }
        if len(identities) != 1:
            raise ValueError("SFFS step rows must share one fold identity")
        with self._lock, self._connect() as connection:
            try:
                connection.execute("BEGIN TRANSACTION")
                for row in rows:
                    values = list(asdict(row).values())
                    key = values[:7]
                    existing = connection.execute(
                        """
                        SELECT fold_id, profile_hash, source_build_id, state_count,
                               step_number, action, candidate, selected_tuple_hash,
                               forecast_score, calibration_score, stability_score,
                               robustness_score, total_score, eligible, rejection_reason
                        FROM sffs_steps
                        WHERE fold_id = ? AND profile_hash = ? AND source_build_id = ?
                          AND state_count = ? AND step_number = ? AND action = ?
                          AND candidate = ?
                        """,
                        key,
                    ).fetchone()
                    if existing is not None and tuple(existing) != tuple(values):
                        raise ValueError("conflicting immutable SFFS step identity")
                    connection.execute(
                        """
                        INSERT INTO sffs_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT DO NOTHING
                        """,
                        values,
                    )
                connection.execute("COMMIT")
                return True
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def commit_fold_feature_stats(self, rows: tuple[FoldFeatureStat, ...]) -> bool:
        """Persist feature participation and unclipped ablation losses."""

        if not rows:
            return False
        identities = {(row.fold_id, row.profile_hash, row.source_build_id) for row in rows}
        if len(identities) != 1:
            raise ValueError("feature-stat rows must share one fold identity")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                for row in rows:
                    connection.execute(
                        """
                        INSERT INTO fold_feature_stats
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (fold_id, profile_hash, source_build_id, feature_name)
                        DO UPDATE SET
                            eligible = excluded.eligible,
                            quality_reason = excluded.quality_reason,
                            direct_participation = excluded.direct_participation,
                            pc_participation = excluded.pc_participation,
                            pca_credit = excluded.pca_credit,
                            representative = excluded.representative,
                            sffs_participation = excluded.sffs_participation,
                            final_selection = excluded.final_selection,
                            ablation_loss = excluded.ablation_loss
                        """,
                        list(asdict(row).values()),
                    )
                connection.execute("COMMIT")
                return True
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def commit_pca_loadings(self, rows: tuple[PcaLoading, ...]) -> bool:
        """Persist immutable PCA loading evidence for one fold."""

        if not rows:
            return False
        identities = {(row.fold_id, row.profile_hash, row.source_build_id) for row in rows}
        if len(identities) != 1:
            raise ValueError("PCA loading rows must share one fold identity")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                for row in rows:
                    values = list(asdict(row).values())
                    key = values[:6]
                    existing = connection.execute(
                        """
                        SELECT fold_id, profile_hash, source_build_id, family,
                               pc_ordinal, source_feature, loading, squared_loading,
                               explained_variance
                        FROM pca_loadings
                        WHERE fold_id = ? AND profile_hash = ? AND source_build_id = ?
                          AND family = ? AND pc_ordinal = ? AND source_feature = ?
                        """,
                        key,
                    ).fetchone()
                    if existing is not None and tuple(existing) != tuple(values):
                        raise ValueError("conflicting immutable PCA loading identity")
                    connection.execute(
                        """
                        INSERT INTO pca_loadings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT DO NOTHING
                        """,
                        values,
                    )
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

    def lifecycle_report(
        self,
        *,
        feature_selection_profile_hash: str,
        policy: object | None = None,
    ) -> object:
        """Build a recommendation report from this store's committed rows only."""

        from market_regime_engine.feature_discovery.lifecycle_recommendations import (
            FeatureLifecyclePolicy,
            build_lifecycle_report,
        )

        resolved_policy = FeatureLifecyclePolicy() if policy is None else policy
        if not isinstance(resolved_policy, FeatureLifecyclePolicy):
            raise TypeError("policy must be a FeatureLifecyclePolicy")
        with self._lock, self._connect() as connection:
            registry_values = connection.execute(
                """
                SELECT feature_identity, source_dataset, source_build_id,
                       source_catalog_hash, feature_name, role, family,
                       transformation_name, transformation_parameters_json,
                       first_seen_utc, lifecycle_status
                FROM feature_registry ORDER BY feature_name
                """
            ).fetchall()
            stat_values = connection.execute(
                """
                SELECT fold_id, profile_hash, source_build_id, feature_name,
                       eligible, quality_reason, direct_participation,
                       pc_participation, pca_credit, representative,
                       sffs_participation, final_selection, ablation_loss
                FROM fold_feature_stats ORDER BY fold_id, feature_name
                """
            ).fetchall()
        registry_rows = tuple(
            FeatureRegistryRow(
                identity,
                dataset,
                build,
                catalog_hash,
                name,
                role,
                family,
                transformation,
                json.loads(parameters),
                first_seen.astimezone(UTC),
                status,
            )
            for (
                identity,
                dataset,
                build,
                catalog_hash,
                name,
                role,
                family,
                transformation,
                parameters,
                first_seen,
                status,
            ) in registry_values
        )
        stat_rows = tuple(FoldFeatureStat(*values) for values in stat_values)
        return build_lifecycle_report(
            registry_rows,
            stat_rows,
            feature_selection_profile_hash=feature_selection_profile_hash,
            policy=resolved_policy,
        )

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
    "apply_ablation_to_feature_stats",
    "apply_pca_credit_to_feature_stats",
    "sffs_step_records",
]
