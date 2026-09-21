"""Pure, catalog-bound quality filtering for one Outer-TRAIN snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import fsum, isfinite
from tempfile import TemporaryDirectory

import numpy as np

from market_regime_engine.feature_discovery.contracts import (
    MIN_ELIGIBLE_FEATURES,
    MIN_FEATURE_COVERAGE,
    MIN_FEATURE_VARIANCE,
    DiscoveryStatus,
    FeatureQuality,
    QualityFilterResult,
)
from market_regime_engine.feature_discovery.feature_roles import FeatureRoleContract
from market_regime_engine.feature_discovery.metadata_store import FoldFeatureStat
from market_regime_engine.features.ports import (
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)
from market_regime_engine.runtime.parallel import (
    FoldParallelExecutor,
    ParallelExecutionPlan,
    ReadOnlyMatrix,
)


def _require_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _population_variance(values: tuple[float, ...]) -> float:
    """Return finite population variance without changing or imputing values."""

    if not values:
        return 0.0
    count = len(values)
    mean = fsum(values) / count
    try:
        variance = fsum((value - mean) ** 2 for value in values) / count
    except OverflowError as exc:
        raise ValueError("population variance must be finite") from exc
    if not isfinite(variance):
        raise ValueError("population variance must be finite")
    return variance


@dataclass(frozen=True, slots=True)
class _QualityFeatureTask:
    """Pickle-safe independent quality calculation for one feature."""

    feature_name: str
    source_position: int
    source_count: int
    column_position: int
    matrix_path: str | None = None
    matrix_shape: tuple[int, int] | None = None
    matrix_dtype: str | None = None
    values: tuple[float, ...] | None = None


def _quality_feature(task: _QualityFeatureTask) -> FeatureQuality:
    if task.matrix_path is not None:
        if task.matrix_shape is None or task.matrix_dtype is None:
            raise ValueError("shared quality matrix metadata is incomplete")
        mapped = np.memmap(
            task.matrix_path,
            dtype=np.dtype(task.matrix_dtype),
            mode="r",
            shape=task.matrix_shape,
        )
        values = tuple(
            float(value) for value in mapped[:, task.column_position] if np.isfinite(value)
        )
        del mapped
    elif task.values is not None:
        values = task.values
    else:
        raise ValueError("quality task has no numeric values")
    finite_count = len(values)
    coverage = finite_count / task.source_count
    variance = _population_variance(values)
    if coverage < MIN_FEATURE_COVERAGE:
        eligible = False
        reason = "coverage_below_minimum"
    elif variance <= MIN_FEATURE_VARIANCE:
        eligible = False
        reason = "variance_below_or_equal_minimum"
    else:
        eligible = True
        reason = None
    return FeatureQuality(
        feature_name=task.feature_name,
        source_position=task.source_position,
        train_observation_count=task.source_count,
        finite_observation_count=finite_count,
        coverage=coverage,
        population_variance=variance,
        eligible=eligible,
        rejection_reason=reason,
    )


def _materialize_train_rows(
    snapshot: FeatureSnapshot,
    catalog: FeatureCatalogSnapshot,
    train_start: datetime,
    train_end: datetime,
) -> tuple[FeatureRow, ...]:
    """Copy and validate the exact source rows supplied for Outer TRAIN."""

    if not isinstance(snapshot, FeatureSnapshot):
        raise TypeError("quality filtering requires a FeatureSnapshot")
    if not isinstance(catalog, FeatureCatalogSnapshot):
        raise TypeError("quality filtering requires a FeatureCatalogSnapshot")
    if train_start > train_end:
        raise ValueError("train_start must not be after train_end")
    if snapshot.lineage != catalog.lineage:
        raise ValueError("feature snapshot and catalog lineage must match")
    if snapshot.feature_names != catalog.feature_names:
        raise ValueError("feature snapshot columns must match catalog order")
    if snapshot.skipped_incomplete_row_count:
        raise ValueError("quality filtering requires all source rows; skipped rows are forbidden")

    copied_rows: list[FeatureRow] = []
    previous: datetime | None = None
    expected_dimension = len(catalog.entries)
    for row in snapshot.rows:
        # Copy the value tuple before doing any calculation. A caller cannot
        # change the quality result by mutating an object used to build a row
        # after this boundary has been crossed.
        copied = FeatureRow(row.timestamp, tuple(row.values))
        if not train_start <= copied.timestamp <= train_end:
            raise ValueError("FeatureSnapshot contains a row outside supplied TRAIN bounds")
        if previous is not None and copied.timestamp <= previous:
            raise ValueError("Outer-TRAIN rows must be unique and strictly increasing")
        if len(copied.values) != expected_dimension:
            raise ValueError("Outer-TRAIN row dimension does not match catalog")
        previous = copied.timestamp
        copied_rows.append(copied)
    if not copied_rows:
        raise ValueError("Outer-TRAIN source rows cannot be empty")
    return tuple(copied_rows)


def filter_outer_train_quality(
    catalog: FeatureCatalogSnapshot,
    snapshot: FeatureSnapshot,
    train_start: datetime,
    train_end: datetime,
    max_workers: int | None = None,
) -> QualityFilterResult:
    """Filter every catalog feature using only one exact Outer-TRAIN snapshot.

    The input snapshot is already the source-row universe for the requested
    TRAIN window. Bounds are still checked fail-closed: a row outside the
    supplied inclusive interval is an invocation error rather than a row that
    can silently influence a different fold. None is the only accepted
    missing value and counts against the exact source-row denominator.
    """

    _require_utc(train_start, "train_start")
    _require_utc(train_end, "train_end")
    rows = _materialize_train_rows(snapshot, catalog, train_start, train_end)
    source_count = len(rows)
    matrix = np.full((source_count, len(catalog.entries)), np.nan, dtype=np.float64)

    for row_index, row in enumerate(rows):
        for index, value in enumerate(row.values):
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("non-null feature values must be numeric") from exc
            if not isfinite(numeric):
                # NaN/Inf is a source-contract failure for the whole
                # invocation, never a feature-level rejection.
                raise ValueError("non-null feature values must be finite")
            matrix[row_index, index] = numeric

    with (
        TemporaryDirectory(prefix="regime-quality-") as matrix_directory,
        ReadOnlyMatrix.create(matrix, matrix_directory) as shared,
    ):
        tasks = tuple(
            _QualityFeatureTask(
                feature_name=entry.feature_name,
                source_position=entry.canonical_ordinal,
                source_count=source_count,
                column_position=index,
                matrix_path=str(shared.path),
                matrix_shape=(source_count, len(catalog.entries)),
                matrix_dtype=matrix.dtype.str,
            )
            for index, entry in enumerate(catalog.entries)
        )
        plan = ParallelExecutionPlan.create(
            len(tasks),
            requested_workers=max_workers,
            shared_matrix_identity=shared.identity,
        )
        executor: FoldParallelExecutor[_QualityFeatureTask, FeatureQuality] = FoldParallelExecutor(
            plan
        )
        with executor:
            quality = executor.map_ordered(_quality_feature, tasks)

    features = tuple(quality)
    eligible_features = tuple(item.feature_name for item in features if item.eligible)
    if len(eligible_features) < MIN_ELIGIBLE_FEATURES:
        raise ValueError(
            f"quality filter produced {len(eligible_features)} eligible features; "
            f"requires at least {MIN_ELIGIBLE_FEATURES}"
        )
    return QualityFilterResult(
        source_build_id=catalog.lineage.source_build_id,
        train_source_observation_count=source_count,
        catalog_hash=catalog.catalog_hash,
        features=features,
        eligible_features=eligible_features,
        status=DiscoveryStatus.VALID,
    )


# The shorter name is useful at orchestration call sites while retaining one
# implementation and one contract.
filter_feature_quality = filter_outer_train_quality


def quality_to_fold_feature_stats(
    result: QualityFilterResult,
    *,
    fold_id: str,
    profile_hash: str,
    role_contract: FeatureRoleContract,
) -> tuple[FoldFeatureStat, ...]:
    """Convert immutable TRAIN quality evidence to metadata-store rows.

    This adapter carries no values or vectors. It only records the quality
    decision and the role boundary for each catalog feature, leaving later PCA,
    SFFS, and HMM participation fields unset until their own stages run.
    """

    if not fold_id or fold_id.strip() != fold_id:
        raise ValueError("fold_id must be a non-empty trimmed string")
    if len(profile_hash) != 64 or any(char not in "0123456789abcdef" for char in profile_hash):
        raise ValueError("profile_hash must be a lowercase SHA-256")
    rows: list[FoldFeatureStat] = []
    for item in result.features:
        assignment = role_contract.assignment(item.feature_name)
        rows.append(
            FoldFeatureStat(
                fold_id=fold_id,
                profile_hash=profile_hash,
                source_build_id=result.source_build_id,
                feature_name=item.feature_name,
                eligible=item.eligible,
                quality_reason=item.rejection_reason,
                direct_participation=assignment.direct_hmm_candidate,
                pc_participation=assignment.family_pca_input,
                pca_credit=None,
                representative=False,
                sffs_participation=False,
                final_selection=False,
                ablation_loss=None,
            )
        )
    return tuple(rows)


__all__ = [
    "filter_feature_quality",
    "filter_outer_train_quality",
    "quality_to_fold_feature_stats",
]
