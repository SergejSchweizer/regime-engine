"""Pure, catalog-bound quality filtering for one Outer-TRAIN snapshot."""

from __future__ import annotations

from datetime import UTC, datetime
from math import fsum, isfinite

from market_regime_engine.feature_discovery.contracts import (
    MIN_ELIGIBLE_FEATURES,
    MIN_FEATURE_COVERAGE,
    MIN_FEATURE_VARIANCE,
    DiscoveryStatus,
    FeatureQuality,
    QualityFilterResult,
)
from market_regime_engine.features.ports import (
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
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
    values_by_feature: list[list[float]] = [[] for _ in catalog.entries]
    finite_counts = [0] * len(catalog.entries)

    for row in rows:
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
            values_by_feature[index].append(numeric)
            finite_counts[index] += 1

    quality: list[FeatureQuality] = []
    for entry, values, finite_count in zip(
        catalog.entries, values_by_feature, finite_counts, strict=True
    ):
        coverage = finite_count / source_count
        variance = _population_variance(tuple(values))
        if coverage < MIN_FEATURE_COVERAGE:
            eligible = False
            reason = "coverage_below_minimum"
        elif variance <= MIN_FEATURE_VARIANCE:
            eligible = False
            reason = "variance_below_or_equal_minimum"
        else:
            eligible = True
            reason = None
        quality.append(
            FeatureQuality(
                feature_name=entry.feature_name,
                source_position=entry.canonical_ordinal,
                train_observation_count=source_count,
                finite_observation_count=finite_count,
                coverage=coverage,
                population_variance=variance,
                eligible=eligible,
                rejection_reason=reason,
            )
        )

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


__all__ = ["filter_feature_quality", "filter_outer_train_quality"]
