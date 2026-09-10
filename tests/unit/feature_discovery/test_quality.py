from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import sqrt

import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.feature_discovery.quality import (
    _population_variance,
    filter_outer_train_quality,
)
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _lineage(build: str = "build-1") -> SourceLineage:
    return SourceLineage(
        source_dataset="xetra_gold",
        source_build_id=build,
        data_sha256="a" * 64,
        schema_version=2,
        feature_version=4,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=BASE,
        row_count=100,
        min_timestamp=BASE,
        max_timestamp=BASE + timedelta(days=100),
    )


def _catalog(
    names: tuple[str, ...], lineage: SourceLineage | None = None
) -> FeatureCatalogSnapshot:
    bound_lineage = lineage or _lineage()
    entries = tuple(FeatureCatalogEntry(name, index + 1) for index, name in enumerate(names))
    return FeatureCatalogSnapshot.from_entries(bound_lineage, "timestamp_m1", reversed(entries))


def _snapshot(
    catalog: FeatureCatalogSnapshot,
    values: tuple[tuple[float | None, ...], ...],
    *,
    start: datetime = BASE,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        lineage=catalog.lineage,
        feature_names=catalog.feature_names,
        rows=tuple(
            FeatureRow(start + timedelta(days=index), row_values)
            for index, row_values in enumerate(values)
        ),
    )


def test_quality_uses_exact_source_denominator_and_catalog_order() -> None:
    catalog = _catalog(("f0", "f1", "f2", "low_coverage", "constant"))
    values = tuple(
        (
            float(index),
            float(index) if index < 9 else None,
            float(2 * index),
            float(index) if index < 8 else None,
            1.0,
        )
        for index in range(10)
    )
    result = filter_outer_train_quality(
        catalog, _snapshot(catalog, values), BASE, BASE + timedelta(days=9)
    )

    assert result.train_source_observation_count == 10
    assert result.eligible_features == ("f0", "f1", "f2")
    assert tuple(item.feature_name for item in result.features) == catalog.feature_names
    assert result.features[1].finite_observation_count == 9
    assert result.features[1].coverage == pytest.approx(0.9, abs=0.0)
    assert result.features[1].rejection_reason is None
    assert result.features[3].rejection_reason == "coverage_below_minimum"
    assert result.features[4].rejection_reason == "variance_below_or_equal_minimum"
    expected_variance = sum(index * index for index in range(10)) / 10 - 4.5**2
    assert result.features[0].population_variance == pytest.approx(expected_variance)


def test_variance_is_population_ddof_zero_and_strictly_above_threshold() -> None:
    names = ("f0", "f1", "f2", "near_zero", "above_zero")
    catalog = _catalog(names)
    below = sqrt(0.5e-12)
    above = sqrt(2.0e-12)
    values = tuple(
        (
            float(index),
            float(index + 1),
            float(2 * index + 1),
            below if index % 2 == 0 else -below,
            above if index % 2 == 0 else -above,
        )
        for index in range(10)
    )
    result = filter_outer_train_quality(
        catalog, _snapshot(catalog, values), BASE, BASE + timedelta(days=9)
    )

    assert result.features[3].population_variance == pytest.approx(0.5e-12)
    assert result.features[3].rejection_reason == "variance_below_or_equal_minimum"
    assert result.features[4].population_variance == pytest.approx(2.0e-12)
    assert result.features[4].eligible is True


def test_nonfinite_value_fails_the_invocation() -> None:
    catalog = _catalog(("f0", "f1", "f2"))
    values = tuple((float(index), float(index + 1), float(index + 2)) for index in range(10))
    broken = list(values)
    broken[4] = (float("nan"), 5.0, 6.0)

    with pytest.raises(ValueError, match="must be finite"):
        filter_outer_train_quality(
            catalog,
            _snapshot(catalog, tuple(broken)),
            BASE,
            BASE + timedelta(days=9),
        )


def test_rows_outside_train_bounds_are_rejected_and_result_is_stable() -> None:
    catalog = _catalog(("f0", "f1", "f2"))
    values = tuple((float(index), float(index + 1), float(index + 2)) for index in range(10))
    snapshot = _snapshot(catalog, values)

    with pytest.raises(ValueError, match="outside supplied TRAIN bounds"):
        filter_outer_train_quality(
            catalog, snapshot, BASE + timedelta(days=1), BASE + timedelta(days=9)
        )

    result = filter_outer_train_quality(catalog, snapshot, BASE, BASE + timedelta(days=9))
    # Reusing the source object cannot alter the copied, deterministic result.
    assert (
        result.result_hash
        == filter_outer_train_quality(catalog, snapshot, BASE, BASE + timedelta(days=9)).result_hash
    )


def test_lineage_mismatch_and_too_few_eligible_features_fail_closed() -> None:
    catalog = _catalog(("f0", "f1", "f2"))
    values = tuple((float(index), float(index + 1), float(index + 2)) for index in range(10))
    foreign = _snapshot(_catalog(catalog.feature_names, _lineage("build-2")), values)
    with pytest.raises(ValueError, match="lineage"):
        filter_outer_train_quality(catalog, foreign, BASE, BASE + timedelta(days=9))

    too_few_catalog = _catalog(("f0", "f1", "f2"))
    too_few_values = tuple((float(index), 1.0, None) for index in range(10))
    with pytest.raises(ValueError, match="at least 3"):
        filter_outer_train_quality(
            too_few_catalog,
            _snapshot(too_few_catalog, too_few_values),
            BASE,
            BASE + timedelta(days=9),
        )


def test_input_contract_rejects_invalid_bounds_types_columns_and_skipped_rows() -> None:
    catalog = _catalog(("f0", "f1", "f2"))
    values = tuple((float(index), float(index + 1), float(index + 2)) for index in range(10))
    snapshot = _snapshot(catalog, values)

    with pytest.raises(ValueError, match="timezone-aware"):
        filter_outer_train_quality(catalog, snapshot, datetime(2026, 1, 1), BASE)
    with pytest.raises(ValueError, match="timezone-aware"):
        filter_outer_train_quality(catalog, snapshot, BASE, datetime(2026, 1, 2))
    with pytest.raises(ValueError, match="must not be after"):
        filter_outer_train_quality(catalog, snapshot, BASE + timedelta(days=2), BASE)
    with pytest.raises(TypeError, match="FeatureSnapshot"):
        filter_outer_train_quality(catalog, object(), BASE, BASE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FeatureCatalogSnapshot"):
        filter_outer_train_quality(object(), snapshot, BASE, BASE)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="columns"):
        filter_outer_train_quality(
            _catalog(("f0", "f1", "other")),
            snapshot,
            BASE,
            BASE + timedelta(days=9),
        )
    with pytest.raises(ValueError, match="skipped"):
        filter_outer_train_quality(
            catalog,
            replace(snapshot, skipped_incomplete_row_count=1),
            BASE,
            BASE + timedelta(days=9),
        )


def test_input_contract_rejects_empty_unsorted_and_malformed_rows() -> None:
    catalog = _catalog(("f0", "f1", "f2"))
    values = tuple((float(index), float(index + 1), float(index + 2)) for index in range(10))
    snapshot = _snapshot(catalog, values)

    with pytest.raises(ValueError, match="cannot be empty"):
        filter_outer_train_quality(
            catalog,
            replace(snapshot, rows=(), materialized_feature_data_sha256=None),
            BASE,
            BASE,
        )
    unsorted_snapshot = object.__new__(FeatureSnapshot)
    object.__setattr__(unsorted_snapshot, "lineage", catalog.lineage)
    object.__setattr__(unsorted_snapshot, "feature_names", catalog.feature_names)
    object.__setattr__(
        unsorted_snapshot,
        "rows",
        (snapshot.rows[1], snapshot.rows[0], *snapshot.rows[2:]),
    )
    object.__setattr__(unsorted_snapshot, "skipped_incomplete_row_count", 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        filter_outer_train_quality(catalog, unsorted_snapshot, BASE, BASE + timedelta(days=9))
    malformed = FeatureRow(BASE, (1.0, 2.0))
    broken_snapshot = object.__new__(FeatureSnapshot)
    object.__setattr__(broken_snapshot, "lineage", catalog.lineage)
    object.__setattr__(broken_snapshot, "feature_names", catalog.feature_names)
    object.__setattr__(broken_snapshot, "rows", (malformed,))
    object.__setattr__(broken_snapshot, "skipped_incomplete_row_count", 0)
    with pytest.raises(ValueError, match="dimension"):
        filter_outer_train_quality(catalog, broken_snapshot, BASE, BASE)  # type: ignore[arg-type]


def test_non_numeric_values_and_overflowing_variance_fail_closed() -> None:
    catalog = _catalog(("f0", "f1", "f2"))
    values = tuple((float(index), float(index + 1), float(index + 2)) for index in range(10))
    broken = list(values)
    broken[2] = (object(), 3.0, 4.0)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="must be numeric"):
        filter_outer_train_quality(
            catalog,
            _snapshot(catalog, tuple(broken)),  # type: ignore[arg-type]
            BASE,
            BASE + timedelta(days=9),
        )
    with pytest.raises(ValueError, match="population variance must be finite"):
        _population_variance((1.0e308, -1.0e308))


def test_fifty_feature_mixed_fixture_preserves_every_catalog_feature() -> None:
    names = tuple(f"feature_{index:02d}" for index in range(50))
    catalog = _catalog(names)
    values = tuple(
        tuple(
            None if feature_index % 5 == 0 and row_index == 0 else float(row_index + feature_index)
            for feature_index in range(50)
        )
        for row_index in range(20)
    )
    result = filter_outer_train_quality(
        catalog, _snapshot(catalog, values), BASE, BASE + timedelta(days=19)
    )

    assert len(result.features) == 50
    assert result.features[0].coverage == pytest.approx(19 / 20)
    assert result.eligible_features == names
