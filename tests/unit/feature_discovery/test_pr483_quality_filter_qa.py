from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.feature_discovery.quality import filter_outer_train_quality
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def catalog(names: tuple[str, ...]) -> FeatureCatalogSnapshot:
    lineage = SourceLineage(
        source_dataset="macro_loader.macro_features",
        source_build_id="build-qa",
        data_sha256="a" * 64,
        schema_version=6,
        feature_version=5,
        source_table="macro_loader.macro_features",
        synced_at_utc=BASE,
        row_count=100,
        min_timestamp=BASE,
        max_timestamp=BASE + timedelta(days=100),
    )
    entries = tuple(FeatureCatalogEntry(name, index + 1) for index, name in enumerate(names))
    return FeatureCatalogSnapshot.from_entries(lineage, "timestamp_m1", entries)


def snapshot(
    feature_catalog: FeatureCatalogSnapshot,
    values: tuple[tuple[float | None, ...], ...],
    *,
    start: datetime = BASE,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        lineage=feature_catalog.lineage,
        feature_names=feature_catalog.feature_names,
        rows=tuple(
            FeatureRow(start + timedelta(days=index), row_values)
            for index, row_values in enumerate(values)
        ),
    )


def test_reference_quality_statistics_and_reason_codes_cover_edge_cases() -> None:
    names = ("valid_a", "valid_b", "valid_c", "all_null", "near_constant", "short_history")
    feature_catalog = catalog(names)
    values = tuple(
        (
            float(index),
            float(index + 1),
            float(2 * index),
            None,
            1.0 + (1.0e-8 if index % 2 else 0.0),
            float(index) if index < 2 else None,
        )
        for index in range(10)
    )

    result = filter_outer_train_quality(
        feature_catalog,
        snapshot(feature_catalog, values),
        BASE,
        BASE + timedelta(days=9),
        max_workers=1,
    )
    by_name = {item.feature_name: item for item in result.features}
    assert by_name["valid_a"].population_variance == pytest.approx(8.25)
    assert by_name["all_null"].rejection_reason == "coverage_below_minimum"
    assert by_name["short_history"].rejection_reason == "coverage_below_minimum"
    assert by_name["near_constant"].rejection_reason == "variance_below_or_equal_minimum"
    assert result.eligible_features == ("valid_a", "valid_b", "valid_c")


def test_outer_test_rows_are_rejected_before_they_can_change_train_quality() -> None:
    names = ("f0", "f1", "f2")
    feature_catalog = catalog(names)
    train_values = tuple((float(index), float(index + 1), float(index + 2)) for index in range(10))
    train = snapshot(feature_catalog, train_values)
    baseline = filter_outer_train_quality(
        feature_catalog, train, BASE, BASE + timedelta(days=9), max_workers=1
    )

    with_test_row = snapshot(
        feature_catalog,
        (*train_values, (-1.0e300, -1.0e300, -1.0e300)),
    )
    with pytest.raises(ValueError, match="outside supplied TRAIN bounds"):
        filter_outer_train_quality(
            feature_catalog, with_test_row, BASE, BASE + timedelta(days=9), max_workers=1
        )
    assert (
        baseline.result_hash
        == filter_outer_train_quality(
            feature_catalog, train, BASE, BASE + timedelta(days=9), max_workers=1
        ).result_hash
    )


def test_large_candidate_matrix_has_no_pairwise_quality_allocation() -> None:
    names = tuple(f"feature_{index:04d}" for index in range(5_000))
    feature_catalog = catalog(names)
    values = tuple(
        tuple(float(row_index + column_index) for column_index in range(len(names)))
        for row_index in range(5)
    )

    result = filter_outer_train_quality(
        feature_catalog,
        snapshot(feature_catalog, values),
        BASE,
        BASE + timedelta(days=4),
        max_workers=1,
    )
    assert len(result.features) == 5_000
    assert result.eligible_features == names


def test_reversing_input_columns_and_process_execution_preserves_result() -> None:
    names = ("f0", "f1", "f2", "f3")
    feature_catalog = catalog(names)
    values = tuple(
        tuple(float(row_index + column_index) for column_index in range(len(names)))
        for row_index in range(10)
    )
    serial = filter_outer_train_quality(
        feature_catalog,
        snapshot(feature_catalog, values),
        BASE,
        BASE + timedelta(days=9),
        max_workers=1,
    )
    parallel = filter_outer_train_quality(
        feature_catalog,
        snapshot(feature_catalog, values),
        BASE,
        BASE + timedelta(days=9),
        max_workers=2,
    )
    assert parallel == serial

    reversed_names = tuple(reversed(names))
    reversed_catalog = catalog(reversed_names)
    reversed_values = tuple(tuple(reversed(row)) for row in values)
    reversed_result = filter_outer_train_quality(
        reversed_catalog,
        snapshot(reversed_catalog, reversed_values),
        BASE,
        BASE + timedelta(days=9),
        max_workers=1,
    )
    assert reversed_result.result_hash != serial.result_hash
    assert {
        item.feature_name: (item.coverage, item.population_variance, item.rejection_reason)
        for item in reversed_result.features
    } == {
        item.feature_name: (item.coverage, item.population_variance, item.rejection_reason)
        for item in serial.features
    }
