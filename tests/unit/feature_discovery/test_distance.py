from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.feature_discovery.contracts import (
    DiscoveryStatus,
    FeatureQuality,
    QualityFilterResult,
)
from market_regime_engine.feature_discovery.distance import (
    _average_ranks,
    _clip_correlation,
    _rank_pearson_correlation,
    _rank_pearson_correlation_array,
    compute_distance_matrix,
    global_absolute_spearman_distance,
)
from market_regime_engine.feature_discovery.quality import filter_outer_train_quality
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)
OBSERVATIONS = 504


def _lineage(build: str = "build-1") -> SourceLineage:
    return SourceLineage(
        source_dataset="xetra_gold",
        source_build_id=build,
        data_sha256="a" * 64,
        schema_version=2,
        feature_version=4,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=BASE,
        row_count=OBSERVATIONS,
        min_timestamp=BASE,
        max_timestamp=BASE + timedelta(days=OBSERVATIONS - 1),
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
) -> FeatureSnapshot:
    return FeatureSnapshot(
        lineage=catalog.lineage,
        feature_names=catalog.feature_names,
        rows=tuple(
            FeatureRow(BASE + timedelta(days=index), row_values)
            for index, row_values in enumerate(values)
        ),
    )


def _unsafe_snapshot(
    catalog: FeatureCatalogSnapshot,
    rows: tuple[FeatureRow, ...],
    *,
    feature_names: tuple[str, ...] | None = None,
    skipped_incomplete_row_count: int = 0,
) -> FeatureSnapshot:
    snapshot = object.__new__(FeatureSnapshot)
    object.__setattr__(snapshot, "lineage", catalog.lineage)
    object.__setattr__(snapshot, "feature_names", feature_names or catalog.feature_names)
    object.__setattr__(snapshot, "rows", rows)
    object.__setattr__(snapshot, "skipped_incomplete_row_count", skipped_incomplete_row_count)
    return snapshot


def _quality_and_snapshot(
    names: tuple[str, ...],
    values: tuple[tuple[float | None, ...], ...],
) -> tuple[FeatureSnapshot, QualityFilterResult]:
    catalog = _catalog(names)
    snapshot = _snapshot(catalog, values)
    quality = filter_outer_train_quality(
        catalog, snapshot, BASE, BASE + timedelta(days=OBSERVATIONS - 1)
    )
    return snapshot, quality


def test_average_ranks_use_average_ties_and_correlation_clipping_is_fail_closed() -> None:
    assert _average_ranks((2.0, 1.0, 1.0, 3.0)) == (3.0, 1.5, 1.5, 4.0)
    assert _clip_correlation(1.0 + 0.5e-12) == 1.0
    assert _clip_correlation(-1.0 - 0.5e-12) == -1.0
    with pytest.raises(ValueError, match="beyond tolerance"):
        _clip_correlation(1.0 + 2.0e-12)
    with pytest.raises(ValueError, match="beyond tolerance"):
        _clip_correlation(-1.0 - 2.0e-12)
    with pytest.raises(ValueError, match="finite"):
        _clip_correlation(float("nan"))


def test_native_array_correlation_matches_reference_rank_pearson() -> None:
    left = (1.0, 2.0, 2.0, 4.0, 5.0)
    right = (5.0, 1.0, 3.0, 3.0, 2.0)
    expected = _rank_pearson_correlation(left, right)
    actual = _rank_pearson_correlation_array(np.asarray(left), np.asarray(right))
    assert actual == pytest.approx(expected, abs=1.0e-15)


def test_perfect_positive_and_negative_monotonic_pairs_have_exact_zero_distance() -> None:
    values = tuple(
        (float(index), 2.0 * index + 1.0, -float(index), float(index // 3))
        for index in range(OBSERVATIONS)
    )
    snapshot, quality = _quality_and_snapshot(("ascending", "positive", "negative", "tied"), values)

    result = global_absolute_spearman_distance(snapshot, quality)
    assert result.feature_order == ("ascending", "positive", "negative", "tied")
    assert result.distances[0][1] == 0.0
    assert result.distances[0][2] == 0.0
    assert result.distances[0][3] > 0.0
    assert result.spearman_correlations[0][2] == -1.0
    assert result.pairwise_support[0][3] == OBSERVATIONS
    assert result.matrix_hash


def test_pairwise_support_is_exact_and_missing_values_are_not_imputed() -> None:
    values = tuple(
        (
            float(index),
            float(index + 1) if index != 0 else None,
            float(index * 2 + 1),
        )
        for index in range(OBSERVATIONS)
    )
    snapshot, quality = _quality_and_snapshot(("f0", "f1", "f2"), values)
    with pytest.raises(ValueError, match=r"support.*503"):
        global_absolute_spearman_distance(snapshot, quality)


def test_fifty_feature_fixture_is_canonical_and_feature_permutations_do_not_change_result() -> None:
    names = tuple(f"feature_{index:02d}" for index in range(50))
    values = tuple(
        tuple(float((row_index * (feature_index + 3)) % 997) for feature_index in range(50))
        for row_index in range(OBSERVATIONS)
    )
    catalog = _catalog(names)
    snapshot = _snapshot(catalog, values)
    quality = filter_outer_train_quality(
        catalog, snapshot, BASE, BASE + timedelta(days=OBSERVATIONS - 1)
    )
    result = compute_distance_matrix(snapshot, quality)

    permuted_catalog = FeatureCatalogSnapshot.from_entries(
        catalog.lineage,
        "timestamp_m1",
        tuple(reversed(catalog.entries)),
    )
    permuted_snapshot = _snapshot(permuted_catalog, values)
    permuted_quality = filter_outer_train_quality(
        permuted_catalog,
        permuted_snapshot,
        BASE,
        BASE + timedelta(days=OBSERVATIONS - 1),
    )
    permuted_result = compute_distance_matrix(permuted_snapshot, permuted_quality)
    assert result.feature_order == names
    assert len(result.distances) == 50
    assert result.pairwise_support[0][49] == OBSERVATIONS
    assert result.matrix_hash == permuted_result.matrix_hash


def test_nonfinite_and_invalid_quality_evidence_fail_before_distance() -> None:
    names = ("f0", "f1", "f2")
    values = tuple(
        (float(index), float(index + 1), float(index + 2)) for index in range(OBSERVATIONS)
    )
    catalog = _catalog(names)
    snapshot = _snapshot(catalog, values)
    quality = filter_outer_train_quality(
        catalog, snapshot, BASE, BASE + timedelta(days=OBSERVATIONS - 1)
    )
    broken_values = list(values)
    broken_values[3] = (float("inf"), 4.0, 5.0)
    with pytest.raises(ValueError, match="must be finite"):
        compute_distance_matrix(_snapshot(catalog, tuple(broken_values)), quality)
    with pytest.raises(ValueError, match="valid quality"):
        compute_distance_matrix(snapshot, replace(quality, status=DiscoveryStatus.INVALID))
    with pytest.raises(ValueError, match="columns"):
        compute_distance_matrix(
            FeatureSnapshot(
                catalog.lineage,
                ("wrong", "f1", "f2"),
                snapshot.rows,
            ),
            quality,
        )


def test_math_and_snapshot_validation_paths_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        _average_ranks(())
    with pytest.raises(ValueError, match="finite"):
        _average_ranks((float("inf"),))
    with pytest.raises(ValueError, match="at least two"):
        _rank_pearson_correlation((1.0,), (1.0,))
    with pytest.raises(ValueError, match="undefined"):
        _rank_pearson_correlation((1.0, 1.0), (1.0, 2.0))

    values = tuple(
        (float(index), float(index + 1), float(index + 2)) for index in range(OBSERVATIONS)
    )
    catalog = _catalog(("f0", "f1", "f2"))
    snapshot = _snapshot(catalog, values)
    quality = filter_outer_train_quality(
        catalog, snapshot, BASE, BASE + timedelta(days=OBSERVATIONS - 1)
    )
    with pytest.raises(TypeError, match="FeatureSnapshot"):
        compute_distance_matrix(object(), quality)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="QualityFilterResult"):
        compute_distance_matrix(snapshot, object())  # type: ignore[arg-type]

    duplicate_positions = replace(
        quality,
        features=tuple(replace(item, source_position=1) for item in quality.features),
    )
    with pytest.raises(ValueError, match="canonical"):
        compute_distance_matrix(snapshot, duplicate_positions)
    with pytest.raises(ValueError, match="build IDs"):
        compute_distance_matrix(snapshot, replace(quality, source_build_id="other"))
    with pytest.raises(ValueError, match="denominator"):
        compute_distance_matrix(
            snapshot,
            replace(quality, train_source_observation_count=OBSERVATIONS + 1),
        )
    with pytest.raises(ValueError, match="all source rows"):
        compute_distance_matrix(
            _unsafe_snapshot(
                catalog,
                snapshot.rows,
                skipped_incomplete_row_count=1,
            ),
            quality,
        )
    empty_quality = object.__new__(QualityFilterResult)
    for field, value in (
        ("source_build_id", quality.source_build_id),
        ("train_source_observation_count", 0),
        ("catalog_hash", quality.catalog_hash),
        ("features", quality.features),
        ("eligible_features", quality.eligible_features),
        ("status", DiscoveryStatus.VALID),
    ):
        object.__setattr__(empty_quality, field, value)
    with pytest.raises(ValueError, match="non-empty"):
        compute_distance_matrix(_unsafe_snapshot(catalog, ()), empty_quality)
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_distance_matrix(
            _unsafe_snapshot(catalog, (snapshot.rows[1], snapshot.rows[0], *snapshot.rows[2:])),
            quality,
        )
    one_row_quality = object.__new__(QualityFilterResult)
    for field, value in (
        ("source_build_id", quality.source_build_id),
        ("train_source_observation_count", 1),
        ("catalog_hash", quality.catalog_hash),
        ("features", quality.features),
        ("eligible_features", quality.eligible_features),
        ("status", DiscoveryStatus.VALID),
    ):
        object.__setattr__(one_row_quality, field, value)
    with pytest.raises(ValueError, match="dimension"):
        compute_distance_matrix(
            _unsafe_snapshot(
                catalog,
                (FeatureRow(BASE, (1.0, 2.0)),),
            ),
            one_row_quality,
        )
    with pytest.raises(ValueError, match="must be numeric"):
        compute_distance_matrix(
            _unsafe_snapshot(
                catalog,
                (FeatureRow(BASE, (object(), 2.0, 3.0)),),  # type: ignore[arg-type]
            ),
            one_row_quality,
        )

    few_quality = object.__new__(QualityFilterResult)
    for field, value in (
        ("source_build_id", quality.source_build_id),
        ("train_source_observation_count", quality.train_source_observation_count),
        ("catalog_hash", quality.catalog_hash),
        ("features", quality.features),
        ("eligible_features", ("f0",)),
        ("status", DiscoveryStatus.VALID),
    ):
        object.__setattr__(few_quality, field, value)
    with pytest.raises(ValueError, match="at least two"):
        compute_distance_matrix(snapshot, few_quality)


def test_quality_evidence_type_is_checked_before_attribute_access() -> None:
    catalog = _catalog(("f0", "f1", "f2"))
    values = tuple(
        (float(index), float(index + 1), float(index + 2)) for index in range(OBSERVATIONS)
    )
    snapshot = _snapshot(catalog, values)
    assert isinstance(
        filter_outer_train_quality(
            catalog, snapshot, BASE, BASE + timedelta(days=OBSERVATIONS - 1)
        ).features[0],
        FeatureQuality,
    )
