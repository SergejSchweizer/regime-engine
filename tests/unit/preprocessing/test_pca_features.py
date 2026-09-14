from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.features.ports import FeatureCatalogEntry, FeatureCatalogSnapshot
from market_regime_engine.preprocessing import (
    PCAFitResult,
    fit_pca_inner_train,
    materialize_pca_generated_features,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def _raw(count: int = 80) -> tuple[FeatureCatalogSnapshot, tuple[datetime, ...], np.ndarray]:
    index = np.arange(count, dtype=np.float64)
    timestamps = tuple(START + timedelta(days=int(value)) for value in index)
    rows = np.column_stack((index, 2.0 * index + 1.0, np.sin(index / 4.0)))
    lineage = SourceLineage(
        source_dataset="synthetic-raw",
        source_build_id="build-1",
        data_sha256="a" * 64,
        schema_version=4,
        feature_version=4,
        source_table="regime_loader.features",
        synced_at_utc=START,
        row_count=count,
        min_timestamp=timestamps[0],
        max_timestamp=timestamps[-1],
    )
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        tuple(
            FeatureCatalogEntry(name, ordinal) for ordinal, name in enumerate(("a", "b", "c"), 1)
        ),
    )
    return catalog, timestamps, rows


def _fit(
    catalog: FeatureCatalogSnapshot,
    timestamps: tuple[datetime, ...],
    rows: np.ndarray,
) -> PCAFitResult:
    return fit_pca_inner_train(
        timestamps,
        rows,
        feature_order=catalog.feature_names,
        inner_fold_id="inner_fold_001",
        fit_start=START,
        fit_end=START + timedelta(days=30),
    )


def test_pca_components_are_first_class_catalogued_features_with_provenance() -> None:
    raw_catalog, timestamps, rows = _raw()
    rows[12, 1] = np.nan
    fit = _fit(raw_catalog, timestamps, rows)
    generated = materialize_pca_generated_features(raw_catalog, fit, timestamps, rows)

    assert generated.catalog.feature_names == (
        "a",
        "b",
        "c",
        *fit.artifact.generated_feature_names,
    )
    assert generated.generated_feature_names == fit.artifact.generated_feature_names
    assert generated.skipped_incomplete_row_count == 1
    assert generated.snapshot.row_count == len(rows)
    assert generated.snapshot.rows[12].values[1] is None
    assert generated.snapshot.rows[12].values[-1] is None
    assert all(
        entry.schema_name == "regime_engine"
        and entry.relation_name == "pca_generated_features"
        and entry.relation_kind == "VIEW"
        for entry in generated.catalog.entries[-fit.artifact.retained_component_count :]
    )
    assert generated.pca_fit_hash == fit.fit_hash
    assert len(generated.provenance_hash) == 64
    assert generated.to_canonical_json() == generated.to_canonical_json()


def test_generated_values_match_frozen_transform_and_mismatch_fails_closed() -> None:
    raw_catalog, timestamps, rows = _raw()
    fit = _fit(raw_catalog, timestamps, rows)
    generated = materialize_pca_generated_features(raw_catalog, fit, timestamps, rows)
    complete = np.all(np.isfinite(rows), axis=1)
    expected = fit.transform(rows[complete])
    actual = np.asarray(
        [
            row.values[-fit.artifact.retained_component_count :]
            for row, is_complete in zip(generated.snapshot.rows, complete, strict=True)
            if is_complete
        ]
    )
    assert np.array_equal(actual, expected)

    wrong_catalog = FeatureCatalogSnapshot.from_entries(
        raw_catalog.lineage,
        "timestamp_m1",
        tuple(
            FeatureCatalogEntry(name, ordinal) for ordinal, name in enumerate(("x", "y", "z"), 1)
        ),
    )
    with pytest.raises(ValueError, match="feature order"):
        materialize_pca_generated_features(wrong_catalog, fit, timestamps, rows)
