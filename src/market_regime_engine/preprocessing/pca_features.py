"""Materialize PCA components as lineage-bound generated feature columns."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)
from market_regime_engine.preprocessing.pca_policy import PCAFitResult, fit_pca_inner_train

ArrayF64 = npt.NDArray[np.float64]
_GENERATED_SCHEMA = "regime_engine"
_GENERATED_RELATION = "pca_generated_features"


def validate_pca_feature_universe(
    catalog: FeatureCatalogSnapshot,
    *,
    component_count: int,
) -> tuple[str, ...]:
    """Validate and return the raw order of the mandatory v4 universe.

    Canonical v4 never evaluates a raw-only catalog.  PCA columns are ordinary
    catalogued features for discovery and selection, but their generated
    provenance must still be structurally identifiable so fold-local PCA can
    be refit without leaking the source snapshot fit.
    """

    if component_count < 1:
        raise ValueError("PCA component_count must be positive")
    generated = tuple(name for name in catalog.feature_names if name.startswith("pca_pc_"))
    expected = tuple(f"pca_pc_{index:03d}" for index in range(1, component_count + 1))
    if generated != expected:
        raise ValueError(
            "canonical v4 requires the complete raw-plus-PCA feature universe "
            f"({', '.join(expected)})"
        )
    raw = tuple(name for name in catalog.feature_names if not name.startswith("pca_pc_"))
    if not raw:
        raise ValueError("canonical v4 PCA feature universe requires raw features")
    generated_entries = {
        entry.feature_name: entry
        for entry in catalog.entries
        if entry.feature_name.startswith("pca_pc_")
    }
    if any(
        generated_entries[name].schema_name != _GENERATED_SCHEMA
        or generated_entries[name].relation_name != _GENERATED_RELATION
        or generated_entries[name].relation_kind not in {"VIEW", "MATERIALIZED VIEW"}
        for name in expected
    ):
        raise ValueError("PCA feature entries must retain generated-feature provenance")
    return raw


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value


@dataclass(frozen=True, slots=True)
class PCAGeneratedFeatureSet:
    """Combined raw/generated catalog and materialized snapshot with provenance."""

    raw_catalog_hash: str
    pca_fit_hash: str
    catalog: FeatureCatalogSnapshot
    snapshot: FeatureSnapshot
    skipped_incomplete_row_count: int

    def __post_init__(self) -> None:
        if len(self.raw_catalog_hash) != 64 or len(self.pca_fit_hash) != 64:
            raise ValueError("PCA generated provenance hashes must be SHA-256 digests")
        if self.snapshot.feature_names != self.catalog.feature_names:
            raise ValueError("generated snapshot columns must match generated catalog")
        if self.skipped_incomplete_row_count < 0:
            raise ValueError("generated skipped row count cannot be negative")
        if self.snapshot.skipped_incomplete_row_count != self.skipped_incomplete_row_count:
            raise ValueError("generated skipped row count does not match snapshot")

    @property
    def generated_feature_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.catalog.feature_names if name.startswith("pca_pc_"))

    @property
    def raw_feature_names(self) -> tuple[str, ...]:
        """Return the immutable raw source order used by the PCA fit."""

        generated = set(self.generated_feature_names)
        raw = tuple(name for name in self.catalog.feature_names if name not in generated)
        return raw

    @property
    def provenance_hash(self) -> str:
        payload = {
            "catalog_hash": self.catalog.catalog_hash,
            "pca_fit_hash": self.pca_fit_hash,
            "raw_catalog_hash": self.raw_catalog_hash,
            "snapshot_hash": self.snapshot.materialized_feature_data_sha256,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_canonical_json(self) -> str:
        return json.dumps(
            {
                "catalog_hash": self.catalog.catalog_hash,
                "generated_feature_names": list(self.generated_feature_names),
                "pca_fit_hash": self.pca_fit_hash,
                "provenance_hash": self.provenance_hash,
                "raw_catalog_hash": self.raw_catalog_hash,
                "skipped_incomplete_row_count": self.skipped_incomplete_row_count,
                "snapshot_hash": self.snapshot.materialized_feature_data_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )


def _derived_lineage(
    raw_catalog: FeatureCatalogSnapshot,
    fit: PCAFitResult,
    *,
    row_count: int,
    min_timestamp: datetime,
    max_timestamp: datetime,
) -> SourceLineage:
    identity = {
        "fit_hash": fit.fit_hash,
        "raw_catalog_hash": raw_catalog.catalog_hash,
        "source_data_sha256": raw_catalog.lineage.data_sha256,
        "source_build_id": raw_catalog.lineage.source_build_id,
    }
    data_hash = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return replace(
        raw_catalog.lineage,
        source_table=f"{_GENERATED_SCHEMA}.{_GENERATED_RELATION}",
        data_sha256=data_hash,
        row_count=row_count,
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
    )


def materialize_pca_generated_features(
    raw_catalog: FeatureCatalogSnapshot,
    fit: PCAFitResult,
    timestamps: Sequence[datetime],
    source_rows: npt.ArrayLike,
) -> PCAGeneratedFeatureSet:
    """Append PCA components while preserving the complete source clock.

    PCA is fit on complete TRAIN rows, but generated columns are materialized
    for every source timestamp.  Incomplete rows retain their raw values and
    receive null PCA values; they are therefore handled by the same coverage
    and complete-case rules as every other feature.
    """

    if fit.feature_order != raw_catalog.feature_names:
        raise ValueError("PCA fit feature order must equal the raw catalog feature order")
    timestamp_values = tuple(_utc(value, "source timestamp") for value in timestamps)
    if any(current <= previous for previous, current in pairwise(timestamp_values)):
        raise ValueError("PCA generated timestamps must be strictly increasing and unique")
    matrix = np.asarray(source_rows, dtype=np.float64)
    expected_shape = (len(timestamp_values), len(raw_catalog.feature_names))
    if matrix.ndim != 2 or matrix.shape != expected_shape:
        raise ValueError("PCA generated source rows must match the raw catalog exactly")
    complete = np.all(np.isfinite(matrix), axis=1)
    complete_rows = matrix[complete]
    if complete_rows.shape[0] == 0:
        raise ValueError("PCA generated materialization has no complete raw rows")
    generated = fit.transform(complete_rows)
    generated_names = fit.artifact.generated_feature_names
    combined_names = raw_catalog.feature_names + generated_names
    generated_by_row = np.full(
        (len(timestamp_values), len(generated_names)), np.nan, dtype=np.float64
    )
    generated_by_row[complete] = generated
    lineage = _derived_lineage(
        raw_catalog,
        fit,
        row_count=len(timestamp_values),
        min_timestamp=timestamp_values[0],
        max_timestamp=timestamp_values[-1],
    )
    rows = tuple(
        FeatureRow(
            timestamp,
            tuple(None if not np.isfinite(value) else float(value) for value in raw_values)
            + tuple(None if np.isnan(value) else float(value) for value in pca_values),
        )
        for timestamp, raw_values, pca_values in zip(
            timestamp_values, matrix, generated_by_row, strict=True
        )
    )
    snapshot = FeatureSnapshot(
        lineage=lineage,
        feature_names=combined_names,
        rows=rows,
        skipped_incomplete_row_count=int(len(timestamp_values) - len(complete_rows)),
    )
    next_ordinal = max(entry.canonical_ordinal for entry in raw_catalog.entries)
    generated_entries = tuple(
        FeatureCatalogEntry(
            feature_name=name,
            canonical_ordinal=next_ordinal + index,
            schema_name=_GENERATED_SCHEMA,
            relation_name=_GENERATED_RELATION,
            relation_kind="VIEW",
            ordinal_position=index,
        )
        for index, name in enumerate(generated_names, start=1)
    )
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        raw_catalog.timestamp_column,
        (*raw_catalog.entries, *generated_entries),
    ).with_materialization(snapshot)
    return PCAGeneratedFeatureSet(
        raw_catalog_hash=raw_catalog.catalog_hash,
        pca_fit_hash=fit.fit_hash,
        catalog=catalog,
        snapshot=snapshot,
        skipped_incomplete_row_count=snapshot.skipped_incomplete_row_count,
    )


def fit_and_materialize_pca_source(
    raw_catalog: FeatureCatalogSnapshot,
    raw_snapshot: FeatureSnapshot,
    *,
    variance_threshold: float = 0.90,
    component_count: int = 8,
) -> PCAGeneratedFeatureSet:
    """Create the fixed raw-plus-PCA source universe for one snapshot.

    The snapshot fit is only a transport representation.  Outer-fold
    discovery refits PCA on each TRAIN interval; this fit supplies stable
    component names and the source/catalog lineage used by the evidence
    contract.
    """

    if raw_snapshot.feature_names != raw_catalog.feature_names:
        raise ValueError("PCA source snapshot columns must match the raw catalog")
    if not raw_snapshot.rows:
        raise ValueError("PCA source snapshot cannot be empty")
    timestamps = tuple(row.timestamp for row in raw_snapshot.rows)
    matrix = np.asarray([row.values for row in raw_snapshot.rows], dtype=np.float64)
    fit = fit_pca_inner_train(
        timestamps,
        matrix,
        feature_order=raw_catalog.feature_names,
        inner_fold_id="source_snapshot",
        fit_start=timestamps[0],
        fit_end=timestamps[-1],
        variance_threshold=variance_threshold,
        component_count=component_count,
    )
    return materialize_pca_generated_features(raw_catalog, fit, timestamps, matrix)


__all__ = [
    "PCAGeneratedFeatureSet",
    "fit_and_materialize_pca_source",
    "materialize_pca_generated_features",
    "validate_pca_feature_universe",
]
