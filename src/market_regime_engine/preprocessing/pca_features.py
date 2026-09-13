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
from market_regime_engine.preprocessing.pca_policy import PCAFitResult

ArrayF64 = npt.NDArray[np.float64]
_GENERATED_SCHEMA = "regime_engine"
_GENERATED_RELATION = "pca_generated_features"


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
    """Append PCA components to complete raw rows without imputing missing data."""

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
    combined = np.column_stack((complete_rows, generated))
    generated_names = fit.artifact.generated_feature_names
    combined_names = raw_catalog.feature_names + generated_names
    selected_timestamps = tuple(
        timestamp
        for timestamp, is_complete in zip(timestamp_values, complete, strict=True)
        if is_complete
    )
    lineage = _derived_lineage(
        raw_catalog,
        fit,
        row_count=len(selected_timestamps),
        min_timestamp=selected_timestamps[0],
        max_timestamp=selected_timestamps[-1],
    )
    rows = tuple(
        FeatureRow(timestamp, tuple(float(value) for value in values))
        for timestamp, values in zip(selected_timestamps, combined, strict=True)
    )
    snapshot = FeatureSnapshot(
        lineage=lineage,
        feature_names=combined_names,
        rows=rows,
        skipped_incomplete_row_count=int(len(timestamp_values) - len(selected_timestamps)),
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


__all__ = ["PCAGeneratedFeatureSet", "materialize_pca_generated_features"]
