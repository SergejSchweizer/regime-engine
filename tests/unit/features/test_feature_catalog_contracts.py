from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.features.ports import FeatureCatalogEntry, FeatureCatalogSnapshot


def lineage() -> SourceLineage:
    now = datetime(2026, 9, 9, tzinfo=UTC)
    return SourceLineage(
        source_dataset="xetra_gold",
        source_build_id="build-1",
        data_sha256="a" * 64,
        schema_version=2,
        feature_version=4,
        source_table="gold_features",
        synced_at_utc=now,
        row_count=100,
        min_timestamp=now,
        max_timestamp=now,
    )


def entries() -> tuple[FeatureCatalogEntry, ...]:
    return (
        FeatureCatalogEntry("feature_a", 1),
        FeatureCatalogEntry("feature_b", 2),
        FeatureCatalogEntry("feature_c", 3),
    )


def test_catalog_is_lineage_bound_and_canonically_ordered() -> None:
    catalog = FeatureCatalogSnapshot.from_entries(lineage(), "timestamp_m1", reversed(entries()))
    assert catalog.feature_names == ("feature_a", "feature_b", "feature_c")
    assert catalog.entries == entries()
    assert len(catalog.catalog_hash) == 64
    assert catalog == FeatureCatalogSnapshot(lineage(), "timestamp_m1", entries())


def test_physical_mapping_order_does_not_change_hash_but_identity_does() -> None:
    base = FeatureCatalogSnapshot.from_entries(lineage(), "timestamp_m1", entries())
    reordered = FeatureCatalogSnapshot.from_entries(
        lineage(), "timestamp_m1", (entries()[2], entries()[0], entries()[1])
    )
    assert reordered.catalog_hash == base.catalog_hash

    changed_ordinal = FeatureCatalogSnapshot.from_entries(
        lineage(), "timestamp_m1", (FeatureCatalogEntry("feature_a", 4), entries()[1], entries()[2])
    )
    assert changed_ordinal.catalog_hash != base.catalog_hash

    changed_name = FeatureCatalogSnapshot.from_entries(
        lineage(), "timestamp_m1", (FeatureCatalogEntry("feature_z", 1), entries()[1], entries()[2])
    )
    assert changed_name.catalog_hash != base.catalog_hash

    changed_lineage = FeatureCatalogSnapshot.from_entries(
        SourceLineage(
            "xetra_gold",
            "build-2",
            "b" * 64,
            2,
            4,
            "gold_features",
            datetime(2026, 9, 9, tzinfo=UTC),
            row_count=100,
            min_timestamp=datetime(2026, 9, 9, tzinfo=UTC),
            max_timestamp=datetime(2026, 9, 9, tzinfo=UTC),
        ),
        "timestamp_m1",
        entries(),
    )
    assert changed_lineage.catalog_hash != base.catalog_hash


def test_catalog_rejects_unsafe_or_incompatible_schema() -> None:
    with pytest.raises(ValueError, match="safe"):
        FeatureCatalogEntry("Feature-A", 1)
    with pytest.raises(ValueError, match="timestamp_m1"):
        FeatureCatalogEntry("timestamp_m1", 1)
    with pytest.raises(ValueError, match="DOUBLE PRECISION"):
        FeatureCatalogEntry("feature_a", 1, "NUMERIC")
    with pytest.raises(ValueError, match="positive"):
        FeatureCatalogEntry("feature_a", 0)

    valid = FeatureCatalogSnapshot(lineage(), "timestamp_m1", entries())
    with pytest.raises(ValueError, match="timestamp_m1"):
        FeatureCatalogSnapshot(lineage(), "timestamp", entries())
    with pytest.raises(ValueError, match="at least one"):
        FeatureCatalogSnapshot(lineage(), "timestamp_m1", ())
    with pytest.raises(ValueError, match="ordinal ordered"):
        FeatureCatalogSnapshot(
            lineage(), "timestamp_m1", (entries()[1], entries()[0], entries()[2])
        )
    with pytest.raises(ValueError, match="unique"):
        FeatureCatalogSnapshot(lineage(), "timestamp_m1", (entries()[0], entries()[0]))
    with pytest.raises(ValueError, match="FeatureCatalogEntry"):
        FeatureCatalogSnapshot(lineage(), "timestamp_m1", (valid.entries[0], "feature_b"))  # type: ignore[arg-type]
