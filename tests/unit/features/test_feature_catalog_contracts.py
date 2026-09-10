from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRequest,
    FeatureRow,
    FeatureSnapshot,
    SourceMode,
)


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


def test_catalog_origin_and_materialization_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="schema_name"):
        FeatureCatalogEntry("feature_a", 1, schema_name="Bad-Schema")
    with pytest.raises(ValueError, match="relation_name"):
        FeatureCatalogEntry("feature_a", 1, relation_name="Bad-Relation")
    with pytest.raises(ValueError, match="relation kind"):
        FeatureCatalogEntry("feature_a", 1, relation_kind="FOREIGN TABLE")
    with pytest.raises(ValueError, match="ordinal_position"):
        FeatureCatalogEntry("feature_a", 1, ordinal_position=0)

    catalog_lineage = lineage()
    entries_value = entries()
    with pytest.raises(ValueError, match="materialized feature data hash is required"):
        FeatureCatalogSnapshot(
            catalog_lineage,
            "timestamp_m1",
            entries_value,
            materialized_row_count=1,
            materialized_min_timestamp=catalog_lineage.min_timestamp,
            materialized_max_timestamp=catalog_lineage.max_timestamp,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        FeatureCatalogSnapshot(
            catalog_lineage,
            "timestamp_m1",
            entries_value,
            materialized_feature_data_sha256="x" * 64,
            materialized_row_count=1,
            materialized_min_timestamp=catalog_lineage.min_timestamp,
            materialized_max_timestamp=catalog_lineage.max_timestamp,
        )
    with pytest.raises(ValueError, match="non-negative"):
        FeatureCatalogSnapshot(
            catalog_lineage,
            "timestamp_m1",
            entries_value,
            materialized_feature_data_sha256="a" * 64,
            materialized_row_count=-1,
            materialized_min_timestamp=catalog_lineage.min_timestamp,
            materialized_max_timestamp=catalog_lineage.max_timestamp,
        )
    with pytest.raises(ValueError, match="cannot have timestamp bounds"):
        FeatureCatalogSnapshot(
            catalog_lineage,
            "timestamp_m1",
            entries_value,
            materialized_feature_data_sha256="a" * 64,
            materialized_row_count=0,
            materialized_min_timestamp=catalog_lineage.min_timestamp,
        )
    with pytest.raises(ValueError, match="requires timestamp bounds"):
        FeatureCatalogSnapshot(
            catalog_lineage,
            "timestamp_m1",
            entries_value,
            materialized_feature_data_sha256="a" * 64,
            materialized_row_count=1,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        FeatureCatalogSnapshot(
            catalog_lineage,
            "timestamp_m1",
            entries_value,
            materialized_feature_data_sha256="a" * 64,
            materialized_row_count=1,
            materialized_min_timestamp=datetime(2026, 9, 9),
            materialized_max_timestamp=catalog_lineage.max_timestamp,
        )
    with pytest.raises(ValueError, match="inverted"):
        FeatureCatalogSnapshot(
            catalog_lineage,
            "timestamp_m1",
            entries_value,
            materialized_feature_data_sha256="a" * 64,
            materialized_row_count=1,
            materialized_min_timestamp=datetime(2026, 9, 10, tzinfo=UTC),
            materialized_max_timestamp=datetime(2026, 9, 9, tzinfo=UTC),
        )
    snapshot = FeatureSnapshot(
        catalog_lineage,
        ("feature_a", "feature_b", "feature_c"),
        (FeatureRow(catalog_lineage.min_timestamp, (1.0, 2.0, 3.0)),),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="columns"):
        FeatureCatalogSnapshot.from_entries(
            catalog_lineage,
            "timestamp_m1",
            entries_value[:2],
        ).with_materialization(snapshot)


def test_request_and_snapshot_contracts_cover_dynamic_and_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="duplicate-free"):
        FeatureRequest(("feature_a", "feature_a"), None, None, SourceMode.FEATURE_SELECTION)
    with pytest.raises(ValueError, match="only valid"):
        FeatureRequest((), None, None, SourceMode.RESOLVED_MODEL)
    with pytest.raises(ValueError, match="after"):
        FeatureRequest(
            ("feature_a",),
            datetime(2026, 9, 10, tzinfo=UTC),
            datetime(2026, 9, 9, tzinfo=UTC),
            SourceMode.FEATURE_SELECTION,
        )
    assert FeatureRequest.all_features().feature_names == ()

    now = datetime(2026, 9, 9, tzinfo=UTC)
    with pytest.raises(ValueError, match="non-empty"):
        FeatureSnapshot(lineage(), (), ())
    with pytest.raises(ValueError, match="negative"):
        FeatureSnapshot(lineage(), ("feature_a",), (), skipped_incomplete_row_count=-1)
    with pytest.raises(ValueError, match="do not match feature_names"):
        FeatureSnapshot(lineage(), ("feature_a",), (FeatureRow(now, (1.0, 2.0)),))
    with pytest.raises(ValueError, match="strictly increasing"):
        FeatureSnapshot(
            lineage(),
            ("feature_a",),
            (FeatureRow(now, (1.0,)), FeatureRow(now, (2.0,))),
        )
    valid = FeatureSnapshot(lineage(), ("feature_a",), (FeatureRow(now, (1.0,)),))
    with pytest.raises(ValueError, match="does not match"):
        FeatureSnapshot(
            lineage(),
            ("feature_a",),
            valid.rows,
            materialized_feature_data_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="finite"):
        FeatureSnapshot(lineage(), ("feature_a",), (FeatureRow(now, (float("inf"),)),))
    assert valid.row_count == 1
    assert valid.min_timestamp == now
    assert valid.max_timestamp == now
    empty = object.__new__(FeatureSnapshot)
    object.__setattr__(empty, "rows", ())
    assert empty.row_count == 0
    assert empty.min_timestamp is None
    assert empty.max_timestamp is None
