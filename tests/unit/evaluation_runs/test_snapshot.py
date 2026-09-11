from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation_runs.contracts import DatasetSnapshotIdentity
from market_regime_engine.evaluation_runs.snapshot import ArrowDatasetSnapshotStore
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def snapshot_and_identity() -> tuple[FeatureSnapshot, DatasetSnapshotIdentity]:
    rows = (
        FeatureRow(START, (-0.0, None)),
        FeatureRow(START + timedelta(days=1), (0.1, -2.5)),
    )
    lineage = SourceLineage(
        source_dataset="gold",
        source_build_id="build-1",
        data_sha256="a" * 64,
        schema_version=1,
        feature_version=1,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=START,
        row_count=2,
        min_timestamp=START,
        max_timestamp=START + timedelta(days=1),
    )
    snapshot = FeatureSnapshot(lineage, ("f0", "f1"), rows)
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (
            FeatureCatalogEntry("f0", 1),
            FeatureCatalogEntry("f1", 2),
        ),
    ).with_materialization(snapshot)
    return snapshot, DatasetSnapshotIdentity.from_catalog(catalog)


def test_arrow_snapshot_roundtrip_preserves_nulls_float_bits_and_manifest(tmp_path: Path) -> None:
    snapshot, identity = snapshot_and_identity()
    store = ArrowDatasetSnapshotStore(tmp_path)
    store.finalize(identity, snapshot)
    restored = store.load(identity)

    assert restored == snapshot
    assert restored.rows[0].values[0].hex() == (-0.0).hex()
    assert restored.rows[0].values[1] is None
    assert (tmp_path / identity.key / "snapshot.arrow").is_file()
    assert (tmp_path / identity.key / "manifest.json").is_file()


def test_arrow_snapshot_is_immutable_and_corruption_fails_closed(tmp_path: Path) -> None:
    snapshot, identity = snapshot_and_identity()
    store = ArrowDatasetSnapshotStore(tmp_path)
    store.finalize(identity, snapshot)
    arrow_path = tmp_path / identity.key / "snapshot.arrow"
    original = arrow_path.read_bytes()
    arrow_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="hash does not match"):
        store.load(identity)

    arrow_path.write_bytes(original)
    store.finalize(identity, snapshot)
    manifest_path = tmp_path / identity.key / "manifest.json"
    manifest_path.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        store.load(identity)


def test_snapshot_persists_catalog_and_identity_for_source_free_resume(tmp_path: Path) -> None:
    snapshot, identity = snapshot_and_identity()
    catalog = FeatureCatalogSnapshot.from_entries(
        snapshot.lineage,
        "timestamp_m1",
        (
            FeatureCatalogEntry("f0", 1),
            FeatureCatalogEntry("f1", 2),
        ),
    ).with_materialization(snapshot)
    store = ArrowDatasetSnapshotStore(tmp_path)
    store.finalize(identity, snapshot, catalog=catalog)

    assert store.load_identity(identity.key) == identity
    assert store.load_catalog(identity) == catalog
