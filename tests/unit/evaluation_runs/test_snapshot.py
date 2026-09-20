from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation_runs import snapshot as snapshot_module
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


def test_snapshot_missing_arrow_fails_closed_for_identity_and_rows(tmp_path: Path) -> None:
    snapshot, identity = snapshot_and_identity()
    store = ArrowDatasetSnapshotStore(tmp_path)
    store.finalize(identity, snapshot)

    (tmp_path / identity.key / "snapshot.arrow").unlink()

    with pytest.raises(ValueError, match="missing or incomplete"):
        store.load_identity(identity.key)
    with pytest.raises(ValueError, match="missing or incomplete"):
        store.load(identity)


def test_snapshot_invalid_arrow_bytes_fail_closed_during_finalize(tmp_path: Path) -> None:
    snapshot, identity = snapshot_and_identity()
    store = ArrowDatasetSnapshotStore(tmp_path)
    store.finalize(identity, snapshot)

    arrow_path = tmp_path / identity.key / "snapshot.arrow"
    arrow_path.write_bytes(b"not an Arrow IPC file")

    with pytest.raises(ValueError, match="Arrow file is invalid"):
        store.finalize(identity, snapshot)


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


def test_snapshot_filesystem_boundary_never_exposes_partial_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, identity = snapshot_and_identity()
    store = ArrowDatasetSnapshotStore(tmp_path)

    def fail_directory_sync(_path: Path) -> None:
        raise RuntimeError("forced filesystem boundary")

    monkeypatch.setattr(snapshot_module, "_fsync_directory", fail_directory_sync)
    with pytest.raises(RuntimeError, match="forced filesystem boundary"):
        store.finalize(identity, snapshot)
    with pytest.raises(ValueError, match="incomplete"):
        store.load(identity)


def test_snapshot_identity_and_lineage_contracts_reject_drift(tmp_path: Path) -> None:
    snapshot, identity = snapshot_and_identity()
    store = ArrowDatasetSnapshotStore(tmp_path)
    with pytest.raises(ValueError, match="lineage does not match"):
        store.finalize(replace(identity, source_build_id="other"), snapshot)
    with pytest.raises(ValueError, match="matrix hash"):
        store.finalize(replace(identity, materialized_feature_data_sha256="f" * 64), snapshot)
    with pytest.raises(ValueError, match="row count"):
        store.finalize(replace(identity, materialized_row_count=3), snapshot)
    with pytest.raises(ValueError, match="dataset_snapshot_key"):
        store.load_identity("short")


def test_snapshot_manifest_and_catalog_fail_closed_on_schema_drift(tmp_path: Path) -> None:
    snapshot, identity = snapshot_and_identity()
    catalog = FeatureCatalogSnapshot.from_entries(
        snapshot.lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("f0", 1), FeatureCatalogEntry("f1", 2)),
    ).with_materialization(snapshot)
    store = ArrowDatasetSnapshotStore(tmp_path)
    store.finalize(identity, snapshot, catalog=catalog)
    manifest_path = tmp_path / identity.key / "manifest.json"
    original = manifest_path.read_bytes()

    manifest_path.write_bytes(b"not json")
    with pytest.raises(ValueError, match="manifest is invalid"):
        store.load(identity)
    manifest_path.write_bytes(original)
    payload = json.loads(original)
    payload["catalog"] = None
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="catalog is missing"):
        store.load_catalog(identity)
    manifest_path.write_bytes(original)
    payload = json.loads(original)
    payload["catalog"]["entries"] = ["invalid"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="catalog is invalid"):
        store.load_catalog(identity)


def test_snapshot_load_identity_rejects_manifest_identity_and_arrow_hash_drift(
    tmp_path: Path,
) -> None:
    snapshot, identity = snapshot_and_identity()
    store = ArrowDatasetSnapshotStore(tmp_path)
    store.finalize(identity, snapshot)
    manifest_path = tmp_path / identity.key / "manifest.json"
    original = manifest_path.read_bytes()
    payload = json.loads(original)
    payload["dataset_snapshot_key"] = "b" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="key does not match manifest"):
        store.load_identity(identity.key)
    manifest_path.write_bytes(original)
    payload = json.loads(original)
    payload["arrow_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Arrow hash does not match"):
        store.load_identity(identity.key)


@pytest.mark.parametrize(
    ("value", "field", "message"),
    [
        (None, "timestamp", "must be an ISO timestamp"),
        ("not-a-timestamp", "timestamp", "not a valid ISO timestamp"),
        ("2026-01-01T00:00:00", "timestamp", "timezone-aware UTC"),
        ("2026-01-01T01:00:00+01:00", "timestamp", "timezone-aware UTC"),
    ],
)
def test_snapshot_timestamp_parser_rejects_non_utc_values(
    value: object, field: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        snapshot_module._parse_timestamp(value, field)


def test_snapshot_lineage_parser_rejects_missing_or_malformed_payload() -> None:
    with pytest.raises(ValueError, match="lineage is missing"):
        snapshot_module._lineage_from_payload(None)
    with pytest.raises(ValueError, match="lineage is invalid"):
        snapshot_module._lineage_from_payload({"source_dataset": "only-field"})


def test_snapshot_finalize_rejects_catalog_and_timestamp_identity_drift(tmp_path: Path) -> None:
    snapshot, identity = snapshot_and_identity()
    store = ArrowDatasetSnapshotStore(tmp_path)
    catalog = FeatureCatalogSnapshot.from_entries(
        snapshot.lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("f0", 1), FeatureCatalogEntry("f1", 2)),
    ).with_materialization(snapshot)
    with pytest.raises(ValueError, match="minimum timestamp"):
        store.finalize(
            replace(identity, materialized_min_timestamp=START + timedelta(days=1)), snapshot
        )
    with pytest.raises(ValueError, match="timestamp column must be timestamp_m1"):
        store.finalize(identity, snapshot, catalog=replace(catalog, timestamp_column="other"))
