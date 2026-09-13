"""Deterministic crash-boundary evidence for immutable dataset snapshots."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation_runs import snapshot as snapshot_module
from market_regime_engine.evaluation_runs.contracts import DatasetSnapshotIdentity
from market_regime_engine.evaluation_runs.snapshot import ArrowDatasetSnapshotStore
from market_regime_engine.features.ports import FeatureRow, FeatureSnapshot

pytestmark = pytest.mark.integration

START = datetime(2026, 1, 1, tzinfo=UTC)


def _snapshot_and_identity() -> tuple[FeatureSnapshot, DatasetSnapshotIdentity]:
    rows = (
        FeatureRow(START, (1.0, None)),
        FeatureRow(START + timedelta(days=1), (2.0, -3.5)),
    )
    lineage = SourceLineage(
        source_dataset="gold",
        source_build_id="crash-boundary-build",
        data_sha256="a" * 64,
        schema_version=1,
        feature_version=1,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=START,
        row_count=len(rows),
        min_timestamp=rows[0].timestamp,
        max_timestamp=rows[-1].timestamp,
    )
    snapshot = FeatureSnapshot(lineage, ("feature_a", "feature_b"), rows)
    identity = DatasetSnapshotIdentity(
        source_dataset=lineage.source_dataset,
        source_build_id=lineage.source_build_id,
        data_sha256=lineage.data_sha256,
        schema_version=lineage.schema_version,
        feature_version=lineage.feature_version,
        data_time_semantics=lineage.data_time_semantics,
        row_count=lineage.row_count or 0,
        min_timestamp=lineage.min_timestamp or START,
        max_timestamp=lineage.max_timestamp or START,
        source_catalog_hash="b" * 64,
        materialized_feature_data_sha256=snapshot.materialized_feature_data_sha256 or "",
        materialized_row_count=len(snapshot.rows),
        materialized_min_timestamp=snapshot.min_timestamp,
        materialized_max_timestamp=snapshot.max_timestamp,
        source_table=lineage.source_table,
    )
    return snapshot, identity


def _snapshot_paths(root: Path, identity: DatasetSnapshotIdentity) -> tuple[Path, Path]:
    directory = root / identity.key
    return directory / "snapshot.arrow", directory / "manifest.json"


def _temporary_files(root: Path) -> tuple[Path, ...]:
    return tuple(root.rglob("*.tmp"))


@pytest.mark.parametrize("failure", ("arrow_fsync", "arrow_rename", "manifest_rename"))
def test_interrupted_publication_never_exposes_a_partial_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    snapshot, identity = _snapshot_and_identity()
    store = ArrowDatasetSnapshotStore(tmp_path)
    arrow_path, manifest_path = _snapshot_paths(tmp_path, identity)

    if failure == "arrow_fsync":
        real_fsync = snapshot_module.os.fsync
        calls = 0

        def fail_arrow_fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("simulated interrupted temp-file fsync")
            real_fsync(descriptor)

        monkeypatch.setattr(snapshot_module.os, "fsync", fail_arrow_fsync)
        expected = "interrupted temp-file fsync"
    elif failure == "arrow_rename":

        def fail_arrow_rename(
            source: str | os.PathLike[str], destination: str | os.PathLike[str]
        ) -> None:
            raise OSError("simulated interrupted Arrow rename")

        monkeypatch.setattr(snapshot_module.os, "replace", fail_arrow_rename)
        expected = "interrupted Arrow rename"
    else:
        real_replace = snapshot_module.os.replace
        calls = 0

        def fail_manifest_rename(
            source: str | os.PathLike[str], destination: str | os.PathLike[str]
        ) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated interrupted manifest rename")
            real_replace(source, destination)

        monkeypatch.setattr(snapshot_module.os, "replace", fail_manifest_rename)
        expected = "interrupted manifest rename"

    with pytest.raises(OSError, match=expected):
        store.finalize(identity, snapshot)

    if failure == "manifest_rename":
        assert arrow_path.is_file()
        assert not manifest_path.exists()
    else:
        assert not arrow_path.exists()
        assert not manifest_path.exists()
    assert _temporary_files(tmp_path) == ()

    with pytest.raises(ValueError, match="missing or incomplete"):
        ArrowDatasetSnapshotStore(tmp_path).load(identity)


def test_restart_rejects_corrupt_manifest_and_arrow_hash(
    tmp_path: Path,
) -> None:
    snapshot, identity = _snapshot_and_identity()
    store = ArrowDatasetSnapshotStore(tmp_path)
    store.finalize(identity, snapshot)
    arrow_path, manifest_path = _snapshot_paths(tmp_path, identity)
    original_manifest = manifest_path.read_bytes()
    original_arrow = arrow_path.read_bytes()

    manifest = json.loads(manifest_path.read_bytes())
    manifest["arrow_sha256"] = "0" * 64
    manifest_path.write_bytes(json.dumps(manifest).encode("utf-8"))
    with pytest.raises(ValueError, match="Arrow hash does not match manifest"):
        ArrowDatasetSnapshotStore(tmp_path).load_identity(identity.key)

    manifest_path.write_bytes(b"{not-valid-json")
    with pytest.raises(ValueError, match="manifest is invalid"):
        ArrowDatasetSnapshotStore(tmp_path).load(identity)

    manifest_path.write_bytes(original_manifest)
    arrow_path.write_bytes(original_arrow[:-1] + bytes([original_arrow[-1] ^ 1]))
    with pytest.raises(ValueError, match="Arrow hash does not match manifest"):
        ArrowDatasetSnapshotStore(tmp_path).load(identity)


def test_restart_reads_only_durable_snapshot_not_live_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, identity = _snapshot_and_identity()
    ArrowDatasetSnapshotStore(tmp_path).finalize(identity, snapshot)

    def rematerialization_is_forbidden(_snapshot: FeatureSnapshot) -> object:
        raise AssertionError("restart attempted to reread live data")

    monkeypatch.setattr(snapshot_module, "_arrow_table", rematerialization_is_forbidden)
    restarted = ArrowDatasetSnapshotStore(tmp_path)

    assert restarted.load_identity(identity.key) == identity
    assert restarted.load(identity) == snapshot
