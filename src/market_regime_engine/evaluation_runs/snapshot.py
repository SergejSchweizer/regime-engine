"""Crash-safe immutable PyArrow IPC dataset snapshots."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.ipc as ipc  # type: ignore[import-untyped]

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation_runs.contracts import (
    DatasetSnapshotIdentity,
    canonical_json,
)
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
    materialized_feature_data_hash,
)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _lineage_payload(lineage: SourceLineage) -> dict[str, object]:
    return {
        "source_dataset": lineage.source_dataset,
        "source_build_id": lineage.source_build_id,
        "data_sha256": lineage.data_sha256,
        "schema_version": lineage.schema_version,
        "feature_version": lineage.feature_version,
        "source_table": lineage.source_table,
        "synced_at_utc": lineage.synced_at_utc.isoformat(),
        "data_time_semantics": lineage.data_time_semantics,
        "row_count": lineage.row_count,
        "min_timestamp": (
            None if lineage.min_timestamp is None else lineage.min_timestamp.isoformat()
        ),
        "max_timestamp": (
            None if lineage.max_timestamp is None else lineage.max_timestamp.isoformat()
        ),
    }


def _lineage_from_payload(payload: object) -> SourceLineage:
    if not isinstance(payload, dict):
        raise ValueError("dataset snapshot lineage is missing")
    try:
        return SourceLineage(
            source_dataset=str(payload["source_dataset"]),
            source_build_id=str(payload["source_build_id"]),
            data_sha256=str(payload["data_sha256"]),
            schema_version=int(payload["schema_version"]),
            feature_version=int(payload["feature_version"]),
            source_table=str(payload["source_table"]),
            synced_at_utc=_parse_timestamp(payload["synced_at_utc"], "synced_at_utc"),
            data_time_semantics=str(payload["data_time_semantics"]),
            row_count=int(payload["row_count"]),
            min_timestamp=_parse_timestamp(payload["min_timestamp"], "min_timestamp"),
            max_timestamp=_parse_timestamp(payload["max_timestamp"], "max_timestamp"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("dataset snapshot lineage is invalid") from exc


def _arrow_table(snapshot: FeatureSnapshot) -> pa.Table:
    columns: dict[str, pa.Array] = {
        "timestamp_m1": pa.array(
            [row.timestamp for row in snapshot.rows],
            type=pa.timestamp("us", tz="UTC"),
        )
    }
    for index, feature in enumerate(snapshot.feature_names):
        columns[feature] = pa.array(
            [row.values[index] for row in snapshot.rows],
            type=pa.float64(),
        )
    return pa.table(columns)


def _write_arrow(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            with ipc.new_file(handle, table.schema) as writer:
                writer.write_table(table)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return parsed


class ArrowDatasetSnapshotStore:
    """Persist and verify one immutable Arrow IPC snapshot per dataset key."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _paths(self, identity: DatasetSnapshotIdentity) -> tuple[Path, Path]:
        directory = self._root / identity.key
        return directory / "snapshot.arrow", directory / "manifest.json"

    @staticmethod
    def _manifest(
        identity: DatasetSnapshotIdentity,
        snapshot: FeatureSnapshot,
        table: pa.Table,
        arrow_sha256: str,
        catalog: FeatureCatalogSnapshot | None,
    ) -> bytes:
        schema_hex = table.schema.serialize().to_pybytes().hex()
        payload = {
            "format_version": 1,
            "dataset_snapshot_key": identity.key,
            "dataset_snapshot_identity": identity.as_dict(),
            "arrow_sha256": arrow_sha256,
            "arrow_schema_hex": schema_hex,
            "feature_names": list(snapshot.feature_names),
            "row_count": len(snapshot.rows),
            "first_timestamp": (
                None if not snapshot.rows else snapshot.rows[0].timestamp.isoformat()
            ),
            "last_timestamp": (
                None if not snapshot.rows else snapshot.rows[-1].timestamp.isoformat()
            ),
            "lineage": _lineage_payload(snapshot.lineage),
            "skipped_incomplete_row_count": snapshot.skipped_incomplete_row_count,
            "materialized_feature_data_sha256": snapshot.materialized_feature_data_sha256,
            "catalog": (
                None
                if catalog is None
                else {
                    "catalog_hash": catalog.catalog_hash,
                    "timestamp_column": catalog.timestamp_column,
                    "entries": [
                        {
                            "feature_name": entry.feature_name,
                            "canonical_ordinal": entry.canonical_ordinal,
                            "postgres_type": entry.postgres_type,
                            "schema_name": entry.schema_name,
                            "relation_name": entry.relation_name,
                            "relation_kind": entry.relation_kind,
                            "ordinal_position": entry.ordinal_position,
                        }
                        for entry in catalog.entries
                    ],
                }
            ),
        }
        return canonical_json(payload)

    def finalize(
        self,
        identity: DatasetSnapshotIdentity,
        snapshot: FeatureSnapshot,
        *,
        catalog: FeatureCatalogSnapshot | None = None,
    ) -> None:
        lineage = snapshot.lineage
        if any(
            (
                value != expected
                for value, expected in (
                    (lineage.source_dataset, identity.source_dataset),
                    (lineage.source_build_id, identity.source_build_id),
                    (lineage.data_sha256, identity.data_sha256),
                    (lineage.schema_version, identity.schema_version),
                    (lineage.feature_version, identity.feature_version),
                    (lineage.data_time_semantics, identity.data_time_semantics),
                    (lineage.row_count, identity.row_count),
                    (lineage.min_timestamp, identity.min_timestamp),
                    (lineage.max_timestamp, identity.max_timestamp),
                    (lineage.source_table, identity.source_table),
                )
            )
        ):
            raise ValueError("snapshot lineage does not match dataset identity")
        if snapshot.materialized_feature_data_sha256 != identity.materialized_feature_data_sha256:
            raise ValueError("snapshot matrix hash does not match dataset identity")
        if len(snapshot.rows) != identity.materialized_row_count:
            raise ValueError("snapshot row count does not match dataset identity")
        if snapshot.min_timestamp != identity.materialized_min_timestamp:
            raise ValueError("snapshot minimum timestamp does not match dataset identity")
        if snapshot.max_timestamp != identity.materialized_max_timestamp:
            raise ValueError("snapshot maximum timestamp does not match dataset identity")
        if catalog is not None and DatasetSnapshotIdentity.from_catalog(catalog) != identity:
            raise ValueError("snapshot catalog does not match dataset identity")
        table = _arrow_table(snapshot)
        arrow_path, manifest_path = self._paths(identity)
        if arrow_path.exists() or manifest_path.exists():
            if not arrow_path.is_file() or not manifest_path.is_file():
                raise ValueError("dataset snapshot is incomplete")
            self._read_arrow_bytes(snapshot, arrow_path)
            expected_manifest = self._manifest(
                identity,
                snapshot,
                table,
                sha256(arrow_path.read_bytes()).hexdigest(),
                catalog,
            )
            if manifest_path.read_bytes() != expected_manifest:
                raise ValueError("dataset snapshot manifest is immutable")
            self.load(identity)
            return
        _write_arrow(arrow_path, table)
        arrow_sha256 = sha256(arrow_path.read_bytes()).hexdigest()
        _atomic_write(
            manifest_path,
            self._manifest(identity, snapshot, table, arrow_sha256, catalog),
        )

    def load_identity(self, dataset_snapshot_key: str) -> DatasetSnapshotIdentity:
        """Load the immutable dataset identity without consulting the source."""

        if len(dataset_snapshot_key) != 64:
            raise ValueError("dataset_snapshot_key must be a SHA-256 digest")
        manifest_path = self._root / dataset_snapshot_key / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("dataset snapshot is missing or incomplete")
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("dataset snapshot manifest is invalid") from exc
        if not isinstance(manifest, dict):
            raise ValueError("dataset snapshot manifest must be an object")
        raw_identity = manifest.get("dataset_snapshot_identity")
        if not isinstance(raw_identity, dict):
            raise ValueError("dataset snapshot identity is missing from manifest")
        identity = DatasetSnapshotIdentity.from_dict(raw_identity)
        if identity.key != dataset_snapshot_key:
            raise ValueError("dataset snapshot identity hash does not match requested key")
        if manifest.get("dataset_snapshot_key") != dataset_snapshot_key:
            raise ValueError("dataset snapshot key does not match manifest")
        return identity

    @staticmethod
    def _read_arrow_bytes(snapshot: FeatureSnapshot, path: Path) -> bytes:
        expected = _arrow_table(snapshot)
        actual = ipc.open_file(path).read_all()
        if not actual.equals(expected):
            raise ValueError("dataset snapshot rows differ from immutable snapshot")
        return path.read_bytes()

    def load(self, identity: DatasetSnapshotIdentity) -> FeatureSnapshot:
        arrow_path, manifest_path = self._paths(identity)
        if not arrow_path.is_file() or not manifest_path.is_file():
            raise ValueError("dataset snapshot is missing or incomplete")
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("dataset snapshot manifest is invalid") from exc
        if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
            raise ValueError("unsupported dataset snapshot manifest")
        if manifest.get("dataset_snapshot_key") != identity.key:
            raise ValueError("dataset snapshot key does not match manifest")
        raw_identity = manifest.get("dataset_snapshot_identity")
        if not isinstance(raw_identity, dict):
            raise ValueError("dataset snapshot identity is missing from manifest")
        if DatasetSnapshotIdentity.from_dict(raw_identity) != identity:
            raise ValueError("dataset snapshot identity does not match manifest")
        arrow_bytes = arrow_path.read_bytes()
        if sha256(arrow_bytes).hexdigest() != manifest.get("arrow_sha256"):
            raise ValueError("dataset snapshot Arrow hash does not match manifest")
        try:
            table = ipc.open_file(arrow_path).read_all()
        except (OSError, pa.ArrowException) as exc:
            raise ValueError("dataset snapshot Arrow file is invalid") from exc
        schema_hex = manifest.get("arrow_schema_hex")
        if not isinstance(schema_hex, str) or (
            table.schema.serialize().to_pybytes().hex() != schema_hex
        ):
            raise ValueError("dataset snapshot Arrow schema does not match manifest")
        feature_names = manifest.get("feature_names")
        if not isinstance(feature_names, list) or any(
            not isinstance(name, str) for name in feature_names
        ):
            raise ValueError("dataset snapshot feature names are invalid")
        if table.column_names != ["timestamp_m1", *feature_names]:
            raise ValueError("dataset snapshot Arrow columns do not match manifest")
        if table.num_rows != identity.materialized_row_count or table.num_rows != manifest.get(
            "row_count"
        ):
            raise ValueError("dataset snapshot row count does not match identity")
        timestamps = table["timestamp_m1"].to_pylist()
        rows = tuple(
            FeatureRow(
                _parse_timestamp(timestamp.isoformat(), "snapshot timestamp"),
                tuple(
                    None if (value := table[name][row_index].as_py()) is None else float(value)
                    for name in feature_names
                ),
            )
            for row_index, timestamp in enumerate(timestamps)
        )
        if rows and (
            rows[0].timestamp != identity.materialized_min_timestamp
            or rows[-1].timestamp != identity.materialized_max_timestamp
        ):
            raise ValueError("dataset snapshot timestamp bounds do not match identity")
        if manifest.get("first_timestamp") != (
            None if not rows else rows[0].timestamp.isoformat()
        ) or manifest.get("last_timestamp") != (
            None if not rows else rows[-1].timestamp.isoformat()
        ):
            raise ValueError("dataset snapshot manifest timestamp bounds are invalid")
        lineage = _lineage_from_payload(manifest.get("lineage"))
        if any(
            (
                value != expected
                for value, expected in (
                    (lineage.source_dataset, identity.source_dataset),
                    (lineage.source_build_id, identity.source_build_id),
                    (lineage.data_sha256, identity.data_sha256),
                    (lineage.schema_version, identity.schema_version),
                    (lineage.feature_version, identity.feature_version),
                    (lineage.data_time_semantics, identity.data_time_semantics),
                    (lineage.row_count, identity.row_count),
                    (lineage.min_timestamp, identity.min_timestamp),
                    (lineage.max_timestamp, identity.max_timestamp),
                    (lineage.source_table, identity.source_table),
                )
            )
        ):
            raise ValueError("dataset snapshot lineage does not match identity")
        matrix_hash = materialized_feature_data_hash(tuple(feature_names), rows)
        if matrix_hash != identity.materialized_feature_data_sha256:
            raise ValueError("dataset snapshot matrix hash does not match identity")
        skipped = manifest.get("skipped_incomplete_row_count", 0)
        if not isinstance(skipped, int) or skipped < 0:
            raise ValueError("dataset snapshot skipped row count is invalid")
        return FeatureSnapshot(
            lineage=lineage,
            feature_names=tuple(feature_names),
            rows=rows,
            skipped_incomplete_row_count=skipped,
            materialized_feature_data_sha256=matrix_hash,
        )

    def load_catalog(self, identity: DatasetSnapshotIdentity) -> FeatureCatalogSnapshot:
        """Load the complete catalog captured with an immutable snapshot."""

        _arrow_path, manifest_path = self._paths(identity)
        if not manifest_path.is_file():
            raise ValueError("dataset snapshot is missing or incomplete")
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("dataset snapshot manifest is invalid") from exc
        if not isinstance(manifest, dict):
            raise ValueError("dataset snapshot manifest must be an object")
        raw_catalog = manifest.get("catalog")
        if not isinstance(raw_catalog, dict):
            raise ValueError("dataset snapshot catalog is missing")
        raw_entries = raw_catalog.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("dataset snapshot catalog entries are invalid")
        try:
            if any(not isinstance(item, dict) for item in raw_entries):
                raise ValueError("catalog entries must be objects")
            entries = tuple(
                FeatureCatalogEntry(
                    feature_name=str(item["feature_name"]),
                    canonical_ordinal=int(item["canonical_ordinal"]),
                    postgres_type=str(item["postgres_type"]),
                    schema_name=str(item["schema_name"]),
                    relation_name=str(item["relation_name"]),
                    relation_kind=str(item["relation_kind"]),
                    ordinal_position=int(item["ordinal_position"]),
                )
                for item in raw_entries
            )
            snapshot = self.load(identity)
            catalog = FeatureCatalogSnapshot.from_entries(
                snapshot.lineage,
                str(raw_catalog["timestamp_column"]),
                entries,
            ).with_materialization(snapshot)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("dataset snapshot catalog is invalid") from exc
        if catalog.catalog_hash != identity.source_catalog_hash:
            raise ValueError("dataset snapshot catalog hash does not match identity")
        if raw_catalog.get("catalog_hash") != catalog.catalog_hash:
            raise ValueError("dataset snapshot catalog manifest hash does not match catalog")
        return catalog


__all__ = ["ArrowDatasetSnapshotStore"]
