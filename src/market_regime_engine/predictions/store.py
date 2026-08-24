"""Immutable atomic Parquet prediction builds with explicit lineage."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from market_regime_engine.contracts import DATA_TIME_SEMANTICS, PredictionMode


@dataclass(frozen=True, slots=True)
class PredictionBuildManifest:
    build_id: str
    profile_id: str
    prediction_mode: PredictionMode
    source_build_id: str
    source_data_sha256: str
    source_schema_version: int
    source_feature_version: int
    source_synced_at_utc: str
    data_time_semantics: str
    feature_contract_hash: str
    feature_selection_definition_hash: str | None
    feature_selection_execution_hash: str | None
    parquet_sha256: str
    row_count: int
    created_at_utc: str

    def __post_init__(self) -> None:
        identity_fields = ("build_id", "profile_id", "source_build_id", "feature_contract_hash")
        for field_name in identity_fields:
            value = getattr(self, field_name)
            if not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a non-empty trimmed string")
        if self.data_time_semantics != DATA_TIME_SEMANTICS:
            raise ValueError("unsupported data_time_semantics")
        if self.row_count < 0:
            raise ValueError("row_count cannot be negative")
        for field_name in ("source_data_sha256", "parquet_sha256"):
            value = getattr(self, field_name)
            if len(value) != 64:
                raise ValueError(f"{field_name} must be a SHA-256 digest")


class PredictionStore:
    """Filesystem store where each build ID is write-once and explicit on read."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        *,
        build_id: str,
        profile_id: str,
        prediction_mode: PredictionMode,
        rows: list[dict[str, Any]],
        source_build_id: str,
        source_data_sha256: str,
        source_schema_version: int,
        source_feature_version: int,
        source_synced_at_utc: datetime,
        feature_contract_hash: str,
        feature_selection_definition_hash: str | None,
        feature_selection_execution_hash: str | None,
        created_at_utc: datetime,
    ) -> PredictionBuildManifest:
        final_dir = self.root / profile_id / build_id
        if final_dir.exists():
            raise FileExistsError(f"prediction build already exists: {profile_id}/{build_id}")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=final_dir.parent))
        try:
            parquet_path = temp_dir / "predictions.parquet"
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, parquet_path, compression="zstd", version="2.6")
            parquet_digest = _sha256_file(parquet_path)
            manifest = PredictionBuildManifest(
                build_id=build_id,
                profile_id=profile_id,
                prediction_mode=prediction_mode,
                source_build_id=source_build_id,
                source_data_sha256=source_data_sha256,
                source_schema_version=source_schema_version,
                source_feature_version=source_feature_version,
                source_synced_at_utc=_utc_text(source_synced_at_utc),
                data_time_semantics=DATA_TIME_SEMANTICS,
                feature_contract_hash=feature_contract_hash,
                feature_selection_definition_hash=feature_selection_definition_hash,
                feature_selection_execution_hash=feature_selection_execution_hash,
                parquet_sha256=parquet_digest,
                row_count=table.num_rows,
                created_at_utc=_utc_text(created_at_utc),
            )
            (temp_dir / "manifest.json").write_text(
                json.dumps(_manifest_dict(manifest), sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_dir, final_dir)
            return manifest
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def load_manifest(self, profile_id: str, build_id: str) -> PredictionBuildManifest:
        path = self.root / profile_id / build_id / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"unknown prediction build: {profile_id}/{build_id}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["prediction_mode"] = PredictionMode(raw["prediction_mode"])
        return PredictionBuildManifest(**raw)

    def read_table(self, profile_id: str, build_id: str) -> pa.Table:
        manifest = self.load_manifest(profile_id, build_id)
        path = self.root / profile_id / build_id / "predictions.parquet"
        if _sha256_file(path) != manifest.parquet_sha256:
            raise ValueError("prediction Parquet checksum mismatch")
        table = pq.read_table(path)
        if table.num_rows != manifest.row_count:
            raise ValueError("prediction row count differs from immutable manifest")
        return table


def _manifest_dict(manifest: PredictionBuildManifest) -> dict[str, Any]:
    raw = asdict(manifest)
    raw["prediction_mode"] = manifest.prediction_mode.value
    return raw


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("timestamps must be timezone-aware UTC")
    return value.isoformat().replace("+00:00", "Z")
