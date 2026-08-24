from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from market_regime_engine.contracts import PredictionMode
from market_regime_engine.predictions import PredictionStore

NOW = datetime(2026, 8, 24, 8, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64


def publish(store: PredictionStore, build_id: str = "build-1") -> None:
    store.publish(
        build_id=build_id,
        profile_id="xetra",
        prediction_mode=PredictionMode.WALK_FORWARD_OOS,
        rows=[
            {"timestamp": NOW, "state_0": 0.25, "state_1": 0.75},
            {"timestamp": NOW.replace(day=25), "state_0": 0.20, "state_1": 0.80},
        ],
        source_build_id="source-7",
        source_data_sha256=HASH_A,
        source_schema_version=1,
        source_feature_version=1,
        source_synced_at_utc=NOW,
        feature_contract_hash=HASH_B,
        feature_selection_definition_hash=HASH_A,
        feature_selection_execution_hash=HASH_B,
        created_at_utc=NOW,
    )


def test_publish_is_atomic_explicit_and_immutable(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    publish(store)
    manifest = store.load_manifest("xetra", "build-1")
    assert manifest.prediction_mode is PredictionMode.WALK_FORWARD_OOS
    assert manifest.row_count == 2
    assert store.read_table("xetra", "build-1").num_rows == 2
    with pytest.raises(FileExistsError):
        publish(store)


def test_research_read_requires_explicit_build_id(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    publish(store, "immutable-a")
    with pytest.raises(FileNotFoundError, match="unknown prediction build"):
        store.load_manifest("xetra", "latest")


def test_checksum_detects_mutation(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    publish(store)
    parquet = tmp_path / "xetra" / "build-1" / "predictions.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        store.read_table("xetra", "build-1")
