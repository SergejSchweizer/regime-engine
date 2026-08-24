from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from market_regime_engine.contracts import PREDICTION_SCHEMA_VERSION, PredictionMode
from market_regime_engine.predictions.store import PredictionStore
from market_regime_engine.serving.oos_handler import OOSPredictionHandler


def ts(day: int) -> datetime:
    return datetime(2026, 2, day, tzinfo=UTC)


def publish(store: PredictionStore) -> None:
    rows = [
        {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "timestamp": ts(day),
            "state_ids": ["state_0", "state_1"],
            "state_probabilities": [0.25, 0.75],
            "dominant_state": "state_1",
            "confidence": 0.75,
            "entropy": 0.5623351446188083,
            "prediction_mode": "walk_forward_oos",
            "profile_id": "xetra",
            "profile_config_version": 1,
            "candidate_id": "gaussian_hmm_k2_full",
            "fold_id": "fold_001",
            "evaluation_plan_hash": "a" * 64,
            "feature_selection_definition_hash": "b" * 64,
            "feature_selection_execution_hash": "c" * 64,
        }
        for day in (1, 2)
    ]
    store.publish(
        build_id="explicit-build",
        profile_id="xetra",
        prediction_mode=PredictionMode.WALK_FORWARD_OOS,
        rows=rows,
        source_build_id="source-1",
        source_data_sha256="d" * 64,
        source_schema_version=1,
        source_feature_version=1,
        source_synced_at_utc=ts(3),
        feature_contract_hash="e" * 64,
        feature_selection_definition_hash="b" * 64,
        feature_selection_execution_hash="c" * 64,
        created_at_utc=ts(3),
    )


def test_handler_returns_explicit_bounded_build_with_full_lineage(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    publish(store)
    response = OOSPredictionHandler(store).handle(
        profile_id="xetra",
        build_id="explicit-build",
        start=ts(2),
        end=ts(2),
    )
    assert response["profile_id"] == "xetra"
    assert response["build_id"] == "explicit-build"
    assert response["prediction_mode"] == "walk_forward_oos"
    assert response["requested_start"] == "2026-02-02T00:00:00Z"
    assert response["requested_end"] == "2026-02-02T00:00:00Z"
    assert response["source_build_id"] == "source-1"
    assert response["source_data_sha256"] == "d" * 64
    assert response["source_schema_version"] == 1
    assert response["source_feature_version"] == 1
    assert response["data_time_semantics"] == "current_vintage_observation_day"
    assert response["feature_contract_hash"] == "e" * 64
    assert response["row_count"] == 1
    predictions = response["predictions"]
    assert isinstance(predictions, list)
    assert predictions == [
        {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "timestamp": "2026-02-02T00:00:00Z",
            "state_ids": ["state_0", "state_1"],
            "state_probabilities": [0.25, 0.75],
            "dominant_state": "state_1",
            "confidence": 0.75,
            "entropy": 0.5623351446188083,
            "candidate_id": "gaussian_hmm_k2_full",
            "fold_id": "fold_001",
            "evaluation_plan_hash": "a" * 64,
            "feature_selection_definition_hash": "b" * 64,
            "feature_selection_execution_hash": "c" * 64,
        }
    ]
    assert "latest" not in response
    assert "model_alias" not in response
    assert "model_version" not in response


def test_handler_allows_an_explicit_empty_slice_without_truncation(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    publish(store)
    response = OOSPredictionHandler(store).handle(
        profile_id="xetra",
        build_id="explicit-build",
        start=datetime(2027, 1, 1, tzinfo=UTC),
    )
    assert response["row_count"] == 0
    assert response["predictions"] == []
    assert response["build_id"] == "explicit-build"
