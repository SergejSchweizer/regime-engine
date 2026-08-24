from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.contracts import DATA_TIME_SEMANTICS, PredictionMode, SourceLineage
from market_regime_engine.evaluation.walk_forward import WalkForwardEvaluation, WalkForwardFoldResult
from market_regime_engine.predictions.oos_publication import publish_walk_forward_oos
from market_regime_engine.predictions.store import PredictionStore
from market_regime_engine.states.alignment import StateAlignment


def alignment() -> StateAlignment:
    return StateAlignment(
        persistent_state_ids=("state_0", "state_1"),
        persistent_to_fitted=(0, 1),
        aligned_signatures=((0.0,), (1.0,)),
        matched_rms=(0.0, 0.0),
        total_cost=0.0,
        max_drift=0.0,
        initial_alignment=True,
    )


def valid_fold() -> WalkForwardFoldResult:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return WalkForwardFoldResult(
        fold_id="fold_001",
        fold_index=1,
        valid=True,
        failure_reason=None,
        train_source_observation_count=1260,
        test_source_observation_count=63,
        train_model_observation_count=1260,
        test_model_observation_count=2,
        skipped_train_incomplete_count=0,
        skipped_test_incomplete_count=61,
        scaler_artifact=object(),  # type: ignore[arg-type]
        multistart_result=object(),  # type: ignore[arg-type]
        model_artifact=object(),  # type: ignore[arg-type]
        alignment=alignment(),
        train_log_likelihood=-100.0,
        oos_predictive_log_likelihood=-2.0,
        oos_predictive_log_likelihood_per_observation=-1.0,
        aic=250.0,
        bic=300.0,
        multistart_success_rate=1.0,
        train_hard_occupancy=(0.5, 0.5),
        train_soft_occupancy=(0.5, 0.5),
        oos_hard_occupancy=(0.5, 0.5),
        oos_soft_occupancy=(0.5, 0.5),
        max_state_signature_drift=0.0,
        mean_state_duration=1.0,
        switches_per_year=365.2425,
        oos_entropy_mean=0.5,
        oos_confidence_mean=0.7,
        oos_timestamps=(start, start + timedelta(days=1)),
        oos_filtered_probabilities=((0.8, 0.2), (0.3, 0.7)),
    )


def evaluation() -> WalkForwardEvaluation:
    return WalkForwardEvaluation(
        profile_id="xetra",
        profile_config_version=1,
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        source_build_id="source-build-1",
        feature_order=("f0", "f1"),
        feature_selection_definition_hash="b" * 64,
        feature_selection_execution_hash="c" * 64,
        evaluation_plan_hash="d" * 64,
        evaluation_cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        folds=(valid_fold(),),
    )


def source() -> SourceLineage:
    return SourceLineage(
        source_dataset="regime_loader.regime_features_daily",
        source_build_id="source-build-1",
        data_sha256="a" * 64,
        schema_version=1,
        feature_version=1,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=datetime(2026, 1, 3, tzinfo=UTC),
        data_time_semantics=DATA_TIME_SEMANTICS,
        row_count=1323,
        min_timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        max_timestamp=datetime(2026, 1, 2, tzinfo=UTC),
    )


def test_publication_is_walk_forward_oos_and_contains_regime_prediction_rows(tmp_path) -> None:
    store = PredictionStore(tmp_path)
    result = publish_walk_forward_oos(
        store,
        evaluation(),
        source(),
        feature_contract_hash="feature-contract-v1",
        created_at_utc=datetime(2026, 1, 4, tzinfo=UTC),
    )
    assert result.build_id.startswith("walk-forward-oos-")
    assert result.manifest.prediction_mode is PredictionMode.WALK_FORWARD_OOS
    assert result.manifest.source_build_id == "source-build-1"
    assert result.manifest.feature_selection_definition_hash == "b" * 64
    assert result.manifest.feature_selection_execution_hash == "c" * 64
    assert len(result.predictions) == 2
    assert result.predictions[0].dominant_state == "state_0"
    assert result.predictions[1].dominant_state == "state_1"
    table = store.read_table("xetra", result.build_id).to_pylist()
    assert {row["fold_id"] for row in table} == {"fold_001"}
    assert {row["candidate_id"] for row in table} == {"gaussian_hmm_k2_full"}
    assert {row["evaluation_plan_hash"] for row in table} == {"d" * 64}
    assert {row["prediction_mode"] for row in table} == {"walk_forward_oos"}


def test_same_inputs_are_idempotent_and_source_or_plan_changes_build_id(tmp_path) -> None:
    store = PredictionStore(tmp_path)
    kwargs = {
        "feature_contract_hash": "feature-contract-v1",
        "created_at_utc": datetime(2026, 1, 4, tzinfo=UTC),
    }
    first = publish_walk_forward_oos(store, evaluation(), source(), **kwargs)
    second = publish_walk_forward_oos(store, evaluation(), source(), **kwargs)
    assert second.build_id == first.build_id
    assert second.manifest == first.manifest

    changed = evaluation()
    object.__setattr__(changed, "evaluation_plan_hash", "e" * 64)
    third = publish_walk_forward_oos(store, changed, source(), **kwargs)
    assert third.build_id != first.build_id


def test_publication_rejects_source_mismatch_naive_time_and_no_valid_predictions(tmp_path) -> None:
    store = PredictionStore(tmp_path)
    wrong_source = source()
    object.__setattr__(wrong_source, "source_build_id", "other-build")
    with pytest.raises(ValueError, match="build identity"):
        publish_walk_forward_oos(
            store,
            evaluation(),
            wrong_source,
            feature_contract_hash="features",
            created_at_utc=datetime(2026, 1, 4, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        publish_walk_forward_oos(
            store,
            evaluation(),
            source(),
            feature_contract_hash="features",
            created_at_utc=datetime(2026, 1, 4),
        )

    invalid = valid_fold()
    object.__setattr__(invalid, "valid", False)
    object.__setattr__(invalid, "failure_reason", "invalid")
    empty_eval = evaluation()
    object.__setattr__(empty_eval, "folds", (invalid,))
    with pytest.raises(ValueError, match="at least one valid prediction"):
        publish_walk_forward_oos(
            store,
            empty_eval,
            source(),
            feature_contract_hash="features",
            created_at_utc=datetime(2026, 1, 4, tzinfo=UTC),
        )
