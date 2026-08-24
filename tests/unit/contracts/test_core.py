from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.contracts import (
    ERROR_SCHEMA_VERSION,
    INVOCATION_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    FeatureSelectionLineage,
    InvocationOperation,
    LatestInvocation,
    ModelIdentity,
    PredictionMode,
    RegimeError,
    RegimeInvocationResponse,
    RegimePrediction,
    ReplayInvocation,
    SourceLineage,
)

NOW = datetime(2026, 8, 24, 8, 0, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64


def source() -> SourceLineage:
    return SourceLineage(
        source_dataset="regime_loader.regime_features_daily",
        source_build_id="build-1",
        data_sha256=HASH_A,
        schema_version=1,
        feature_version=1,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=NOW,
    )


def model() -> ModelIdentity:
    return ModelIdentity(
        profile_id="xetra",
        profile_config_version=1,
        model_name="regime-xetra",
        model_version="7",
        feature_contract_hash=HASH_B,
        feature_order=("f1", "f2"),
        inference_origin_timestamp=NOW - timedelta(days=100),
        trained_through_timestamp=NOW - timedelta(days=2),
        terminal_filtered_probabilities=(0.4, 0.6),
        model_alias="champion",
        alias_resolved_at_utc=NOW,
    )


def prediction() -> RegimePrediction:
    return RegimePrediction(
        schema_version=PREDICTION_SCHEMA_VERSION,
        timestamp=NOW,
        state_ids=("state_0", "state_1"),
        state_probabilities=(0.25, 0.75),
        dominant_state="state_1",
        confidence=0.75,
        entropy=0.562335,
    )


def test_contracts_are_frozen_and_keep_profile_version_separate() -> None:
    identity = model()
    assert identity.profile_id == "xetra"
    assert identity.profile_config_version == 1
    with pytest.raises(FrozenInstanceError):
        identity.profile_id = "other"  # type: ignore[misc]


def test_source_contract_rejects_wrong_time_semantics_and_non_utc() -> None:
    with pytest.raises(ValueError, match="data_time_semantics"):
        SourceLineage(
            source_dataset="dataset",
            source_build_id="build",
            data_sha256=HASH_A,
            schema_version=1,
            feature_version=1,
            source_table="schema.table",
            synced_at_utc=NOW,
            data_time_semantics="release_time",
        )
    with pytest.raises(ValueError, match="UTC"):
        SourceLineage(
            source_dataset="dataset",
            source_build_id="build",
            data_sha256=HASH_A,
            schema_version=1,
            feature_version=1,
            source_table="schema.table",
            synced_at_utc=NOW.replace(tzinfo=None),
        )


def test_selection_hashes_are_distinct_and_validated_independently() -> None:
    lineage = FeatureSelectionLineage(HASH_A, HASH_B)
    assert lineage.feature_selection_definition_hash != lineage.feature_selection_execution_hash
    with pytest.raises(ValueError, match="definition"):
        FeatureSelectionLineage("not-a-hash", HASH_B)


def test_model_contract_rejects_reduced_covariance_and_bad_filter_state() -> None:
    kwargs = dict(
        profile_id="xetra",
        profile_config_version=1,
        model_name="regime-xetra",
        model_version="1",
        feature_contract_hash=HASH_A,
        feature_order=("f1",),
        inference_origin_timestamp=NOW - timedelta(days=2),
        trained_through_timestamp=NOW - timedelta(days=1),
        terminal_filtered_probabilities=(1.0,),
    )
    with pytest.raises(ValueError, match="full"):
        ModelIdentity(**kwargs, covariance_type="diag")
    with pytest.raises(ValueError, match="sum to one"):
        ModelIdentity(**(kwargs | {"terminal_filtered_probabilities": (0.2, 0.2)}))


def test_invocation_time_contracts_are_exclusive_types() -> None:
    latest = LatestInvocation(as_of=NOW, model_version="9")
    replay = ReplayInvocation(start=NOW - timedelta(days=3), end=NOW)
    assert latest.operation is InvocationOperation.LATEST
    assert replay.operation is InvocationOperation.REPLAY
    with pytest.raises(ValueError, match="start"):
        ReplayInvocation(start=NOW, end=NOW - timedelta(seconds=1))


def test_prediction_and_response_validate_modes_and_cardinality() -> None:
    item = prediction()
    response = RegimeInvocationResponse(
        schema_version=INVOCATION_SCHEMA_VERSION,
        request_id="req-1",
        profile_id="xetra",
        operation=InvocationOperation.LATEST,
        prediction_mode=PredictionMode.FIXED_MODEL_LATEST,
        requested_time_fields=(("as_of", NOW.isoformat()),),
        model=model(),
        source=source(),
        selection=FeatureSelectionLineage(HASH_A, HASH_B),
        warmup_observation_count=10,
        skipped_incomplete_row_count=2,
        predictions=(item,),
    )
    assert response.predictions == (item,)
    with pytest.raises(ValueError, match="mode"):
        replace(response, prediction_mode=PredictionMode.FIXED_MODEL_REPLAY)


def test_prediction_probability_shape_and_error_contract() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        RegimePrediction(
            schema_version=PREDICTION_SCHEMA_VERSION,
            timestamp=NOW,
            state_ids=("state_0", "state_1"),
            state_probabilities=(1.0,),
            dominant_state="state_0",
            confidence=1.0,
            entropy=0.0,
        )
    error = RegimeError(
        schema_version=ERROR_SCHEMA_VERSION,
        request_id="req-2",
        error_code="no_complete_observations",
        message="No complete observations were available.",
        retryable=False,
    )
    assert error.retryable is False
