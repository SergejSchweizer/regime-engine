from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import inf, nan

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

NOW = datetime(2026, 8, 24, 8, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64


def source(**changes: object) -> SourceLineage:
    values: dict[str, object] = {
        "source_dataset": "regime_loader.regime_features_daily",
        "source_build_id": "build-1",
        "data_sha256": HASH_A,
        "schema_version": 1,
        "feature_version": 1,
        "source_table": "regime_loader.regime_features_daily",
        "synced_at_utc": NOW,
    }
    values.update(changes)
    return SourceLineage(**values)  # type: ignore[arg-type]


def model(**changes: object) -> ModelIdentity:
    values: dict[str, object] = {
        "profile_id": "xetra",
        "profile_config_version": 1,
        "model_name": "regime-xetra",
        "model_version": "7",
        "feature_contract_hash": HASH_B,
        "feature_order": ("f1", "f2"),
        "inference_origin_timestamp": NOW - timedelta(days=100),
        "trained_through_timestamp": NOW - timedelta(days=2),
        "terminal_filtered_probabilities": (0.4, 0.6),
        "model_alias": "champion",
        "alias_resolved_at_utc": NOW,
    }
    values.update(changes)
    return ModelIdentity(**values)  # type: ignore[arg-type]


def prediction(**changes: object) -> RegimePrediction:
    values: dict[str, object] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "timestamp": NOW,
        "state_ids": ("state_0", "state_1"),
        "state_probabilities": (0.25, 0.75),
        "dominant_state": "state_1",
        "confidence": 0.75,
        "entropy": 0.562335,
    }
    values.update(changes)
    return RegimePrediction(**values)  # type: ignore[arg-type]


def response(**changes: object) -> RegimeInvocationResponse:
    values: dict[str, object] = {
        "schema_version": INVOCATION_SCHEMA_VERSION,
        "request_id": "req-1",
        "profile_id": "xetra",
        "operation": InvocationOperation.LATEST,
        "prediction_mode": PredictionMode.FIXED_MODEL_LATEST,
        "requested_time_fields": (("as_of", NOW.isoformat()),),
        "model": model(),
        "source": source(),
        "selection": FeatureSelectionLineage(HASH_A, HASH_B),
        "warmup_observation_count": 1,
        "skipped_incomplete_row_count": 0,
        "predictions": (prediction(),),
    }
    values.update(changes)
    return RegimeInvocationResponse(**values)  # type: ignore[arg-type]


def test_source_identity_hash_versions_and_bounds_validation() -> None:
    with pytest.raises(ValueError, match="source_dataset"):
        source(source_dataset=" ")
    with pytest.raises(ValueError, match="SHA-256"):
        source(data_sha256="g" * 64)
    with pytest.raises(ValueError, match="versions"):
        source(schema_version=0)

    bounded = source(
        row_count=2,
        min_timestamp=NOW - timedelta(days=1),
        max_timestamp=NOW,
    )
    assert bounded.row_count == 2

    with pytest.raises(ValueError, match="supplied together"):
        source(row_count=2)
    with pytest.raises(ValueError, match="cannot be negative"):
        source(row_count=-1, min_timestamp=NOW, max_timestamp=NOW)
    with pytest.raises(ValueError, match=r"min_timestamp.*UTC"):
        source(row_count=1, min_timestamp=NOW.replace(tzinfo=None), max_timestamp=NOW)
    with pytest.raises(ValueError, match=r"max_timestamp.*UTC"):
        source(row_count=1, min_timestamp=NOW, max_timestamp=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="inverted"):
        source(
            row_count=1,
            min_timestamp=NOW,
            max_timestamp=NOW - timedelta(seconds=1),
        )


def test_optional_selection_hashes_and_bad_execution_hash() -> None:
    assert FeatureSelectionLineage(None, None).feature_selection_definition_hash is None
    with pytest.raises(ValueError, match="execution"):
        FeatureSelectionLineage(HASH_A, "bad")


def test_model_identity_rejects_invalid_identity_and_temporal_fields() -> None:
    with pytest.raises(ValueError, match="profile_id"):
        model(profile_id="")
    with pytest.raises(ValueError, match="profile_config_version"):
        model(profile_config_version=0)
    with pytest.raises(ValueError, match="model_name"):
        model(model_name=" bad ")
    with pytest.raises(ValueError, match="feature_contract_hash"):
        model(feature_contract_hash="bad")
    with pytest.raises(ValueError, match="feature_order"):
        model(feature_order=())
    with pytest.raises(ValueError, match="feature_order"):
        model(feature_order=("f1", "f1"))
    with pytest.raises(ValueError, match="origin"):
        model(
            inference_origin_timestamp=NOW,
            trained_through_timestamp=NOW - timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="model_alias"):
        model(model_alias="")
    with pytest.raises(ValueError, match=r"alias_resolved_at_utc.*UTC"):
        model(alias_resolved_at_utc=NOW.replace(tzinfo=None))


def test_probability_contract_rejects_empty_negative_nonfinite_and_wrong_sum() -> None:
    for probabilities, match in (
        ((), "cannot be empty"),
        ((-0.1, 1.1), "non-negative"),
        ((inf, 0.0), "finite"),
        ((nan, 1.0), "finite"),
        ((0.2, 0.2), "sum to one"),
    ):
        with pytest.raises(ValueError, match=match):
            model(terminal_filtered_probabilities=probabilities)


def test_prediction_validates_schema_timestamp_states_and_diagnostics() -> None:
    with pytest.raises(ValueError, match="prediction schema"):
        prediction(schema_version="RegimePrediction.v0")
    with pytest.raises(ValueError, match=r"timestamp.*UTC"):
        prediction(timestamp=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="state_ids"):
        prediction(state_ids=(), state_probabilities=())
    with pytest.raises(ValueError, match="state_ids"):
        prediction(state_ids=("state_0", "state_0"), state_probabilities=(0.5, 0.5))
    with pytest.raises(ValueError, match="dominant_state"):
        prediction(dominant_state="state_9")
    for confidence in (-0.1, 1.1, nan):
        with pytest.raises(ValueError, match="confidence"):
            prediction(confidence=confidence)
    for entropy in (-0.1, nan):
        with pytest.raises(ValueError, match="entropy"):
            prediction(entropy=entropy)


def test_latest_and_replay_validate_operations_utc_and_versions() -> None:
    with pytest.raises(ValueError, match="operation"):
        LatestInvocation(operation=InvocationOperation.REPLAY)
    with pytest.raises(ValueError, match=r"as_of.*UTC"):
        LatestInvocation(as_of=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="model_version"):
        LatestInvocation(model_version=" ")

    with pytest.raises(ValueError, match="operation"):
        ReplayInvocation(start=NOW, end=NOW, operation=InvocationOperation.LATEST)
    with pytest.raises(ValueError, match=r"start.*UTC"):
        ReplayInvocation(start=NOW.replace(tzinfo=None), end=NOW)
    with pytest.raises(ValueError, match=r"end.*UTC"):
        ReplayInvocation(start=NOW, end=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="model_version"):
        ReplayInvocation(start=NOW, end=NOW, model_version="")


def test_response_contract_covers_success_and_all_hard_validation_paths() -> None:
    replay = response(
        operation=InvocationOperation.REPLAY,
        prediction_mode=PredictionMode.FIXED_MODEL_REPLAY,
    )
    assert replay.operation is InvocationOperation.REPLAY

    with pytest.raises(ValueError, match="invocation schema"):
        response(schema_version="RegimeInvocationResponse.v0")
    with pytest.raises(ValueError, match="request_id"):
        response(request_id="")
    with pytest.raises(ValueError, match="profile_id"):
        response(profile_id=" ")
    with pytest.raises(ValueError, match="match model"):
        response(profile_id="other")
    with pytest.raises(ValueError, match="observation counts"):
        response(warmup_observation_count=-1)
    with pytest.raises(ValueError, match="at least one prediction"):
        response(predictions=())
    with pytest.raises(ValueError, match="exactly one"):
        response(predictions=(prediction(), prediction()))
    with pytest.raises(ValueError, match="mismatch"):
        replace(replay, prediction_mode=PredictionMode.FIXED_MODEL_LATEST)


def test_error_schema_and_text_validation_paths() -> None:
    valid = RegimeError(
        schema_version=ERROR_SCHEMA_VERSION,
        request_id="req-2",
        error_code="bad_request",
        message="safe",
        retryable=False,
    )
    assert valid.details == ()
    with pytest.raises(ValueError, match="error schema"):
        replace(valid, schema_version="RegimeError.v0")
    for field_name in ("request_id", "error_code", "message"):
        with pytest.raises(ValueError, match=field_name):
            replace(valid, **{field_name: ""})
