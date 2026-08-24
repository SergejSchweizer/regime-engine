from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from market_regime_engine.contracts import PREDICTION_SCHEMA_VERSION, PredictionMode
from market_regime_engine.predictions.query import OOSQuery, query_oos_build
from market_regime_engine.predictions.store import PredictionStore


def timestamp(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def oos_rows() -> list[dict[str, object]]:
    return [
        {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "timestamp": timestamp(day),
            "state_ids": ["state_0", "state_1"],
            "state_probabilities": [0.8, 0.2],
            "dominant_state": "state_0",
            "confidence": 0.8,
            "entropy": 0.5004024235381879,
            "prediction_mode": PredictionMode.WALK_FORWARD_OOS.value,
            "profile_id": "xetra",
            "profile_config_version": 1,
            "candidate_id": "gaussian_hmm_k2_full",
            "fold_id": "fold_001",
            "evaluation_plan_hash": "a" * 64,
            "feature_selection_definition_hash": "b" * 64,
            "feature_selection_execution_hash": "c" * 64,
        }
        for day in (1, 2, 3)
    ]


def publish(
    store: PredictionStore,
    *,
    build_id: str = "oos-build",
    mode: PredictionMode = PredictionMode.WALK_FORWARD_OOS,
    rows: list[dict[str, object]] | None = None,
) -> None:
    store.publish(
        build_id=build_id,
        profile_id="xetra",
        prediction_mode=mode,
        rows=oos_rows() if rows is None else rows,
        source_build_id="source-build",
        source_data_sha256="d" * 64,
        source_schema_version=1,
        source_feature_version=1,
        source_synced_at_utc=datetime(2026, 1, 4, tzinfo=UTC),
        feature_contract_hash="e" * 64,
        feature_selection_definition_hash="b" * 64,
        feature_selection_execution_hash="c" * 64,
        created_at_utc=datetime(2026, 1, 4, tzinfo=UTC),
    )


def test_explicit_build_query_returns_complete_ordered_oos_build(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    publish(store)
    result = query_oos_build(store, OOSQuery(profile_id="xetra", build_id="oos-build"))
    assert result.build_id == "oos-build"
    assert result.profile_id == "xetra"
    assert result.prediction_mode is PredictionMode.WALK_FORWARD_OOS
    assert [row.prediction.timestamp for row in result.rows] == [
        timestamp(1),
        timestamp(2),
        timestamp(3),
    ]
    assert all(row.candidate_id == "gaussian_hmm_k2_full" for row in result.rows)
    assert all(row.fold_id == "fold_001" for row in result.rows)


def test_bounded_query_is_inclusive_and_does_not_resolve_latest(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    publish(store, build_id="older-build")
    publish(store, build_id="newer-build")
    result = query_oos_build(
        store,
        OOSQuery(
            profile_id="xetra",
            build_id="older-build",
            start=timestamp(2),
            end=timestamp(2),
        ),
    )
    assert result.build_id == "older-build"
    assert tuple(row.prediction.timestamp for row in result.rows) == (timestamp(2),)
    empty = query_oos_build(
        store,
        OOSQuery(
            profile_id="xetra",
            build_id="older-build",
            start=timestamp(3) + timedelta(seconds=1),
        ),
    )
    assert empty.rows == ()


def test_query_validation_requires_explicit_identity_and_utc_bounds() -> None:
    with pytest.raises(ValueError, match="profile_id"):
        OOSQuery(profile_id="", build_id="build")
    with pytest.raises(ValueError, match="build_id"):
        OOSQuery(profile_id="xetra", build_id=" build ")
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        OOSQuery(profile_id="xetra", build_id="build", start=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="start <= end"):
        OOSQuery(
            profile_id="xetra",
            build_id="build",
            start=timestamp(3),
            end=timestamp(2),
        )


def test_unknown_build_and_non_oos_build_fail_closed(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="unknown prediction build"):
        query_oos_build(store, OOSQuery(profile_id="xetra", build_id="missing"))
    publish(store, build_id="replay-build", mode=PredictionMode.FIXED_MODEL_REPLAY)
    with pytest.raises(ValueError, match="not walk_forward_oos"):
        query_oos_build(store, OOSQuery(profile_id="xetra", build_id="replay-build"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "RegimePrediction.v0", "schema version"),
        ("prediction_mode", "fixed_model_replay", "prediction_mode"),
        ("profile_id", "other", "profile_id"),
        ("dominant_state", 7, "dominant_state"),
        ("state_ids", "state_0", "state_ids"),
        ("state_probabilities", "bad", "state_probabilities"),
        ("candidate_id", "", "candidate_id"),
    ],
)
def test_corrupt_immutable_oos_rows_are_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    store = PredictionStore(tmp_path)
    rows = oos_rows()
    rows[0][field] = value
    publish(store, rows=rows)
    with pytest.raises(ValueError, match=message):
        query_oos_build(store, OOSQuery(profile_id="xetra", build_id="oos-build"))


def test_duplicate_or_decreasing_oos_timestamps_are_rejected(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    rows = oos_rows()
    rows[1]["timestamp"] = timestamp(1)
    publish(store, rows=rows)
    with pytest.raises(ValueError, match="globally unique and strictly increasing"):
        query_oos_build(store, OOSQuery(profile_id="xetra", build_id="oos-build"))
