"""Deterministic publication of immutable walk-forward OOS prediction builds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import log
from typing import Any

from market_regime_engine.contracts import (
    PREDICTION_SCHEMA_VERSION,
    PredictionMode,
    RegimePrediction,
    SourceLineage,
)
from market_regime_engine.evaluation.walk_forward import WalkForwardEvaluation
from market_regime_engine.predictions.store import PredictionBuildManifest, PredictionStore


@dataclass(frozen=True, slots=True)
class OOSPublication:
    build_id: str
    manifest: PredictionBuildManifest
    predictions: tuple[RegimePrediction, ...]


def _entropy(probabilities: tuple[float, ...]) -> float:
    return -sum(value * log(value) for value in probabilities if value > 0.0)


def _canonical_build_id(
    evaluation: WalkForwardEvaluation,
    source: SourceLineage,
    feature_contract_hash: str,
) -> str:
    payload = {
        "candidate_id": evaluation.candidate_id,
        "evaluation_plan_hash": evaluation.evaluation_plan_hash,
        "feature_contract_hash": feature_contract_hash,
        "feature_order": list(evaluation.feature_order),
        "feature_selection_definition_hash": evaluation.feature_selection_definition_hash,
        "feature_selection_execution_hash": evaluation.feature_selection_execution_hash,
        "profile_config_version": evaluation.profile_config_version,
        "profile_id": evaluation.profile_id,
        "source_build_id": source.source_build_id,
        "source_data_sha256": source.data_sha256,
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"walk-forward-oos-{digest}"


def _rows_and_predictions(
    evaluation: WalkForwardEvaluation,
) -> tuple[list[dict[str, Any]], tuple[RegimePrediction, ...]]:
    rows: list[dict[str, Any]] = []
    predictions: list[RegimePrediction] = []
    for fold in evaluation.folds:
        if not fold.valid:
            continue
        if fold.alignment is None:
            raise ValueError("valid OOS fold is missing persistent state alignment")
        state_ids = fold.alignment.persistent_state_ids
        if len(fold.oos_timestamps) != len(fold.oos_filtered_probabilities):
            raise ValueError("OOS timestamp/probability counts differ")
        for timestamp, raw_probabilities in zip(
            fold.oos_timestamps,
            fold.oos_filtered_probabilities,
            strict=True,
        ):
            probabilities = tuple(float(value) for value in raw_probabilities)
            dominant_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            prediction = RegimePrediction(
                schema_version=PREDICTION_SCHEMA_VERSION,
                timestamp=timestamp,
                state_ids=state_ids,
                state_probabilities=probabilities,
                dominant_state=state_ids[dominant_index],
                confidence=probabilities[dominant_index],
                entropy=_entropy(probabilities),
            )
            predictions.append(prediction)
            rows.append(
                {
                    "schema_version": prediction.schema_version,
                    "timestamp": prediction.timestamp,
                    "state_ids": list(prediction.state_ids),
                    "state_probabilities": list(prediction.state_probabilities),
                    "dominant_state": prediction.dominant_state,
                    "confidence": prediction.confidence,
                    "entropy": prediction.entropy,
                    "prediction_mode": PredictionMode.WALK_FORWARD_OOS.value,
                    "profile_id": evaluation.profile_id,
                    "profile_config_version": evaluation.profile_config_version,
                    "candidate_id": evaluation.candidate_id,
                    "fold_id": fold.fold_id,
                    "evaluation_plan_hash": evaluation.evaluation_plan_hash,
                    "feature_selection_definition_hash": (
                        evaluation.feature_selection_definition_hash
                    ),
                    "feature_selection_execution_hash": (
                        evaluation.feature_selection_execution_hash
                    ),
                }
            )
    if not predictions:
        raise ValueError("walk-forward OOS publication requires at least one valid prediction")
    timestamps = tuple(prediction.timestamp for prediction in predictions)
    if timestamps != tuple(sorted(timestamps)) or len(set(timestamps)) != len(timestamps):
        raise ValueError("walk-forward OOS timestamps must be globally unique and increasing")
    return rows, tuple(predictions)


def publish_walk_forward_oos(
    store: PredictionStore,
    evaluation: WalkForwardEvaluation,
    source: SourceLineage,
    *,
    feature_contract_hash: str,
    created_at_utc: datetime,
) -> OOSPublication:
    """Publish an idempotent immutable walk_forward_oos build for one candidate evaluation."""

    if evaluation.source_build_id != source.source_build_id:
        raise ValueError("evaluation/source build identity mismatch")
    if created_at_utc.tzinfo is None or created_at_utc.utcoffset() != UTC.utcoffset(created_at_utc):
        raise ValueError("created_at_utc must be timezone-aware UTC")
    rows, predictions = _rows_and_predictions(evaluation)
    build_id = _canonical_build_id(evaluation, source, feature_contract_hash)
    try:
        manifest = store.publish(
            build_id=build_id,
            profile_id=evaluation.profile_id,
            prediction_mode=PredictionMode.WALK_FORWARD_OOS,
            rows=rows,
            source_build_id=source.source_build_id,
            source_data_sha256=source.data_sha256,
            source_schema_version=source.schema_version,
            source_feature_version=source.feature_version,
            source_synced_at_utc=source.synced_at_utc,
            feature_contract_hash=feature_contract_hash,
            feature_selection_definition_hash=evaluation.feature_selection_definition_hash,
            feature_selection_execution_hash=evaluation.feature_selection_execution_hash,
            created_at_utc=created_at_utc,
        )
    except FileExistsError:
        manifest = store.load_manifest(evaluation.profile_id, build_id)
        if manifest.prediction_mode is not PredictionMode.WALK_FORWARD_OOS:
            raise ValueError("existing deterministic build has wrong prediction mode")
        existing = store.read_table(evaluation.profile_id, build_id).to_pylist()
        expected = rows
        normalized_existing = [
            {
                **row,
                "timestamp": row["timestamp"].astimezone(UTC),
            }
            for row in existing
        ]
        if normalized_existing != expected:
            raise ValueError("existing deterministic OOS build differs from requested publication")
    return OOSPublication(build_id=build_id, manifest=manifest, predictions=predictions)
