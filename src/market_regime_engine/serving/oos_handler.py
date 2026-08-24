"""Framework-neutral handler for explicit immutable walk-forward OOS retrieval."""

from __future__ import annotations

from datetime import datetime

from market_regime_engine.predictions.query import OOSQuery, query_oos_build
from market_regime_engine.predictions.store import PredictionStore


def _time_text(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


class OOSPredictionHandler:
    """Serve one explicit build ID without resolving aliases or a silent latest build."""

    def __init__(self, store: PredictionStore) -> None:
        self._store = store

    def handle(
        self,
        *,
        profile_id: str,
        build_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, object]:
        result = query_oos_build(
            self._store,
            OOSQuery(profile_id=profile_id, build_id=build_id, start=start, end=end),
        )
        manifest = result.manifest
        predictions = [
            {
                "schema_version": row.prediction.schema_version,
                "timestamp": _time_text(row.prediction.timestamp),
                "state_ids": list(row.prediction.state_ids),
                "state_probabilities": list(row.prediction.state_probabilities),
                "dominant_state": row.prediction.dominant_state,
                "confidence": row.prediction.confidence,
                "entropy": row.prediction.entropy,
                "candidate_id": row.candidate_id,
                "fold_id": row.fold_id,
                "evaluation_plan_hash": row.evaluation_plan_hash,
                "feature_selection_definition_hash": row.feature_selection_definition_hash,
                "feature_selection_execution_hash": row.feature_selection_execution_hash,
            }
            for row in result.rows
        ]
        return {
            "profile_id": result.profile_id,
            "build_id": result.build_id,
            "prediction_mode": result.prediction_mode.value,
            "requested_start": _time_text(result.requested_start),
            "requested_end": _time_text(result.requested_end),
            "source_build_id": manifest.source_build_id,
            "source_data_sha256": manifest.source_data_sha256,
            "source_schema_version": manifest.source_schema_version,
            "source_feature_version": manifest.source_feature_version,
            "source_synced_at_utc": manifest.source_synced_at_utc,
            "data_time_semantics": manifest.data_time_semantics,
            "feature_contract_hash": manifest.feature_contract_hash,
            "feature_selection_definition_hash": manifest.feature_selection_definition_hash,
            "feature_selection_execution_hash": manifest.feature_selection_execution_hash,
            "row_count": len(predictions),
            "predictions": predictions,
        }
