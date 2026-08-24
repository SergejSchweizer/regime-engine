"""Profile-aware fixed-model replay handler with bounded synchronous work."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from math import log
from typing import Any

from market_regime_engine.contracts import (
    INVOCATION_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    FeatureSelectionLineage,
    ModelIdentity,
    PredictionMode,
    RegimeInvocationResponse,
    RegimePrediction,
    ReplayInvocation,
)
from market_regime_engine.features.ports import FeatureRequest, FeatureSource, SourceMode
from market_regime_engine.inference.replay import fixed_model_replay
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.serving.model_resolver import ModelResolver, ResolvedModelLease
from market_regime_engine.serving.replay_admission import ReplayAdmission
from market_regime_engine.serving.replay_limits import ReplayLimits


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("timestamp must be timezone-aware UTC")
    return value.isoformat().replace("+00:00", "Z")


def _feature_contract_hash(artifact: ProductionModelArtifact) -> str:
    payload = json.dumps(list(artifact.feature_order), separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _model_identity(
    lease: ResolvedModelLease,
    *,
    alias_resolved_at_utc: datetime | None,
) -> ModelIdentity:
    artifact = lease.artifact
    return ModelIdentity(
        profile_id=artifact.profile_id,
        profile_config_version=artifact.profile_config_version,
        model_name=artifact.registered_model,
        model_version=lease.exact_version,
        feature_contract_hash=_feature_contract_hash(artifact),
        feature_order=artifact.feature_order,
        inference_origin_timestamp=artifact.inference_origin_timestamp,
        trained_through_timestamp=artifact.trained_through_timestamp,
        terminal_filtered_probabilities=artifact.terminal_filtered_probabilities,
        covariance_type="full",
        model_alias=lease.resolved_via_alias,
        alias_resolved_at_utc=alias_resolved_at_utc,
    )


def _prediction(timestamp: datetime, probabilities: tuple[float, ...]) -> RegimePrediction:
    state_ids = tuple(f"state_{index}" for index in range(len(probabilities)))
    dominant_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    entropy = -sum(value * log(value) for value in probabilities if value > 0.0)
    return RegimePrediction(
        schema_version=PREDICTION_SCHEMA_VERSION,
        timestamp=timestamp,
        state_ids=state_ids,
        state_probabilities=probabilities,
        dominant_state=state_ids[dominant_index],
        confidence=probabilities[dominant_index],
        entropy=entropy,
    )


def _json_default(value: object) -> Any:
    if isinstance(value, datetime):
        return _utc_text(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"unsupported response value: {type(value).__name__}")


def replay_response_json(response: RegimeInvocationResponse) -> str:
    return json.dumps(
        asdict(response),
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _estimate_response_bytes(response_rows: int, state_count: int) -> int:
    if response_rows < 0 or state_count < 1:
        raise ValueError("response estimate inputs are invalid")
    return 4096 + response_rows * (384 + state_count * 64)


class ReplayHandler:
    def __init__(
        self,
        resolver: ModelResolver,
        source: FeatureSource,
        limits: ReplayLimits,
        admission: ReplayAdmission,
    ) -> None:
        self._resolver = resolver
        self._source = source
        self._limits = limits
        self._admission = admission

    def handle(
        self,
        *,
        request_id: str,
        profile_id: str,
        invocation: ReplayInvocation,
        request_time_utc: datetime,
    ) -> RegimeInvocationResponse:
        self._limits.validate_interval(invocation.start, invocation.end)
        _utc_text(request_time_utc)
        with self._resolver.resolve(
            profile_id,
            exact_version=invocation.model_version,
        ) as lease:
            artifact = lease.artifact
            internal_start = (
                artifact.trained_through_timestamp
                if invocation.start > artifact.trained_through_timestamp
                else artifact.inference_origin_timestamp
            )
            alias_time = request_time_utc if lease.resolved_via_alias is not None else None
            with self._admission.admit() as permit:
                permit.check_deadline()
                snapshot = self._source.read(
                    FeatureRequest(
                        feature_names=artifact.feature_order,
                        start=internal_start,
                        end=invocation.end,
                        mode=SourceMode.RESOLVED_MODEL,
                    )
                )
                permit.check_deadline()
                response_rows = sum(
                    invocation.start <= row.timestamp <= invocation.end for row in snapshot.rows
                )
                internal_rows = len(snapshot.rows) + snapshot.skipped_incomplete_row_count
                self._limits.validate_estimates(
                    response_rows=response_rows,
                    internal_rows=internal_rows,
                    estimated_response_bytes=_estimate_response_bytes(
                        response_rows,
                        artifact.state_count,
                    ),
                )
                inference = fixed_model_replay(
                    artifact,
                    snapshot,
                    start=invocation.start,
                    end=invocation.end,
                    deadline_check=permit.check_deadline,
                )
                predictions = tuple(
                    _prediction(timestamp, probabilities)
                    for timestamp, probabilities in zip(
                        inference.timestamps,
                        inference.filtered_probabilities,
                        strict=True,
                    )
                )
                response = RegimeInvocationResponse(
                    schema_version=INVOCATION_SCHEMA_VERSION,
                    request_id=request_id,
                    profile_id=profile_id,
                    operation=invocation.operation,
                    prediction_mode=PredictionMode.FIXED_MODEL_REPLAY,
                    requested_time_fields=(
                        ("end", _utc_text(invocation.end)),
                        ("start", _utc_text(invocation.start)),
                    ),
                    model=_model_identity(lease, alias_resolved_at_utc=alias_time),
                    source=snapshot.lineage,
                    selection=FeatureSelectionLineage(
                        artifact.feature_selection_definition_hash,
                        artifact.feature_selection_execution_hash,
                    ),
                    warmup_observation_count=inference.warmup_observation_count,
                    skipped_incomplete_row_count=snapshot.skipped_incomplete_row_count,
                    predictions=predictions,
                )
                self._limits.validate_serialized_size(
                    len(replay_response_json(response).encode("utf-8"))
                )
                permit.check_deadline()
                return response
