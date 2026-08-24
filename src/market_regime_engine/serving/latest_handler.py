"""Profile-aware latest handler with exact source/model freshness semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from math import log

from market_regime_engine.contracts import (
    INVOCATION_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    FeatureSelectionLineage,
    LatestInvocation,
    ModelIdentity,
    PredictionMode,
    RegimeInvocationResponse,
    RegimePrediction,
)
from market_regime_engine.features.ports import FeatureRequest, FeatureSource, SourceMode
from market_regime_engine.inference.latest import latest_prediction
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.serving.model_resolver import ModelResolver, ResolvedModelLease

_SECONDS_PER_DAY = 86_400.0
SOURCE_STALE_WARN_DAYS = 4.0
SOURCE_STALE_FAIL_DAYS = 7.0
MODEL_STALE_WARN_DAYS = 14.0
MODEL_STALE_FAIL_DAYS = 35.0


class FreshnessState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class FreshnessAssessment:
    source_staleness_days: float
    model_staleness_days: float
    state: FreshnessState
    fail_threshold_exceeded: bool


@dataclass(slots=True)
class StaleDefaultChampionError(Exception):
    source_staleness_days: float
    model_staleness_days: float
    status_code: int = 503
    error_code: str = "stale_default_champion"
    retryable: bool = True

    def __str__(self) -> str:
        return "default champion latest exceeds configured source/model staleness limits"


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("timestamp must be timezone-aware UTC")
    return value.isoformat().replace("+00:00", "Z")


def assess_freshness(
    *,
    request_time_utc: datetime,
    prediction_timestamp: datetime,
    trained_through_timestamp: datetime,
) -> FreshnessAssessment:
    _utc_text(request_time_utc)
    _utc_text(prediction_timestamp)
    _utc_text(trained_through_timestamp)
    source_days = (request_time_utc - prediction_timestamp).total_seconds() / _SECONDS_PER_DAY
    model_days = max(
        0.0,
        (prediction_timestamp - trained_through_timestamp).total_seconds() / _SECONDS_PER_DAY,
    )
    failed = source_days >= SOURCE_STALE_FAIL_DAYS or model_days >= MODEL_STALE_FAIL_DAYS
    warned = source_days >= SOURCE_STALE_WARN_DAYS or model_days >= MODEL_STALE_WARN_DAYS
    return FreshnessAssessment(
        source_staleness_days=source_days,
        model_staleness_days=model_days,
        state=FreshnessState.DEGRADED if warned else FreshnessState.HEALTHY,
        fail_threshold_exceeded=failed,
    )


def _feature_contract_hash(artifact: ProductionModelArtifact) -> str:
    payload = json.dumps(list(artifact.feature_order), separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _identity(
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
    return RegimePrediction(
        schema_version=PREDICTION_SCHEMA_VERSION,
        timestamp=timestamp,
        state_ids=state_ids,
        state_probabilities=probabilities,
        dominant_state=state_ids[dominant_index],
        confidence=probabilities[dominant_index],
        entropy=-sum(value * log(value) for value in probabilities if value > 0.0),
    )


class LatestHandler:
    def __init__(self, resolver: ModelResolver, source: FeatureSource) -> None:
        self._resolver = resolver
        self._source = source

    def handle(
        self,
        *,
        request_id: str,
        profile_id: str,
        invocation: LatestInvocation,
        request_time_utc: datetime,
    ) -> RegimeInvocationResponse:
        _utc_text(request_time_utc)
        with self._resolver.resolve(
            profile_id,
            exact_version=invocation.model_version,
        ) as lease:
            artifact = lease.artifact
            start = (
                artifact.inference_origin_timestamp
                if invocation.as_of is None
                or invocation.as_of <= artifact.trained_through_timestamp
                else artifact.trained_through_timestamp
            )
            snapshot = self._source.read(
                FeatureRequest(
                    feature_names=artifact.feature_order,
                    start=start,
                    end=invocation.as_of,
                    mode=SourceMode.RESOLVED_MODEL,
                )
            )
            effective_as_of = invocation.as_of
            if effective_as_of is None:
                if snapshot.lineage.max_timestamp is None:
                    raise ValueError("validated serving source is missing max_timestamp")
                effective_as_of = snapshot.lineage.max_timestamp
            inference = latest_prediction(artifact, snapshot, as_of=effective_as_of)
            freshness = assess_freshness(
                request_time_utc=request_time_utc,
                prediction_timestamp=inference.timestamp,
                trained_through_timestamp=artifact.trained_through_timestamp,
            )
            if lease.resolved_via_alias == "champion" and freshness.fail_threshold_exceeded:
                raise StaleDefaultChampionError(
                    freshness.source_staleness_days,
                    freshness.model_staleness_days,
                )
            prediction = _prediction(inference.timestamp, inference.filtered_probabilities)
            alias_time = request_time_utc if lease.resolved_via_alias is not None else None
            requested_fields = ()
            if invocation.as_of is not None:
                requested_fields = (("as_of", _utc_text(invocation.as_of)),)
            return RegimeInvocationResponse(
                schema_version=INVOCATION_SCHEMA_VERSION,
                request_id=request_id,
                profile_id=profile_id,
                operation=invocation.operation,
                prediction_mode=PredictionMode.FIXED_MODEL_LATEST,
                requested_time_fields=requested_fields,
                model=_identity(lease, alias_resolved_at_utc=alias_time),
                source=snapshot.lineage,
                selection=FeatureSelectionLineage(
                    artifact.feature_selection_definition_hash,
                    artifact.feature_selection_execution_hash,
                ),
                warmup_observation_count=inference.warmup_observation_count,
                skipped_incomplete_row_count=snapshot.skipped_incomplete_row_count,
                predictions=(prediction,),
            )
