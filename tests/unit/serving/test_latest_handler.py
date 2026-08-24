from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.contracts import (
    DATA_TIME_SEMANTICS,
    LatestInvocation,
    PredictionMode,
    SourceLineage,
)
from market_regime_engine.features.ports import FeatureRequest, FeatureRow, FeatureSnapshot
from market_regime_engine.inference.latest import LatestInferenceResult, latest_prediction
from market_regime_engine.mlflow_support.ports import ResolvedModelVersion
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact
from market_regime_engine.serving.latest_handler import (
    FreshnessState,
    LatestHandler,
    StaleDefaultChampionError,
    assess_freshness,
)
from market_regime_engine.serving.model_resolver import ModelResolver

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def artifact() -> ProductionModelArtifact:
    features = ("f0",)
    return ProductionModelArtifact(
        profile_id="xetra",
        profile_config_version=1,
        registered_model="regime-xetra",
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        source_build_id="training-build",
        source_data_sha256="a" * 64,
        source_schema_version=1,
        source_feature_version=1,
        data_time_semantics=DATA_TIME_SEMANTICS,
        feature_selection_definition_hash="b" * 64,
        feature_selection_execution_hash="c" * 64,
        evaluation_plan_hash="d" * 64,
        evaluation_cutoff=BASE + timedelta(days=5),
        feature_order=features,
        scaler=StandardScalerArtifact(features, (0.0,), (1.0,), (1.0,)),
        hmm=GaussianHMMArtifact(
            state_count=2,
            feature_order=features,
            start_probabilities=(0.6, 0.4),
            transition_matrix=((0.85, 0.15), (0.2, 0.8)),
            means=((-1.0,), (1.0,)),
            full_covariances=(((1.0,),), ((1.5,),)),
        ),
        winning_seed=11,
        inference_origin_timestamp=BASE,
        trained_through_timestamp=BASE + timedelta(days=2),
        terminal_filtered_probabilities=(0.7, 0.3),
        retained_observation_count=504,
        skipped_incomplete_observation_count=0,
    )


def lineage(
    *,
    schema: int = 1,
    feature: int = 1,
    maximum: datetime = BASE + timedelta(days=6),
) -> SourceLineage:
    return SourceLineage(
        source_dataset="regime_loader.regime_features_daily",
        source_build_id="serving-build",
        data_sha256="e" * 64,
        schema_version=schema,
        feature_version=feature,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=maximum,
        data_time_semantics=DATA_TIME_SEMANTICS,
        row_count=100,
        min_timestamp=BASE,
        max_timestamp=maximum,
    )


def rows_through(day: int) -> tuple[FeatureRow, ...]:
    return tuple(
        FeatureRow(BASE + timedelta(days=index), (float(index) / 10.0,)) for index in range(day + 1)
    )


def snapshot(
    *,
    rows: tuple[FeatureRow, ...] | None = None,
    source: SourceLineage | None = None,
    feature_names: tuple[str, ...] = ("f0",),
    skipped: int = 0,
) -> FeatureSnapshot:
    actual_rows = rows if rows is not None else rows_through(6)
    return FeatureSnapshot(
        lineage=source or lineage(),
        feature_names=feature_names,
        rows=actual_rows,
        skipped_incomplete_row_count=skipped,
    )


class FakeRegistry:
    def __init__(self) -> None:
        self.alias_calls = 0
        self.package_calls: list[str] = []

    def resolve_alias(self, model_name: str, alias: str) -> ResolvedModelVersion:
        self.alias_calls += 1
        return ResolvedModelVersion(model_name, alias, "7", BASE + timedelta(days=6))

    def get_model_package_uri(self, model_name: str, exact_version: str) -> str:
        self.package_calls.append(exact_version)
        return f"file:///models/{model_name}/{exact_version}"

    def compare_and_swap_alias(
        self,
        *,
        model_name: str,
        alias: str,
        expected_current_version: str | None,
        new_version: str,
        reason: str,
    ) -> bool:
        del model_name, alias, expected_current_version, new_version, reason
        raise AssertionError("latest serving never mutates aliases")


class RecordingSource:
    def __init__(self, data: FeatureSnapshot, events: list[str] | None = None) -> None:
        self.data = data
        self.events = events
        self.requests: list[FeatureRequest] = []

    def read(self, request: FeatureRequest) -> FeatureSnapshot:
        if self.events is not None:
            self.events.append("source")
        self.requests.append(request)
        return self.data


def resolver(
    registry: FakeRegistry | None = None,
    events: list[str] | None = None,
) -> ModelResolver:
    backend = registry or FakeRegistry()

    def loader(uri: str) -> ProductionModelArtifact:
        del uri
        if events is not None:
            events.append("model")
        return artifact()

    return ModelResolver(backend, package_loader=loader)


def test_omitted_as_of_pins_model_first_and_uses_validated_source_maximum() -> None:
    events: list[str] = []
    registry = FakeRegistry()
    source = RecordingSource(snapshot(skipped=2), events)
    handler = LatestHandler(resolver(registry, events), source)  # type: ignore[arg-type]
    response = handler.handle(
        request_id="request-1",
        profile_id="xetra",
        invocation=LatestInvocation(),
        request_time_utc=BASE + timedelta(days=6, hours=12),
    )
    assert events[:2] == ["model", "source"]
    assert registry.alias_calls == 1
    assert source.requests[0].end is None
    assert response.prediction_mode is PredictionMode.FIXED_MODEL_LATEST
    assert response.predictions[0].timestamp == BASE + timedelta(days=6)
    assert response.source.source_build_id == "serving-build"
    assert response.model.model_version == "7"
    assert response.model.model_alias == "champion"
    assert response.skipped_incomplete_row_count == 2
    assert response.requested_time_fields == ()


def test_explicit_as_of_after_training_continues_from_stored_terminal_alpha() -> None:
    data = snapshot()
    direct = latest_prediction(
        artifact(),
        data,
        as_of=BASE + timedelta(days=5),
    )
    assert direct.timestamp == BASE + timedelta(days=5)
    assert direct.warmup_observation_count == 2

    registry = FakeRegistry()
    source = RecordingSource(data)
    handler = LatestHandler(resolver(registry), source)  # type: ignore[arg-type]
    response = handler.handle(
        request_id="request-2",
        profile_id="xetra",
        invocation=LatestInvocation(
            as_of=BASE + timedelta(days=5),
            model_version="99",
        ),
        request_time_utc=BASE + timedelta(days=5),
    )
    assert registry.alias_calls == 0
    assert registry.package_calls == ["99"]
    assert source.requests[0].start == artifact().trained_through_timestamp
    assert source.requests[0].end == BASE + timedelta(days=5)
    assert response.model.model_alias is None
    assert response.requested_time_fields == (("as_of", "2026-01-06T00:00:00Z"),)
    assert response.predictions[0].state_probabilities == direct.filtered_probabilities


def test_latest_at_or_before_training_refilters_from_inference_origin() -> None:
    result = latest_prediction(
        artifact(),
        snapshot(),
        as_of=BASE + timedelta(days=1),
    )
    assert result.timestamp == BASE + timedelta(days=1)
    assert result.warmup_observation_count == 1
    assert result.filtered_probabilities != artifact().terminal_filtered_probabilities


def test_latest_without_subsequent_observation_can_use_stored_training_terminal() -> None:
    data = snapshot(rows=rows_through(2), source=lineage(maximum=BASE + timedelta(days=2)))
    result = latest_prediction(
        artifact(),
        data,
        as_of=BASE + timedelta(days=3),
    )
    assert result.timestamp == artifact().trained_through_timestamp
    assert result.filtered_probabilities == artifact().terminal_filtered_probabilities
    assert result.warmup_observation_count == 0


def test_freshness_formula_and_exact_thresholds() -> None:
    prediction = BASE + timedelta(days=100)
    healthy = assess_freshness(
        request_time_utc=prediction + timedelta(days=3, seconds=86_399),
        prediction_timestamp=prediction,
        trained_through_timestamp=prediction,
    )
    assert healthy.state is FreshnessState.HEALTHY
    assert not healthy.fail_threshold_exceeded

    source_warn = assess_freshness(
        request_time_utc=prediction + timedelta(days=4),
        prediction_timestamp=prediction,
        trained_through_timestamp=prediction,
    )
    assert source_warn.source_staleness_days == 4.0
    assert source_warn.state is FreshnessState.DEGRADED
    assert not source_warn.fail_threshold_exceeded

    source_fail = assess_freshness(
        request_time_utc=prediction + timedelta(days=7),
        prediction_timestamp=prediction,
        trained_through_timestamp=prediction,
    )
    assert source_fail.source_staleness_days == 7.0
    assert source_fail.fail_threshold_exceeded

    model_warn = assess_freshness(
        request_time_utc=prediction,
        prediction_timestamp=prediction,
        trained_through_timestamp=prediction - timedelta(days=14),
    )
    assert model_warn.model_staleness_days == 14.0
    assert model_warn.state is FreshnessState.DEGRADED
    assert not model_warn.fail_threshold_exceeded

    model_fail = assess_freshness(
        request_time_utc=prediction,
        prediction_timestamp=prediction,
        trained_through_timestamp=prediction - timedelta(days=35),
    )
    assert model_fail.model_staleness_days == 35.0
    assert model_fail.fail_threshold_exceeded

    future_training = assess_freshness(
        request_time_utc=prediction,
        prediction_timestamp=prediction,
        trained_through_timestamp=prediction + timedelta(days=1),
    )
    assert future_training.model_staleness_days == 0.0


def test_default_champion_latest_fails_stale_but_explicit_version_does_not() -> None:
    stale_data = snapshot()
    alias_handler = LatestHandler(
        resolver(),
        RecordingSource(stale_data),  # type: ignore[arg-type]
    )
    with pytest.raises(StaleDefaultChampionError) as exc:
        alias_handler.handle(
            request_id="request-3",
            profile_id="xetra",
            invocation=LatestInvocation(),
            request_time_utc=BASE + timedelta(days=14),
        )
    assert exc.value.status_code == 503
    assert exc.value.error_code == "stale_default_champion"

    registry = FakeRegistry()
    explicit_handler = LatestHandler(
        resolver(registry),
        RecordingSource(stale_data),  # type: ignore[arg-type]
    )
    response = explicit_handler.handle(
        request_id="request-4",
        profile_id="xetra",
        invocation=LatestInvocation(model_version="7"),
        request_time_utc=BASE + timedelta(days=14),
    )
    assert response.model.model_alias is None
    assert registry.alias_calls == 0


def test_latest_inference_fails_closed_on_source_contract_and_incomplete_rows() -> None:
    with pytest.raises(ValueError, match="feature order"):
        latest_prediction(
            artifact(),
            snapshot(feature_names=("other",)),
            as_of=BASE + timedelta(days=1),
        )
    with pytest.raises(ValueError, match="schema version"):
        latest_prediction(
            artifact(),
            snapshot(source=lineage(schema=2)),
            as_of=BASE + timedelta(days=1),
        )
    with pytest.raises(ValueError, match="feature version"):
        latest_prediction(
            artifact(),
            snapshot(source=lineage(feature=2)),
            as_of=BASE + timedelta(days=1),
        )
    bad_lineage = lineage()
    object.__setattr__(bad_lineage, "data_time_semantics", "other")
    with pytest.raises(ValueError, match="time semantics"):
        latest_prediction(
            artifact(),
            snapshot(source=bad_lineage),
            as_of=BASE + timedelta(days=1),
        )
    with pytest.raises(ValueError, match="incomplete"):
        latest_prediction(
            artifact(),
            snapshot(rows=(FeatureRow(BASE, (None,)),)),
            as_of=BASE,
        )
    with pytest.raises(ValueError, match="no_complete_observations"):
        latest_prediction(
            artifact(),
            snapshot(rows=()),
            as_of=BASE,
        )


def test_latest_result_and_freshness_validate_inputs() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        LatestInferenceResult(BASE, (), 0)
    with pytest.raises(ValueError, match="warmup"):
        LatestInferenceResult(BASE, (0.5, 0.5), -1)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        assess_freshness(
            request_time_utc=datetime(2026, 1, 1),
            prediction_timestamp=BASE,
            trained_through_timestamp=BASE,
        )
