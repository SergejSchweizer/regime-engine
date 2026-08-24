from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.contracts import (
    DATA_TIME_SEMANTICS,
    PredictionMode,
    ReplayInvocation,
    SourceLineage,
)
from market_regime_engine.features.ports import FeatureRow, FeatureSnapshot
from market_regime_engine.inference.replay import ReplayInferenceResult, fixed_model_replay
from market_regime_engine.mlflow_support.ports import ResolvedModelVersion
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact
from market_regime_engine.serving.model_resolver import ModelResolver
from market_regime_engine.serving.replay_admission import ReplayAdmission
from market_regime_engine.serving.replay_handler import ReplayHandler, replay_response_json
from market_regime_engine.serving.replay_limits import ReplayGuardrailError, ReplayLimits

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
    semantics: str = DATA_TIME_SEMANTICS,
) -> SourceLineage:
    return SourceLineage(
        source_dataset="regime_loader.regime_features_daily",
        source_build_id="serving-build",
        data_sha256="e" * 64,
        schema_version=schema,
        feature_version=feature,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=BASE + timedelta(days=10),
        data_time_semantics=semantics,
        row_count=10,
        min_timestamp=BASE,
        max_timestamp=BASE + timedelta(days=9),
    )


def snapshot(
    rows: tuple[FeatureRow, ...] | None = None,
    *,
    feature_names: tuple[str, ...] = ("f0",),
    source: SourceLineage | None = None,
    skipped: int = 0,
) -> FeatureSnapshot:
    if rows is None:
        rows = tuple(
            FeatureRow(BASE + timedelta(days=day), (float(day) / 10.0,))
            for day in range(0, 7)
        )
    return FeatureSnapshot(
        lineage=source or lineage(),
        feature_names=feature_names,
        rows=rows,
        skipped_incomplete_row_count=skipped,
    )


class FakeRegistry:
    def __init__(self) -> None:
        self.alias_calls = 0
        self.package_calls: list[str] = []

    def resolve_alias(self, model_name: str, alias: str) -> ResolvedModelVersion:
        self.alias_calls += 1
        return ResolvedModelVersion(model_name, alias, "7", BASE + timedelta(days=9))

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
        raise AssertionError("serving must never mutate aliases")


class RecordingSource:
    def __init__(self, data: FeatureSnapshot, events: list[str] | None = None) -> None:
        self.data = data
        self.requests: list[object] = []
        self.events = events

    def read(self, request: object) -> FeatureSnapshot:
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


def test_fixed_model_replay_is_start_invariant_after_training_boundary() -> None:
    data = snapshot()
    first = fixed_model_replay(
        artifact(),
        data,
        start=BASE + timedelta(days=4),
        end=BASE + timedelta(days=6),
    )
    second = fixed_model_replay(
        artifact(),
        data,
        start=BASE + timedelta(days=5),
        end=BASE + timedelta(days=6),
    )
    assert first.timestamps[1:] == second.timestamps
    assert first.filtered_probabilities[1:] == second.filtered_probabilities
    assert first.warmup_observation_count == 1
    assert second.warmup_observation_count == 2


def test_replay_including_training_period_filters_from_persisted_origin() -> None:
    result = fixed_model_replay(
        artifact(),
        snapshot(),
        start=BASE + timedelta(days=1),
        end=BASE + timedelta(days=3),
    )
    assert result.timestamps == tuple(BASE + timedelta(days=day) for day in (1, 2, 3))
    assert result.warmup_observation_count == 1
    assert all(
        abs(sum(probabilities) - 1.0) <= 1e-10
        for probabilities in result.filtered_probabilities
    )


def test_replay_handler_pins_model_before_source_and_returns_current_vintage_lineage() -> None:
    events: list[str] = []
    registry = FakeRegistry()
    source = RecordingSource(snapshot(), events)
    handler = ReplayHandler(
        resolver(registry, events),
        source,  # type: ignore[arg-type]
        ReplayLimits(),
        ReplayAdmission(ReplayLimits()),
    )
    response = handler.handle(
        request_id="request-1",
        profile_id="xetra",
        invocation=ReplayInvocation(
            BASE + timedelta(days=4),
            BASE + timedelta(days=6),
        ),
        request_time_utc=BASE + timedelta(days=9),
    )
    assert events[:2] == ["model", "source"]
    assert registry.alias_calls == 1
    assert response.prediction_mode is PredictionMode.FIXED_MODEL_REPLAY
    assert response.source.source_build_id == "serving-build"
    assert response.model.model_version == "7"
    assert response.model.model_alias == "champion"
    assert response.selection.feature_selection_definition_hash == "b" * 64
    assert response.requested_time_fields == (
        ("end", "2026-01-07T00:00:00Z"),
        ("start", "2026-01-05T00:00:00Z"),
    )
    assert "state_probabilities" in replay_response_json(response)


def test_explicit_model_version_bypasses_alias_resolution() -> None:
    registry = FakeRegistry()
    handler = ReplayHandler(
        resolver(registry),
        RecordingSource(snapshot()),  # type: ignore[arg-type]
        ReplayLimits(),
        ReplayAdmission(ReplayLimits()),
    )
    response = handler.handle(
        request_id="request-2",
        profile_id="xetra",
        invocation=ReplayInvocation(
            BASE + timedelta(days=4),
            BASE + timedelta(days=4),
            model_version="99",
        ),
        request_time_utc=BASE + timedelta(days=9),
    )
    assert registry.alias_calls == 0
    assert registry.package_calls == ["99"]
    assert response.model.model_version == "99"
    assert response.model.model_alias is None
    assert response.model.alias_resolved_at_utc is None


def test_replay_handler_enforces_row_and_internal_limits() -> None:
    handler = ReplayHandler(
        resolver(),
        RecordingSource(snapshot(skipped=3)),  # type: ignore[arg-type]
        ReplayLimits(max_rows=1, max_internal_rows=20),
        ReplayAdmission(ReplayLimits()),
    )
    with pytest.raises(ReplayGuardrailError) as exc:
        handler.handle(
            request_id="request-3",
            profile_id="xetra",
            invocation=ReplayInvocation(
                BASE + timedelta(days=4),
                BASE + timedelta(days=6),
            ),
            request_time_utc=BASE + timedelta(days=9),
        )
    assert (exc.value.status_code, exc.value.error_code) == (413, "replay_row_limit")

    internal_handler = ReplayHandler(
        resolver(),
        RecordingSource(snapshot(skipped=20)),  # type: ignore[arg-type]
        ReplayLimits(max_internal_rows=5),
        ReplayAdmission(ReplayLimits()),
    )
    with pytest.raises(ReplayGuardrailError, match="internal-row"):
        internal_handler.handle(
            request_id="request-4",
            profile_id="xetra",
            invocation=ReplayInvocation(
                BASE + timedelta(days=4),
                BASE + timedelta(days=4),
            ),
            request_time_utc=BASE + timedelta(days=9),
        )


def test_replay_inference_rejects_incompatible_or_incomplete_snapshots() -> None:
    with pytest.raises(ValueError, match="feature order"):
        fixed_model_replay(
            artifact(),
            snapshot(feature_names=("other",)),
            start=BASE,
            end=BASE + timedelta(days=1),
        )
    with pytest.raises(ValueError, match="schema version"):
        fixed_model_replay(
            artifact(),
            snapshot(source=lineage(schema=2)),
            start=BASE,
            end=BASE + timedelta(days=1),
        )
    with pytest.raises(ValueError, match="feature version"):
        fixed_model_replay(
            artifact(),
            snapshot(source=lineage(feature=2)),
            start=BASE,
            end=BASE + timedelta(days=1),
        )
    bad_lineage = lineage()
    object.__setattr__(bad_lineage, "data_time_semantics", "other")
    with pytest.raises(ValueError, match="time semantics"):
        fixed_model_replay(
            artifact(),
            snapshot(source=bad_lineage),
            start=BASE,
            end=BASE + timedelta(days=1),
        )
    incomplete = snapshot((FeatureRow(BASE + timedelta(days=3), (None,)),))
    with pytest.raises(ValueError, match="incomplete"):
        fixed_model_replay(
            artifact(),
            incomplete,
            start=BASE + timedelta(days=3),
            end=BASE + timedelta(days=3),
        )


def test_replay_inference_validates_empty_output_and_result_contract() -> None:
    with pytest.raises(ValueError, match="start must not be after end"):
        fixed_model_replay(
            artifact(),
            snapshot(),
            start=BASE + timedelta(days=2),
            end=BASE,
        )
    with pytest.raises(ValueError, match="no_complete_observations"):
        fixed_model_replay(
            artifact(),
            snapshot(()),
            start=BASE,
            end=BASE,
        )
    with pytest.raises(ValueError, match="matching non-empty"):
        ReplayInferenceResult((), (), 0)
    with pytest.raises(ValueError, match="warmup"):
        ReplayInferenceResult((BASE,), ((0.5, 0.5),), -1)
