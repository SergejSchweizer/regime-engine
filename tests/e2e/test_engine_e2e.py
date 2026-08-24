"""Hermetic proof of the complete Xetra training-to-serving contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation.selection import select_statistical_champion
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.feature_selection.contracts import FeatureBlock, FeatureSelectionPolicy
from market_regime_engine.feature_selection.freeze import freeze_first_train_features
from market_regime_engine.features.ports import FeatureRequest, FeatureRow, FeatureSnapshot
from market_regime_engine.mlflow_app.app import create_app
from market_regime_engine.mlflow_app.dependencies import ReadinessSnapshot, ServiceDependencies
from market_regime_engine.mlflow_support.model_package import save_production_package
from market_regime_engine.mlflow_support.registry import MlflowModelRegistry
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FilterResult, FitResult
from market_regime_engine.predictions.oos_publication import publish_walk_forward_oos
from market_regime_engine.predictions.store import PredictionStore
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import resolve_selected_feature_profile
from market_regime_engine.serving.latest_handler import LatestHandler
from market_regime_engine.serving.model_resolver import ModelResolver
from market_regime_engine.serving.oos_handler import OOSPredictionHandler
from market_regime_engine.serving.replay_admission import ReplayAdmission
from market_regime_engine.serving.replay_handler import ReplayHandler
from market_regime_engine.serving.replay_limits import ReplayLimits
from market_regime_engine.training.candidate_grid import evaluate_candidate_grid
from market_regime_engine.training.final_refit import final_production_refit

pytestmark = pytest.mark.integration

_PROFILE_PATH = Path("configs/profiles/xetra_v1.yaml")
_POLICY_PATH = Path("configs/feature_selection/xetra_semantic_medoid_v1.yaml")
_SOURCE_BUILD_ID = "hermetic-e2e-build"
_SOURCE_SHA = "a" * 64


def _policy() -> FeatureSelectionPolicy:
    raw = yaml.safe_load(_POLICY_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    blocks = tuple(
        FeatureBlock(str(item["block_id"]), tuple(str(value) for value in item["features"]))
        for item in raw["blocks"]
    )
    return FeatureSelectionPolicy(
        policy_id=str(raw["policy_id"]),
        blocks=blocks,
        within_block_method=str(raw["within_block_method"]),
        cross_block_method=str(raw["cross_block_method"]),
        minimum_feature_coverage=float(raw["minimum_feature_coverage"]),
        minimum_nonzero_variance=float(raw["minimum_nonzero_variance"]),
        minimum_block_complete_observations=int(raw["minimum_block_complete_observations"]),
        maximum_cross_block_abs_spearman=float(raw["maximum_cross_block_abs_spearman"]),
        numeric_tie_abs_tolerance=float(raw["numeric_tie_abs_tolerance"]),
    )


def _source_rows(policy: FeatureSelectionPolicy) -> pd.DataFrame:
    row_count = 1_386
    rng = np.random.default_rng(20260824)
    timestamps = [
        datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(row_count)
    ]
    regimes = (np.arange(row_count) % 4).astype(np.float64) - 1.5
    rows: dict[str, object] = {"timestamp_m1": timestamps}
    for position, feature in enumerate(policy.feature_universe):
        # Independent feature noise keeps Stage-2 pruning deterministic but non-degenerate;
        # the shared low-amplitude regime signal yields occupied synthetic HMM states.
        rows[feature] = regimes * (0.2 + position * 0.001) + rng.normal(size=row_count)
    return pd.DataFrame(rows)


def _artifact(state_count: int, features: tuple[str, ...]) -> GaussianHMMArtifact:
    dimension = len(features)
    # Use distinct regions of the first two standardized dimensions.  This gives
    # each K=2/K=3/K=4 synthetic state material occupancy without coupling the
    # 48 source columns and accidentally changing the selection contract.
    coordinates = {
        2: ((-0.8, -0.8), (0.8, 0.8)),
        3: ((-0.9, -0.9), (-0.9, 0.9), (0.9, 0.0)),
        4: ((-0.9, -0.9), (-0.9, 0.9), (0.9, -0.9), (0.9, 0.9)),
    }[state_count]
    means = tuple(
        tuple(coordinate[index] if index < 2 else 0.0 for index in range(dimension))
        for coordinate in coordinates
    )
    off_diagonal = 0.08 / (state_count - 1)
    transition = tuple(
        tuple(0.92 if left == right else off_diagonal for right in range(state_count))
        for left in range(state_count)
    )
    covariance = tuple(
        tuple(
            tuple(1.0 if left == right else 0.0 for right in range(dimension))
            for left in range(dimension)
        )
        for _ in range(state_count)
    )
    return GaussianHMMArtifact(
        state_count=state_count,
        feature_order=features,
        start_probabilities=tuple(1.0 / state_count for _ in range(state_count)),
        transition_matrix=transition,
        means=means,
        full_covariances=covariance,
    )


class _DeterministicAdapter:
    def __init__(self, features: tuple[str, ...]) -> None:
        self._features = features
        self._current = _artifact(2, features)

    def fit(self, train_rows: object, state_count: int, seed: int) -> FitResult:
        matrix = np.asarray(train_rows, dtype=np.float64)
        self._current = _artifact(state_count, self._features)
        return FitResult(
            artifact=self._current,
            train_log_likelihood=-float(np.sum(matrix * matrix)) + seed * 1e-6,
            converged=True,
            iterations=5,
            seed=seed,
        )

    def extract(self) -> GaussianHMMArtifact:
        return self._current

    def reconstruct(self, model_artifact: GaussianHMMArtifact) -> None:
        self._current = model_artifact

    def causal_filter(
        self,
        rows: object,
        initial_filtered_probabilities: tuple[float, ...] | None = None,
    ) -> FilterResult:
        raise AssertionError("the production runner uses the canonical backend-independent filter")


class _FixtureSource:
    def __init__(self, snapshot: FeatureSnapshot) -> None:
        self.snapshot = snapshot
        self.requests: list[FeatureRequest] = []

    def read(self, request: FeatureRequest) -> FeatureSnapshot:
        self.requests.append(request)
        return self.snapshot


def _lineage(rows: pd.DataFrame) -> SourceLineage:
    return SourceLineage(
        source_dataset="regime_loader.regime_features_daily",
        source_build_id=_SOURCE_BUILD_ID,
        data_sha256=_SOURCE_SHA,
        schema_version=1,
        feature_version=1,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=datetime(2026, 8, 24, tzinfo=UTC),
        row_count=len(rows),
        min_timestamp=rows["timestamp_m1"].iloc[0],
        max_timestamp=rows["timestamp_m1"].iloc[-1],
    )


def test_hermetic_xetra_training_registry_and_serving_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the required data, training, package, registry, and serving sequence."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    policy = _policy()
    profile = load_profile(_PROFILE_PATH)
    rows = _source_rows(policy)
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    first_train = rows.iloc[: plan.folds[0].train_source_observations]
    selection = freeze_first_train_features(
        first_train,
        policy,
        source_build_id=_SOURCE_BUILD_ID,
        data_sha256=_SOURCE_SHA,
        evaluation_plan_hash=plan.plan_hash,
    )
    resolved = resolve_selected_feature_profile(
        profile,
        policy,
        selection,
        source_build_id=_SOURCE_BUILD_ID,
    )
    grid = evaluate_candidate_grid(
        rows,
        plan=plan,
        profile=profile,
        resolved_profile=resolved,
        adapter_factory_builder=lambda candidate: (
            lambda: _DeterministicAdapter(candidate.feature_order)
        ),
    )
    assert tuple(item.candidate_id for item in grid.evaluations) == (
        "gaussian_hmm_k2_full",
        "gaussian_hmm_k3_full",
        "gaussian_hmm_k4_full",
    )
    assert all(item.valid_fold_rate == 1.0 for item in grid.evaluations)
    assert all(item.feature_order == selection.final_features for item in grid.evaluations)
    assert all(
        item.feature_selection_execution_hash == selection.feature_selection_execution_hash
        for item in grid.evaluations
    )

    champion = select_statistical_champion(grid)
    winning_evaluation = next(
        item for item in grid.evaluations if item.candidate_id == champion.champion_candidate_id
    )
    winning_candidate = next(
        item for item in resolved.candidates if item.candidate_id == champion.champion_candidate_id
    )
    production = final_production_refit(
        rows,
        lineage=_lineage(rows),
        candidate=winning_candidate,
        winning_evaluation=winning_evaluation,
        adapter_factory_builder=lambda candidate: (
            lambda: _DeterministicAdapter(candidate.feature_order)
        ),
    )
    assert production.candidate_id == champion.champion_candidate_id
    assert production.feature_order == selection.final_features
    assert production.evaluation_cutoff == winning_evaluation.evaluation_cutoff
    assert production.trained_through_timestamp == winning_evaluation.evaluation_cutoff

    package = save_production_package(production, tmp_path / "package")
    database_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
    registry = MlflowModelRegistry(
        MlflowClient(tracking_uri=database_uri, registry_uri=database_uri)
    )
    registered = registry.register_production_model(production, package)
    assert registry.compare_and_swap_alias(
        model_name=production.registered_model,
        alias="champion",
        expected_current_version=None,
        new_version=registered.exact_version,
        reason="hermetic E2E promotion",
    )

    source = _FixtureSource(
        FeatureSnapshot(
            lineage=_lineage(rows),
            feature_names=production.feature_order,
            rows=tuple(
                FeatureRow(timestamp, tuple(float(value) for value in values))
                for timestamp, values in zip(
                    rows["timestamp_m1"],
                    rows.loc[:, list(production.feature_order)].itertuples(index=False, name=None),
                    strict=True,
                )
            ),
        )
    )
    resolver = ModelResolver(registry)
    latest = LatestHandler(resolver, source)
    replay = ReplayHandler(resolver, source, ReplayLimits(), ReplayAdmission(ReplayLimits()))
    store = PredictionStore(tmp_path / "oos")
    publication = publish_walk_forward_oos(
        store,
        winning_evaluation,
        _lineage(rows),
        feature_contract_hash="b" * 64,
        created_at_utc=datetime(2026, 8, 24, tzinfo=UTC),
    )
    app = create_app(
        dependencies=ServiceDependencies(
            latest_handler=latest,
            replay_handler=replay,
            oos_handler=OOSPredictionHandler(store),
            readiness=lambda: ReadinessSnapshot("healthy", True),
            request_id_factory=lambda: "hermetic-e2e-request",
            request_time_factory=lambda: production.trained_through_timestamp,
        )
    )
    client = app.test_client()
    latest_response = client.post(
        "/regime-engine/v1/profiles/xetra/invocations",
        json={"operation": "latest"},
    )
    assert latest_response.status_code == 200
    latest_payload = latest_response.get_json()
    assert latest_payload["model"]["model_version"] == registered.exact_version
    assert latest_payload["model"]["model_alias"] == "champion"
    assert latest_payload["prediction_mode"] == "fixed_model_latest"

    start = production.trained_through_timestamp - timedelta(days=2)
    replay_response = client.post(
        "/regime-engine/v1/profiles/xetra/invocations",
        json={
            "operation": "replay",
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": production.trained_through_timestamp.isoformat().replace("+00:00", "Z"),
        },
    )
    assert replay_response.status_code == 200
    assert replay_response.get_json()["prediction_mode"] == "fixed_model_replay"

    oos_response = client.get(f"/regime-engine/v1/profiles/xetra/oos-builds/{publication.build_id}")
    assert oos_response.status_code == 200
    assert oos_response.get_json()["prediction_mode"] == "walk_forward_oos"
    assert oos_response.get_json()["row_count"] == len(publication.predictions)

    # OOS is a published evaluation artifact, distinct from current-vintage serving.
    assert latest_payload["source"]["source_build_id"] == _SOURCE_BUILD_ID
    assert oos_response.get_json()["build_id"] == publication.build_id
    assert selection.feature_selection_definition_hash != ""
    assert selection.feature_selection_execution_hash != ""
