from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.tracking import MlflowClient

import market_regime_engine.mlflow_support.evaluation_tracking as module
from market_regime_engine.evaluation_statistics.contracts import GlobalV4Evidence
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    FinalSelectedConfiguration,
    OuterFoldResult,
)
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort

HASH = "a" * 64
START = datetime(2024, 1, 1, tzinfo=UTC)


class RecordingPort:
    def __init__(self) -> None:
        self.artifacts: list[tuple[str, str, str]] = []
        self.params: dict[str, dict[str, str]] = {}
        self.finished: list[str] = []
        self.failed: list[str] = []
        self._count = 0

    def start_run(self, *, run_name: str, parent_run_id: str | None = None) -> str:
        del run_name, parent_run_id
        self._count += 1
        return f"run-{self._count}"

    def log_params(self, run_id: str, params: dict[str, str]) -> None:
        self.params.setdefault(run_id, {}).update(params)

    def log_metric_points(self, run_id: str, points: tuple[object, ...]) -> None:
        del run_id, points

    def create_logged_model(self, **kwargs: object) -> str:
        del kwargs
        return "unused"

    def log_model_metric_points(self, model_id: str, points: tuple[object, ...]) -> None:
        del model_id, points

    def log_model_artifacts(self, model_id: str, local_dir: str) -> None:
        del model_id, local_dir

    def finalize_logged_model(self, model_id: str, *, failed: bool = False) -> None:
        del model_id, failed

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self.artifacts.append((run_id, local_path, artifact_path))

    def end_run(self, run_id: str) -> None:
        self.finished.append(run_id)

    def fail_run(self, run_id: str) -> None:
        self.failed.append(run_id)


def _configuration() -> FinalSelectedConfiguration:
    return FinalSelectedConfiguration(
        feature_order=("feature_a", "feature_b"),
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        model_family="gaussian_hmm",
        selected_prefix_length=2,
        feature_discovery_hash=HASH,
        source_build_id="build-1",
        catalog_hash=HASH,
        selection_definition_hash=HASH,
        selection_execution_hash=HASH,
    )


def _result() -> AdaptiveEvaluationResult:
    fold = OuterFoldResult(
        fold_index=1,
        train_start=START,
        train_end=START + timedelta(days=4),
        test_start=START + timedelta(days=5),
        test_end=START + timedelta(days=9),
        final_configuration=_configuration(),
        oos_predictive_loglik_per_observation=-1.2,
        oos_timestamps=(START + timedelta(days=5),),
        oos_filtered_probabilities=((0.5, 0.5),),
        teacher_reference_hash=HASH,
        outer_teacher_final_soft_nmi=0.6,
        outer_shared_timestamp_count=63,
    )
    return AdaptiveEvaluationResult(
        source_build_id="build-1",
        catalog_hash=HASH,
        validation_evaluation_cutoff=fold.test_end,
        outer_folds=(fold,),
        valid_fold_count=1,
        valid_fold_rate=1.0,
        soft_nmi_mean=0.6,
        soft_nmi_population_std=0.0,
        soft_nmi_worst=0.6,
        latest_complete_fold_valid=True,
        production_eligible=False,
        policy_hash=HASH,
    )


def _selection() -> SimpleNamespace:
    candidate_ids = (
        *(f"gaussian_hmm_k{count}_full" for count in (2, 3, 4, 5)),
        *(f"gmm_hmm_k{count}_m2_full" for count in (2, 3, 4, 5)),
        *(f"student_t_hmm_k{count}_full" for count in (2, 3, 4, 5)),
    )
    aggregates = tuple(
        SimpleNamespace(
            candidate_id=candidate_id,
            state_count=2,
            planned_fold_count=1,
            valid_fold_count=1,
            invalid_fold_count=0,
            valid_fold_rate=1.0,
            passes_valid_fold_rate_gate=True,
            oos_predictive_loglik_mean=-1.5 + position / 100.0,
            oos_predictive_loglik_std=0.0,
            oos_predictive_loglik_worst_fold=-1.5,
            oos_predictive_loglik_best_fold=-1.5,
            bic_mean=10.0 + position,
            aic_mean=9.0 + position,
        )
        for position, candidate_id in enumerate(candidate_ids)
    )
    features = tuple(
        SimpleNamespace(
            feature_name=name,
            canonical_ordinal=index + 1,
            finite_observation_count=1260,
            coverage=1.0,
            population_variance=1.0,
            eligible=True,
            rejection_reason=None,
        )
        for index, name in enumerate(("feature_a", "feature_b", "feature_c"))
    )
    scores = tuple(
        SimpleNamespace(
            feature_name=item.feature_name,
            canonical_ordinal=item.canonical_ordinal,
            coverage=1.0,
            observation_count=1260,
            state_information_ratio=0.3,
            eta_squared=0.2,
            eligible=True,
            exclusion_reason=None,
        )
        for item in features
    )
    return SimpleNamespace(
        source_build_id="build-1",
        catalog_hash=HASH,
        feature_discovery_hash=HASH,
        quality=SimpleNamespace(
            result_hash=HASH,
            train_source_observation_count=1260,
            eligible_features=("feature_a", "feature_b", "feature_c"),
            features=features,
        ),
        distance=SimpleNamespace(
            feature_order=("feature_a", "feature_b", "feature_c"),
            matrix_hash=HASH,
            minimum_pairwise_observations=504,
            distances=((0.0, 0.2, 0.3), (0.2, 0.0, 0.4), (0.3, 0.4, 0.0)),
            pairwise_support=((1260, 1260, 1260),) * 3,
            spearman_correlations=((1.0, 0.8, 0.7), (0.8, 1.0, 0.6), (0.7, 0.6, 1.0)),
        ),
        clusters=SimpleNamespace(
            solution_hash=HASH,
            candidate_count=3,
            selected_count=2,
            silhouette_curve=((2, 0.4),),
            selected_silhouette=0.4,
            singleton_count=1,
            memberships=(
                ("cluster_000", ("feature_a", "feature_b")),
                ("cluster_001", ("feature_c",)),
            ),
        ),
        prototypes=SimpleNamespace(
            temporary=True,
            cluster_ids=("cluster_000", "cluster_001"),
            prototypes=("feature_a", "feature_c"),
            mean_distances=(("feature_a", 0.1), ("feature_c", 0.0)),
        ),
        teacher_evaluation=SimpleNamespace(
            provisional_candidate_id="gaussian_hmm_k2_full",
            provisional_state_count=2,
            no_selection_reason=None,
        ),
        teacher_reference=SimpleNamespace(
            candidate_id="gaussian_hmm_k2_full",
            state_count=2,
            reference_hash=HASH,
            prototype_features=("feature_a", "feature_c"),
            valid_inner_fold_ids=("fold_001",),
            inner_plan_hash=HASH,
        ),
        feature_scores=scores,
        winner_selection=SimpleNamespace(
            ranked_features=("feature_a", "feature_c"),
            winners=(
                SimpleNamespace(
                    cluster_id="cluster_000",
                    winner_feature="feature_a",
                    member_features=("feature_a", "feature_b"),
                ),
                SimpleNamespace(
                    cluster_id="cluster_001",
                    winner_feature="feature_c",
                    member_features=("feature_c",),
                ),
            ),
        ),
        prefix_search=SimpleNamespace(
            ranked_features=("feature_a", "feature_c"),
            selected_prefix_length=2,
            selected_candidate_id="gaussian_hmm_k2_full",
            evaluations=(
                SimpleNamespace(
                    prefix_length=2,
                    feature_order=("feature_a", "feature_c"),
                    candidate_id="gaussian_hmm_k2_full",
                    shared_timestamp_count=126,
                    shared_teacher_coverage=1.0,
                    soft_regime_nmi=0.6,
                    valid=True,
                    invalid_reason=None,
                ),
            ),
        ),
        final_grid=SimpleNamespace(
            candidate_grid=SimpleNamespace(
                feature_order=("feature_a", "feature_b"),
                evaluation_plan_hash=HASH,
                aggregates=aggregates,
            ),
            selection=SimpleNamespace(
                champion_candidate_id="gaussian_hmm_k2_full",
                champion_state_count=2,
                ranked_candidate_ids=candidate_ids,
            ),
        ),
        final_candidate=SimpleNamespace(
            candidate_id="gaussian_hmm_k2_full",
            feature_order=("feature_a", "feature_b"),
        ),
    )


def _evidence() -> GlobalV4Evidence:
    groups = {
        "identity": {"policy_id": "xetra_global_regime_v4"},
        "lineage": {"source_build_id": "build-1", "catalog_hash": HASH},
        "input": {"feature_order": ["feature_a", "feature_b"]},
        "quality": {"eligible_features": ["feature_a", "feature_b", "feature_c"]},
        "distance": {"matrix_hash": HASH},
        "clustering": {"selected_m": 2, "silhouette_curve": [[2, 0.4]]},
        "prototypes": {"features": ["feature_a", "feature_c"]},
        "teacher": {"candidate_id": "gaussian_hmm_k2_full"},
        "feature_scores": {"scores": [{"feature": "feature_a", "eta_squared": 0.2}]},
        "prefix_search": {"selected_l": 2, "nmi_by_l": [[2, 0.6]]},
        "final_grid": {"candidate_ids": ["gaussian_hmm_k2_full"]},
        "outer_folds": [{"fold_id": "outer_fold_001", "valid": True}],
        "agreement": {"soft_nmi": [0.6]},
        "validity": {"valid_fold_rate": 1.0, "production_eligible": False},
        "stability": {"adjacent_cluster_stability": []},
    }
    return GlobalV4Evidence("build-1", HASH, HASH, HASH, HASH, HASH, groups)


def test_global_v4_tracking_preserves_canonical_evidence_and_parent_fold_hierarchy(
    tmp_path: Path,
) -> None:
    port = RecordingPort()
    evidence = _evidence()
    tracked = module.track_global_v4_evaluation(
        port,
        StatisticsWriter(tmp_path),
        evidence=evidence,
        result=_result(),
        selections={1: _selection()},
    )

    parent_dir = tmp_path / "evaluations" / "global_regime_v4" / "run-1"
    canonical_path = parent_dir / "global_v4_evidence.json"
    assert tracked.parent_run_id == "run-1"
    assert tracked.outer_fold_run_ids == (("outer_fold_001", "run-2"),)
    assert canonical_path.read_bytes() == evidence.canonical_json()
    assert sha256(canonical_path.read_bytes()).hexdigest() == evidence.evidence_hash
    assert port.params["run-1"]["global_v4_evidence_sha256"] == evidence.evidence_hash
    assert (
        sha256((parent_dir / "statistics.json").read_bytes()).hexdigest()
        == port.params["run-1"]["statistics_sha256"]
    )
    child_dir = tmp_path / "evaluations" / "global_regime_v4" / "run-2"
    assert (
        sha256((child_dir / "statistics.json").read_bytes()).hexdigest()
        == port.params["run-2"]["statistics_sha256"]
    )
    manifest = json.loads(Path(tracked.plot_manifest_path).read_text(encoding="utf-8"))
    assert len(manifest["entries"]) >= 10
    assert all(item["png_path"].startswith("plots/") for item in manifest["entries"])
    assert not tuple(tmp_path.rglob("*.svg"))
    assert port.failed == []
    assert port.finished == ["run-2", "run-1"]


def test_global_v4_plot_failure_fails_the_parent_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        raise RuntimeError("plot rendering failed")

    monkeypatch.setattr(module, "render_global_v4_diagnostics", fail)
    port = RecordingPort()
    with pytest.raises(RuntimeError, match="plot rendering failed"):
        module.track_global_v4_evaluation(
            port,
            StatisticsWriter(tmp_path),
            evidence=_evidence(),
            result=_result(),
            selections={1: _selection()},
        )
    assert port.finished == ["run-2"]
    assert port.failed == ["run-1"]


def test_global_v4_tracking_persists_file_store_parent_child_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "mlruns").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="global-v4-test")
    tracked = module.track_global_v4_evaluation(
        port,
        StatisticsWriter(tmp_path / "statistics"),
        evidence=_evidence(),
        result=_result(),
        selections={1: _selection()},
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    parent = client.get_run(tracked.parent_run_id)
    child_id = tracked.outer_fold_run_ids[0][1]
    child = client.get_run(child_id)
    assert child.data.tags["mlflow.parentRunId"] == tracked.parent_run_id
    assert parent.data.params["global_v4_evidence_sha256"] == _evidence().evidence_hash
    assert [item.path for item in client.list_artifacts(tracked.parent_run_id, "evidence")] == [
        "evidence/global_v4_evidence.json"
    ]
    assert {item.path for item in client.list_artifacts(tracked.parent_run_id, "plots")} >= {
        "plots/global_v4_plot_manifest.json",
        "plots/quality_eligibility.png",
        "plots/final_12_model_same_vector_outer_fold_001.png",
    }
    assert [item.path for item in client.list_artifacts(child_id, "statistics")] == [
        "statistics/statistics.json"
    ]
