from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mlflow.tracking import MlflowClient

import market_regime_engine.mlflow_support.evaluation_tracking as module
import market_regime_engine.mlflow_support.tracking as tracking_module
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation_statistics.contracts import GlobalV4Evidence, RunType, Status
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    FinalSelectedConfiguration,
    OuterFoldResult,
)
from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.mlflow_support.tracking import (
    FileMlflowTrackingPort,
    _safe_logged_model_name,
)
from tests.unit.mlflow_support.test_all_plotting_functions import (
    _evaluation as plotting_evaluation,
)
from tests.unit.mlflow_support.test_all_plotting_functions import (
    _plan as plotting_plan,
)

HASH = "a" * 64
START = datetime(2024, 1, 1, tzinfo=UTC)


def test_logged_model_name_encoding_preserves_logical_key_without_mlflow_delimiters() -> None:
    logical = "global_regime_v4:run/outer_fold_001.candidate"
    encoded = _safe_logged_model_name(logical)

    assert encoded == "global_regime_v4_x3a_run_x2f_outer_fold_001_x2e_candidate"
    assert all(character.isalnum() or character in "_-" for character in encoded)


@pytest.mark.parametrize("logical_name", ["", " candidate", "candidate ", " "])
def test_logged_model_name_rejects_empty_or_untrimmed_keys(logical_name: str) -> None:
    with pytest.raises(ValueError, match="non-empty and trimmed"):
        _safe_logged_model_name(logical_name)


def test_file_tracking_port_rejects_duplicate_models_and_missing_model_source_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = SimpleNamespace(model_id="model-1", name="candidate", model_type="hmm", tags={})
    client = SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(
            experiment_id="exp-1", lifecycle_stage="active"
        ),
        search_logged_models=lambda ids: [duplicate, duplicate],
        get_logged_model=lambda model_id: SimpleNamespace(source_run_id=None, metrics=()),
    )
    monkeypatch.setattr(tracking_module, "MlflowClient", lambda tracking_uri: client)
    port = FileMlflowTrackingPort("file:///tmp/mlflow")
    with pytest.raises(ValueError, match="multiple LoggedModels"):
        port.create_logged_model(name="candidate", source_run_id="run-1", model_type="hmm", tags={})

    client.search_logged_models = lambda ids: []
    with pytest.raises(ValueError, match="no source run"):
        port.log_model_metric_points("unknown", (MetricPoint("score", 1.0, 0, 0),))


def test_file_tracking_port_reads_model_metrics_from_client_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        model_id="model-1",
        name="candidate",
        model_type="hmm",
        tags={},
        source_run_id="run-1",
        metrics=(SimpleNamespace(key="score", value=0.5, step=2, timestamp=3),),
    )
    client = SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(
            experiment_id="exp-1", lifecycle_stage="active"
        ),
        search_logged_models=lambda ids: [model],
        get_logged_model=lambda model_id: model,
        _tracking_client=SimpleNamespace(store=None),
    )
    monkeypatch.setattr(tracking_module, "MlflowClient", lambda tracking_uri: client)
    port = FileMlflowTrackingPort("file:///tmp/mlflow")
    assert port.get_model_metric_points("model-1") == (MetricPoint("score", 0.5, 2, 3),)


def test_tracking_validation_rejects_empty_selection_and_identity_mismatches(
    tmp_path: Path,
) -> None:
    evaluation = plotting_evaluation(valid_fold_count=1)
    plan = plotting_plan()
    lineage = SourceLineage(
        source_dataset="macro_loader.macro_features_daily",
        source_build_id="synthetic-build",
        data_sha256=HASH,
        schema_version=1,
        feature_version=1,
        source_table="macro_loader.macro_features_daily",
        synced_at_utc=START,
        row_count=1260,
        min_timestamp=plan.folds[0].train_start,
        max_timestamp=plan.folds[-1].test_end,
    )
    kwargs = dict(
        port=RecordingPort(),
        source_lineage=lineage,
        plan=plan,
        evaluations=(evaluation,),
        statistical_selection_result="selected",
        artifact_root=tmp_path,
        max_workers=1,
    )
    with pytest.raises(ValueError, match="at least one candidate"):
        tracking_module.track_walk_forward_evaluations(**{**kwargs, "evaluations": ()})
    with pytest.raises(ValueError, match="non-empty trimmed"):
        tracking_module.track_walk_forward_evaluations(
            **{**kwargs, "statistical_selection_result": " selected"}
        )
    with pytest.raises(ValueError, match="source build"):
        tracking_module.track_walk_forward_evaluations(
            **{
                **kwargs,
                "evaluations": (replace(evaluation, source_build_id="other-build"),),
            }
        )
    with pytest.raises(ValueError, match="dataset_snapshot_key"):
        tracking_module.track_walk_forward_evaluations(
            **{**kwargs, "dataset_snapshot_key": " invalid"}
        )


def test_file_tracking_port_exercises_run_batch_and_logged_model_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def __init__(self, tracking_uri: str) -> None:
            assert tracking_uri == "file:///tmp/mlflow"
            self.experiment = SimpleNamespace(experiment_id="exp-1", lifecycle_stage="deleted")
            self.calls: list[tuple[str, object]] = []
            self.models: list[object] = []
            self._tracking_client = SimpleNamespace(store=None)

        def get_experiment_by_name(self, name: str) -> object:
            assert name == "macro-regime-evaluation"
            return self.experiment

        def restore_experiment(self, experiment_id: str) -> None:
            self.calls.append(("restore", experiment_id))

        def create_run(self, experiment_id: str, *, tags: dict[str, str]) -> object:
            self.calls.append(("create_run", (experiment_id, tags)))
            return SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

        def log_batch(self, run_id: str, **kwargs: object) -> None:
            self.calls.append(("batch", (run_id, kwargs)))

        def search_logged_models(self, experiment_ids: list[str]) -> list[object]:
            self.calls.append(("search", experiment_ids))
            return self.models

        def create_logged_model(self, experiment_id: str, **kwargs: object) -> object:
            self.calls.append(("create_model", (experiment_id, kwargs)))
            model = SimpleNamespace(model_id="model-1", **kwargs)
            self.models.append(model)
            return model

        def get_logged_model(self, model_id: str) -> object:
            assert model_id == "model-1"
            return SimpleNamespace(
                source_run_id="run-1",
                metrics=(SimpleNamespace(key="score", value=0.5, step=2, timestamp=3),),
            )

        def log_model_artifacts(self, model_id: str, local_dir: str) -> None:
            self.calls.append(("model_artifacts", (model_id, local_dir)))

        def finalize_logged_model(self, model_id: str, status: str) -> None:
            self.calls.append(("finalize", (model_id, status)))

        def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
            self.calls.append(("artifact", (run_id, local_path, artifact_path)))

        def set_terminated(self, run_id: str, *, status: str) -> None:
            self.calls.append(("terminate", (run_id, status)))

    client = Client("file:///tmp/mlflow")
    monkeypatch.setattr(tracking_module, "MlflowClient", lambda tracking_uri: client)
    port = FileMlflowTrackingPort("file:///tmp/mlflow")
    assert port.start_run(run_name="parent") == "run-1"
    assert port.start_run(run_name="child", parent_run_id="parent") == "run-1"
    port.log_params("run-1", {"z": "2", "a": "1"})
    points = tuple(MetricPoint("score", float(index), index, index) for index in range(1001))
    port.log_metric_points("run-1", points)
    model_id = port.create_logged_model(
        name="candidate",
        source_run_id="run-1",
        model_type="hmm",
        tags={"candidate": "k2"},
    )
    assert model_id == "model-1"
    assert (
        port.create_logged_model(
            name="candidate",
            source_run_id="run-2",
            model_type="hmm",
            tags={"candidate": "k2"},
        )
        == "model-1"
    )
    assert port.get_model_metric_points(model_id)[0].key == "score"
    port.log_model_metric_points(model_id, (MetricPoint("score", 1.0, 0, 0),))
    port.log_model_artifacts(model_id, "/tmp/model")
    port.finalize_logged_model(model_id)
    port.log_artifact("run-1", "/tmp/evidence", "evaluation")
    port.end_run("run-1")
    port.fail_run("run-1")
    assert ("restore", "exp-1") in client.calls
    assert sum(name == "batch" for name, _ in client.calls) >= 3


def test_file_tracking_port_rejects_logged_model_identity_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = SimpleNamespace(
        model_id="model-1", name="candidate", model_type="other", tags={"candidate": "k2"}
    )
    client = SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(
            experiment_id="exp-1", lifecycle_stage="active"
        ),
        create_run=lambda *args, **kwargs: SimpleNamespace(info=SimpleNamespace(run_id="run-1")),
        search_logged_models=lambda ids: [existing],
    )
    monkeypatch.setattr(tracking_module, "MlflowClient", lambda tracking_uri: client)
    port = FileMlflowTrackingPort("file:///tmp/mlflow")
    with pytest.raises(ValueError, match="conflicting identity"):
        port.create_logged_model(
            name="candidate",
            source_run_id="run-1",
            model_type="hmm",
            tags={"candidate": "k2"},
        )


def test_tracking_evidence_helpers_cover_valid_and_invalid_fold_payloads() -> None:
    evaluation = plotting_evaluation(valid_fold_count=1)
    plan = plotting_plan()
    fold = evaluation.folds[0]

    scalar = tracking_module._scalar_fold_metrics(fold)
    assert scalar["fold_train_loglik"] == pytest.approx(-100.0 / 1260.0)
    assert tracking_module._timestamp_ms(plan.folds[0]) > 0
    candidate_points = tracking_module._candidate_metric_points(evaluation, plan)
    aggregate_points = tracking_module._aggregate_metric_points(evaluation)
    assert candidate_points
    assert {point.key for point in aggregate_points} >= {
        "candidate_valid_fold_rate",
        "candidate_oos_predictive_loglik_mean",
    }
    tags = tracking_module._candidate_model_tags(
        evaluation,
        source_build_id="synthetic-build",
        plan=plan,
        dataset_snapshot_key="dataset-1",
        evaluation_run_key="evaluation-1",
        outer_fold_id="fold_001",
    )
    assert tags["regime_engine.pca_mandatory"] == "true"
    assert tags["regime_engine.outer_fold_id"] == "fold_001"
    assert tracking_module._timeline_rows(evaluation, plan)[1]["valid"] is False
    assert tracking_module._metric_rows(evaluation, plan)[0]["fold_aic"] == 100.0
    assert tracking_module._aligned_parameter_payload(evaluation, fold)["candidate_id"] == (
        evaluation.candidate_id
    )
    candidate_id, evidence = tracking_module._prepare_candidate_tracking_evidence(evaluation, plan)
    assert candidate_id == evaluation.candidate_id
    assert evidence.candidate_points == candidate_points

    invalid = evaluation.folds[1]
    with pytest.raises(ValueError, match="aligned model evidence"):
        tracking_module._aligned_parameter_payload(evaluation, invalid)


def test_walk_forward_tracking_orchestrator_is_hermetic_and_complete(tmp_path: Path) -> None:
    evaluation = plotting_evaluation(valid_fold_count=1)
    plan = plotting_plan()
    lineage = SourceLineage(
        source_dataset="macro_loader.macro_features_daily",
        source_build_id="synthetic-build",
        data_sha256=HASH,
        schema_version=1,
        feature_version=1,
        source_table="macro_loader.macro_features_daily",
        synced_at_utc=START,
        row_count=1260,
        min_timestamp=plan.folds[0].train_start,
        max_timestamp=plan.folds[-1].test_end,
    )
    port = RecordingPort()

    result = tracking_module.track_walk_forward_evaluations(
        port,
        source_lineage=lineage,
        plan=plan,
        evaluations=(evaluation,),
        statistical_selection_result="gaussian_hmm_k2_full",
        artifact_root=tmp_path,
        max_workers=1,
    )

    assert result.parent_run_id == "run-1"
    assert result.candidate_run_ids == ((evaluation.candidate_id, "run-2"),)
    assert result.parent_manifest_path == str(tmp_path / "parent" / "plot_manifest.json")
    assert port.params["run-1"]["source_build_id"] == lineage.source_build_id
    assert port.params["run-1"]["evaluation_plan_hash"] == plan.plan_hash
    assert (tmp_path / evaluation.candidate_id / "fold_timeline.parquet").is_file()
    assert (tmp_path / evaluation.candidate_id / "fold_metrics.parquet").is_file()
    assert port.finished == ["run-3", "run-4", "run-2", "run-1"]


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

    def get_model_metric_points(self, model_id: str) -> tuple[MetricPoint, ...]:
        del model_id
        return ()

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


def _selection() -> Any:
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


def test_outer_fold_evidence_worker_matches_inline_payload() -> None:
    fold = _result().outer_folds[0]
    selection = _selection()
    inline = module._build_fold_evidence(fold, selection)
    with cpu_process_pool(2) as executor:
        parallel = executor.submit(module._build_fold_evidence, fold, selection).result()
    assert parallel == inline


def test_adjacent_cluster_stability_is_derived_from_train_memberships() -> None:
    selection = _selection()
    first = SimpleNamespace(fold_index=1, valid=True)
    second = SimpleNamespace(fold_index=2, valid=True)
    result = SimpleNamespace(outer_folds=(first, second))

    stability = module._adjacent_cluster_stability(result, {1: selection, 2: selection})

    assert stability == [
        {
            "outer_fold_pair": [1, 2],
            "mean_best_cluster_jaccard": 1.0,
        }
    ]


def test_tracking_preparation_worker_matches_inline_payload() -> None:
    fold = _result().outer_folds[0]
    selection = _selection()
    inline = module._prepare_global_v4_fold_tracking(fold, selection)
    with cpu_process_pool(2) as executor:
        parallel = executor.submit(
            module._prepare_global_v4_fold_tracking,
            fold,
            selection,
        ).result()
    assert parallel == inline


def test_candidate_tracking_artifact_worker_writes_immutable_evidence_file(
    tmp_path: Path,
) -> None:
    candidate = SimpleNamespace(candidate_id="gaussian_hmm_k2_full", folds=())
    candidate_dir = tmp_path / "logged_models" / "outer_fold_001" / candidate.candidate_id

    with cpu_process_pool(2) as executor:
        executor.submit(
            module._materialize_candidate_tracking_artifacts,
            candidate,
            b'{"candidate":1}\n',
            candidate_dir,
        ).result()

    assert (candidate_dir / "candidate_evidence.json").read_bytes() == b'{"candidate":1}\n'
    assert not tuple(candidate_dir.glob("*.png"))


def test_plot_preparation_pool_covers_candidate_and_parent_tasks_in_canonical_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluations = (
        SimpleNamespace(candidate_id="candidate-b"),
        SimpleNamespace(candidate_id="candidate-a"),
    )
    worker_requests: list[tuple[int | None, int]] = []
    pool_sizes: list[int] = []
    submissions: list[tuple[str, str]] = []

    class ImmediateFuture:
        def __init__(self, value: object) -> None:
            self._value = value

        def result(self) -> object:
            return self._value

    class RecordingExecutor:
        def submit(self, function: Any, *args: object) -> ImmediateFuture:
            if function is tracking_module._render_candidate_artifacts:
                evaluation = args[0]
                assert isinstance(evaluation, SimpleNamespace)
                candidate_id = str(evaluation.candidate_id)
                submissions.append(("candidate", candidate_id))
                return ImmediateFuture((f"candidate:{candidate_id}",))
            plot_type = str(args[0])
            submissions.append(("parent", plot_type))
            return ImmediateFuture(f"parent:{plot_type}")

    def record_worker_count(requested: int | None, *, task_count: int) -> int:
        worker_requests.append((requested, task_count))
        return task_count

    @contextmanager
    def recording_pool(max_workers: int):
        pool_sizes.append(max_workers)
        yield RecordingExecutor()

    monkeypatch.setattr(tracking_module, "cpu_worker_count", record_worker_count)
    monkeypatch.setattr(tracking_module, "cpu_process_pool", recording_pool)

    by_candidate, parent_entries = tracking_module._prepare_plot_entries(
        evaluations,
        SimpleNamespace(),
        tmp_path,
        "candidate-a",
        None,
    )

    assert worker_requests == [(None, 5)]
    assert pool_sizes == [5]
    assert submissions == [
        ("candidate", "candidate-b"),
        ("candidate", "candidate-a"),
        ("parent", "candidate_comparison"),
        ("parent", "candidate_oos_gap_heatmap"),
        ("parent", "candidate_oos_summary"),
    ]
    assert tuple(by_candidate) == ("candidate-b", "candidate-a")
    assert by_candidate == {
        "candidate-b": ("candidate:candidate-b",),
        "candidate-a": ("candidate:candidate-a",),
    }
    assert parent_entries == (
        "parent:candidate_comparison",
        "parent:candidate_oos_gap_heatmap",
        "parent:candidate_oos_summary",
    )


def test_candidate_tracking_evidence_uses_bounded_process_tasks_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluations = (
        SimpleNamespace(candidate_id="candidate-b"),
        SimpleNamespace(candidate_id="candidate-a"),
    )
    worker_requests: list[tuple[int | None, int]] = []
    submissions: list[str] = []

    def fake_worker_count(requested: int | None, *, task_count: int) -> int:
        worker_requests.append((requested, task_count))
        return task_count

    def fake_prepare(evaluation: Any, plan: Any) -> tuple[str, object]:
        del plan
        submissions.append(str(evaluation.candidate_id))
        return str(evaluation.candidate_id), f"evidence:{evaluation.candidate_id}"

    class ImmediateFuture:
        def __init__(self, value: object) -> None:
            self._value = value

        def result(self) -> object:
            return self._value

    class RecordingExecutor:
        def submit(self, function: Any, *args: object) -> ImmediateFuture:
            return ImmediateFuture(function(*args))

    @contextmanager
    def recording_pool(max_workers: int):
        assert max_workers == 2
        yield RecordingExecutor()

    monkeypatch.setattr(tracking_module, "cpu_worker_count", fake_worker_count)
    monkeypatch.setattr(tracking_module, "cpu_process_pool", recording_pool)
    monkeypatch.setattr(tracking_module, "_prepare_candidate_tracking_evidence", fake_prepare)

    prepared = tracking_module._prepare_candidate_tracking_evidence_batch(
        evaluations,
        SimpleNamespace(),
        None,
    )

    assert worker_requests == [(None, 2)]
    assert submissions == ["candidate-b", "candidate-a"]
    assert tuple(prepared) == ("candidate-b", "candidate-a")
    assert prepared == {
        "candidate-b": "evidence:candidate-b",
        "candidate-a": "evidence:candidate-a",
    }


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
    assert port.params["run-1"]["regime_engine.runtime_scope"] == "tracking_run"
    assert float(port.params["run-1"]["regime_engine.runtime_seconds"]) >= 0.0
    assert port.params["run-2"]["regime_engine.runtime_scope"] == "tracking_run"
    assert float(port.params["run-2"]["regime_engine.runtime_seconds"]) >= 0.0
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


def test_global_v4_tracking_records_full_evaluation_runtime_metadata(
    tmp_path: Path,
) -> None:
    port = RecordingPort()
    started_at = datetime.now(UTC) - timedelta(seconds=2)
    started_monotonic = module.time.perf_counter() - 2.0

    module.track_global_v4_evaluation(
        port,
        StatisticsWriter(tmp_path),
        evidence=_evidence(),
        result=_result(),
        selections={1: _selection()},
        evaluation_started_at=started_at,
        evaluation_start_monotonic=started_monotonic,
    )

    assert port.params["run-1"]["regime_engine.runtime_scope"] == "full_evaluation"
    assert port.params["run-1"]["regime_engine.runtime_started_at_utc"] == started_at.isoformat()
    assert float(port.params["run-1"]["regime_engine.runtime_seconds"]) >= 2.0


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


def test_tracking_helpers_cover_failed_folds_and_validation_contracts(tmp_path: Path) -> None:
    fold = replace(_result().outer_folds[0], valid=False, failure_reason="synthetic failure")
    failed = module._failed_fold_evidence(fold)
    assert failed["validity"] == {"valid": False, "failure_reason": "synthetic failure"}
    prepared = module._prepare_global_v4_fold_tracking(fold, None)
    assert prepared.final_grid_plan is None
    assert prepared.candidates == ()
    assert module._json_bytes({"b": 1, "a": 2}).decode().startswith('{"a":2')

    with pytest.raises(ValueError, match="non-empty memberships"):
        module._cluster_membership_jaccard(
            SimpleNamespace(memberships=()), SimpleNamespace(memberships=(("cluster", ("f0",)),))
        )
    with pytest.raises(ValueError, match="every valid outer fold"):
        module._validate_inputs(_evidence(), _result(), {})
    with pytest.raises(ValueError, match="source build"):
        module._validate_inputs(
            _evidence(),
            replace(_result(), source_build_id="different-build"),
            {},
        )

    statistics = module._running_statistics("failed", RunType.PARENT, {"identity": {}})
    with pytest.raises(RuntimeError, match="emitter failure"):
        module.track_statistics_run(
            RecordingPort(),
            StatisticsWriter(tmp_path / "failed"),
            run_name="failed",
            statistics=statistics,
            payload_emitter=lambda *_args: (_ for _ in ()).throw(RuntimeError("emitter failure")),
        )
    with pytest.raises(ValueError, match="ended_at"):
        module.track_statistics_run(
            RecordingPort(),
            StatisticsWriter(tmp_path / "invalid"),
            run_name="invalid",
            statistics=replace(statistics, status=Status.FINISHED),
        )


def test_statistics_tracking_writes_runtime_metadata_and_finalizes(tmp_path: Path) -> None:
    port = RecordingPort()
    statistics = module._running_statistics("successful", RunType.PARENT, {"identity": {}})
    run_id, digest = module.track_statistics_run(
        port,
        StatisticsWriter(tmp_path / "successful"),
        run_name="successful",
        statistics=statistics,
        runtime_started_at=START,
        runtime_start_monotonic=0.0,
    )
    assert run_id == "run-1"
    assert len(digest) == 64
    assert port.finished == [run_id]
    assert port.params[run_id]["regime_engine.runtime_scope"] == "tracking_run"
    assert float(port.params[run_id]["regime_engine.runtime_seconds"]) >= 0.0


@pytest.mark.parametrize(
    "field",
    [
        "source_build_id",
        "catalog_hash",
    ],
)
def test_tracking_validation_rejects_unknown_and_mismatched_selections(
    field: str,
) -> None:
    evidence = _evidence()
    result = _result()
    selection = _selection()
    if field == "source_build_id":
        selection = SimpleNamespace(**{**vars(selection), "source_build_id": "other"})
    else:
        selection = SimpleNamespace(**{**vars(selection), "catalog_hash": "other"})
    with pytest.raises(ValueError, match="lineage"):
        module._validate_inputs(evidence, result, {1: selection})
    with pytest.raises(ValueError, match="unknown outer folds"):
        module._validate_inputs(evidence, result, {99: selection})


def test_candidate_tracking_preparation_builds_canonical_payload_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = SimpleNamespace(
        candidate_id="gaussian_hmm_k2_full",
        feature_order=("f0",),
        source_build_id="build-1",
        evaluation_plan_hash=HASH,
        folds=(),
    )
    aggregate = SimpleNamespace(candidate_id=candidate.candidate_id)
    selection = SimpleNamespace(
        final_grid=SimpleNamespace(
            candidate_grid=SimpleNamespace(evaluations=(candidate,), aggregates=(aggregate,))
        ),
        feature_discovery_hash=HASH,
    )
    fold = SimpleNamespace(fold_index=1, final_configuration=candidate)
    plan = SimpleNamespace(folds=(SimpleNamespace(test_end=START),))
    monkeypatch.setattr(module, "require_pca_artifacts", lambda value: ())
    monkeypatch.setattr(
        module, "_aggregate_record", lambda value: {"candidate_id": value.candidate_id}
    )
    monkeypatch.setattr(
        module,
        "_candidate_metric_points",
        lambda *args: (MetricPoint("metric", 1.0, 1, 0),),
    )
    monkeypatch.setattr(module, "_aggregate_metric_points", lambda *args: ())
    monkeypatch.setattr(module, "model_metric_points", lambda *args, **kwargs: ())
    monkeypatch.setattr(module, "outer_selection_metric_points", lambda *args: ())
    monkeypatch.setattr(module, "validate_metric_points", lambda points: None)

    prepared = module._prepare_global_v4_candidate_tracking(
        fold,
        selection,
        {"identity": {"fold_index": 1}},
        candidate,
        plan,
    )
    payload = json.loads(prepared.evidence_json)
    assert payload["candidate_id"] == candidate.candidate_id
    assert payload["aggregate"] == {"candidate_id": candidate.candidate_id}
    assert payload["selection_context"] == {"identity": {"fold_index": 1}}
    assert len(prepared.metric_points) == 1


def test_tracking_prepares_candidates_and_artifacts_in_worker_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _selection()
    candidate = SimpleNamespace(candidate_id="gaussian_hmm_k2_full")
    grid = SimpleNamespace(
        candidate_grid=SimpleNamespace(evaluations=(candidate,)),
    )
    selection = SimpleNamespace(
        **{**vars(base), "final_grid": grid, "final_grid_plan": SimpleNamespace()}
    )
    submissions: list[str] = []
    materialized: list[str] = []

    class ImmediateFuture:
        def __init__(self, value: object) -> None:
            self.value = value

        def result(self) -> object:
            return self.value

    class RecordingExecutor:
        def submit(self, function: Any, *args: object) -> ImmediateFuture:
            return ImmediateFuture(function(*args))

    @contextmanager
    def recording_pool(max_workers: int):
        assert max_workers == 2
        yield RecordingExecutor()

    def prepare(*args: object) -> module._PreparedCandidateTracking:
        submissions.append(str(args[3].candidate_id))
        return module._PreparedCandidateTracking(candidate, b"{}", ())

    def materialize(*args: object) -> None:
        materialized.append(str(args[0].candidate_id))

    monkeypatch.setattr(module, "cpu_worker_count", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(module, "cpu_process_pool", recording_pool)
    monkeypatch.setattr(module, "_build_fold_evidence", lambda *_args: {"fold": 1})
    monkeypatch.setattr(module, "_prepare_global_v4_candidate_tracking", prepare)
    monkeypatch.setattr(module, "_materialize_candidate_tracking_artifacts", materialize)
    monkeypatch.setattr(
        module, "_track_global_v4_fold", lambda *args, **kwargs: (("fold", "child"), ())
    )
    monkeypatch.setattr(module, "render_global_v4_diagnostics", lambda *args, **kwargs: ())

    tracked = module.track_global_v4_evaluation(
        RecordingPort(),
        StatisticsWriter(tmp_path),
        evidence=_evidence(),
        result=_result(),
        selections={1: selection},
    )
    assert submissions == [candidate.candidate_id]
    assert materialized == [candidate.candidate_id]
    assert tracked.outer_fold_run_ids == (("fold", "child"),)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("source_build_id", "source build differs"),
        ("catalog_hash", "catalog differs"),
        ("snapshot_lineage", "snapshot lineage"),
        ("feature_names", "snapshot columns"),
        ("repository_commit_sha", "non-empty and trimmed"),
    ],
)
def test_global_evidence_rejects_lineage_and_repository_identity_drift(
    change: str, message: str
) -> None:
    result = SimpleNamespace(source_build_id="build-1", catalog_hash=HASH, outer_folds=())
    lineage = SimpleNamespace(source_build_id="build-1")
    catalog = SimpleNamespace(lineage=lineage, catalog_hash=HASH, feature_names=("f0",))
    snapshot = SimpleNamespace(lineage=lineage, feature_names=("f0",), rows=())
    if change == "source_build_id":
        result.source_build_id = "other"
    elif change == "catalog_hash":
        result.catalog_hash = "other"
    elif change == "snapshot_lineage":
        snapshot.lineage = SimpleNamespace(source_build_id="other")
    elif change == "feature_names":
        snapshot.feature_names = ("other",)
    commit = " " if change == "repository_commit_sha" else "f" * 40
    with pytest.raises(ValueError, match=message):
        module.build_global_v4_evidence(
            result,
            catalog=catalog,
            snapshot=snapshot,
            profile=SimpleNamespace(walk_forward=SimpleNamespace()),
            selections={},
            repository_commit_sha=commit,
        )
