from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import nan

import pytest

from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.feature_discovery.feature_subset_score import (
    FeatureSubsetFoldEvidence,
)
from market_regime_engine.feature_discovery.k_sffs import select_k_slot_sffs
from market_regime_engine.feature_discovery.sffs import (
    FrontierFeatureSubsetEvaluator,
    FrontierFoldJob,
    select_sffs,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.runtime.task_frontier import (
    FrontierMetrics,
    FrontierResult,
    SharedTaskFrontier,
)
from market_regime_engine.training.multistart import (
    MINIMUM_SUCCESS_RATE,
    MINIMUM_VALID_STARTS,
    MULTISTART_SEEDS,
    TRAIN_LOGLIK_TIE_ABS_TOLERANCE,
    MultistartBatchJob,
    MultistartResult,
    StartDiagnostic,
    _anchored_winner,
    _evaluate_start,
    _FrontierStartPayload,
    run_multistart,
    run_multistart_batch,
)


def artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("x",),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.9, 0.1), (0.1, 0.9)),
        means=((-1.0,), (1.0,)),
        full_covariances=(((1.0,),), ((1.0,),)),
    )


def fit_result(seed: int, loglik: float, *, converged: bool = True) -> FitResult:
    return FitResult(
        artifact=artifact(),
        train_log_likelihood=loglik,
        converged=converged,
        iterations=17,
        seed=seed,
        em_log_likelihood_history=tuple(float(index) for index in range(17)),
    )


class FakeAdapter:
    def __init__(self, outcomes: Mapping[int, FitResult | Exception]) -> None:
        self._outcomes = outcomes

    def fit(self, train_rows: object, state_count: int, seed: int) -> FitResult:
        del train_rows, state_count
        outcome = self._outcomes[seed]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def extract(self) -> GaussianHMMArtifact:
        raise AssertionError("unused")

    def reconstruct(self, artifact: GaussianHMMArtifact) -> None:
        raise AssertionError(f"unused: {artifact}")


@dataclass(frozen=True)
class PickleableAdapterFactory:
    outcomes: Mapping[int, FitResult | Exception]

    def __call__(self) -> FakeAdapter:
        return FakeAdapter(self.outcomes)

    def causal_filter(
        self,
        rows: object,
        initial_filtered_probabilities: tuple[float, ...] | None = None,
    ) -> object:
        raise AssertionError(f"unused: {rows}, {initial_filtered_probabilities}")


def _frontier_sffs_jobs(state_count: int, features: tuple[str, ...]) -> tuple[FrontierFoldJob, ...]:
    outcomes = {seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS}
    return tuple(
        FrontierFoldJob(
            features,
            fold_id,
            MultistartBatchJob(
                f"{fold_id}:{','.join(features)}",
                [[0.0], [1.0]],
                state_count,
                PickleableAdapterFactory(outcomes),
            ),
        )
        for fold_id in ("fold-001", "fold-002", "fold-003")
    )


def _frontier_sffs_evidence(
    job: FrontierFoldJob, result: MultistartResult
) -> FeatureSubsetFoldEvidence:
    assert result.valid_start_count == 8
    return FeatureSubsetFoldEvidence(
        job.fold_id,
        True,
        latest=job.fold_id == "fold-003",
        target_log_score=float(len(job.candidate_subset)),
        baseline_target_log_score=0.0,
        calibration_error=0.2,
        stability_score=0.8,
        support_score=0.9,
    )


def factory(outcomes: Mapping[int, FitResult | Exception]):
    return lambda: FakeAdapter(outcomes)


def test_exact_seed_set_gates_and_train_loglik_winner() -> None:
    assert MULTISTART_SEEDS == (11, 23, 37, 53, 71, 89, 107, 131)
    assert MINIMUM_VALID_STARTS == 6
    assert MINIMUM_SUCCESS_RATE == 0.75
    assert TRAIN_LOGLIK_TIE_ABS_TOLERANCE == 1e-12
    outcomes: dict[int, FitResult | Exception] = {
        seed: fit_result(seed, float(index)) for index, seed in enumerate(MULTISTART_SEEDS)
    }
    outcomes[107] = RecoverableEvaluationInvalidity("numerical failure")
    outcomes[131] = fit_result(131, 100.0, converged=False)

    result = run_multistart(
        [[0.0], [1.0]], state_count=2, adapter_factory=factory(outcomes), max_workers=1
    )
    assert result.valid_start_count == 6
    assert result.success_rate == 0.75
    assert result.winner.seed == 89
    assert result.winner.em_log_likelihood_history == tuple(float(index) for index in range(17))
    assert tuple(item.seed for item in result.diagnostics) == MULTISTART_SEEDS
    assert result.diagnostics[6].failure_reason == (
        "RecoverableEvaluationInvalidity: numerical failure"
    )
    assert result.diagnostics[7].failure_reason == "not converged"


def test_numeric_tie_within_1e12_prefers_lower_seed() -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: fit_result(seed, -100.0) for seed in MULTISTART_SEEDS
    }
    outcomes[11] = fit_result(11, 10.0)
    outcomes[23] = fit_result(23, 10.0 + 0.5e-12)
    result = run_multistart(
        [[0.0]], state_count=2, adapter_factory=factory(outcomes), max_workers=1
    )
    assert result.winner.seed == 11


def test_difference_above_tolerance_selects_higher_loglik() -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: fit_result(seed, -100.0) for seed in MULTISTART_SEEDS
    }
    outcomes[11] = fit_result(11, 10.0)
    outcomes[23] = fit_result(23, 10.0 + 2e-12)
    result = run_multistart(
        [[0.0]], state_count=2, adapter_factory=factory(outcomes), max_workers=1
    )
    assert result.winner.seed == 23


def test_global_anchor_rejects_pairwise_likelihood_tolerance_chain() -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: fit_result(seed, -100.0) for seed in MULTISTART_SEEDS
    }
    outcomes[11] = fit_result(11, 10.0)
    outcomes[23] = fit_result(23, 10.0 + 0.75e-12)
    outcomes[37] = fit_result(37, 10.0 + 1.5e-12)

    result = run_multistart(
        [[0.0]], state_count=2, adapter_factory=factory(outcomes), max_workers=1
    )

    assert result.winner.seed == 23


def test_anchored_winner_requires_successful_starts() -> None:
    with pytest.raises(ValueError, match="requires successful starts"):
        _anchored_winner([])


def test_fewer_than_six_valid_starts_fails_with_failure_evidence() -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: RecoverableEvaluationInvalidity(f"fail-{seed}") for seed in MULTISTART_SEEDS
    }
    for seed in MULTISTART_SEEDS[:5]:
        outcomes[seed] = fit_result(seed, float(seed))
    with pytest.raises(ValueError, match="valid_starts=5/8") as exc_info:
        run_multistart([[0.0]], state_count=3, adapter_factory=factory(outcomes), max_workers=1)
    assert "fail-89" in str(exc_info.value)


def test_invalid_result_paths_are_counted_as_failed_starts() -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS
    }
    outcomes[11] = replace(outcomes[11], iterations=0, em_log_likelihood_history=())  # type: ignore[arg-type]
    outcomes[23] = replace(outcomes[23], train_log_likelihood=nan)  # type: ignore[arg-type]
    outcomes[37] = replace(outcomes[37], converged=False)
    with pytest.raises(ValueError, match="valid_starts=5/8"):
        run_multistart([[0.0]], state_count=4, adapter_factory=factory(outcomes), max_workers=1)


def test_invalid_state_count_fails_before_adapter_use() -> None:
    with pytest.raises(ValueError, match="K=2,3,4,5"):
        run_multistart([[0.0]], state_count=6, adapter_factory=factory({}))


def test_pickleable_custom_adapter_factory_uses_the_shared_frontier() -> None:
    outcomes = {seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS}
    result = run_multistart(
        [[0.0], [1.0]],
        state_count=2,
        adapter_factory=PickleableAdapterFactory(outcomes),
        max_workers=2,
    )

    assert result.winner.seed == 131


def test_multistart_can_use_a_caller_owned_shared_frontier() -> None:
    outcomes = {seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS}
    with SharedTaskFrontier(max_workers=2) as frontier:
        result = run_multistart(
            [[0.0], [1.0]],
            state_count=2,
            adapter_factory=PickleableAdapterFactory(outcomes),
            max_workers=2,
            frontier=frontier,
        )

    assert result.winner.seed == 131
    assert tuple(item.seed for item in result.diagnostics) == MULTISTART_SEEDS


def test_multistart_batch_flattens_all_fold_seeds_on_one_frontier() -> None:
    outcomes = {seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS}
    jobs = (
        MultistartBatchJob("fold-001", [[0.0], [1.0]], 2, PickleableAdapterFactory(outcomes)),
        MultistartBatchJob("fold-002", [[2.0], [3.0]], 2, PickleableAdapterFactory(outcomes)),
    )

    with SharedTaskFrontier(max_workers=2) as frontier:
        parallel = run_multistart_batch(jobs, max_workers=2, frontier=frontier)
    serial = run_multistart_batch(jobs, max_workers=1)

    assert tuple(item.winner.seed for item in parallel) == (131, 131)
    assert parallel == serial


def test_multistart_batch_worker_budget_matrix_preserves_two_fold_results() -> None:
    outcomes = {seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS}
    jobs = (
        MultistartBatchJob(
            "fold-budget-001", [[0.0], [1.0]], 2, PickleableAdapterFactory(outcomes)
        ),
        MultistartBatchJob(
            "fold-budget-002", [[2.0], [3.0]], 2, PickleableAdapterFactory(outcomes)
        ),
    )

    results = tuple(
        run_multistart_batch(jobs, max_workers=worker_budget)
        for worker_budget in (1, 8, 32, 64, None)
    )
    assert all(result == results[0] for result in results)


def test_batch_typed_invalidity_is_fold_evidence_when_requested() -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: RecoverableEvaluationInvalidity(f"fail-{seed}") for seed in MULTISTART_SEEDS
    }
    for seed in MULTISTART_SEEDS[:5]:
        outcomes[seed] = fit_result(seed, float(seed))
    job = MultistartBatchJob("fold-invalid", [[0.0], [1.0]], 2, PickleableAdapterFactory(outcomes))

    with pytest.raises(RecoverableEvaluationInvalidity):
        run_multistart_batch((job,), max_workers=1)
    assert run_multistart_batch((job,), max_workers=1, allow_invalid=True) == (None,)


def test_batch_frontier_tasks_carry_matrix_metadata_not_full_rows() -> None:
    outcomes = {seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS}
    job = MultistartBatchJob("fold-metadata", [[0.0], [1.0]], 2, PickleableAdapterFactory(outcomes))
    captured: list[object] = []

    class CaptureFrontier:
        def map(self, tasks, worker):
            ordered = tuple(tasks)
            captured.extend(task.payload for task in ordered)
            values = tuple((task, worker(task)) for task in ordered)
            return FrontierResult(
                values,
                FrontierMetrics(
                    submitted_count=len(values),
                    completed_count=len(values),
                    max_queue_depth=len(values),
                    runnable_tasks=len(values),
                    elapsed_seconds=0.0,
                    worker_utilization_proxy=1.0,
                ),
            )

    result = run_multistart_batch((job,), max_workers=2, frontier=CaptureFrontier())
    assert result[0] is not None
    assert len(captured) == len(MULTISTART_SEEDS)
    assert all(isinstance(payload, _FrontierStartPayload) for payload in captured)
    assert all(not hasattr(payload, "train_rows") for payload in captured)


def test_frontier_feature_subset_evaluator_feeds_sffs_from_fold_evidence() -> None:
    with SharedTaskFrontier(max_workers=2) as frontier:
        evaluator = FrontierFeatureSubsetEvaluator(
            _frontier_sffs_jobs,
            _frontier_sffs_evidence,
            "a" * 64,
            "build-001",
            "b" * 64,
            "fold-003",
            2,
            max_workers=2,
            frontier=frontier,
        )
        result = select_sffs(("a", "b"), evaluator, max_features=2, max_workers=2)

    assert result.selected_features == ("a", "b")
    assert len(result.evaluations) >= 3
    serial = select_sffs(
        ("a", "b"),
        FrontierFeatureSubsetEvaluator(
            _frontier_sffs_jobs,
            _frontier_sffs_evidence,
            "a" * 64,
            "build-001",
            "b" * 64,
            "fold-003",
            2,
            max_workers=1,
        ),
        max_features=2,
        max_workers=1,
    )
    assert result == serial


def test_frontier_feature_subset_evaluator_supports_independent_k_slots() -> None:
    with SharedTaskFrontier(max_workers=2) as frontier:
        evaluator = FrontierFeatureSubsetEvaluator(
            _frontier_sffs_jobs,
            _frontier_sffs_evidence,
            "a" * 64,
            "build-001",
            "b" * 64,
            "fold-003",
            2,
            max_workers=2,
            frontier=frontier,
        )
        results = select_k_slot_sffs(
            ("a", "b"),
            evaluator,
            state_counts=(2, 3),
            max_features=2,
            max_workers=2,
            frontier=frontier,
        )

    assert tuple(item.state_count for item in results) == (2, 3)
    assert all(item.selected_features == ("a", "b") for item in results)


def test_non_pickleable_adapter_factory_fails_before_thread_fallback() -> None:
    outcomes = {seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS}

    with pytest.raises(
        TypeError,
        match="CPU-bound multistart requires a pickleable adapter_factory",
    ):
        run_multistart(
            [[0.0], [1.0]],
            state_count=2,
            adapter_factory=lambda: FakeAdapter(outcomes),
            max_workers=2,
        )


def test_non_pickleable_adapter_factory_can_run_serially() -> None:
    outcomes = {seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS}

    result = run_multistart(
        [[0.0], [1.0]],
        state_count=2,
        adapter_factory=lambda: FakeAdapter(outcomes),
        max_workers=1,
    )

    assert result.winner.seed == 131


def test_multistart_contracts_reject_invalid_diagnostics_and_results() -> None:
    with pytest.raises(ValueError, match="outside"):
        StartDiagnostic(999, False, False, None, None, None, "bad")
    with pytest.raises(ValueError, match="successful start"):
        StartDiagnostic(11, True, False, 1, 1.0, artifact(), None)
    with pytest.raises(ValueError, match="positive iterations"):
        StartDiagnostic(11, True, True, 0, 1.0, artifact(), None)
    with pytest.raises(ValueError, match="finite"):
        StartDiagnostic(11, True, True, 1, nan, artifact(), None)
    with pytest.raises(ValueError, match="failure reason"):
        StartDiagnostic(11, False, False, None, None, None, None)

    diagnostics = tuple(
        StartDiagnostic(seed, True, True, 1, 1.0, artifact(), None) for seed in MULTISTART_SEEDS
    )
    winner = fit_result(11, 1.0)
    with pytest.raises(ValueError, match="state_count"):
        MultistartResult(1, winner, diagnostics)
    with pytest.raises(ValueError, match="winner"):
        MultistartResult(2, fit_result(999, 1.0), diagnostics)  # type: ignore[arg-type]


def test_evaluate_start_does_not_hide_unexpected_adapter_contract_failures() -> None:
    mismatch = fit_result(23, 1.0)
    with pytest.raises(ValueError, match="mismatched seed"):
        _evaluate_start(
            [[0.0]],
            state_count=2,
            adapter_factory=factory({11: mismatch}),
            seed=11,
        )


@pytest.mark.parametrize(
    "message",
    (
        "array must not contain infs or NaNs",
        "transition row must sum to one within 1e-10",
    ),
)
def test_evaluate_start_counts_numerical_backend_errors_as_failed_starts(
    message: str,
) -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS
    }
    outcomes.update(
        {
            seed: ValueError(message)
            for seed in MULTISTART_SEEDS[:3]
        }
    )
    with pytest.raises(ValueError, match="valid_starts=5/8"):
        run_multistart(
            [[0.0]],
            state_count=2,
            adapter_factory=factory(outcomes),
            max_workers=1,
        )
