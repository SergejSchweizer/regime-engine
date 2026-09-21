from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import nan

import pytest

import market_regime_engine.training.multistart as multistart_module
from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.evaluations.task_frontier import SharedTaskFrontier
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult
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


def test_checkpointed_technical_start_failure_remains_retryable() -> None:
    error = RuntimeError("temporary backend failure")

    with pytest.raises(RuntimeError, match="temporary backend failure"):
        multistart_module._evaluate_start(
            [[0.0]],
            state_count=2,
            adapter_factory=factory({MULTISTART_SEEDS[0]: error}),
            seed=MULTISTART_SEEDS[0],
            retryable_technical_failure=True,
        )


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
