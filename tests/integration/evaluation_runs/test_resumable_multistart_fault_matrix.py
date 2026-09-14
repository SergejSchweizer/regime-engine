from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from market_regime_engine.evaluation_runs.contracts import EvaluationRunIdentity
from market_regime_engine.evaluation_runs.hmm_units import (
    HMMSeedCheckpoint,
    SeedFitOutcome,
)
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult, GaussianHMMAdapter
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    run_multistart,
)

pytestmark = pytest.mark.integration


def _identity() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        "global_regime_v4",
        "xetra",
        4,
        "a" * 64,
        1,
        "b" * 64,
        "c" * 64,
        datetime(2026, 1, 1, tzinfo=UTC),
        "d" * 40,
        "e" * 64,
        "3.14.7",
    )


def _artifact(model_family: str = "gaussian_hmm") -> GaussianHMMArtifact:
    mixture_weights = ((0.25, 0.75), (0.6, 0.4)) if model_family == "gmm_hmm" else None
    mixture_means = (((-1.0,), (1.0,)), ((-0.5,), (1.5,))) if model_family == "gmm_hmm" else None
    mixture_covariances = (
        ((((1.0,),), ((1.0,),)), (((1.0,),), ((1.0,),))) if model_family == "gmm_hmm" else None
    )
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("feature_0",),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.9, 0.1), (0.1, 0.9)),
        means=((-1.0,), (1.0,)),
        full_covariances=(((1.0,),), ((1.0,),)),
        model_family=model_family,
        mixture_weights=mixture_weights,
        mixture_means=mixture_means,
        mixture_full_covariances=mixture_covariances,
        degrees_of_freedom=(5.5, 9.0) if model_family == "student_t_hmm" else None,
    )


class _RecordingAdapter:
    def __init__(self, calls: list[int], model_family: str = "gaussian_hmm") -> None:
        self._calls = calls
        self._model_family = model_family

    def fit(self, rows: object, state_count: int, seed: int) -> FitResult:
        del rows
        assert state_count == 2
        self._calls.append(seed)
        return FitResult(
            artifact=_artifact(self._model_family),
            train_log_likelihood=100.0 - seed / 1000.0,
            converged=True,
            iterations=3,
            seed=seed,
        )


def _adapter_factory(
    calls: list[int], model_family: str = "gaussian_hmm"
) -> Callable[[], GaussianHMMAdapter]:
    return cast(
        Callable[[], GaussianHMMAdapter],
        lambda: _RecordingAdapter(calls, model_family),
    )


class _CheckpointBoundaryFault:
    """Inject one interruption immediately before or after a durable save."""

    def __init__(
        self,
        checkpoint: HMMSeedCheckpoint,
        *,
        failure_seed: int,
        mode: str,
    ) -> None:
        self._checkpoint = checkpoint
        self._failure_seed = failure_seed
        self._mode = mode
        self._save_calls = 0

    def load(self, seed: int) -> SeedFitOutcome | None:
        return self._checkpoint.load(seed)

    def save(self, seed: int, outcome: SeedFitOutcome) -> None:
        self._save_calls += 1
        if seed == self._failure_seed and self._mode == "before_save":
            raise KeyboardInterrupt(f"fault before seed {seed} checkpoint")
        self._checkpoint.save(seed, outcome)
        if seed == self._failure_seed and self._mode == "after_save":
            raise KeyboardInterrupt(f"fault after seed {seed} checkpoint")


@pytest.mark.parametrize(
    "failure_seed",
    (MULTISTART_SEEDS[0], MULTISTART_SEEDS[2], MULTISTART_SEEDS[5]),
    ids=("seed-position-1", "seed-position-3", "seed-position-6"),
)
@pytest.mark.parametrize("mode", ("before_save", "after_save"))
@pytest.mark.parametrize("model_family", ("gaussian_hmm", "gmm_hmm", "student_t_hmm"))
def test_seed_checkpoint_interruption_matrix_resumes_without_duplicate_terminal_work(
    tmp_path: Path,
    failure_seed: int,
    mode: str,
    model_family: str,
) -> None:
    """Every production model family resumes to the uninterrupted golden result."""

    identity = _identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    store.open_run(identity)
    checkpoint = HMMSeedCheckpoint(
        run_identity=identity,
        store=store,
        candidate_id="gaussian_hmm_k2_full",
        fold_id="fold_001",
        state_count=2,
    )
    boundary = MULTISTART_SEEDS.index(failure_seed) + 1
    persisted_before_failure = boundary - (mode == "before_save")
    expected_persisted = set(MULTISTART_SEEDS[:persisted_before_failure])

    first_calls: list[int] = []
    fault_checkpoint = _CheckpointBoundaryFault(
        checkpoint,
        failure_seed=failure_seed,
        mode=mode,
    )
    with pytest.raises(KeyboardInterrupt, match="seed"):
        run_multistart(
            [[0.0], [1.0]],
            state_count=2,
            adapter_factory=_adapter_factory(first_calls, model_family),
            max_workers=1,
            checkpoint=cast(HMMSeedCheckpoint, fault_checkpoint),
        )

    assert first_calls == list(MULTISTART_SEEDS[:boundary])
    assert {
        seed for seed in MULTISTART_SEEDS if checkpoint.load(seed) is not None
    } == expected_persisted

    restart_calls: list[int] = []
    resumed = run_multistart(
        [[0.0], [1.0]],
        state_count=2,
        adapter_factory=_adapter_factory(restart_calls, model_family),
        max_workers=1,
        checkpoint=checkpoint,
    )
    assert resumed.winner.seed == MULTISTART_SEEDS[0]
    assert restart_calls == [seed for seed in MULTISTART_SEEDS if seed not in expected_persisted]

    # Every terminal seed has one durable claim/commit attempt. In particular,
    # an after-save interruption must not cause the committed seed to be fit
    # again on restart.
    for seed in MULTISTART_SEEDS:
        state = store.work_unit_state(identity, checkpoint.unit(seed))
        assert state is not None
        assert state.status.value == "COMPLETE"
        assert state.payload_hash is not None
        assert state.attempt_count == 1

    golden = run_multistart(
        [[0.0], [1.0]],
        state_count=2,
        adapter_factory=_adapter_factory([], model_family),
        max_workers=1,
    )
    assert resumed == golden
    assert resumed.winner.artifact.model_family == model_family
    assert all(
        diagnostic.artifact is None or diagnostic.artifact.model_family == model_family
        for diagnostic in resumed.diagnostics
    )

    replay_calls: list[int] = []
    replayed = run_multistart(
        [[0.0], [1.0]],
        state_count=2,
        adapter_factory=_adapter_factory(replay_calls, model_family),
        max_workers=1,
        checkpoint=checkpoint,
    )
    assert replayed == resumed
    assert replay_calls == []
