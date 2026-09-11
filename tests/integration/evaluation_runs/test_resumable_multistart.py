from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from market_regime_engine.evaluation_runs.contracts import EvaluationRunIdentity
from market_regime_engine.evaluation_runs.hmm_units import HMMSeedCheckpoint
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    MultistartResult,
    run_multistart,
)

pytestmark = pytest.mark.integration


def run_identity() -> EvaluationRunIdentity:
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


def artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("f0",),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.9, 0.1), (0.1, 0.9)),
        means=((-1.0,), (1.0,)),
        full_covariances=(((1.0,),), ((1.0,),)),
    )


class Adapter:
    def __init__(self, calls: list[int], interrupted: set[int]) -> None:
        self.calls = calls
        self.interrupted = interrupted

    def fit(self, rows: object, state_count: int, seed: int) -> FitResult:
        del rows
        assert state_count == 2
        self.calls.append(seed)
        if seed in self.interrupted:
            self.interrupted.remove(seed)
            raise KeyboardInterrupt("forced interruption")
        return FitResult(artifact(), 100.0 - seed / 1000.0, True, 3, seed)


def test_resume_reuses_completed_seed_fits_after_interruption(tmp_path: Path) -> None:
    identity = run_identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    store.open_run(identity)
    checkpoint = HMMSeedCheckpoint(
        run_identity=identity,
        store=store,
        candidate_id="gaussian_hmm_k2_full",
        fold_id="fold_001",
        state_count=2,
    )
    calls: list[int] = []
    interrupted = {37}
    with pytest.raises(KeyboardInterrupt):
        run_multistart(
            [[0.0], [1.0]],
            state_count=2,
            adapter_factory=lambda: Adapter(calls, interrupted),
            checkpoint=checkpoint,
        )
    # With the production default, all independent seeds are submitted to the
    # full CPU worker pool. The interrupted seed may therefore be observed in
    # any position, while the other seeds can finish and checkpoint before the
    # interruption is surfaced to the caller.
    assert set(calls) == set(MULTISTART_SEEDS)
    assert calls.count(37) == 1

    result = run_multistart(
        [[0.0], [1.0]],
        state_count=2,
        adapter_factory=lambda: Adapter(calls, interrupted),
        checkpoint=checkpoint,
    )
    assert result.winner.seed == 11
    assert calls.count(37) == 2
    assert all(calls.count(seed) == 1 for seed in MULTISTART_SEEDS if seed != 37)


def test_concurrent_multistart_workers_commit_one_consistent_seed_payload(
    tmp_path: Path,
) -> None:
    identity = run_identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    store.open_run(identity)
    checkpoint = HMMSeedCheckpoint(
        run_identity=identity,
        store=store,
        candidate_id="gaussian_hmm_k2_full",
        fold_id="fold_001",
        state_count=2,
    )

    def execute() -> MultistartResult:
        return run_multistart(
            [[0.0], [1.0]],
            state_count=2,
            adapter_factory=lambda: Adapter([], set()),
            checkpoint=checkpoint,
            max_workers=8,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = tuple(pool.map(lambda _index: execute(), (1, 2)))

    assert first.winner.seed == second.winner.seed
    assert all(checkpoint.load(seed) is not None for seed in MULTISTART_SEEDS)
