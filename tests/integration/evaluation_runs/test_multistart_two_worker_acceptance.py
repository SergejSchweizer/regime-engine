from __future__ import annotations

import hashlib
import multiprocessing
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from market_regime_engine.evaluation_runs.contracts import EvaluationRunIdentity
from market_regime_engine.evaluation_runs.hmm_units import (
    HMMSeedCheckpoint,
    SeedFitOutcome,
)
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore, WorkUnitStatus
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    StartDiagnostic,
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


def _artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("feature_0",),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.9, 0.1), (0.1, 0.9)),
        means=((-1.0,), (1.0,)),
        full_covariances=(((1.0,),), ((1.0,),)),
    )


class _DeterministicAdapter:
    def __init__(self, first_seed_barrier: Any) -> None:
        self._first_seed_barrier = first_seed_barrier

    def fit(self, rows: object, state_count: int, seed: int) -> FitResult:
        del rows
        if state_count != 2:
            raise AssertionError("acceptance fixture must use K=2")
        if seed == MULTISTART_SEEDS[0]:
            self._first_seed_barrier.wait(timeout=30)
        return FitResult(
            artifact=_artifact(),
            train_log_likelihood=10_000.0 - seed,
            converged=True,
            iterations=3,
            seed=seed,
        )


class _DeterministicFactory:
    def __init__(self, first_seed_barrier: Any) -> None:
        self._first_seed_barrier = first_seed_barrier

    def __call__(self) -> _DeterministicAdapter:
        return _DeterministicAdapter(self._first_seed_barrier)


class _NoFitAdapter:
    def fit(self, rows: object, state_count: int, seed: int) -> FitResult:
        del rows, state_count, seed
        raise AssertionError("replay unexpectedly performed a fit")


class _NoFitFactory:
    def __call__(self) -> _NoFitAdapter:
        return _NoFitAdapter()


def _checkpoint(root: str, identity: EvaluationRunIdentity) -> HMMSeedCheckpoint:
    return HMMSeedCheckpoint(
        run_identity=identity,
        store=SQLiteEvaluationRunStore(root),
        candidate_id="gaussian_hmm_k2_full",
        fold_id="fold_001",
        state_count=2,
    )


def _multistart_worker(
    root: str,
    start_event: Any,
    ready_queue: Any,
    first_seed_barrier: Any,
    result_queue: Any,
) -> None:
    identity = _identity()
    ready_queue.put(os.getpid())
    if not start_event.wait(timeout=30):
        raise RuntimeError("worker start barrier timed out")
    result = run_multistart(
        [[0.0], [1.0]],
        state_count=2,
        adapter_factory=_DeterministicFactory(first_seed_barrier),
        max_workers=1,
        checkpoint=_checkpoint(root, identity),
    )
    result_queue.put((os.getpid(), result.winner.seed))


def _expected_payload(seed: int) -> bytes:
    artifact = _artifact()
    result = FitResult(artifact, 10_000.0 - seed, True, 3, seed)
    diagnostic = StartDiagnostic(
        seed=seed,
        success=True,
        converged=True,
        iterations=3,
        train_log_likelihood=10_000.0 - seed,
        artifact=artifact,
        failure_reason=None,
    )
    return pickle.dumps(SeedFitOutcome(diagnostic, result), protocol=pickle.HIGHEST_PROTOCOL)


def test_two_worker_same_candidate_fold_commits_canonical_seed_payloads_once(
    tmp_path: Path,
) -> None:
    """Two processes share one candidate/fold without duplicate terminal commits."""

    identity = _identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    store.open_run(identity)
    store.enable_write_ahead_logging()

    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    first_seed_barrier = context.Barrier(2)
    ready_queue = context.Queue()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_multistart_worker,
            args=(
                str(tmp_path),
                start_event,
                ready_queue,
                first_seed_barrier,
                result_queue,
            ),
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        assert len({ready_queue.get(timeout=30) for _ in processes}) == 2
        start_event.set()
        for process in processes:
            process.join(timeout=60)
        assert [process.exitcode for process in processes] == [0, 0]
        assert sorted(result_queue.get(timeout=10)[1] for _ in processes) == [11, 11]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    checkpoint = _checkpoint(str(tmp_path), identity)
    payloads = store.load_completed_work_unit_payloads(identity, "hmm_seed/")
    assert len(payloads) == len(MULTISTART_SEEDS)
    assert {key.rsplit("=", 1)[-1] for key, _payload in payloads} == {
        str(seed) for seed in MULTISTART_SEEDS
    }

    for seed in MULTISTART_SEEDS:
        unit = checkpoint.unit(seed)
        state = store.work_unit_state(identity, unit)
        assert state is not None
        assert state.status is WorkUnitStatus.COMPLETE
        assert state.attempt_count == 1
        payload = store.load_completed_work_unit(identity, unit)
        assert payload is not None
        assert payload == _expected_payload(seed)
        assert state.payload_hash == hashlib.sha256(payload).hexdigest()

    replay = run_multistart(
        [[0.0], [1.0]],
        state_count=2,
        adapter_factory=_NoFitFactory(),
        max_workers=2,
        checkpoint=checkpoint,
    )
    assert replay.winner.seed == MULTISTART_SEEDS[0]
    assert replay.valid_start_count == len(MULTISTART_SEEDS)
