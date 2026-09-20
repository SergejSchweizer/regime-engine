"""Adversarial QA matrix for PR-453 failure classification boundaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import sleep
from typing import NoReturn

import pandas as pd
import pytest

from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold
from market_regime_engine.evaluations.k_feature_selection import run_k_feature_selection
from market_regime_engine.training.multistart import _evaluate_start
from tests.unit.evaluations import test_k_champion_outer as outer_fixtures

_CUTOFF = datetime(2024, 1, 1, tzinfo=UTC)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"timestamp_m1": [_CUTOFF], "f0": [1.0]})


def _typed_selector(_rows: pd.DataFrame, *, state_count: int, **_: object) -> NoReturn:
    delay = (5 - state_count) * 0.01
    sleep(delay)
    raise RecoverableEvaluationInvalidity(f"statistical gate rejected k{state_count}")


def _raise_runtime(*_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError("unexpected runtime failure")


def _raise_key(*_args: object, **_kwargs: object) -> NoReturn:
    raise KeyError("unexpected key failure")


def _raise_assertion(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("unexpected assertion failure")


def _raise_type(*_args: object, **_kwargs: object) -> NoReturn:
    raise TypeError("unexpected type failure")


def _raise_value(*_args: object, **_kwargs: object) -> NoReturn:
    raise ValueError("unexpected value failure")


def test_recoverable_k_evidence_is_completion_order_independent() -> None:
    kwargs = dict(
        train_rows=_frame(),
        source_snapshot_id="snapshot",
        validation_cutoff=_CUTOFF,
        selector=_typed_selector,
        requested_state_counts=(2, 3, 4, 5),
    )
    serial = run_k_feature_selection(**kwargs, max_workers=1)
    parallel = run_k_feature_selection(**kwargs, max_workers=4)

    assert serial == parallel
    assert tuple(item.state_count for item in parallel) == (2, 3, 4, 5)
    assert tuple(item.rejection_reason for item in parallel) == (
        "RecoverableEvaluationInvalidity: statistical gate rejected k2",
        "RecoverableEvaluationInvalidity: statistical gate rejected k3",
        "RecoverableEvaluationInvalidity: statistical gate rejected k4",
        "RecoverableEvaluationInvalidity: statistical gate rejected k5",
    )
    assert all(not item.eligible for item in parallel)


@pytest.mark.parametrize(
    ("callback", "exception_type", "message"),
    (
        (_raise_runtime, RuntimeError, "unexpected runtime failure"),
        (_raise_key, KeyError, "unexpected key failure"),
        (_raise_assertion, AssertionError, "unexpected assertion failure"),
        (_raise_type, TypeError, "unexpected type failure"),
        (_raise_value, ValueError, "unexpected value failure"),
    ),
    ids=("runtime", "key", "assertion", "type", "value"),
)
@pytest.mark.parametrize("max_workers", (1, 4))
def test_unexpected_k_failures_escape_in_serial_and_process_modes(
    callback: object,
    exception_type: type[BaseException],
    message: str,
    max_workers: int,
) -> None:
    with pytest.raises(exception_type, match=message):
        run_k_feature_selection(
            _frame(),
            source_snapshot_id="snapshot",
            validation_cutoff=_CUTOFF,
            selector=callback,  # type: ignore[arg-type]
            requested_state_counts=(2,),
            max_workers=max_workers,
        )


def _outer_failure_after_first_fold(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    selection: object,
    slot_id: str,
    fold: WalkForwardFold,
) -> object:
    if fold.fold_index == 1:
        raise RuntimeError("unexpected outer failure")
    return outer_fixtures._evaluate(
        train_rows,
        test_rows,
        selection=selection,
        slot_id=slot_id,
        fold=fold,
    )


def _outer_kwargs(rows: pd.DataFrame) -> dict[str, object]:
    plan = outer_fixtures.plan()
    return {
        "source_rows": rows,
        "plan": plan,
        "source_snapshot_id": "snapshot-1",
        "profile_id": "xetra",
        "profile_config_version": 4,
        "validation_cutoff": plan.evaluation_cutoff,
        "selector": outer_fixtures._selection,
        "evaluator": _outer_failure_after_first_fold,
    }


@pytest.mark.parametrize("max_workers", (1, 2))
def test_one_unexpected_outer_fold_failure_cannot_be_hidden_by_valid_folds(
    max_workers: int,
) -> None:
    timestamps = tuple(outer_fixtures.BASE + timedelta(minutes=index) for index in range(1449))
    rows = pd.DataFrame({"timestamp_m1": timestamps, "f0": range(1449)})

    with pytest.raises(RuntimeError, match="unexpected outer failure"):
        outer_fixtures.run_k_champion_outer_policy(**_outer_kwargs(rows), max_workers=max_workers)


class _RaisingAdapter:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def fit(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise self._error


def test_typed_multistart_invalidity_is_recorded_at_the_l_boundary() -> None:
    diagnostic, result = _evaluate_start(
        [[0.0]],
        state_count=2,
        adapter_factory=lambda: _RaisingAdapter(
            RecoverableEvaluationInvalidity("insufficient statistical support")
        ),  # type: ignore[arg-type]
        seed=11,
    )

    assert result is None
    assert diagnostic.success is False
    assert diagnostic.failure_reason == (
        "RecoverableEvaluationInvalidity: insufficient statistical support"
    )


@pytest.mark.parametrize(
    ("error", "exception_type"),
    ((KeyboardInterrupt("interrupt"), KeyboardInterrupt), (SystemExit("exit"), SystemExit)),
    ids=("keyboard-interrupt", "system-exit"),
)
def test_multistart_hard_interrupts_propagate_from_the_l_boundary(
    error: BaseException,
    exception_type: type[BaseException],
) -> None:
    with pytest.raises(exception_type):
        _evaluate_start(
            [[0.0]],
            state_count=2,
            adapter_factory=lambda: _RaisingAdapter(error),  # type: ignore[arg-type]
            seed=11,
        )
