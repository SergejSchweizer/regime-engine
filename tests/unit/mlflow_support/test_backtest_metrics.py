from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.evaluations.backtest_metrics import (
    BacktestContract,
    build_backtest_evidence,
)
from market_regime_engine.mlflow_support.backtest_projection import backtest_model_metric_points


def _timestamps(count: int) -> tuple[datetime, ...]:
    return tuple(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(count))


def _contract(**overrides: object) -> BacktestContract:
    values: dict[str, object] = {
        "strategy_identity": "state-position-v1",
        "state_positions": (1.0, -1.0),
        "data_snapshot_identity": "snapshot-sha256:fixture",
        "transaction_cost_bps": 10.0,
        "slippage_bps": 5.0,
    }
    values.update(overrides)
    return BacktestContract(**values)  # type: ignore[arg-type]


def test_backtest_shifts_signal_one_observation_and_applies_costs() -> None:
    evidence = build_backtest_evidence(
        timestamps=_timestamps(4),
        filtered_probabilities=((1.0, 0.0), (0.0, 1.0), (0.0, 1.0), (1.0, 0.0)),
        asset_returns=(None, 0.10, 0.20, -0.10),
        contract=_contract(),
    )

    assert evidence.status == "available"
    assert [item.state_id for item in evidence.observations] == [0, 1, 1]
    assert [item.position for item in evidence.observations] == [1.0, -1.0, -1.0]
    assert [item.gross_return for item in evidence.observations] == pytest.approx(
        [0.10, -0.20, 0.10]
    )
    assert [item.transaction_cost_return for item in evidence.observations] == pytest.approx(
        [0.0015, 0.003, 0.0]
    )
    assert [item.net_return for item in evidence.observations] == pytest.approx(
        [0.0985, -0.203, 0.10]
    )
    assert evidence.summary is not None
    assert evidence.summary.turnover_total == pytest.approx(3.0)
    assert evidence.summary.transaction_cost_total == pytest.approx(0.0045)
    assert backtest_model_metric_points(evidence)


def test_future_return_mutation_cannot_change_prior_observation() -> None:
    kwargs = {
        "timestamps": _timestamps(5),
        "filtered_probabilities": ((1.0, 0.0),) * 5,
        "contract": _contract(transaction_cost_bps=0.0, slippage_bps=0.0),
    }
    first = build_backtest_evidence(asset_returns=(None, 0.1, 0.2, 0.3, 0.4), **kwargs)
    changed = build_backtest_evidence(asset_returns=(None, 0.1, 0.2, 99.0, 0.4), **kwargs)
    assert first.observations[:2] == changed.observations[:2]
    assert first.observations[2] != changed.observations[2]


def test_missing_returns_are_explicitly_counted_and_missing_all_is_unavailable() -> None:
    evidence = build_backtest_evidence(
        timestamps=_timestamps(4),
        filtered_probabilities=((1.0, 0.0),) * 4,
        asset_returns=(None, None, 0.1, None),
        contract=_contract(),
    )
    assert evidence.status == "available"
    assert evidence.summary is not None
    assert evidence.summary.missing_return_count == 2
    unavailable = build_backtest_evidence(
        timestamps=_timestamps(3),
        filtered_probabilities=((1.0, 0.0),) * 3,
        asset_returns=(None, None, None),
        contract=_contract(),
    )
    assert unavailable.status == "not_available"
    assert unavailable.metric_points == ()


def test_benchmark_is_separate_and_invalid_alignment_fails_closed() -> None:
    evidence = build_backtest_evidence(
        timestamps=_timestamps(3),
        filtered_probabilities=((1.0, 0.0),) * 3,
        asset_returns=(None, 0.1, 0.1),
        benchmark_returns=(None, 0.05, 0.05),
        contract=_contract(benchmark_identity="benchmark-fixture-v1"),
    )
    assert evidence.summary is not None
    assert evidence.summary.benchmark_cumulative_return == pytest.approx(0.1025)
    assert evidence.summary.active_cumulative_return is not None
    with pytest.raises(ValueError, match="benchmark returns must align"):
        build_backtest_evidence(
            timestamps=_timestamps(3),
            filtered_probabilities=((1.0, 0.0),) * 3,
            asset_returns=(None, 0.1, 0.1),
            benchmark_returns=(None, 0.05),
            contract=_contract(benchmark_identity="benchmark-fixture-v1"),
        )


def test_benchmark_requires_a_declared_identity() -> None:
    with pytest.raises(ValueError, match="benchmark identity is required"):
        build_backtest_evidence(
            timestamps=_timestamps(3),
            filtered_probabilities=((1.0, 0.0),) * 3,
            asset_returns=(None, 0.1, 0.1),
            benchmark_returns=(None, 0.05, 0.05),
            contract=_contract(),
        )
