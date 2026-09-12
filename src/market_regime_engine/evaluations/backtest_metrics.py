"""Leak-free event-driven backtest metrics for explicit downstream inputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from math import isfinite, sqrt

import numpy as np

from market_regime_engine.mlflow_support.metric_catalog import validate_metric_points
from market_regime_engine.mlflow_support.ports import MetricPoint

_BASIS_POINTS = 10_000.0
_MIN_PERIODS_PER_YEAR = 1


def _timestamp_ms(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("backtest timestamps must be timezone-aware UTC")
    result = int(value.timestamp() * 1000)
    if result < 0:
        raise ValueError("backtest timestamps must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class BacktestContract:
    """Immutable assumptions for translating state output into a position."""

    strategy_identity: str
    state_positions: tuple[float, ...]
    data_snapshot_identity: str
    window_identity: str = "backtest"
    execution_timing: str = "next_observation_return"
    return_units: str = "simple_return"
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    leverage_limit: float = 1.0
    cash_handling: str = "cash_residual"
    benchmark_identity: str = "none"
    risk_free_rate_annualized: float = 0.0
    periods_per_year: int = 252
    allowed_instruments: tuple[str, ...] = ("single_instrument",)
    missing_return_policy: str = "omit_missing_return"
    selection_policy: str = "diagnostic_only"

    def __post_init__(self) -> None:
        if not self.strategy_identity or self.strategy_identity.strip() != self.strategy_identity:
            raise ValueError("strategy identity must be non-empty and trimmed")
        if not self.state_positions:
            raise ValueError("state position mapping cannot be empty")
        if any(not isfinite(value) for value in self.state_positions):
            raise ValueError("state positions must be finite")
        for name, value in (
            ("data snapshot identity", self.data_snapshot_identity),
            ("window identity", self.window_identity),
        ):
            if not value or value.strip() != value:
                raise ValueError(f"{name} must be non-empty and trimmed")
        if self.execution_timing != "next_observation_return":
            raise ValueError("execution timing must use the next observation return")
        if self.return_units != "simple_return":
            raise ValueError("backtest returns must be simple returns")
        for field_name, field_value in (
            ("transaction_cost_bps", self.transaction_cost_bps),
            ("slippage_bps", self.slippage_bps),
            ("risk_free_rate_annualized", self.risk_free_rate_annualized),
        ):
            if not isfinite(field_value):
                raise ValueError(f"{field_name} must be finite")
        if self.transaction_cost_bps < 0.0 or self.slippage_bps < 0.0:
            raise ValueError("transaction costs and slippage cannot be negative")
        if self.risk_free_rate_annualized <= -1.0:
            raise ValueError("annualized risk-free rate must be greater than -1")
        if not isfinite(self.leverage_limit) or self.leverage_limit <= 0.0:
            raise ValueError("leverage limit must be positive and finite")
        if any(abs(value) > self.leverage_limit for value in self.state_positions):
            raise ValueError("state position exceeds leverage limit")
        if self.cash_handling != "cash_residual":
            raise ValueError("unsupported cash handling policy")
        if self.benchmark_identity != "none" and (
            not self.benchmark_identity
            or self.benchmark_identity.strip() != self.benchmark_identity
        ):
            raise ValueError("benchmark identity must be non-empty and trimmed when supplied")
        if not self.allowed_instruments or len(set(self.allowed_instruments)) != len(
            self.allowed_instruments
        ):
            raise ValueError("allowed instruments must be non-empty and duplicate-free")
        if self.missing_return_policy != "omit_missing_return":
            raise ValueError("unsupported missing-return policy")
        if self.selection_policy != "diagnostic_only":
            raise ValueError("backtest metrics cannot drive HMM model selection")
        if isinstance(self.periods_per_year, bool) or self.periods_per_year < _MIN_PERIODS_PER_YEAR:
            raise ValueError("periods per year must be a positive integer")


@dataclass(frozen=True, slots=True)
class BacktestObservation:
    timestamp: datetime
    signal_timestamp: datetime
    state_id: int
    position: float
    turnover: float
    asset_return: float
    gross_return: float
    transaction_cost_return: float
    net_return: float
    equity: float
    drawdown: float
    benchmark_return: float | None


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    observation_count: int
    missing_return_count: int
    cumulative_net_return: float
    annualized_return: float
    volatility_annualized: float
    sharpe: float | None
    sortino: float | None
    maximum_drawdown: float
    calmar: float | None
    hit_rate: float
    turnover_mean: float
    turnover_total: float
    transaction_cost_total: float
    mean_abs_exposure: float
    maximum_abs_exposure: float
    benchmark_cumulative_return: float | None
    active_cumulative_return: float | None


@dataclass(frozen=True, slots=True)
class BacktestEvidence:
    status: str
    contract: BacktestContract
    observations: tuple[BacktestObservation, ...]
    summary: BacktestSummary | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "not_available"}:
            raise ValueError("backtest status must be available or not_available")
        if self.status == "available" and (not self.observations or self.summary is None):
            raise ValueError("available backtest evidence requires observations and summary")
        if self.status == "not_available" and not self.unavailable_reason:
            raise ValueError("unavailable backtest evidence requires a reason")

    @property
    def metric_points(self) -> tuple[MetricPoint, ...]:
        if self.status != "available" or self.summary is None:
            return ()
        points: list[MetricPoint] = []
        for step, observation in enumerate(self.observations, start=1):
            timestamp_ms = _timestamp_ms(observation.timestamp)
            values: tuple[tuple[str, float | None], ...] = (
                ("backtest_position", observation.position),
                ("backtest_turnover", observation.turnover),
                ("backtest_asset_return", observation.asset_return),
                ("backtest_gross_return", observation.gross_return),
                ("backtest_transaction_cost_return", observation.transaction_cost_return),
                ("backtest_net_return", observation.net_return),
                ("backtest_equity", observation.equity),
                ("backtest_drawdown", observation.drawdown),
                ("backtest_benchmark_return", observation.benchmark_return),
            )
            for key, value in values:
                if value is not None:
                    points.append(
                        MetricPoint(key=key, value=value, step=step, timestamp_ms=timestamp_ms)
                    )
        summary_values: tuple[tuple[str, float | None], ...] = (
            ("backtest_observation_count", float(self.summary.observation_count)),
            ("backtest_missing_return_count", float(self.summary.missing_return_count)),
            ("backtest_cumulative_net_return", self.summary.cumulative_net_return),
            ("backtest_annualized_return", self.summary.annualized_return),
            ("backtest_volatility_annualized", self.summary.volatility_annualized),
            ("backtest_sharpe", self.summary.sharpe),
            ("backtest_sortino", self.summary.sortino),
            ("backtest_maximum_drawdown", self.summary.maximum_drawdown),
            ("backtest_calmar", self.summary.calmar),
            ("backtest_hit_rate", self.summary.hit_rate),
            ("backtest_turnover_mean", self.summary.turnover_mean),
            ("backtest_turnover_total", self.summary.turnover_total),
            ("backtest_transaction_cost_total", self.summary.transaction_cost_total),
            ("backtest_mean_abs_exposure", self.summary.mean_abs_exposure),
            ("backtest_maximum_abs_exposure", self.summary.maximum_abs_exposure),
            ("backtest_benchmark_cumulative_return", self.summary.benchmark_cumulative_return),
            ("backtest_active_cumulative_return", self.summary.active_cumulative_return),
        )
        timestamp_ms = _timestamp_ms(self.observations[-1].timestamp)
        points.extend(
            MetricPoint(key=key, value=value, step=0, timestamp_ms=timestamp_ms)
            for key, value in summary_values
            if value is not None
        )
        result = tuple(sorted(points, key=lambda point: (point.key, point.step)))
        validate_metric_points(result)
        return result

    @property
    def canonical_json(self) -> str:
        payload = {
            "status": self.status,
            "contract": asdict(self.contract),
            "observations": [
                {
                    **asdict(item),
                    "timestamp": item.timestamp.isoformat().replace("+00:00", "Z"),
                    "signal_timestamp": item.signal_timestamp.isoformat().replace("+00:00", "Z"),
                    "position_hex": item.position.hex(),
                    "asset_return_hex": item.asset_return.hex(),
                    "gross_return_hex": item.gross_return.hex(),
                    "transaction_cost_return_hex": item.transaction_cost_return.hex(),
                    "net_return_hex": item.net_return.hex(),
                    "equity_hex": item.equity.hex(),
                    "drawdown_hex": item.drawdown.hex(),
                    "benchmark_return_hex": (
                        None if item.benchmark_return is None else item.benchmark_return.hex()
                    ),
                }
                for item in self.observations
            ],
            "summary": asdict(self.summary) if self.summary is not None else None,
            "unavailable_reason": self.unavailable_reason,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def source_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _validate_inputs(
    timestamps: Sequence[datetime],
    probabilities: np.ndarray,
    asset_returns: Sequence[float | None],
    benchmark_returns: Sequence[float | None] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not timestamps or len(timestamps) != len(asset_returns):
        raise ValueError("backtest timestamps and asset returns must be non-empty and aligned")
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("backtest timestamps must be strictly increasing")
    for timestamp in timestamps:
        _timestamp_ms(timestamp)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(timestamps):
        raise ValueError("backtest state probabilities must align with timestamps")
    if probabilities.shape[1] < 1 or not np.all(np.isfinite(probabilities)):
        raise ValueError("backtest state probabilities must be finite and non-empty")
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("backtest state probabilities must be non-negative rows summing to one")
    if any(value is not None and not isfinite(value) for value in asset_returns):
        raise ValueError("asset returns must be finite or None")
    if benchmark_returns is None:
        return np.asarray(asset_returns, dtype=object), None
    if len(benchmark_returns) != len(timestamps):
        raise ValueError("benchmark returns must align with timestamps")
    if any(value is not None and not isfinite(value) for value in benchmark_returns):
        raise ValueError("benchmark returns must be finite or None")
    return np.asarray(asset_returns, dtype=object), np.asarray(benchmark_returns, dtype=object)


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def build_backtest_evidence(
    *,
    timestamps: Sequence[datetime],
    filtered_probabilities: Sequence[Sequence[float]],
    asset_returns: Sequence[float | None],
    contract: BacktestContract,
    benchmark_returns: Sequence[float | None] | None = None,
) -> BacktestEvidence:
    """Run a causal one-instrument backtest from already-produced model output.

    The state observed at timestamp ``t-1`` determines the position applied to
    the simple return ending at timestamp ``t``.  No target return is read while
    creating the signal or position, and trading metrics are diagnostic-only.
    """

    probabilities = np.asarray(filtered_probabilities, dtype=np.float64)
    returns, benchmarks = _validate_inputs(
        timestamps, probabilities, asset_returns, benchmark_returns
    )
    if probabilities.shape[1] != len(contract.state_positions):
        raise ValueError("state position mapping must match the model state count")
    if benchmark_returns is not None and contract.benchmark_identity == "none":
        raise ValueError("benchmark identity is required when benchmark returns are supplied")
    state_ids = np.argmax(probabilities, axis=1).astype(np.intp)
    cost_rate = (contract.transaction_cost_bps + contract.slippage_bps) / _BASIS_POINTS
    positions = np.asarray(contract.state_positions, dtype=np.float64)
    observations: list[BacktestObservation] = []
    equity = 1.0
    previous_position = 0.0
    peak_equity = 1.0
    missing_count = 0
    for index in range(1, len(timestamps)):
        position = float(positions[state_ids[index - 1]])
        turnover = abs(position - previous_position)
        asset_return = returns[index]
        benchmark_return = None if benchmarks is None else benchmarks[index]
        if asset_return is None:
            missing_count += 1
            continue
        previous_position = position
        typed_return = float(asset_return)
        gross_return = position * typed_return
        transaction_cost_return = cost_rate * turnover
        net_return = gross_return - transaction_cost_return
        equity *= 1.0 + net_return
        if equity <= 0.0:
            raise ValueError(
                "backtest equity became non-positive; annualized metrics are undefined"
            )
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0
        observations.append(
            BacktestObservation(
                timestamp=timestamps[index],
                signal_timestamp=timestamps[index - 1],
                state_id=int(state_ids[index - 1]),
                position=position,
                turnover=turnover,
                asset_return=typed_return,
                gross_return=gross_return,
                transaction_cost_return=transaction_cost_return,
                net_return=net_return,
                equity=equity,
                drawdown=drawdown,
                benchmark_return=None if benchmark_return is None else float(benchmark_return),
            )
        )
    if not observations:
        return BacktestEvidence(
            status="not_available",
            contract=contract,
            observations=(),
            summary=None,
            unavailable_reason="no complete return observations after causal signal shift",
        )
    net_returns = np.asarray([item.net_return for item in observations], dtype=np.float64)
    turnover_values = np.asarray([item.turnover for item in observations], dtype=np.float64)
    exposure_values = np.asarray([abs(item.position) for item in observations], dtype=np.float64)
    risk_free_period = (1.0 + contract.risk_free_rate_annualized) ** (
        1.0 / contract.periods_per_year
    ) - 1.0
    excess = net_returns - risk_free_period
    volatility = float(np.std(net_returns, ddof=0) * sqrt(contract.periods_per_year))
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(
        np.sqrt(np.mean(downside * downside)) * sqrt(contract.periods_per_year)
    )
    cumulative = float(equity - 1.0)
    annualized = float(equity ** (contract.periods_per_year / len(net_returns)) - 1.0)
    maximum_drawdown = float(min(item.drawdown for item in observations))
    benchmark_values = [item.benchmark_return for item in observations]
    has_benchmark = all(value is not None for value in benchmark_values)
    benchmark_cumulative = (
        float(np.prod(1.0 + np.asarray(benchmark_values, dtype=np.float64)) - 1.0)
        if has_benchmark
        else None
    )
    active_cumulative = (
        cumulative - benchmark_cumulative if benchmark_cumulative is not None else None
    )
    summary = BacktestSummary(
        observation_count=len(observations),
        missing_return_count=missing_count,
        cumulative_net_return=cumulative,
        annualized_return=annualized,
        volatility_annualized=volatility,
        sharpe=_optional_ratio(
            float(np.mean(excess) * sqrt(contract.periods_per_year)), volatility
        ),
        sortino=_optional_ratio(
            float(np.mean(excess) * sqrt(contract.periods_per_year)), downside_deviation
        ),
        maximum_drawdown=maximum_drawdown,
        calmar=_optional_ratio(annualized, abs(maximum_drawdown)),
        hit_rate=float(np.mean(net_returns > 0.0)),
        turnover_mean=float(np.mean(turnover_values)),
        turnover_total=float(np.sum(turnover_values)),
        transaction_cost_total=float(sum(item.transaction_cost_return for item in observations)),
        mean_abs_exposure=float(np.mean(exposure_values)),
        maximum_abs_exposure=float(np.max(exposure_values)),
        benchmark_cumulative_return=benchmark_cumulative,
        active_cumulative_return=active_cumulative,
    )
    return BacktestEvidence(
        status="available",
        contract=contract,
        observations=tuple(observations),
        summary=summary,
    )


__all__ = [
    "BacktestContract",
    "BacktestEvidence",
    "BacktestObservation",
    "BacktestSummary",
    "build_backtest_evidence",
]
