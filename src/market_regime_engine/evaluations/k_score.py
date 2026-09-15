"""Dimension-independent cross-K scoring for the K-champion portfolio.

The score compares K-specific feature sets only through a common scalar target
forecast on the identical Outer-TEST folds.  Raw vector likelihood, AIC and
BIC are intentionally absent: those quantities are not comparable when the
K-specific feature tuple changes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite, tanh
from statistics import fmean

from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.mlflow_support.metric_catalog import validate_metric_points
from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.runtime.cpu import cpu_worker_count

K_SCORE_VERSION = "cross_k_score.v1"
LEGAL_STATE_COUNTS = (2, 3, 4, 5)
VALID_FOLD_RATE_GATE = 0.80
MIN_VALID_FOLD_COUNT = 3
SCORE_ABS_TOLERANCE = 1.0e-12
FORECAST_WEIGHT = 0.50
CALIBRATION_WEIGHT = 0.20
STABILITY_WEIGHT = 0.20
ROBUSTNESS_WEIGHT = 0.10
COMPLEXITY_PENALTY_PER_EXTRA_STATE = 0.01


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")


def _require_sha256(value: str, field: str) -> None:
    _require_text(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")


def _require_unit_interval(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and in [0, 1]")
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class KScoreFoldEvidence:
    """One K candidate's evidence on one common Outer-TEST fold."""

    fold_id: str
    valid: bool
    target_log_score: float | None = None
    baseline_target_log_score: float | None = None
    calibration_error: float | None = None
    stability_score: float | None = None
    support_score: float | None = None
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.fold_id, "fold_id")
        if not isinstance(self.valid, bool):
            raise ValueError("K-score fold validity must be a boolean")
        if not self.valid:
            _require_text(self.invalid_reason or "", "invalid_reason")
            if any(
                value is not None
                for value in (
                    self.target_log_score,
                    self.baseline_target_log_score,
                    self.calibration_error,
                    self.stability_score,
                    self.support_score,
                )
            ):
                raise ValueError("invalid K-score folds cannot contain partial score values")
            return
        if self.invalid_reason is not None:
            raise ValueError("valid K-score folds cannot contain an invalid reason")
        score_values = (
            self.target_log_score,
            self.baseline_target_log_score,
            self.calibration_error,
            self.stability_score,
            self.support_score,
        )
        if any(
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in score_values
        ):
            raise ValueError("valid K-score folds require finite score values")
        assert self.calibration_error is not None
        assert self.stability_score is not None
        assert self.support_score is not None
        _require_unit_interval(self.calibration_error, "calibration_error")
        _require_unit_interval(self.stability_score, "stability_score")
        _require_unit_interval(self.support_score, "support_score")


@dataclass(frozen=True, slots=True)
class KScoreCandidate:
    """A K-specific candidate whose folds share one comparison contract."""

    state_count: int
    feature_order_hash: str
    source_build_id: str
    evaluation_plan_hash: str
    target_metric_id: str
    target_horizon: str
    folds: tuple[KScoreFoldEvidence, ...]

    def __post_init__(self) -> None:
        if isinstance(self.state_count, bool) or self.state_count not in LEGAL_STATE_COUNTS:
            raise ValueError("K-score state_count must be one of 2, 3, 4 or 5")
        _require_sha256(self.feature_order_hash, "feature_order_hash")
        _require_text(self.source_build_id, "source_build_id")
        _require_sha256(self.evaluation_plan_hash, "evaluation_plan_hash")
        _require_text(self.target_metric_id, "target_metric_id")
        _require_text(self.target_horizon, "target_horizon")
        if not self.folds:
            raise ValueError("K-score candidate requires at least one planned fold")
        if not isinstance(self.folds, tuple) or any(
            not isinstance(fold, KScoreFoldEvidence) for fold in self.folds
        ):
            raise ValueError("K-score candidate folds must be a tuple of fold evidence")
        fold_ids = tuple(fold.fold_id for fold in self.folds)
        if len(set(fold_ids)) != len(fold_ids):
            raise ValueError("K-score fold IDs must be unique")
        # Completion order is operational metadata, not score evidence.  Keep
        # the fold order canonical before fmean() so process/serial execution
        # produces identical floating-point evidence and hashes.
        object.__setattr__(self, "folds", tuple(sorted(self.folds, key=lambda fold: fold.fold_id)))


@dataclass(frozen=True, slots=True)
class KScoreBreakdown:
    """Auditable score components and eligibility for one K."""

    state_count: int
    feature_order_hash: str
    planned_fold_count: int
    valid_fold_count: int
    valid_fold_rate: float
    latest_fold_valid: bool
    forecast_score: float | None
    calibration_score: float | None
    stability_score: float | None
    robustness_score: float | None
    worst_fold_forecast_score: float | None
    mean_support_score: float | None
    complexity_penalty: float
    total_score: float | None
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.state_count, bool) or self.state_count not in LEGAL_STATE_COUNTS:
            raise ValueError("K-score breakdown has an illegal state count")
        _require_sha256(self.feature_order_hash, "feature_order_hash")
        if (
            isinstance(self.planned_fold_count, bool)
            or isinstance(self.valid_fold_count, bool)
            or not isinstance(self.planned_fold_count, int)
            or not isinstance(self.valid_fold_count, int)
            or self.planned_fold_count < 1
            or not 0 <= self.valid_fold_count <= self.planned_fold_count
        ):
            raise ValueError("K-score fold counts are inconsistent")
        _require_unit_interval(self.valid_fold_rate, "valid_fold_rate")
        expected_rate = self.valid_fold_count / self.planned_fold_count
        if abs(self.valid_fold_rate - expected_rate) > SCORE_ABS_TOLERANCE:
            raise ValueError("K-score valid-fold rate does not reconcile")
        if isinstance(self.latest_fold_valid, bool) is False:
            raise ValueError("K-score latest-fold validity must be a boolean")
        if (
            isinstance(self.complexity_penalty, bool)
            or not isinstance(self.complexity_penalty, (int, float))
            or not isfinite(self.complexity_penalty)
            or self.complexity_penalty < 0.0
        ):
            raise ValueError("K-score complexity penalty must be finite and non-negative")
        expected_penalty = COMPLEXITY_PENALTY_PER_EXTRA_STATE * (self.state_count - 2)
        if abs(self.complexity_penalty - expected_penalty) > SCORE_ABS_TOLERANCE:
            raise ValueError("K-score complexity penalty does not reconcile")
        for field_name in (
            "forecast_score",
            "calibration_score",
            "stability_score",
            "robustness_score",
            "worst_fold_forecast_score",
            "mean_support_score",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_unit_interval(value, field_name)
        if not isinstance(self.eligible, bool):
            raise ValueError("K-score eligibility must be a boolean")
        if not isinstance(self.rejection_reasons, tuple) or any(
            not isinstance(reason, str) or not reason or reason.strip() != reason
            for reason in self.rejection_reasons
        ):
            raise ValueError("K-score rejection reasons must be non-empty trimmed strings")
        if len(set(self.rejection_reasons)) != len(self.rejection_reasons):
            raise ValueError("K-score rejection reasons must be unique")
        if self.eligible != (not self.rejection_reasons):
            raise ValueError("K-score eligibility must match rejection reasons")
        if self.eligible and self.total_score is None:
            raise ValueError("eligible K-score must have a total score")
        if self.eligible and any(
            getattr(self, field_name) is None
            for field_name in (
                "forecast_score",
                "calibration_score",
                "stability_score",
                "robustness_score",
                "worst_fold_forecast_score",
                "mean_support_score",
            )
        ):
            raise ValueError("eligible K-score must contain every score component")
        if not self.eligible and self.total_score is not None:
            raise ValueError("ineligible K-score cannot have a total score")
        if self.total_score is not None:
            if isinstance(self.total_score, bool) or not isinstance(self.total_score, (int, float)):
                raise ValueError("K-score total must be finite")
            if not isfinite(self.total_score):
                raise ValueError("K-score total must be finite")
            assert self.forecast_score is not None
            assert self.calibration_score is not None
            assert self.stability_score is not None
            assert self.robustness_score is not None
            expected_total = (
                FORECAST_WEIGHT * self.forecast_score
                + CALIBRATION_WEIGHT * self.calibration_score
                + STABILITY_WEIGHT * self.stability_score
                + ROBUSTNESS_WEIGHT * self.robustness_score
                - self.complexity_penalty
            )
            if abs(self.total_score - expected_total) > SCORE_ABS_TOLERANCE:
                raise ValueError("K-score total does not reconcile")


@dataclass(frozen=True, slots=True)
class KScoreRanking:
    """Cross-K ranking with explicit ineligible-slot evidence."""

    score_version: str
    ranked_state_counts: tuple[int, ...]
    selected_state_count: int | None
    scores: tuple[KScoreBreakdown, ...]
    no_winner_reason: str | None = None

    def __post_init__(self) -> None:
        if self.score_version != K_SCORE_VERSION:
            raise ValueError("unsupported K-score version")
        if len({score.state_count for score in self.scores}) != len(self.scores):
            raise ValueError("K-score state counts must be unique")
        if len(set(self.ranked_state_counts)) != len(self.ranked_state_counts):
            raise ValueError("K-score ranked state counts must be unique")
        if set(self.ranked_state_counts) != {
            score.state_count for score in self.scores if score.eligible
        }:
            raise ValueError("K-score ranking must contain exactly all eligible state counts")
        if self.selected_state_count is not None:
            if (
                isinstance(self.selected_state_count, bool)
                or self.selected_state_count not in LEGAL_STATE_COUNTS
            ):
                raise ValueError("selected K is illegal")
            if (
                not self.ranked_state_counts
                or self.ranked_state_counts[0] != self.selected_state_count
            ):
                raise ValueError("selected K must be the first eligible ranked K")
            if self.no_winner_reason is not None:
                raise ValueError("a selected K cannot have a no-winner reason")
        elif self.no_winner_reason is None or not self.no_winner_reason.strip():
            raise ValueError("missing K winner requires an explicit reason")
        elif self.no_winner_reason.strip() != self.no_winner_reason:
            raise ValueError("K no-winner reason must be trimmed")

    @property
    def canonical_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def source_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def metric_points(self, *, timestamp_ms: int, step: int = 0) -> tuple[MetricPoint, ...]:
        """Project the score catalog onto MLflow Model Metrics without recomputation."""

        if isinstance(timestamp_ms, bool) or timestamp_ms < 0:
            raise ValueError("K-score metric timestamp must be a non-negative integer")
        if isinstance(step, bool) or step < 0:
            raise ValueError("K-score metric step must be a non-negative integer")
        points: list[MetricPoint] = []
        fields = (
            "valid_fold_rate",
            "forecast_score",
            "calibration_score",
            "stability_score",
            "robustness_score",
            "worst_fold_forecast_score",
            "mean_support_score",
            "complexity_penalty",
            "total_score",
        )
        for score in sorted(self.scores, key=lambda item: item.state_count):
            for field_name in fields:
                value = getattr(score, field_name)
                if value is not None:
                    points.append(
                        MetricPoint(
                            key=f"k_score_{field_name}_k{score.state_count}",
                            value=float(value),
                            step=step,
                            timestamp_ms=timestamp_ms,
                        )
                    )
            points.append(
                MetricPoint(
                    key=f"k_score_eligible_k{score.state_count}",
                    value=float(score.eligible),
                    step=step,
                    timestamp_ms=timestamp_ms,
                )
            )
        validate_metric_points(tuple(points))
        return tuple(points)


def _forecast_score(target_log_score: float, baseline_target_log_score: float) -> float:
    """Map common-target improvement to a bounded, monotonic score."""

    return 0.5 + 0.5 * tanh(target_log_score - baseline_target_log_score)


def score_k_candidate(candidate: KScoreCandidate, *, latest_fold_id: str) -> KScoreBreakdown:
    """Score one K using only its valid common-target Outer-TEST folds."""

    _require_text(latest_fold_id, "latest_fold_id")
    latest = next((fold for fold in candidate.folds if fold.fold_id == latest_fold_id), None)
    if latest is None:
        raise ValueError("latest_fold_id is not present in every K-score candidate")
    valid_folds = tuple(fold for fold in candidate.folds if fold.valid)
    valid_rate = len(valid_folds) / len(candidate.folds)
    reasons: list[str] = []
    if not valid_folds:
        reasons.append("zero valid outer folds")
    if valid_rate < VALID_FOLD_RATE_GATE:
        reasons.append("valid-fold rate below 0.80")
    if len(valid_folds) < MIN_VALID_FOLD_COUNT:
        reasons.append("fewer than three valid outer folds")
    if not latest.valid:
        reasons.append("latest complete outer fold is invalid")
    complexity_penalty = COMPLEXITY_PENALTY_PER_EXTRA_STATE * (candidate.state_count - 2)
    if not valid_folds:
        return KScoreBreakdown(
            state_count=candidate.state_count,
            feature_order_hash=candidate.feature_order_hash,
            planned_fold_count=len(candidate.folds),
            valid_fold_count=0,
            valid_fold_rate=valid_rate,
            latest_fold_valid=latest.valid,
            forecast_score=None,
            calibration_score=None,
            stability_score=None,
            robustness_score=None,
            worst_fold_forecast_score=None,
            mean_support_score=None,
            complexity_penalty=complexity_penalty,
            total_score=None,
            eligible=False,
            rejection_reasons=tuple(reasons),
        )

    forecast_values: list[float] = []
    calibration_values: list[float] = []
    stability_values: list[float] = []
    support_values: list[float] = []
    for fold in valid_folds:
        assert fold.target_log_score is not None
        assert fold.baseline_target_log_score is not None
        assert fold.calibration_error is not None
        assert fold.stability_score is not None
        assert fold.support_score is not None
        forecast_values.append(
            _forecast_score(fold.target_log_score, fold.baseline_target_log_score)
        )
        calibration_values.append(1.0 - fold.calibration_error)
        stability_values.append(fold.stability_score)
        support_values.append(fold.support_score)

    forecast = fmean(forecast_values) if forecast_values else None
    calibration = fmean(calibration_values) if calibration_values else None
    stability = fmean(stability_values) if stability_values else None
    support = fmean(support_values) if support_values else None
    worst_forecast = min(forecast_values) if forecast_values else None
    robustness = (
        0.50 * valid_rate + 0.25 * worst_forecast + 0.25 * support
        if worst_forecast is not None and support is not None
        else None
    )
    if any(value is None for value in (forecast, calibration, stability, robustness)):
        raise ValueError("valid K-score candidate cannot be scored without all components")
    assert forecast is not None
    assert calibration is not None
    assert stability is not None
    assert robustness is not None
    total = (
        FORECAST_WEIGHT * forecast
        + CALIBRATION_WEIGHT * calibration
        + STABILITY_WEIGHT * stability
        + ROBUSTNESS_WEIGHT * robustness
        - complexity_penalty
        if not reasons
        else None
    )
    return KScoreBreakdown(
        state_count=candidate.state_count,
        feature_order_hash=candidate.feature_order_hash,
        planned_fold_count=len(candidate.folds),
        valid_fold_count=len(valid_folds),
        valid_fold_rate=valid_rate,
        latest_fold_valid=latest.valid,
        forecast_score=forecast,
        calibration_score=calibration,
        stability_score=stability,
        robustness_score=robustness,
        worst_fold_forecast_score=worst_forecast,
        mean_support_score=support,
        complexity_penalty=complexity_penalty,
        total_score=total,
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _score_candidate_process(task: tuple[KScoreCandidate, str]) -> KScoreBreakdown:
    candidate, latest_fold_id = task
    return score_k_candidate(candidate, latest_fold_id=latest_fold_id)


def _score_candidates(
    candidates: tuple[KScoreCandidate, ...],
    *,
    latest_fold_id: str,
    max_workers: int | None,
) -> tuple[KScoreBreakdown, ...]:
    """Score independent K candidates in processes, preserving canonical order."""

    worker_limit = cpu_worker_count(max_workers, task_count=len(candidates))
    tasks = tuple((candidate, latest_fold_id) for candidate in candidates)
    if worker_limit <= 1:
        return tuple(_score_candidate_process(task) for task in tasks)
    with cpu_process_pool(worker_limit) as executor:
        return tuple(executor.map(_score_candidate_process, tasks))


def _anchored_partitions(
    items: tuple[KScoreBreakdown, ...], field_name: str
) -> tuple[tuple[KScoreBreakdown, ...], ...]:
    remaining = list(items)
    partitions: list[tuple[KScoreBreakdown, ...]] = []
    while remaining:
        values = tuple(getattr(item, field_name) for item in remaining)
        if any(value is None for value in values):
            raise ValueError("eligible K-score ranking fields must be finite")
        numeric_values = tuple(float(value) for value in values)
        anchor = max(numeric_values)
        tied = tuple(
            item
            for item in remaining
            if float(getattr(item, field_name)) >= anchor - SCORE_ABS_TOLERANCE
        )
        partitions.append(tied)
        tied_ids = {item.state_count for item in tied}
        remaining = [item for item in remaining if item.state_count not in tied_ids]
    return tuple(partitions)


def _rank_eligible(scores: tuple[KScoreBreakdown, ...]) -> tuple[KScoreBreakdown, ...]:
    groups: tuple[tuple[KScoreBreakdown, ...], ...] = (scores,)
    for field_name in (
        "total_score",
        "forecast_score",
        "worst_fold_forecast_score",
        "calibration_score",
        "stability_score",
        "robustness_score",
    ):
        groups = tuple(
            subgroup for group in groups for subgroup in _anchored_partitions(group, field_name)
        )
    return tuple(
        item for group in groups for item in sorted(group, key=lambda item: item.state_count)
    )


def rank_k_candidates(
    candidates: Sequence[KScoreCandidate],
    *,
    latest_fold_id: str,
    max_workers: int | None = None,
) -> KScoreRanking:
    """Rank eligible K values under one common source/target evaluation contract.

    Each K score is independent and therefore runs in GIL-independent worker
    processes by default.  ``max_workers=1`` is an explicit serial reference
    mode for deterministic QA and constrained callers.
    """

    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise ValueError("cross-K ranking requires at least one candidate")
    if len({candidate.state_count for candidate in candidate_tuple}) != len(candidate_tuple):
        raise ValueError("cross-K ranking requires unique state counts")
    common_contract = (
        candidate_tuple[0].source_build_id,
        candidate_tuple[0].evaluation_plan_hash,
        candidate_tuple[0].target_metric_id,
        candidate_tuple[0].target_horizon,
        tuple(sorted(fold.fold_id for fold in candidate_tuple[0].folds)),
    )
    for candidate in candidate_tuple[1:]:
        current_contract = (
            candidate.source_build_id,
            candidate.evaluation_plan_hash,
            candidate.target_metric_id,
            candidate.target_horizon,
            tuple(sorted(fold.fold_id for fold in candidate.folds)),
        )
        if current_contract != common_contract:
            raise ValueError(
                "cross-K ranking requires identical source, plan, target, horizon and folds"
            )
    baseline_by_fold: dict[str, float] = {}
    for candidate in candidate_tuple:
        for fold in candidate.folds:
            if fold.valid:
                assert fold.baseline_target_log_score is not None
                previous = baseline_by_fold.get(fold.fold_id)
                if previous is not None and fold.baseline_target_log_score != previous:
                    raise ValueError("cross-K ranking requires one common baseline per fold")
                baseline_by_fold[fold.fold_id] = fold.baseline_target_log_score

    scores = _score_candidates(
        tuple(sorted(candidate_tuple, key=lambda item: item.state_count)),
        latest_fold_id=latest_fold_id,
        max_workers=max_workers,
    )
    eligible = tuple(score for score in scores if score.eligible)
    ranked = _rank_eligible(eligible) if eligible else ()
    return KScoreRanking(
        score_version=K_SCORE_VERSION,
        ranked_state_counts=tuple(score.state_count for score in ranked),
        selected_state_count=ranked[0].state_count if ranked else None,
        scores=scores,
        no_winner_reason=None if ranked else "no K passes the cross-K eligibility gates",
    )
