"""Dimension-independent score contract for inner-fold feature selection.

This contract is deliberately separate from the Outer-TEST ``cross_k_score``
contract.  It contains only the common forecast, calibration, stability,
support and valid-fold evidence needed while selecting a feature tuple.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite, tanh
from statistics import fmean

FEATURE_SUBSET_SCORE_VERSION = "feature_subset_score.v1"
INNER_FOLD_PLAN_ID = "monthly_inner_within_outer_train.v1"
VALID_FOLD_RATE_GATE = 0.80
MIN_VALID_FOLD_COUNT = 3
SCORE_ABS_TOLERANCE = 1.0e-12
FORECAST_WEIGHT = 0.50
CALIBRATION_WEIGHT = 0.20
STABILITY_WEIGHT = 0.20
ROBUSTNESS_WEIGHT = 0.10


def _text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")


def _sha256(value: str, field: str) -> None:
    _text(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")


def _unit(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and in [0, 1]")
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")


def _required(value: float | None, field: str) -> float:
    if value is None:
        raise ValueError(f"{field} is missing")
    return value


@dataclass(frozen=True, slots=True)
class FeatureSubsetFoldEvidence:
    """One candidate tuple's evidence on one monthly inner fold."""

    fold_id: str
    valid: bool
    latest: bool = False
    target_log_score: float | None = None
    baseline_target_log_score: float | None = None
    calibration_error: float | None = None
    stability_score: float | None = None
    support_score: float | None = None
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.fold_id, "fold_id")
        if not isinstance(self.valid, bool) or not isinstance(self.latest, bool):
            raise ValueError("feature-subset fold flags must be boolean")
        values = (
            self.target_log_score,
            self.baseline_target_log_score,
            self.calibration_error,
            self.stability_score,
            self.support_score,
        )
        if not self.valid:
            _text(self.invalid_reason or "", "invalid_reason")
            if any(value is not None for value in values):
                raise ValueError("invalid feature-subset folds cannot contain score values")
            return
        if self.invalid_reason is not None:
            raise ValueError("valid feature-subset folds cannot contain an invalid reason")
        if any(value is None or isinstance(value, bool) or not isfinite(value) for value in values):
            raise ValueError("valid feature-subset folds require finite score values")
        assert self.calibration_error is not None
        assert self.stability_score is not None
        assert self.support_score is not None
        _unit(self.calibration_error, "calibration_error")
        _unit(self.stability_score, "stability_score")
        _unit(self.support_score, "support_score")


@dataclass(frozen=True, slots=True)
class FeatureSubsetCandidate:
    """Immutable candidate and monthly inner-fold evaluation contract."""

    feature_names: tuple[str, ...]
    feature_order_hash: str
    source_build_id: str
    evaluation_plan_hash: str
    folds: tuple[FeatureSubsetFoldEvidence, ...]
    fold_plan_id: str = INNER_FOLD_PLAN_ID
    outer_test_accessed: bool = False

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature-subset candidates require unique features")
        if any(
            not isinstance(name, str) or not name or name.strip() != name
            for name in self.feature_names
        ):
            raise ValueError("feature-subset feature names must be non-empty and trimmed")
        _sha256(self.feature_order_hash, "feature_order_hash")
        _text(self.source_build_id, "source_build_id")
        _sha256(self.evaluation_plan_hash, "evaluation_plan_hash")
        if self.fold_plan_id != INNER_FOLD_PLAN_ID:
            raise ValueError(
                "feature-subset scoring requires monthly inner folds within outer TRAIN"
            )
        if self.outer_test_accessed:
            raise ValueError("feature-subset scoring may not access Outer TEST")
        if not self.folds or any(
            not isinstance(fold, FeatureSubsetFoldEvidence) for fold in self.folds
        ):
            raise ValueError("feature-subset candidate requires inner-fold evidence")
        if len({fold.fold_id for fold in self.folds}) != len(self.folds):
            raise ValueError("feature-subset fold IDs must be unique")
        latest_count = sum(fold.latest for fold in self.folds)
        if latest_count != 1:
            raise ValueError("feature-subset candidate requires exactly one latest inner fold")


@dataclass(frozen=True, slots=True)
class FeatureSubsetBreakdown:
    """Auditable score components and eligibility for one feature tuple."""

    feature_names: tuple[str, ...]
    score_version: str
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
    total_score: float | None
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.score_version != FEATURE_SUBSET_SCORE_VERSION:
            raise ValueError("unsupported feature-subset score version")
        if self.planned_fold_count < 1 or not 0 <= self.valid_fold_count <= self.planned_fold_count:
            raise ValueError("feature-subset fold counts are inconsistent")
        _unit(self.valid_fold_rate, "valid_fold_rate")
        if (
            abs(self.valid_fold_rate - self.valid_fold_count / self.planned_fold_count)
            > SCORE_ABS_TOLERANCE
        ):
            raise ValueError("feature-subset valid-fold rate does not reconcile")
        if not isinstance(self.latest_fold_valid, bool):
            raise ValueError("latest-fold validity must be boolean")
        for field in (
            "forecast_score",
            "calibration_score",
            "stability_score",
            "robustness_score",
            "worst_fold_forecast_score",
            "mean_support_score",
        ):
            value = getattr(self, field)
            if value is not None:
                _unit(value, field)
        if any(
            not isinstance(reason, str) or not reason or reason.strip() != reason
            for reason in self.rejection_reasons
        ):
            raise ValueError("feature-subset rejection reasons must be non-empty strings")
        if len(set(self.rejection_reasons)) != len(self.rejection_reasons):
            raise ValueError("feature-subset rejection reasons must be unique")
        if self.eligible != (not self.rejection_reasons):
            raise ValueError("feature-subset eligibility must match rejection reasons")
        if self.eligible != (self.total_score is not None):
            raise ValueError("eligible feature subsets require exactly one total score")
        if self.total_score is not None and not isfinite(self.total_score):
            raise ValueError("feature-subset total score must be finite")


def _forecast_score(target: float, baseline: float) -> float:
    return 0.5 + 0.5 * tanh(target - baseline)


def score_feature_subset(
    candidate: FeatureSubsetCandidate, *, latest_fold_id: str
) -> FeatureSubsetBreakdown:
    """Score one tuple using only eligible monthly inner folds.

    Unlike cross-K selection, this score has no state-count or feature-dimension
    complexity penalty.  Raw likelihood/AIC/BIC never enter the returned total.
    """

    _text(latest_fold_id, "latest_fold_id")
    latest = next((fold for fold in candidate.folds if fold.fold_id == latest_fold_id), None)
    if latest is None:
        raise ValueError("latest_fold_id is not present in the candidate")
    valid = tuple(fold for fold in candidate.folds if fold.valid)
    valid_rate = len(valid) / len(candidate.folds)
    reasons: list[str] = []
    if not valid:
        reasons.append("zero valid inner folds")
    if valid_rate < VALID_FOLD_RATE_GATE:
        reasons.append("valid-fold rate below 0.80")
    if len(valid) < MIN_VALID_FOLD_COUNT:
        reasons.append("fewer than three valid inner folds")
    if not latest.valid:
        reasons.append("latest inner fold is invalid")
    if not valid:
        return FeatureSubsetBreakdown(
            candidate.feature_names,
            FEATURE_SUBSET_SCORE_VERSION,
            len(candidate.folds),
            0,
            valid_rate,
            latest.valid,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            tuple(reasons),
        )
    for fold in valid:
        assert fold.target_log_score is not None
        assert fold.baseline_target_log_score is not None
        assert fold.calibration_error is not None
        assert fold.stability_score is not None
        assert fold.support_score is not None
    forecast_values = tuple(
        _forecast_score(
            _required(fold.target_log_score, "target_log_score"),
            _required(fold.baseline_target_log_score, "baseline_target_log_score"),
        )
        for fold in valid
    )
    calibration = fmean(
        1.0 - _required(fold.calibration_error, "calibration_error") for fold in valid
    )
    stability = fmean(_required(fold.stability_score, "stability_score") for fold in valid)
    support = fmean(_required(fold.support_score, "support_score") for fold in valid)
    forecast = fmean(forecast_values)
    worst = min(forecast_values)
    robustness = 0.50 * valid_rate + 0.25 * worst + 0.25 * support
    total = (
        FORECAST_WEIGHT * forecast
        + CALIBRATION_WEIGHT * calibration
        + STABILITY_WEIGHT * stability
        + ROBUSTNESS_WEIGHT * robustness
        if not reasons
        else None
    )
    return FeatureSubsetBreakdown(
        candidate.feature_names,
        FEATURE_SUBSET_SCORE_VERSION,
        len(candidate.folds),
        len(valid),
        valid_rate,
        latest.valid,
        forecast,
        calibration,
        stability,
        robustness,
        worst,
        support,
        total,
        not reasons,
        tuple(reasons),
    )


def rank_feature_subset_scores(
    scores: Iterable[FeatureSubsetBreakdown],
) -> tuple[FeatureSubsetBreakdown, ...]:
    """Rank eligible tuples by the canonical score tie-break sequence."""

    eligible = tuple(score for score in scores if score.eligible)
    if not eligible:
        return ()
    if any(score.total_score is None for score in eligible):
        raise ValueError("eligible feature-subset scores require totals")
    return tuple(
        sorted(
            eligible,
            key=lambda score: (
                -_required(score.total_score, "total_score"),
                -_required(score.forecast_score, "forecast_score"),
                -_required(score.worst_fold_forecast_score, "worst_fold_forecast_score"),
                -_required(score.calibration_score, "calibration_score"),
                -_required(score.stability_score, "stability_score"),
                -_required(score.robustness_score, "robustness_score"),
                len(score.feature_names),
                score.feature_names,
            ),
        )
    )


__all__ = [
    "FEATURE_SUBSET_SCORE_VERSION",
    "INNER_FOLD_PLAN_ID",
    "FeatureSubsetBreakdown",
    "FeatureSubsetCandidate",
    "FeatureSubsetFoldEvidence",
    "rank_feature_subset_scores",
    "score_feature_subset",
]
