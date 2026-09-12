"""Central registry for metrics emitted to MLflow LoggedModels."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from market_regime_engine.mlflow_support.ports import MetricPoint

METRIC_CATALOG_VERSION = 1


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    key: str
    human_label: str
    description: str
    unit: str
    direction: str
    scope: str
    step_semantics: str
    aggregation: str
    value_kind: str = "scalar"
    comparison_domain: str = "same_source_plan"
    source_field: str = ""
    model_metrics_visible: bool = True


def _definition(
    key: str,
    label: str,
    description: str,
    unit: str,
    direction: str,
    scope: str,
    step: str,
    aggregation: str,
    *,
    value_kind: str = "scalar",
    comparison_domain: str = "same_source_plan",
    source_field: str = "",
    model_metrics_visible: bool = True,
) -> MetricDefinition:
    if comparison_domain == "same_source_plan" and any(
        token in key for token in ("loglik", "aic", "bic", "hqc")
    ):
        comparison_domain = "same_feature_vector_source_plan"
    return MetricDefinition(
        key=key,
        human_label=label,
        description=description,
        unit=unit,
        direction=direction,
        scope=scope,
        step_semantics=step,
        aggregation=aggregation,
        value_kind=value_kind,
        comparison_domain=comparison_domain,
        source_field=source_field or key,
        model_metrics_visible=model_metrics_visible,
    )


_DEFINITIONS = (
    _definition(
        "train_loglik_total",
        "TRAIN log likelihood",
        "Fitted TRAIN log likelihood.",
        "log-likelihood",
        "higher",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "train_loglik_per_obs",
        "TRAIN log likelihood / observation",
        "TRAIN log likelihood normalized by retained TRAIN observations.",
        "log-likelihood / observation",
        "higher",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "oos_predictive_loglik_total",
        "OOS predictive log likelihood",
        "Continued TEST predictive log likelihood.",
        "log-likelihood",
        "higher",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "oos_predictive_loglik_per_obs",
        "OOS predictive log likelihood / observation",
        "Continued TEST predictive log likelihood normalized by retained TEST observations.",
        "log-likelihood / observation",
        "higher",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "aic",
        "AIC",
        "Akaike information criterion.",
        "criterion",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "bic",
        "BIC",
        "Bayesian information criterion.",
        "criterion",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "hqc",
        "HQC",
        "Hannan-Quinn information criterion.",
        "criterion",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "valid_fold_rate",
        "Valid fold rate",
        "Fraction of planned folds that completed.",
        "fraction",
        "higher",
        "candidate",
        "aggregate",
        "none",
    ),
    _definition(
        "valid_fold_count",
        "Valid fold count",
        "Number of valid folds.",
        "count",
        "higher",
        "candidate",
        "aggregate",
        "count",
    ),
    _definition(
        "invalid_fold_count",
        "Invalid fold count",
        "Number of invalid folds.",
        "count",
        "lower",
        "candidate",
        "aggregate",
        "count",
    ),
    _definition(
        "oos_predictive_loglik_mean",
        "Mean OOS predictive log likelihood",
        "Mean valid-fold OOS score.",
        "log-likelihood / observation",
        "higher",
        "candidate",
        "aggregate",
        "mean",
    ),
    _definition(
        "oos_predictive_loglik_std",
        "OOS predictive log likelihood spread",
        "Population standard deviation of valid-fold scores.",
        "log-likelihood / observation",
        "lower",
        "candidate",
        "aggregate",
        "population_std",
    ),
    _definition(
        "oos_predictive_loglik_worst_fold",
        "Worst-fold OOS score",
        "Minimum valid-fold OOS score.",
        "log-likelihood / observation",
        "higher",
        "candidate",
        "aggregate",
        "min",
    ),
    _definition(
        "oos_predictive_loglik_best_fold",
        "Best-fold OOS score",
        "Maximum valid-fold OOS score.",
        "log-likelihood / observation",
        "higher",
        "candidate",
        "aggregate",
        "max",
    ),
    _definition(
        "multistart_attempt_count",
        "Multistart attempts",
        "Number of optimization starts.",
        "count",
        "neutral",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "multistart_success_count",
        "Successful multistarts",
        "Number of successful starts.",
        "count",
        "higher",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "multistart_failure_count",
        "Failed multistarts",
        "Number of failed starts.",
        "count",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "multistart_success_rate",
        "Multistart success rate",
        "Successful starts divided by attempts.",
        "fraction",
        "higher",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "em_iteration_count",
        "EM iterations",
        "Iterations for one optimization seed.",
        "iterations",
        "lower",
        "seed",
        "seed ordinal",
        "none",
    ),
    _definition(
        "em_converged",
        "EM converged",
        "Convergence status represented as 0/1.",
        "boolean",
        "neutral",
        "seed",
        "seed ordinal",
        "none",
    ),
    _definition(
        "seed_train_loglik",
        "Seed TRAIN log likelihood",
        "TRAIN score for one seed.",
        "log-likelihood",
        "higher",
        "seed",
        "seed ordinal",
        "none",
    ),
    _definition(
        "covariance_min_eigenvalue",
        "Minimum covariance eigenvalue",
        "Smallest fitted covariance eigenvalue.",
        "variance",
        "higher",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "covariance_max_eigenvalue",
        "Maximum covariance eigenvalue",
        "Largest fitted covariance eigenvalue.",
        "variance",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "covariance_condition_number",
        "Covariance condition number",
        "Largest divided by smallest positive eigenvalue.",
        "ratio",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "covariance_min_diagonal_variance",
        "Minimum covariance diagonal variance",
        "Smallest fitted marginal variance.",
        "variance",
        "higher",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "soft_regime_nmi",
        "Soft regime NMI",
        "Agreement with the common teacher.",
        "fraction",
        "higher",
        "outer policy",
        "outer fold",
        "none",
    ),
    _definition(
        "shared_timestamp_count",
        "Shared teacher timestamps",
        "Timestamps shared with the teacher.",
        "count",
        "higher",
        "outer policy",
        "outer fold",
        "none",
    ),
    _definition(
        "selected_M",
        "Selected cluster count",
        "Selected feature clusters.",
        "count",
        "neutral",
        "feature discovery",
        "outer fold",
        "none",
    ),
    _definition(
        "selected_L",
        "Selected feature count",
        "Selected prefix length.",
        "count",
        "neutral",
        "feature discovery",
        "outer fold",
        "none",
    ),
    _definition(
        "selected_K",
        "Selected state count",
        "Number of model states.",
        "count",
        "neutral",
        "candidate",
        "aggregate",
        "none",
    ),
    _definition(
        "eligible_feature_count_N",
        "Eligible feature count (N)",
        "Number of statistically eligible raw features in the outer TRAIN sample.",
        "count",
        "neutral",
        "outer policy",
        "outer fold",
        "none",
    ),
    _definition(
        "selected_cluster_count_M",
        "Selected cluster count (M*)",
        "Global redundancy-cluster count selected by silhouette.",
        "count",
        "neutral",
        "outer policy",
        "outer fold",
        "none",
    ),
    _definition(
        "selected_feature_count_L",
        "Selected feature count (L*)",
        "Final regime-feature prefix length selected by soft NMI.",
        "count",
        "neutral",
        "outer policy",
        "outer fold",
        "none",
    ),
    _definition(
        "selected_state_count_K",
        "Selected state count (K*)",
        "Final hidden-state count selected by the candidate grid.",
        "count",
        "neutral",
        "outer policy",
        "outer fold",
        "none",
    ),
    _definition(
        "outer_selected_M",
        "Outer selected cluster count",
        "Outer-policy selected global cluster count.",
        "count",
        "neutral",
        "outer policy",
        "outer fold",
        "none",
    ),
    _definition(
        "outer_selected_L",
        "Outer selected feature count",
        "Outer-policy selected final feature count.",
        "count",
        "neutral",
        "outer policy",
        "outer fold",
        "none",
    ),
    _definition(
        "outer_selected_K",
        "Outer selected state count",
        "Outer-policy selected hidden-state count.",
        "count",
        "neutral",
        "outer policy",
        "outer fold",
        "none",
    ),
    _definition(
        "outer_soft_regime_nmi",
        "Outer soft regime NMI",
        "Outer final model agreement with the causal teacher.",
        "fraction",
        "higher",
        "outer policy",
        "outer fold",
        "none",
    ),
    _definition(
        "aic_per_train_obs",
        "AIC / TRAIN observation",
        "AIC normalized by retained TRAIN observations.",
        "criterion / observation",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "bic_per_train_obs",
        "BIC / TRAIN observation",
        "BIC normalized by retained TRAIN observations.",
        "criterion / observation",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "seed_train_loglik_per_obs",
        "Seed TRAIN log likelihood / observation",
        "One-seed TRAIN log likelihood normalized by retained TRAIN observations.",
        "log-likelihood / observation",
        "higher",
        "seed",
        "seed ordinal",
        "none",
    ),
    _definition(
        "seed_rank",
        "Seed rank",
        "Descending TRAIN log-likelihood rank among successful seeds.",
        "rank",
        "lower",
        "seed",
        "seed ordinal",
        "none",
    ),
    _definition(
        "best_seed_loglik",
        "Best seed TRAIN log likelihood",
        "Maximum successful-seed TRAIN log likelihood.",
        "log-likelihood",
        "higher",
        "fold",
        "fold index",
        "max",
    ),
    _definition(
        "median_seed_loglik",
        "Median seed TRAIN log likelihood",
        "Median successful-seed TRAIN log likelihood.",
        "log-likelihood",
        "higher",
        "fold",
        "fold index",
        "median",
    ),
    _definition(
        "worst_seed_loglik",
        "Worst seed TRAIN log likelihood",
        "Minimum successful-seed TRAIN log likelihood.",
        "log-likelihood",
        "higher",
        "fold",
        "fold index",
        "min",
    ),
    _definition(
        "seed_loglik_spread",
        "Seed TRAIN log-likelihood spread",
        "Maximum minus minimum successful-seed TRAIN log likelihood.",
        "log-likelihood",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "em_final_loglik",
        "Final EM log likelihood",
        "Last value of the winning seed EM history.",
        "log-likelihood",
        "higher",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "em_final_delta",
        "Final EM delta",
        "Difference between the last two winning-seed EM log-likelihood values.",
        "log-likelihood",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "em_monotonicity_violation_count",
        "EM monotonicity violations",
        "Count of decreases in the winning-seed EM log-likelihood history.",
        "count",
        "lower",
        "fold",
        "fold index",
        "none",
    ),
    _definition(
        "filtered_entropy_mean",
        "Filtered entropy mean",
        "Mean entropy of retained filtered state probabilities.",
        "nats",
        "lower",
        "fold",
        "fold index",
        "mean",
    ),
    _definition(
        "filtered_entropy_std",
        "Filtered entropy spread",
        "Population standard deviation of retained filtered entropy.",
        "nats",
        "lower",
        "fold",
        "fold index",
        "population_std",
    ),
    _definition(
        "filtered_entropy_min",
        "Minimum filtered entropy",
        "Minimum retained filtered entropy.",
        "nats",
        "lower",
        "fold",
        "fold index",
        "min",
    ),
    _definition(
        "filtered_entropy_max",
        "Maximum filtered entropy",
        "Maximum retained filtered entropy.",
        "nats",
        "lower",
        "fold",
        "fold index",
        "max",
    ),
    _definition(
        "dominant_state_confidence_mean",
        "Dominant-state confidence mean",
        "Mean maximum filtered-state probability.",
        "fraction",
        "higher",
        "fold",
        "fold index",
        "mean",
    ),
    _definition(
        "dominant_state_confidence_std",
        "Dominant-state confidence spread",
        "Population standard deviation of maximum filtered-state probability.",
        "fraction",
        "lower",
        "fold",
        "fold index",
        "population_std",
    ),
    _definition(
        "transition_min_probability",
        "Minimum transition probability",
        "Minimum aligned transition probability.",
        "fraction",
        "higher",
        "fold",
        "fold index",
        "min",
    ),
    _definition(
        "transition_max_probability",
        "Maximum transition probability",
        "Maximum aligned transition probability.",
        "fraction",
        "lower",
        "fold",
        "fold index",
        "max",
    ),
    _definition(
        "feature_dimension",
        "Feature dimension",
        "Number of features in the candidate vector.",
        "count",
        "neutral",
        "candidate",
        "aggregate",
        "count",
    ),
    _definition(
        "parameter_count",
        "Free parameter count",
        "Number of free fitted model parameters.",
        "count",
        "lower",
        "candidate",
        "aggregate",
        "count",
    ),
)

METRIC_CATALOG: Mapping[str, MetricDefinition] = MappingProxyType(
    {item.key: item for item in _DEFINITIONS}
)

_DYNAMIC_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"candidate_(?:valid_fold_rate|oos_predictive_loglik_(?:mean|std|worst_fold|best_fold)|bic_mean|aic_mean)",
        r"fold_(?:train_loglik|oos_predictive_loglik|oos_predictive_loglik_per_obs|aic|bic|aic_per_train_obs|bic_per_train_obs|multistart_success_rate|min_train_hard_occupancy|min_train_soft_occupancy|max_state_signature_drift|mean_state_duration|switches_per_year|oos_entropy_mean|oos_confidence_mean)",
        r"fold_(?:train_hard_occupancy|train_soft_occupancy|oos_hard_occupancy|oos_soft_occupancy)_state_[0-9]+",
        r"fold_self_transition_state_[0-9]+",
        r"seed_(?:train_loglik|train_loglik_per_obs|rank)_fold_[0-9]+",
        r"em_(?:iteration_count|converged|loglik_per_obs)_fold_[0-9]+",
        r"state_(?:occupancy_(?:train|oos)_(?:hard|soft)|filtered_probability_(?:mean|std)|expected_duration)_state_[0-9]+",
        r"transition_(?:row_entropy_state_[0-9]+|probability_state_[0-9]+_to_state_[0-9]+|self_probability_state_[0-9]+)",
        r"emission_(?:mean|variance)_state_[0-9]+_feature_[0-9]+",
        r"oos_filtered_probability_state_[0-9]+",
        r"viterbi_state",
        r"state_diag_(?:train|oos)_(?:hard|soft)_occupancy_state_[0-9]+",
        r"state_diag_(?:expected_duration|covariance_condition_number)_state_[0-9]+",
        r"state_diag_covariance_eigenvalue_state_[0-9]+_[0-9]+",
        r"state_diag_(?:posterior_entropy|posterior_confidence)",
        r"state_diag_posterior_probability_state_[0-9]+",
        r"state_diag_viterbi_state",
        r"state_diag_transition_row_entropy_state_[0-9]+",
        r"state_diag_transition_probability_state_[0-9]+_to_state_[0-9]+",
        r"state_diag_emission_(?:mean|variance)_state_[0-9]+_feature_[0-9]+",
        r"state_diag_low_confidence_(?:count|rate)",
        r"fit_quality_(?:train_loglik_total|train_loglik_per_obs|oos_predictive_loglik_total|oos_predictive_loglik_per_obs|aic|bic|hqc|aic_per_train_obs|bic_per_train_obs|hqc_per_train_obs|train_observation_count|valid_fold_count|invalid_fold_count|valid_fold_rate|parameter_count|feature_dimension)(?:_(?:mean|std|minimum|maximum|median|count))?",
        r"predictive_forecast_h[0-9]+_(?:predicted|actual|residual)",
        r"predictive_h[0-9]+_(?:observation_count|missing_target_count|rmse|mae|mape|r2)",
        r"(?:selected_M_silhouette|prefix_soft_regime_nmi|outer_teacher_final_soft_nmi|outer_shared_timestamp_count|outer_oos_predictive_loglik_per_obs|eligible_feature_count)",
    )
)


def metric_definition(key: str) -> MetricDefinition | None:
    definition = METRIC_CATALOG.get(key)
    if definition is not None:
        return definition
    if any(pattern.fullmatch(key) is not None for pattern in _DYNAMIC_PATTERNS):
        comparison_domain = (
            "fold_local_only" if key.startswith("outer_oos_") else "same_source_plan"
        )
        return _definition(
            key,
            key,
            "Indexed metric emitted from deterministic v4 evidence.",
            "see key",
            "neutral",
            "indexed",
            "metric-family-defined",
            "none",
            value_kind="history",
            comparison_domain=comparison_domain,
            source_field="indexed_evidence_metric",
        )
    return None


def require_metric_definition(key: str) -> MetricDefinition:
    definition = metric_definition(key)
    if definition is None:
        raise KeyError(f"metric key is not registered: {key}")
    return definition


def validate_metric_points(points: tuple[MetricPoint, ...]) -> None:
    """Fail closed if a tracker tries to emit an unregistered metric key."""

    seen: set[tuple[str, int]] = set()
    for point in points:
        if not isinstance(point, MetricPoint):
            raise TypeError("metric projection must contain MetricPoint values")
        require_metric_definition(point.key)
        if not isfinite(point.value):
            raise ValueError(f"metric point {point.key} must be finite")
        identity = (point.key, point.step)
        if identity in seen:
            raise ValueError(f"duplicate metric point identity: {point.key}@{point.step}")
        seen.add(identity)


__all__ = [
    "METRIC_CATALOG",
    "METRIC_CATALOG_VERSION",
    "MetricDefinition",
    "metric_definition",
    "require_metric_definition",
    "validate_metric_points",
]
