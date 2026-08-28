"""Label-invariant agreement between one univariate and multivariate winner."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import log

import numpy as np

from market_regime_engine.evaluation.walk_forward import WalkForwardEvaluation


@dataclass(frozen=True, slots=True)
class UnivariateAgreement:
    feature_name: str
    shared_fold_ids: tuple[str, ...]
    shared_fold_count: int
    shared_fold_rate: float
    shared_timestamp_count: int
    dominant_state_nmi: float | None
    permutation_hard_agreement: float | None
    permutation_mapping: tuple[int, ...] | None
    unavailable_reason: str | None


def _nmi(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    count = len(left)
    left_counts = {value: left.count(value) for value in set(left)}
    right_counts = {value: right.count(value) for value in set(right)}
    joint = {
        (x, y): sum(1 for a, b in zip(left, right, strict=True) if (a, b) == (x, y))
        for x in left_counts
        for y in right_counts
    }
    mutual = sum(
        (value / count) * log((value * count) / (left_counts[x] * right_counts[y]))
        for (x, y), value in joint.items()
        if value
    )
    entropy_left = -sum((value / count) * log(value / count) for value in left_counts.values())
    entropy_right = -sum((value / count) * log(value / count) for value in right_counts.values())
    return (
        1.0
        if entropy_left + entropy_right == 0.0
        else 2.0 * mutual / (entropy_left + entropy_right)
    )


def compare_univariate_to_multivariate(
    feature_name: str,
    univariate: WalkForwardEvaluation,
    multivariate: WalkForwardEvaluation,
) -> UnivariateAgreement:
    """Compare valid OOS support only; unavailable support does not invalidate either winner."""

    if (
        not feature_name
        or len(univariate.feature_order) != 1
        or univariate.feature_order[0] != feature_name
    ):
        raise ValueError("agreement requires one valid univariate feature winner")
    if (
        univariate.source_build_id != multivariate.source_build_id
        or univariate.evaluation_plan_hash != multivariate.evaluation_plan_hash
    ):
        raise ValueError("agreement winners must share source build and walk-forward plan")
    shared_fold_ids = tuple(
        fold.fold_id
        for fold in univariate.valid_folds
        if fold.fold_id in {item.fold_id for item in multivariate.valid_folds}
    )
    rate = len(shared_fold_ids) / len(univariate.folds)
    values: list[tuple[int, int]] = []
    multi_by_fold = {fold.fold_id: fold for fold in multivariate.valid_folds}
    for fold in univariate.valid_folds:
        other = multi_by_fold.get(fold.fold_id)
        if other is None:
            continue
        univariate_states = {
            time: int(np.argmax(probability))
            for time, probability in zip(
                fold.oos_timestamps, fold.oos_filtered_probabilities, strict=True
            )
        }
        multivariate_states = {
            time: int(np.argmax(probability))
            for time, probability in zip(
                other.oos_timestamps, other.oos_filtered_probabilities, strict=True
            )
        }
        values.extend(
            (univariate_states[time], multivariate_states[time])
            for time in fold.oos_timestamps
            if time in multivariate_states
        )
    if rate < 0.80 or not values:
        reason = (
            "shared valid-fold support below 0.80" if rate < 0.80 else "zero shared OOS timestamps"
        )
        return UnivariateAgreement(
            feature_name,
            shared_fold_ids,
            len(shared_fold_ids),
            rate,
            len(values),
            None,
            None,
            None,
            reason,
        )
    univariate_sequence, multivariate_sequence = zip(*values, strict=True)
    permutation_agreement: float | None = None
    mapping: tuple[int, ...] | None = None
    if univariate.state_count == multivariate.state_count:
        candidates = tuple(permutations(range(univariate.state_count)))
        scores = tuple(
            sum(
                mapping[value] == observed
                for value, observed in zip(univariate_sequence, multivariate_sequence, strict=True)
            )
            / len(univariate_sequence)
            for mapping in candidates
        )
        best = max(scores)
        mapping = min(
            candidate for candidate, score in zip(candidates, scores, strict=True) if score == best
        )
        permutation_agreement = best
    return UnivariateAgreement(
        feature_name,
        shared_fold_ids,
        len(shared_fold_ids),
        rate,
        len(values),
        _nmi(univariate_sequence, multivariate_sequence),
        permutation_agreement,
        mapping,
        None,
    )
