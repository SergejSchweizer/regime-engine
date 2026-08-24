"""Causal filtered-state inference on the retained-observation clock."""

from __future__ import annotations

import numpy.typing as npt

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.gaussian_hmm import forward_filter
from market_regime_engine.models.protocols import FilterResult


def causal_filter(
    rows: npt.ArrayLike,
    artifact: GaussianHMMArtifact,
    *,
    initial_filtered_probabilities: tuple[float, ...] | None = None,
) -> FilterResult:
    """Filter retained observations with exactly one transition between retained rows.

    When ``initial_filtered_probabilities`` is supplied, the first row uses
    ``initial_filtered_probabilities @ A`` as its prior. This is the required
    TRAIN-to-TEST continuation rule; no calendar-gap transition powers are used.
    """

    return forward_filter(
        rows,
        artifact,
        terminal_train_alpha=initial_filtered_probabilities,
    )
