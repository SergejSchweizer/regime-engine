"""Deterministic persistent-state alignment for K<=4 Gaussian HMMs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import isfinite

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact
from market_regime_engine.states.signatures import (
    StateSignature,
    rms_distance,
    signature_sort_key,
    state_signatures,
    state_signatures_in_alignment_coordinate,
)

ALIGNMENT_AMBIGUITY_ABS_TOLERANCE = 1e-10


class StateAlignmentAmbiguityError(ValueError):
    """Raised when the pinned alignment rule has no unique answer."""


@dataclass(frozen=True, slots=True)
class StateAlignment:
    persistent_state_ids: tuple[str, ...]
    persistent_to_fitted: tuple[int, ...]
    aligned_signatures: tuple[StateSignature, ...]
    matched_rms: tuple[float, ...]
    total_cost: float
    max_drift: float
    initial_alignment: bool

    def __post_init__(self) -> None:
        state_count = len(self.persistent_state_ids)
        expected_ids = tuple(f"state_{index}" for index in range(state_count))
        if self.persistent_state_ids != expected_ids:
            raise ValueError("persistent state IDs must be canonical state_0..state_(K-1)")
        if tuple(sorted(self.persistent_to_fitted)) != tuple(range(state_count)):
            raise ValueError("alignment must be a one-to-one fitted-state permutation")
        if len(self.aligned_signatures) != state_count or len(self.matched_rms) != state_count:
            raise ValueError("alignment evidence must have one entry per persistent state")
        if any(value < 0.0 or not isfinite(value) for value in self.matched_rms):
            raise ValueError("matched RMS drift must be finite and non-negative")
        if self.total_cost < 0.0 or not isfinite(self.total_cost):
            raise ValueError("alignment total cost must be finite and non-negative")
        if self.max_drift < 0.0 or not isfinite(self.max_drift):
            raise ValueError("alignment max drift must be finite and non-negative")
        if self.matched_rms:
            if abs(self.total_cost - sum(self.matched_rms)) > 1e-12:
                raise ValueError("alignment total cost must equal summed matched RMS distances")
            if abs(self.max_drift - max(self.matched_rms)) > 1e-12:
                raise ValueError("alignment max drift must equal maximum matched RMS distance")


def _state_ids(state_count: int) -> tuple[str, ...]:
    if state_count not in (2, 3, 4, 5):
        raise ValueError("persistent state alignment supports only K=2,3,4,5")
    return tuple(f"state_{index}" for index in range(state_count))


def align_first_fold(
    artifact: GaussianHMMArtifact,
    fold_scaler: StandardScalerArtifact | None = None,
    reference_scaler: StandardScalerArtifact | None = None,
) -> StateAlignment:
    """Assign persistent IDs by the exact rounded-10-decimal lexicographic rule."""

    signatures = (
        state_signatures(artifact)
        if fold_scaler is None and reference_scaler is None
        else state_signatures_in_alignment_coordinate(
            artifact,
            _require_scaler(fold_scaler, "fold"),
            _require_scaler(reference_scaler, "reference"),
        )
    )
    _state_ids(artifact.state_count)
    keyed = [(signature_sort_key(signature), index) for index, signature in enumerate(signatures)]
    keys = [key for key, _ in keyed]
    if len(set(keys)) != len(keys):
        raise StateAlignmentAmbiguityError(
            "first-fold state signatures have identical rounded-10-decimal sort keys"
        )
    keyed.sort(key=lambda item: item[0])
    mapping = tuple(index for _, index in keyed)
    aligned = tuple(signatures[index] for index in mapping)
    zero_drift = tuple(0.0 for _ in mapping)
    return StateAlignment(
        persistent_state_ids=_state_ids(artifact.state_count),
        persistent_to_fitted=mapping,
        aligned_signatures=aligned,
        matched_rms=zero_drift,
        total_cost=0.0,
        max_drift=0.0,
        initial_alignment=True,
    )


def align_to_reference(
    artifact: GaussianHMMArtifact,
    reference_signatures: tuple[StateSignature, ...],
    fold_scaler: StandardScalerArtifact | None = None,
    reference_scaler: StandardScalerArtifact | None = None,
) -> StateAlignment:
    """Align a later fold or final refit to prior persistent signatures by exhaustive K!."""

    current = (
        state_signatures(artifact)
        if fold_scaler is None and reference_scaler is None
        else state_signatures_in_alignment_coordinate(
            artifact,
            _require_scaler(fold_scaler, "fold"),
            _require_scaler(reference_scaler, "reference"),
        )
    )
    state_count = artifact.state_count
    _state_ids(state_count)
    if len(reference_signatures) != state_count:
        raise ValueError("reference signature count must match fitted state count")
    signature_dimension = len(current[0])
    if any(len(signature) != signature_dimension for signature in reference_signatures):
        raise ValueError("reference signatures must match current signature dimension")

    candidates: list[tuple[float, tuple[int, ...], tuple[float, ...]]] = []
    for mapping in permutations(range(state_count)):
        matched = tuple(
            rms_distance(reference_signatures[persistent], current[fitted])
            for persistent, fitted in enumerate(mapping)
        )
        total = sum(matched)
        if not isfinite(total):
            raise ValueError("alignment candidate cost must be finite")
        candidates.append((total, mapping, matched))
    candidates.sort(key=lambda item: (item[0], item[1]))
    best = candidates[0]
    second = candidates[1]
    if second[0] - best[0] <= ALIGNMENT_AMBIGUITY_ABS_TOLERANCE:
        raise StateAlignmentAmbiguityError(
            "state alignment is ambiguous within absolute total-cost tolerance 1e-10"
        )

    mapping = best[1]
    matched = best[2]
    aligned = tuple(current[index] for index in mapping)
    return StateAlignment(
        persistent_state_ids=_state_ids(state_count),
        persistent_to_fitted=mapping,
        aligned_signatures=aligned,
        matched_rms=matched,
        total_cost=best[0],
        max_drift=max(matched),
        initial_alignment=False,
    )


def _require_scaler(
    scaler: StandardScalerArtifact | None, name: str
) -> StandardScalerArtifact:
    if scaler is None:
        raise ValueError(f"{name} scaler is required for fixed-coordinate alignment")
    return scaler
