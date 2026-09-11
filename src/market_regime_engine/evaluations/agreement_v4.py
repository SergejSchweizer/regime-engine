"""Label-invariant soft agreement for the global Xetra v4 selection chain."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import fsum, isfinite, log

from market_regime_engine.feature_discovery.contracts import content_hash

_PROBABILITY_TOLERANCE = 1.0e-10
_ENTROPY_TOLERANCE = 1.0e-15


@dataclass(frozen=True, slots=True)
class SoftRegimeNmiAgreement:
    """A complete, immutable soft-NMI calculation on common timestamps.

    ``joint_matrix`` is the probability-weighted joint distribution
    ``J[a,b] = mean_t(left[t,a] * right[t,b])``.  It contains no state-label
    matching and therefore remains invariant to independent permutations of
    either model's state columns.
    """

    left_state_count: int
    right_state_count: int
    shared_timestamps: tuple[datetime, ...]
    joint_matrix: tuple[tuple[float, ...], ...]
    left_marginals: tuple[float, ...]
    right_marginals: tuple[float, ...]
    left_entropy: float
    right_entropy: float
    mutual_information: float
    soft_regime_nmi: float
    shared_timestamp_count: int
    shared_timestamp_hash: str

    def __post_init__(self) -> None:
        if self.left_state_count < 1 or self.right_state_count < 1:
            raise ValueError("soft NMI state counts must be positive")
        if not self.shared_timestamps:
            raise ValueError("soft NMI requires non-empty shared timestamps")
        if self.shared_timestamp_count != len(self.shared_timestamps):
            raise ValueError("soft NMI shared timestamp count does not reconcile")
        if any(
            not isinstance(timestamp, datetime) or timestamp.tzinfo is None
            for timestamp in self.shared_timestamps
        ):
            raise ValueError("soft NMI timestamps must be timezone-aware datetimes")
        if any(
            current <= previous
            for previous, current in zip(
                self.shared_timestamps, self.shared_timestamps[1:], strict=False
            )
        ):
            raise ValueError("soft NMI shared timestamps must be strictly increasing")
        if len(self.joint_matrix) != self.left_state_count or any(
            len(row) != self.right_state_count for row in self.joint_matrix
        ):
            raise ValueError("soft NMI joint matrix dimensions are invalid")
        if (
            len(self.left_marginals) != self.left_state_count
            or len(self.right_marginals) != self.right_state_count
        ):
            raise ValueError("soft NMI marginal dimensions are invalid")
        if any(value < 0.0 or not isfinite(value) for row in self.joint_matrix for value in row):
            raise ValueError("soft NMI joint matrix must be finite and non-negative")
        if any(
            value < 0.0 or not isfinite(value)
            for value in (*self.left_marginals, *self.right_marginals)
        ):
            raise ValueError("soft NMI marginals must be finite and non-negative")
        if (
            abs(fsum(value for row in self.joint_matrix for value in row) - 1.0)
            > _PROBABILITY_TOLERANCE
        ):
            raise ValueError("soft NMI joint matrix must sum to one")
        if (
            abs(fsum(self.left_marginals) - 1.0) > _PROBABILITY_TOLERANCE
            or abs(fsum(self.right_marginals) - 1.0) > _PROBABILITY_TOLERANCE
        ):
            raise ValueError("soft NMI marginals must sum to one")
        for index, marginal in enumerate(self.left_marginals):
            if abs(marginal - fsum(self.joint_matrix[index])) > _PROBABILITY_TOLERANCE:
                raise ValueError("left soft NMI marginal does not reconcile")
        for index, marginal in enumerate(self.right_marginals):
            if (
                abs(
                    marginal
                    - fsum(
                        self.joint_matrix[left_index][index]
                        for left_index in range(self.left_state_count)
                    )
                )
                > _PROBABILITY_TOLERANCE
            ):
                raise ValueError("right soft NMI marginal does not reconcile")
        if any(
            not isfinite(value) or value < 0.0
            for value in (
                self.left_entropy,
                self.right_entropy,
                self.mutual_information,
            )
        ):
            raise ValueError("soft NMI information quantities must be finite and non-negative")
        if self.left_entropy <= _ENTROPY_TOLERANCE or self.right_entropy <= _ENTROPY_TOLERANCE:
            raise ValueError("soft NMI requires non-degenerate marginal entropy")
        if not isfinite(self.soft_regime_nmi) or not 0.0 <= self.soft_regime_nmi <= 1.0:
            raise ValueError("soft_regime_nmi must be finite and in [0,1]")
        if len(self.shared_timestamp_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.shared_timestamp_hash
        ):
            raise ValueError("soft NMI shared timestamp hash must be lowercase SHA-256")

    @property
    def normalized_mutual_information(self) -> float:
        """Normalized mutual information for the selected soft regimes."""
        return self.soft_regime_nmi

    @property
    def nmi(self) -> float:
        """Short name for ``soft_regime_nmi``."""
        return self.soft_regime_nmi

    @property
    def entropy_left(self) -> float:
        return self.left_entropy

    @property
    def entropy_right(self) -> float:
        return self.right_entropy

    @property
    def agreement_hash(self) -> str:
        """Hash all persisted result primitives for audit and lineage."""
        return content_hash(self)

    @property
    def result_hash(self) -> str:
        return self.agreement_hash


def _probability_matrix(
    timestamps: Sequence[datetime],
    probabilities: Sequence[Sequence[float]],
    side: str,
) -> tuple[tuple[datetime, ...], tuple[tuple[float, ...], ...], int]:
    timestamp_values = tuple(timestamps)
    rows = tuple(tuple(row) for row in probabilities)
    if not timestamp_values or not rows:
        raise ValueError(f"{side} soft NMI input must be non-empty")
    if len(timestamp_values) != len(rows):
        raise ValueError(f"{side} soft NMI timestamps and probabilities must align")
    if any(
        not isinstance(timestamp, datetime) or timestamp.tzinfo is None
        for timestamp in timestamp_values
    ):
        raise ValueError(f"{side} soft NMI timestamps must be timezone-aware datetimes")
    if len(set(timestamp_values)) != len(timestamp_values):
        raise ValueError(f"{side} soft NMI timestamps must be unique")
    state_count = len(rows[0])
    if state_count < 1 or any(len(row) != state_count for row in rows):
        raise ValueError(f"{side} soft NMI probability rows must have one state dimension")
    normalized_rows: list[tuple[float, ...]] = []
    for row in rows:
        values: list[float] = []
        for value in row:
            if isinstance(value, bool):
                raise ValueError(f"{side} soft NMI probabilities must be numeric")
            try:
                converted = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{side} soft NMI probabilities must be numeric") from exc
            if not isfinite(converted) or converted < 0.0:
                raise ValueError(f"{side} soft NMI probabilities must be finite and non-negative")
            values.append(converted)
        if abs(fsum(values) - 1.0) > _PROBABILITY_TOLERANCE:
            raise ValueError(f"{side} soft NMI probability rows must be normalized within 1e-10")
        normalized_rows.append(tuple(values))
    return timestamp_values, tuple(normalized_rows), state_count


def _entropy(marginals: tuple[float, ...]) -> float:
    return -fsum(value * log(value) for value in marginals if value > 0.0)


def compute_soft_regime_nmi(
    left_timestamps: Sequence[datetime],
    left_probabilities: Sequence[Sequence[float]],
    right_timestamps: Sequence[datetime],
    right_probabilities: Sequence[Sequence[float]],
) -> SoftRegimeNmiAgreement:
    """Compute section-2.9 soft NMI on the exact shared timestamp set.

    Each input may contain additional timestamps.  Rows are aligned by exact
    timezone-aware timestamp equality, then the common rows are ordered
    chronologically before the joint distribution is calculated.
    """

    left_times, left_rows, left_states = _probability_matrix(
        left_timestamps, left_probabilities, "left"
    )
    right_times, right_rows, right_states = _probability_matrix(
        right_timestamps, right_probabilities, "right"
    )
    left_by_time = dict(zip(left_times, left_rows, strict=True))
    right_by_time = dict(zip(right_times, right_rows, strict=True))
    shared = tuple(sorted(set(left_by_time).intersection(right_by_time)))
    if not shared:
        raise ValueError("soft NMI has zero shared timestamp support")
    aligned_left = tuple(left_by_time[timestamp] for timestamp in shared)
    aligned_right = tuple(right_by_time[timestamp] for timestamp in shared)
    count = len(shared)
    joint = tuple(
        tuple(
            fsum(
                left_row[left_index] * right_row[right_index]
                for left_row, right_row in zip(aligned_left, aligned_right, strict=True)
            )
            / count
            for right_index in range(right_states)
        )
        for left_index in range(left_states)
    )
    left_marginals = tuple(fsum(row) for row in joint)
    right_marginals = tuple(
        fsum(joint[left_index][right_index] for left_index in range(left_states))
        for right_index in range(right_states)
    )
    left_entropy = _entropy(left_marginals)
    right_entropy = _entropy(right_marginals)
    if left_entropy <= _ENTROPY_TOLERANCE or right_entropy <= _ENTROPY_TOLERANCE:
        raise ValueError("soft NMI requires non-degenerate marginal entropy")
    mutual_information = fsum(
        joint[left_index][right_index]
        * log(
            joint[left_index][right_index]
            / (left_marginals[left_index] * right_marginals[right_index])
        )
        for left_index in range(left_states)
        for right_index in range(right_states)
        if joint[left_index][right_index] > 0.0
    )
    mutual_information = max(0.0, mutual_information)
    normalized = min(1.0, max(0.0, 2.0 * mutual_information / (left_entropy + right_entropy)))
    return SoftRegimeNmiAgreement(
        left_state_count=left_states,
        right_state_count=right_states,
        shared_timestamps=shared,
        joint_matrix=joint,
        left_marginals=left_marginals,
        right_marginals=right_marginals,
        left_entropy=left_entropy,
        right_entropy=right_entropy,
        mutual_information=mutual_information,
        soft_regime_nmi=normalized,
        shared_timestamp_count=count,
        shared_timestamp_hash=content_hash(shared),
    )


compare_soft_regime_nmi = compute_soft_regime_nmi
soft_regime_nmi = compute_soft_regime_nmi


__all__ = [
    "SoftRegimeNmiAgreement",
    "compare_soft_regime_nmi",
    "compute_soft_regime_nmi",
    "soft_regime_nmi",
]
