"""Process-parallel orchestration boundary for K-specific TRAIN-only selection.

The actual discovery stages are supplied as one process-safe policy callback.
The boundary owns the important invariants: exactly one K per task, immutable
source identity, no TEST frame in the callback signature, and canonical K
ordering.  This keeps the four policies independent and prevents the legacy
single-v4 selector from silently choosing another teacher K.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluations.k_champion_contract import KChampionSelection
from market_regime_engine.evaluations.process_parallel import cpu_process_pool, is_pickleable
from market_regime_engine.runtime.cpu import cpu_worker_count

LEGAL_K = (2, 3, 4, 5)
K_FEATURE_SELECTION_VERSION = "k_specific_feature_selection.v1"


@dataclass(frozen=True, slots=True)
class KFeatureSelectionPayload:
    """Immutable TRAIN-only selection evidence returned by one K policy."""

    selection: KChampionSelection
    discovery_hash: str
    teacher_identity: str
    prefix_evidence_hash: str

    def __post_init__(self) -> None:
        for value, field in (
            (self.discovery_hash, "discovery_hash"),
            (self.prefix_evidence_hash, "prefix_evidence_hash"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if not isinstance(self.teacher_identity, str) or not self.teacher_identity.strip():
            raise ValueError("teacher_identity must be non-empty")
        if self.teacher_identity != self.selection.reference_teacher_id:
            raise ValueError("teacher_identity must match the immutable selection teacher")


KTrainSelector = Callable[..., KFeatureSelectionPayload | None]


def _utc(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


@dataclass(frozen=True, slots=True)
class KFeatureSelectionResult:
    state_count: int
    source_snapshot_id: str
    validation_cutoff: datetime
    selection: KChampionSelection | None
    discovery_hash: str | None
    teacher_identity: str | None
    prefix_evidence_hash: str | None
    eligible: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.state_count not in LEGAL_K:
            raise ValueError("K feature selection accepts only K=2,3,4,5")
        if not self.source_snapshot_id.strip():
            raise ValueError("source_snapshot_id cannot be empty")
        _utc(self.validation_cutoff, "validation_cutoff")
        if self.eligible != (self.selection is not None):
            raise ValueError("selection eligibility must match selection presence")
        if self.selection is not None:
            if self.selection.state_count != self.state_count:
                raise ValueError("selector returned a different K")
            if self.selection.source_snapshot_id != self.source_snapshot_id:
                raise ValueError("selection source snapshot differs from task identity")
            if self.selection.validation_cutoff != self.validation_cutoff:
                raise ValueError("selection validation cutoff differs from task identity")
            if (
                not self.discovery_hash
                or not self.teacher_identity
                or not self.prefix_evidence_hash
            ):
                raise ValueError("eligible K selection requires discovery/teacher/prefix hashes")
            if self.rejection_reason is not None:
                raise ValueError("eligible K selection cannot have a rejection reason")
        elif not self.rejection_reason or self.rejection_reason.strip() != self.rejection_reason:
            raise ValueError("ineligible K selection requires a trimmed rejection reason")

    @property
    def selection_hash(self) -> str | None:
        return None if self.selection is None else self.selection.selection_hash


def _run_task(
    task: tuple[pd.DataFrame, int, str, datetime, KTrainSelector],
) -> KFeatureSelectionResult:
    train_rows, state_count, source_snapshot_id, validation_cutoff, selector = task
    try:
        payload = selector(
            train_rows,
            state_count=state_count,
            source_snapshot_id=source_snapshot_id,
            validation_cutoff=validation_cutoff,
        )
        if payload is None:
            return KFeatureSelectionResult(
                state_count,
                source_snapshot_id,
                validation_cutoff,
                None,
                None,
                None,
                None,
                False,
                "K-specific selector returned no eligible result",
            )
        if not isinstance(payload, KFeatureSelectionPayload):
            raise TypeError("K-specific selector must return KFeatureSelectionPayload or None")
        selection = payload.selection
        return KFeatureSelectionResult(
            state_count,
            source_snapshot_id,
            validation_cutoff,
            selection,
            payload.discovery_hash,
            payload.teacher_identity,
            payload.prefix_evidence_hash,
            True,
        )
    except Exception as exc:
        return KFeatureSelectionResult(
            state_count,
            source_snapshot_id,
            validation_cutoff,
            None,
            None,
            None,
            None,
            False,
            f"{type(exc).__name__}: {exc}",
        )


def run_k_feature_selection(
    train_rows: pd.DataFrame,
    *,
    source_snapshot_id: str,
    validation_cutoff: datetime,
    selector: KTrainSelector,
    requested_state_counts: tuple[int, ...] = LEGAL_K,
    max_workers: int | None = None,
) -> tuple[KFeatureSelectionResult, ...]:
    """Run one independent TRAIN-only selector per requested K."""

    if not isinstance(train_rows, pd.DataFrame):
        raise TypeError("K feature selection requires a pandas DataFrame")
    if not source_snapshot_id.strip():
        raise ValueError("source_snapshot_id cannot be empty")
    validation_cutoff = _utc(validation_cutoff, "validation_cutoff")
    requested = tuple(requested_state_counts)
    if requested != tuple(sorted(set(requested))) or any(k not in LEGAL_K for k in requested):
        raise ValueError("requested_state_counts must be a unique ordered subset of K=2,3,4,5")
    if not requested:
        raise ValueError("at least one K-specific selector task is required")
    if not is_pickleable(selector) and cpu_worker_count(max_workers, task_count=len(requested)) > 1:
        raise TypeError("parallel K selection requires a pickleable TRAIN-only selector")
    tasks = tuple(
        (train_rows.copy(deep=True), state_count, source_snapshot_id, validation_cutoff, selector)
        for state_count in requested
    )
    workers = cpu_worker_count(max_workers, task_count=len(tasks))
    if workers == 1:
        results = tuple(_run_task(task) for task in tasks)
    else:
        with cpu_process_pool(workers) as executor:
            results = tuple(executor.map(_run_task, tasks))
    return tuple(sorted(results, key=lambda item: item.state_count))


__all__ = [
    "K_FEATURE_SELECTION_VERSION",
    "KFeatureSelectionPayload",
    "KFeatureSelectionResult",
    "run_k_feature_selection",
]
