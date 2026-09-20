"""Independent four-slot outer walk-forward validation.

The selector callback receives only Outer-TRAIN rows.  Each ``(fold, K)``
task owns its selection and one subsequent refit/evaluation call; tasks are
process-parallel by default and are assembled in canonical fold/K order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from math import isfinite
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.contracts.core import K_CHAMPION_POLICY_VERSION
from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold, WalkForwardPlan
from market_regime_engine.evaluations.agreement_v4 import compute_soft_regime_nmi
from market_regime_engine.evaluations.k_champion_contract import (
    MIN_VALID_FOLD_COUNT,
    MIN_VALID_FOLD_RATE,
    KChampionSelection,
)
from market_regime_engine.evaluations.process_parallel import cpu_process_pool, is_pickleable
from market_regime_engine.runtime.cpu import cpu_worker_count

_TIMESTAMP_COLUMN = "timestamp_m1"
_SLOT_IDS = ("k2", "k3", "k4", "k5")

KSelectionCallback = Callable[..., KChampionSelection | None]
KFoldEvaluationCallback = Callable[..., "KChampionFoldEvaluation"]


def _utc(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _unit(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and in [0, 1]")
    converted = float(value)
    if not isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return converted


def _validate_timestamps(values: Sequence[datetime], field: str) -> tuple[datetime, ...]:
    result = tuple(_utc(value, f"{field} timestamp") for value in values)
    if any(left >= right for left, right in pairwise(result)):
        raise ValueError(f"{field} timestamps must be strictly increasing")
    if not result:
        raise ValueError(f"{field} timestamps cannot be empty")
    return result


def _validate_probabilities(
    values: Sequence[Sequence[float]], state_count: int, field: str
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if not rows or any(len(row) != state_count for row in rows):
        raise ValueError(f"{field} probabilities have the wrong state dimension")
    if any(
        any(not isfinite(value) or value < 0.0 for value in row) or abs(sum(row) - 1.0) > 1.0e-10
        for row in rows
    ):
        raise ValueError(f"{field} probabilities must be finite and normalized")
    return rows


@dataclass(frozen=True, slots=True)
class KChampionFoldEvaluation:
    """The model-free output of one selected K-slot TEST evaluation."""

    oos_timestamps: tuple[datetime, ...]
    oos_filtered_probabilities: tuple[tuple[float, ...], ...]
    teacher_oos_timestamps: tuple[datetime, ...]
    teacher_oos_filtered_probabilities: tuple[tuple[float, ...], ...]
    stability: float

    def __post_init__(self) -> None:
        _validate_timestamps(self.oos_timestamps, "candidate OOS")
        _validate_timestamps(self.teacher_oos_timestamps, "teacher OOS")
        if len(self.oos_timestamps) != len(self.oos_filtered_probabilities):
            raise ValueError("candidate OOS timestamps/probabilities do not align")
        if len(self.teacher_oos_timestamps) != len(self.teacher_oos_filtered_probabilities):
            raise ValueError("teacher OOS timestamps/probabilities do not align")
        _unit(self.stability, "stability")


@dataclass(frozen=True, slots=True)
class KChampionOuterFoldEvidence:
    """Per-slot/per-fold evidence with no adaptive TEST leakage."""

    slot_id: str
    fold_index: int
    fold_id: str
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    source_snapshot_id: str
    outer_plan_hash: str
    selection: KChampionSelection | None
    selection_hash: str | None
    feature_order_hash: str | None
    oos_timestamps: tuple[datetime, ...]
    oos_filtered_probabilities: tuple[tuple[float, ...], ...]
    teacher_oos_timestamps: tuple[datetime, ...]
    teacher_oos_filtered_probabilities: tuple[tuple[float, ...], ...]
    soft_regime_nmi: float | None
    shared_timestamp_count: int
    common_support: float | None
    stability: float | None
    valid: bool
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if self.slot_id not in _SLOT_IDS:
            raise ValueError("outer evidence slot is not one of k2..k5")
        if self.fold_index < 1 or self.fold_id != f"fold_{self.fold_index:03d}":
            raise ValueError("outer evidence fold identity is invalid")
        for value, field in (
            (self.train_start, "train_start"),
            (self.train_end, "train_end"),
            (self.test_start, "test_start"),
            (self.test_end, "test_end"),
        ):
            _utc(value, field)
        if not self.train_start <= self.train_end < self.test_start <= self.test_end:
            raise ValueError("outer evidence fold bounds are invalid")
        _text(self.source_snapshot_id, "source_snapshot_id")
        if len(self.outer_plan_hash) != 64 or self.outer_plan_hash != self.outer_plan_hash.lower():
            raise ValueError("outer_plan_hash must be a lowercase SHA-256")
        if self.selection is not None:
            if self.selection.slot_id != self.slot_id:
                raise ValueError("outer evidence selection slot mismatch")
            if self.selection_hash != self.selection.selection_hash:
                raise ValueError("outer evidence selection hash does not reconcile")
            if self.feature_order_hash != self.selection.feature_order_hash:
                raise ValueError("outer evidence feature hash does not reconcile")
        elif self.selection_hash is not None or self.feature_order_hash is not None:
            raise ValueError("missing selection cannot have selection provenance")
        if self.valid:
            if self.selection is None or self.invalid_reason is not None:
                raise ValueError("valid outer evidence requires a selection and no failure")
            if (
                self.soft_regime_nmi is None
                or self.common_support is None
                or self.stability is None
            ):
                raise ValueError("valid outer evidence requires agreement/support/stability")
            _unit(self.soft_regime_nmi, "soft_regime_nmi")
            _unit(self.common_support, "common_support")
            _unit(self.stability, "stability")
            if self.shared_timestamp_count < 1:
                raise ValueError("valid outer evidence requires shared timestamp support")
        else:
            if not self.invalid_reason or self.invalid_reason.strip() != self.invalid_reason:
                raise ValueError("invalid outer evidence requires a trimmed reason")
            if any(
                value is not None
                for value in (self.soft_regime_nmi, self.common_support, self.stability)
            ):
                raise ValueError("invalid outer evidence cannot contain partial score values")

    @property
    def result_hash(self) -> str:
        payload = _jsonable(asdict(self))
        return sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class KChampionSlotValidation:
    """Independent eligibility dossier for one K slot."""

    slot_id: str
    source_snapshot_id: str
    outer_plan_hash: str
    planned_fold_count: int
    valid_fold_count: int
    valid_fold_rate: float
    latest_complete_fold_valid: bool
    mean_soft_regime_nmi: float | None
    worst_fold_soft_regime_nmi: float | None
    mean_common_support: float | None
    mean_stability: float | None
    fold_result_hashes: tuple[str, ...]
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.slot_id not in _SLOT_IDS:
            raise ValueError("slot validation slot is not one of k2..k5")
        _text(self.source_snapshot_id, "source_snapshot_id")
        if len(self.outer_plan_hash) != 64 or self.outer_plan_hash != self.outer_plan_hash.lower():
            raise ValueError("outer_plan_hash must be a lowercase SHA-256")
        if self.planned_fold_count < 1 or not 0 <= self.valid_fold_count <= self.planned_fold_count:
            raise ValueError("slot validation fold counts are inconsistent")
        expected_rate = self.valid_fold_count / self.planned_fold_count
        if abs(self.valid_fold_rate - expected_rate) > 1.0e-12:
            raise ValueError("slot validation fold rate does not reconcile")
        if not isinstance(self.latest_complete_fold_valid, bool):
            raise ValueError("latest_complete_fold_valid must be boolean")
        for value, field in (
            (self.mean_soft_regime_nmi, "mean_soft_regime_nmi"),
            (self.worst_fold_soft_regime_nmi, "worst_fold_soft_regime_nmi"),
            (self.mean_common_support, "mean_common_support"),
            (self.mean_stability, "mean_stability"),
        ):
            if value is not None:
                _unit(value, field)
        if len(self.fold_result_hashes) != self.planned_fold_count:
            raise ValueError("slot validation must retain one hash per planned fold")
        for hash_value in self.fold_result_hashes:
            if len(hash_value) != 64 or hash_value != hash_value.lower():
                raise ValueError("fold result hashes must be lowercase SHA-256 values")
        if not isinstance(self.rejection_reasons, tuple) or any(
            not isinstance(reason, str) or not reason or reason.strip() != reason
            for reason in self.rejection_reasons
        ):
            raise ValueError("slot rejection reasons must be trimmed non-empty strings")
        expected = (
            self.valid_fold_rate >= MIN_VALID_FOLD_RATE
            and self.valid_fold_count >= MIN_VALID_FOLD_COUNT
            and self.latest_complete_fold_valid
        )
        if self.eligible != expected or self.eligible != (not self.rejection_reasons):
            raise ValueError("slot eligibility does not reconcile to the pinned gates")
        if self.eligible and any(
            value is None
            for value in (
                self.mean_soft_regime_nmi,
                self.worst_fold_soft_regime_nmi,
                self.mean_common_support,
                self.mean_stability,
            )
        ):
            raise ValueError("eligible slot validation requires all aggregate evidence")


@dataclass(frozen=True, slots=True)
class KChampionOuterPolicyResult:
    """Complete four-slot validation result; slots are never cross-ranked."""

    source_snapshot_id: str
    profile_id: str
    profile_config_version: int
    policy_version: str
    validation_cutoff: datetime
    outer_plan_hash: str
    outer_folds: tuple[KChampionOuterFoldEvidence, ...]
    slots: tuple[KChampionSlotValidation, ...]

    def __post_init__(self) -> None:
        _text(self.source_snapshot_id, "source_snapshot_id")
        _text(self.profile_id, "profile_id")
        if isinstance(self.profile_config_version, bool) or self.profile_config_version < 1:
            raise ValueError("profile_config_version must be positive")
        if self.policy_version != K_CHAMPION_POLICY_VERSION:
            raise ValueError("unsupported K-slot outer policy version")
        _utc(self.validation_cutoff, "validation_cutoff")
        if len(self.outer_plan_hash) != 64 or self.outer_plan_hash != self.outer_plan_hash.lower():
            raise ValueError("outer_plan_hash must be a lowercase SHA-256")
        if not self.outer_folds:
            raise ValueError("outer policy requires fold evidence")
        expected_slots = set(_SLOT_IDS)
        if {slot.slot_id for slot in self.slots} != expected_slots or len(self.slots) != 4:
            raise ValueError("outer policy must contain exactly one dossier for k2..k5")
        fold_counts: dict[int, int] = {}
        for item in self.outer_folds:
            fold_counts[item.fold_index] = fold_counts.get(item.fold_index, 0) + 1
        if not fold_counts or any(count != 4 for count in fold_counts.values()):
            raise ValueError("outer policy must contain one evidence item per slot and fold")
        if any(
            item.source_snapshot_id != self.source_snapshot_id
            or item.outer_plan_hash != self.outer_plan_hash
            for item in self.outer_folds
        ):
            raise ValueError("outer evidence provenance does not match policy")

    @property
    def policy_hash(self) -> str:
        payload = {
            "outer_plan_hash": self.outer_plan_hash,
            "policy_version": self.policy_version,
            "profile_config_version": self.profile_config_version,
            "profile_id": self.profile_id,
            "source_snapshot_id": self.source_snapshot_id,
            "validation_cutoff": self.validation_cutoff.isoformat(),
        }
        return sha256(_canonical(payload)).hexdigest()

    @property
    def result_hash(self) -> str:
        payload = _jsonable(asdict(self))
        return sha256(_canonical(payload)).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value: object) -> bytes:
    return json_dumps(_jsonable(value)).encode("utf-8")


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _invalid_evidence(
    *,
    slot_id: str,
    fold: WalkForwardFold,
    source_snapshot_id: str,
    outer_plan_hash: str,
    reason: str,
    selection: KChampionSelection | None = None,
) -> KChampionOuterFoldEvidence:
    return KChampionOuterFoldEvidence(
        slot_id=slot_id,
        fold_index=fold.fold_index,
        fold_id=fold.fold_id,
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        source_snapshot_id=source_snapshot_id,
        outer_plan_hash=outer_plan_hash,
        selection=selection,
        selection_hash=None if selection is None else selection.selection_hash,
        feature_order_hash=None if selection is None else selection.feature_order_hash,
        oos_timestamps=(),
        oos_filtered_probabilities=(),
        teacher_oos_timestamps=(),
        teacher_oos_filtered_probabilities=(),
        soft_regime_nmi=None,
        shared_timestamp_count=0,
        common_support=None,
        stability=None,
        valid=False,
        invalid_reason=reason,
    )


def _run_outer_task(
    task: tuple[
        WalkForwardFold,
        str,
        pd.DataFrame,
        pd.DataFrame,
        str,
        str,
        int,
        str,
        KSelectionCallback,
        KFoldEvaluationCallback,
    ],
) -> KChampionOuterFoldEvidence:
    (
        fold,
        slot_id,
        train_rows,
        test_rows,
        source_snapshot_id,
        profile_id,
        profile_config_version,
        outer_plan_hash,
        selector,
        evaluator,
    ) = task
    try:
        selection: KChampionSelection | None = None
        selection = selector(train_rows, slot_id=slot_id, fold=fold)
        if selection is None:
            return _invalid_evidence(
                slot_id=slot_id,
                fold=fold,
                source_snapshot_id=source_snapshot_id,
                outer_plan_hash=outer_plan_hash,
                reason="K-slot selector returned no champion",
            )
        if selection.slot_id != slot_id:
            raise RecoverableEvaluationInvalidity("selector returned a different K slot")
        if selection.source_snapshot_id != source_snapshot_id:
            raise RecoverableEvaluationInvalidity(
                "selection source snapshot differs from outer policy"
            )
        if (
            selection.profile_id != profile_id
            or selection.profile_config_version != profile_config_version
        ):
            raise RecoverableEvaluationInvalidity("selection profile differs from outer policy")
        if selection.policy_version != K_CHAMPION_POLICY_VERSION:
            raise RecoverableEvaluationInvalidity(
                "selection policy version differs from outer policy"
            )
        if selection.validation_cutoff != fold.train_end:
            raise RecoverableEvaluationInvalidity(
                "selection validation cutoff must equal the Outer-TRAIN cutoff"
            )
        evaluation = evaluator(
            train_rows,
            test_rows,
            selection=selection,
            slot_id=slot_id,
            fold=fold,
        )
        if not isinstance(evaluation, KChampionFoldEvaluation):
            raise TypeError("K-slot evaluator must return KChampionFoldEvaluation")
        candidate_times = _validate_timestamps(evaluation.oos_timestamps, "candidate OOS")
        teacher_times = _validate_timestamps(evaluation.teacher_oos_timestamps, "teacher OOS")
        candidate_probs = _validate_probabilities(
            evaluation.oos_filtered_probabilities, selection.state_count, "candidate OOS"
        )
        teacher_probs = _validate_probabilities(
            evaluation.teacher_oos_filtered_probabilities, selection.state_count, "teacher OOS"
        )
        agreement = compute_soft_regime_nmi(
            candidate_times,
            candidate_probs,
            teacher_times,
            teacher_probs,
        )
        shared = agreement.shared_timestamp_count
        common_support = shared / len(candidate_times)
        return KChampionOuterFoldEvidence(
            slot_id=slot_id,
            fold_index=fold.fold_index,
            fold_id=fold.fold_id,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            source_snapshot_id=source_snapshot_id,
            outer_plan_hash=outer_plan_hash,
            selection=selection,
            selection_hash=selection.selection_hash,
            feature_order_hash=selection.feature_order_hash,
            oos_timestamps=candidate_times,
            oos_filtered_probabilities=candidate_probs,
            teacher_oos_timestamps=teacher_times,
            teacher_oos_filtered_probabilities=teacher_probs,
            soft_regime_nmi=agreement.soft_regime_nmi,
            shared_timestamp_count=shared,
            common_support=common_support,
            stability=float(evaluation.stability),
            valid=True,
        )
    except RecoverableEvaluationInvalidity as exc:
        return _invalid_evidence(
            slot_id=slot_id,
            fold=fold,
            source_snapshot_id=source_snapshot_id,
            outer_plan_hash=outer_plan_hash,
            reason=f"{type(exc).__name__}: {exc}",
            selection=selection,
        )


def _slot_summary(
    slot_id: str,
    folds: tuple[KChampionOuterFoldEvidence, ...],
    *,
    source_snapshot_id: str,
    outer_plan_hash: str,
) -> KChampionSlotValidation:
    ordered = tuple(sorted(folds, key=lambda item: item.fold_index))
    valid = tuple(item for item in ordered if item.valid)
    valid_rate = len(valid) / len(ordered)
    latest_valid = ordered[-1].valid
    reasons: list[str] = []
    if valid_rate < MIN_VALID_FOLD_RATE:
        reasons.append("valid-fold rate below 0.80")
    if len(valid) < MIN_VALID_FOLD_COUNT:
        reasons.append("fewer than three valid outer folds")
    if not latest_valid:
        reasons.append("latest complete outer fold is invalid")
    if not valid:
        means: tuple[float | None, float | None, float | None, float | None] = (
            None,
            None,
            None,
            None,
        )
    else:
        nmis = tuple(item.soft_regime_nmi for item in valid)
        supports = tuple(item.common_support for item in valid)
        stability = tuple(item.stability for item in valid)
        if any(value is None for value in nmis + supports + stability):
            raise ValueError("valid outer evidence is missing aggregate fields")
        nmi_values = tuple(value for value in nmis if value is not None)
        support_values = tuple(value for value in supports if value is not None)
        stability_values = tuple(value for value in stability if value is not None)
        means = (
            sum(nmi_values) / len(nmi_values),
            min(nmi_values),
            sum(support_values) / len(support_values),
            sum(stability_values) / len(stability_values),
        )
    return KChampionSlotValidation(
        slot_id=slot_id,
        source_snapshot_id=source_snapshot_id,
        outer_plan_hash=outer_plan_hash,
        planned_fold_count=len(ordered),
        valid_fold_count=len(valid),
        valid_fold_rate=valid_rate,
        latest_complete_fold_valid=latest_valid,
        mean_soft_regime_nmi=means[0],
        worst_fold_soft_regime_nmi=means[1],
        mean_common_support=means[2],
        mean_stability=means[3],
        fold_result_hashes=tuple(item.result_hash for item in ordered),
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def run_k_champion_outer_policy(
    source_rows: pd.DataFrame,
    *,
    plan: WalkForwardPlan,
    source_snapshot_id: str,
    profile_id: str,
    profile_config_version: int,
    validation_cutoff: datetime,
    selector: KSelectionCallback,
    evaluator: KFoldEvaluationCallback,
    max_workers: int | None = None,
    timestamp_column: str = _TIMESTAMP_COLUMN,
) -> KChampionOuterPolicyResult:
    """Run all four K slots independently through every outer fold.

    Selection receives a fresh TRAIN-only frame.  The evaluator receives that
    same TRAIN frame plus a separate TEST frame and is called at most once for
    each successful ``(fold, K)`` selection.  A non-pickleable callback is
    rejected when process parallelism is requested instead of silently falling
    back to a GIL-bound serial path.
    """

    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("source_rows must be a pandas DataFrame")
    if not isinstance(plan, WalkForwardPlan) or not plan.folds:
        raise ValueError("four-slot outer policy requires a non-empty walk-forward plan")
    source_snapshot_id = _text(source_snapshot_id, "source_snapshot_id")
    profile_id = _text(profile_id, "profile_id")
    if isinstance(profile_config_version, bool) or profile_config_version < 1:
        raise ValueError("profile_config_version must be positive")
    validation_cutoff = _utc(validation_cutoff, "validation_cutoff")
    if plan.evaluation_cutoff != validation_cutoff:
        raise ValueError("validation_cutoff must equal the outer plan cutoff")
    if timestamp_column not in source_rows.columns:
        raise ValueError(f"source rows require {timestamp_column}")
    timestamps = tuple(_utc(value, "source timestamp") for value in source_rows[timestamp_column])
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("source timestamps must be strictly increasing and unique")
    if not timestamps:
        raise ValueError("source rows cannot be empty")

    tasks: list[tuple[Any, ...]] = []
    for fold in plan.folds:
        train_mask = source_rows[timestamp_column] <= fold.train_end
        test_mask = (source_rows[timestamp_column] >= fold.test_start) & (
            source_rows[timestamp_column] <= fold.test_end
        )
        train_rows = source_rows.loc[train_mask].copy(deep=True)
        test_rows = source_rows.loc[test_mask].copy(deep=True)
        if tuple(train_rows[timestamp_column])[-1] != fold.train_end:
            raise ValueError(f"fold {fold.fold_id} TRAIN does not end at its declared cutoff")
        if tuple(test_rows[timestamp_column]) != tuple(
            timestamp for timestamp in timestamps if fold.test_start <= timestamp <= fold.test_end
        ):
            raise ValueError(f"fold {fold.fold_id} TEST rows do not match its declared window")
        for slot_id in _SLOT_IDS:
            tasks.append(
                (
                    fold,
                    slot_id,
                    train_rows,
                    test_rows,
                    source_snapshot_id,
                    profile_id,
                    profile_config_version,
                    plan.plan_hash,
                    selector,
                    evaluator,
                )
            )

    worker_limit = cpu_worker_count(max_workers, task_count=len(tasks))
    if worker_limit > 1 and (not is_pickleable(selector) or not is_pickleable(evaluator)):
        raise TypeError("parallel four-slot outer policy requires pickleable selector/evaluator")
    if worker_limit <= 1:
        evidence = tuple(_run_outer_task(task) for task in tasks)
    else:
        with cpu_process_pool(worker_limit) as executor:
            evidence = tuple(executor.map(_run_outer_task, tasks))
    evidence = tuple(sorted(evidence, key=lambda item: (item.fold_index, item.slot_id)))
    slots = tuple(
        _slot_summary(
            slot_id,
            tuple(item for item in evidence if item.slot_id == slot_id),
            source_snapshot_id=source_snapshot_id,
            outer_plan_hash=plan.plan_hash,
        )
        for slot_id in _SLOT_IDS
    )
    return KChampionOuterPolicyResult(
        source_snapshot_id=source_snapshot_id,
        profile_id=profile_id,
        profile_config_version=profile_config_version,
        policy_version=K_CHAMPION_POLICY_VERSION,
        validation_cutoff=validation_cutoff,
        outer_plan_hash=plan.plan_hash,
        outer_folds=evidence,
        slots=slots,
    )


__all__ = [
    "KChampionFoldEvaluation",
    "KChampionOuterFoldEvidence",
    "KChampionOuterPolicyResult",
    "KChampionSlotValidation",
    "run_k_champion_outer_policy",
]
