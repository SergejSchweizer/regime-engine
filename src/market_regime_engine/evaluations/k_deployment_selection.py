"""Full-history K-slot deployment selection and one refit per eligible slot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.contracts.core import K_CHAMPION_POLICY_VERSION
from market_regime_engine.evaluations.k_champion_contract import (
    KChampionSelection,
    feature_order_hash,
)
from market_regime_engine.evaluations.k_champion_outer import (
    KChampionOuterPolicyResult,
    KChampionSlotValidation,
)
from market_regime_engine.evaluations.process_parallel import cpu_process_pool, is_pickleable
from market_regime_engine.runtime.cpu import cpu_worker_count

_SLOTS = ("k2", "k3", "k4", "k5")
_TIMESTAMP_COLUMN = "timestamp_m1"

KDeploymentSelector = Callable[..., KChampionSelection | None]
KDeploymentRefitter = Callable[..., "KDeploymentArtifact"]


def _utc(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


@dataclass(frozen=True, slots=True)
class KDeploymentArtifact:
    """Immutable output identity returned by one slot refit."""

    slot_id: str
    selection_hash: str
    artifact_hash: str
    package_directory: str
    source_snapshot_id: str
    deployment_cutoff: datetime
    state_count: int
    model_family: str
    feature_order: tuple[str, ...]
    feature_order_hash: str
    policy_version: str
    source_build_id: str
    source_catalog_hash: str

    def __post_init__(self) -> None:
        if self.slot_id not in _SLOTS:
            raise ValueError("deployment artifact slot must be k2, k3, k4 or k5")
        for value, field in (
            (self.selection_hash, "selection_hash"),
            (self.artifact_hash, "artifact_hash"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if not isinstance(self.package_directory, str) or not self.package_directory.strip():
            raise ValueError("package_directory must be non-empty")
        if not isinstance(self.source_snapshot_id, str) or not self.source_snapshot_id.strip():
            raise ValueError("source_snapshot_id must be non-empty")
        if self.state_count != int(self.slot_id[1:]):
            raise ValueError("deployment artifact state count must match its slot")
        if self.model_family not in {"gaussian_hmm", "gmm_hmm", "student_t_hmm"}:
            raise ValueError("deployment artifact model family is unsupported")
        if not self.feature_order or len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("deployment artifact feature order must be non-empty and unique")
        if self.feature_order_hash != feature_order_hash(self.feature_order):
            raise ValueError("deployment artifact feature-order hash does not reconcile")
        if self.policy_version != K_CHAMPION_POLICY_VERSION:
            raise ValueError("deployment artifact policy version is unsupported")
        if not self.source_build_id or self.source_build_id.strip() != self.source_build_id:
            raise ValueError("deployment artifact source build must be non-empty")
        if (
            len(self.source_catalog_hash) != 64
            or self.source_catalog_hash != self.source_catalog_hash.lower()
            or any(character not in "0123456789abcdef" for character in self.source_catalog_hash)
        ):
            raise ValueError("deployment artifact source catalog hash must be a lowercase SHA-256")
        _utc(self.deployment_cutoff, "deployment_cutoff")


@dataclass(frozen=True, slots=True)
class KDeploymentSlotResult:
    slot_id: str
    eligible: bool
    selection: KChampionSelection | None
    artifact: KDeploymentArtifact | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.slot_id not in _SLOTS:
            raise ValueError("deployment slot must be k2, k3, k4 or k5")
        if self.eligible != (self.selection is not None and self.artifact is not None):
            raise ValueError("deployment eligibility must match selection/artifact presence")
        if self.eligible:
            if self.reason is not None:
                raise ValueError("eligible deployment cannot have a failure reason")
            assert self.selection is not None and self.artifact is not None
            if self.artifact.slot_id != self.slot_id:
                raise ValueError("deployment artifact slot differs from result slot")
            if self.artifact.selection_hash != self.selection.selection_hash:
                raise ValueError("deployment artifact is not bound to the selected configuration")
            if self.artifact.state_count != self.selection.state_count:
                raise ValueError("deployment artifact state count differs from selection")
            if self.artifact.model_family != self.selection.model_family:
                raise ValueError("deployment artifact family differs from selection")
            if self.artifact.feature_order != self.selection.feature_order:
                raise ValueError("deployment artifact feature order differs from selection")
            if self.artifact.feature_order_hash != self.selection.feature_order_hash:
                raise ValueError("deployment artifact feature hash differs from selection")
            if self.artifact.policy_version != self.selection.policy_version:
                raise ValueError("deployment artifact policy differs from selection")
            if self.selection.source_build_id and (
                self.artifact.source_build_id != self.selection.source_build_id
            ):
                raise ValueError("deployment artifact source build differs from selection")
            if self.selection.source_catalog_hash and (
                self.artifact.source_catalog_hash != self.selection.source_catalog_hash
            ):
                raise ValueError("deployment artifact source catalog differs from selection")
        elif not self.reason or self.reason.strip() != self.reason:
            raise ValueError("ineligible deployment requires a trimmed reason")


@dataclass(frozen=True, slots=True)
class KDeploymentSelectionResult:
    source_snapshot_id: str
    validation_cutoff: datetime
    deployment_cutoff: datetime
    slots: tuple[KDeploymentSlotResult, ...]

    def __post_init__(self) -> None:
        if not self.source_snapshot_id.strip():
            raise ValueError("deployment source snapshot ID cannot be empty")
        _utc(self.validation_cutoff, "validation_cutoff")
        _utc(self.deployment_cutoff, "deployment_cutoff")
        if self.validation_cutoff >= self.deployment_cutoff:
            raise ValueError("deployment cutoff must be after validation cutoff")
        if tuple(item.slot_id for item in self.slots) != _SLOTS:
            raise ValueError("deployment result must contain exactly k2,k3,k4,k5 in order")

    @property
    def eligible_artifacts(self) -> tuple[KDeploymentArtifact, ...]:
        return tuple(item.artifact for item in self.slots if item.artifact is not None)


def _run_slot(
    task: tuple[
        pd.DataFrame,
        str,
        KChampionSlotValidation,
        str,
        str,
        str,
        datetime,
        datetime,
        KDeploymentSelector,
        KDeploymentRefitter,
    ],
) -> KDeploymentSlotResult:
    (
        source_rows,
        slot_id,
        validation,
        source_snapshot_id,
        source_build_id,
        source_catalog_hash,
        validation_cutoff,
        cutoff,
        selector,
        refitter,
    ) = task
    if not validation.eligible:
        return KDeploymentSlotResult(
            slot_id=slot_id,
            eligible=False,
            selection=None,
            artifact=None,
            reason="slot failed outer validation: " + "; ".join(validation.rejection_reasons),
        )
    try:
        selection = selector(source_rows, slot_id=slot_id, deployment_cutoff=cutoff)
        if selection is None:
            raise ValueError("deployment selector returned no configuration")
        if selection.slot_id != slot_id:
            raise ValueError("deployment selector returned a different K slot")
        if selection.source_snapshot_id != source_snapshot_id:
            raise ValueError("deployment selection source identity differs from validation")
        if selection.validation_cutoff != validation_cutoff:
            raise ValueError("deployment selection validation cutoff differs from validation")
        if selection.deployment_cutoff != cutoff:
            raise ValueError("deployment selection cutoff differs from source maximum")
        if not selection.source_build_id or not selection.source_catalog_hash:
            raise ValueError("deployment selection is missing source build/catalog identity")
        if selection.source_build_id != source_build_id:
            raise ValueError("deployment selection source build differs from deployment source")
        if selection.source_catalog_hash != source_catalog_hash:
            raise ValueError("deployment selection source catalog differs from deployment source")
        if selection.policy_id != "k_champion_portfolio":
            raise ValueError("deployment selection policy identifier is unsupported")
        missing_features = tuple(
            feature for feature in selection.feature_order if feature not in source_rows.columns
        )
        if missing_features:
            raise ValueError(
                "deployment selection features are absent from the source: "
                + ", ".join(missing_features)
            )
        artifact = refitter(source_rows, selection=selection, slot_id=slot_id)
        if not isinstance(artifact, KDeploymentArtifact):
            raise TypeError("K deployment refitter must return KDeploymentArtifact")
        if artifact.source_snapshot_id != source_snapshot_id:
            raise ValueError("deployment artifact source identity differs from validation")
        if artifact.deployment_cutoff != cutoff:
            raise ValueError("deployment artifact cutoff differs from source maximum")
        return KDeploymentSlotResult(slot_id, True, selection, artifact)
    except Exception as exc:
        return KDeploymentSlotResult(
            slot_id=slot_id,
            eligible=False,
            selection=None,
            artifact=None,
            reason=f"{type(exc).__name__}: {exc}",
        )


def select_k_deployment_packages(
    source_rows: pd.DataFrame,
    *,
    validation: KChampionOuterPolicyResult,
    deployment_cutoff: datetime,
    source_build_id: str,
    source_catalog_hash: str,
    selector: KDeploymentSelector,
    refitter: KDeploymentRefitter,
    max_workers: int | None = None,
    timestamp_column: str = _TIMESTAMP_COLUMN,
) -> KDeploymentSelectionResult:
    """Rerun selection through the source maximum and refit each eligible K once."""

    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("K deployment selection requires a pandas DataFrame")
    if timestamp_column not in source_rows.columns:
        raise ValueError(f"deployment source requires {timestamp_column}")
    deployment_cutoff = _utc(deployment_cutoff, "deployment_cutoff")
    if not source_build_id or source_build_id.strip() != source_build_id:
        raise ValueError("source_build_id must be non-empty and trimmed")
    if (
        len(source_catalog_hash) != 64
        or source_catalog_hash != source_catalog_hash.lower()
        or any(character not in "0123456789abcdef" for character in source_catalog_hash)
    ):
        raise ValueError("source_catalog_hash must be a lowercase SHA-256")
    timestamps = tuple(
        _utc(value, "deployment source timestamp") for value in source_rows[timestamp_column]
    )
    if not timestamps or any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("deployment source timestamps must be strictly increasing")
    if timestamps[-1] != deployment_cutoff:
        raise ValueError("deployment source must extend exactly through deployment cutoff")
    if validation.validation_cutoff >= deployment_cutoff:
        raise ValueError("deployment cutoff must be after validation cutoff")
    workers = cpu_worker_count(max_workers, task_count=len(_SLOTS))
    if workers > 1 and (not is_pickleable(selector) or not is_pickleable(refitter)):
        raise TypeError("parallel K deployment requires pickleable selector and refitter")
    by_slot = {item.slot_id: item for item in validation.slots}
    if set(by_slot) != set(_SLOTS) or len(by_slot) != len(_SLOTS):
        raise ValueError("deployment validation must contain exactly one dossier per K slot")
    tasks = tuple(
        (
            source_rows.copy(deep=True),
            slot_id,
            by_slot[slot_id],
            validation.source_snapshot_id,
            source_build_id,
            source_catalog_hash,
            validation.validation_cutoff,
            deployment_cutoff,
            selector,
            refitter,
        )
        for slot_id in _SLOTS
    )
    if workers == 1:
        results = tuple(_run_slot(task) for task in tasks)
    else:
        with cpu_process_pool(workers) as executor:
            results = tuple(executor.map(_run_slot, tasks))
    return KDeploymentSelectionResult(
        source_snapshot_id=validation.source_snapshot_id,
        validation_cutoff=validation.validation_cutoff,
        deployment_cutoff=deployment_cutoff,
        slots=results,
    )


__all__ = [
    "KDeploymentArtifact",
    "KDeploymentSelectionResult",
    "KDeploymentSlotResult",
    "select_k_deployment_packages",
]
