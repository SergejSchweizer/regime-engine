"""Composition of the canonical TRAIN-only feature-selection stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from market_regime_engine.evaluations.process_parallel import is_pickleable
from market_regime_engine.evaluations.task_frontier import SharedTaskFrontier
from market_regime_engine.feature_discovery.ablation import (
    AblationResult,
    HMMSubsetEvaluator,
    SubsetEvaluator,
    run_one_feature_hmm_ablation,
)
from market_regime_engine.feature_discovery.family_pca import (
    FamilyPCAArtifact,
    FamilyPCAStatisticalInvalid,
    fit_family_pca,
)
from market_regime_engine.feature_discovery.family_reduction import (
    FamilyNearDuplicateResult,
    prune_family_near_duplicates,
)
from market_regime_engine.feature_discovery.feature_roles import (
    FeatureRole,
    FeatureRoleContract,
    FeatureSelectionProfile,
    FeatureStage,
)
from market_regime_engine.feature_discovery.global_reduction import (
    GlobalCorrelationResult,
    prune_global_correlated_features,
)
from market_regime_engine.feature_discovery.k_sffs import (
    LEGAL_K,
    KSlotSFFSResult,
    KSubsetScore,
    select_k_slot_sffs,
)
from market_regime_engine.feature_discovery.metadata_store import (
    FeatureSelectionMetadataStore,
    FoldFeatureStat,
    apply_ablation_to_feature_stats,
    apply_pca_credit_to_feature_stats,
    sffs_step_records,
)
from market_regime_engine.feature_discovery.sffs import SFFSResult, select_sffs
from market_regime_engine.runtime.parallel import (
    FoldParallelExecutor,
    ParallelExecutionPlan,
    ReadOnlyMatrix,
)


def _empty_family_reduction(profile_hash: str) -> FamilyNearDuplicateResult:
    return FamilyNearDuplicateResult((), (), (), profile_hash)


def _complete_vector(name: str, values: Sequence[float | None]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values if value is not None)
    if len(result) != len(values) or any(not isfinite(value) for value in result):
        raise ValueError(
            f"quality-eligible feature {name!r} must have finite complete TRAIN values"
        )
    return result


def _bind_feature_values(callback: object, values: Mapping[str, Sequence[float]]) -> None:
    """Expose pipeline-generated TRAIN values to a callback owner when supported."""

    owner = getattr(callback, "__self__", None)
    binder = getattr(owner, "bind_feature_values", None)
    if callable(binder):
        binder(values)


@dataclass(frozen=True, slots=True)
class FeatureSelectionPipelineResult:
    quality_eligible_features: tuple[str, ...]
    family_reduction: FamilyNearDuplicateResult
    family_pca: tuple[FamilyPCAArtifact, ...]
    global_reduction: GlobalCorrelationResult
    sffs: SFFSResult
    ablation: AblationResult
    profile_hash: str
    role_contract_hash: str
    invalid_families: tuple[str, ...] = ()
    k_sffs: tuple[KSlotSFFSResult, ...] = ()

    @property
    def selected_features(self) -> tuple[str, ...]:
        return self.sffs.selected_features

    @property
    def emission_feature_orders(self) -> tuple[tuple[int, tuple[str, ...]], ...]:
        """Return the frozen per-K tuple reused by every emission family."""

        return tuple((item.state_count, item.selected_features) for item in self.k_sffs)

    @property
    def evidence_metadata(self) -> dict[str, object]:
        return {
            "feature_selection_profile_hash": self.profile_hash,
            "feature_role_contract_hash": self.role_contract_hash,
            "family_reduction_hash": self.family_reduction.result_hash,
            "family_pca_fit_hashes": tuple(item.fit_hash for item in self.family_pca),
            "family_pca_invalid_families": self.invalid_families,
            "global_reduction_hash": self.global_reduction.result_hash,
            "sffs_selected_features": self.sffs.selected_features,
            "sffs_selected_features_by_k": self.emission_feature_orders,
            "ablation_feature_count": len(self.ablation.one_feature_results),
            "ablation_selector_contract_hash": self.ablation.selector_contract_hash,
            "ablation_model_family": self.ablation.model_family,
            "ablation_state_count": self.ablation.state_count,
            "ablation_fit_execution_hashes": self.ablation.fit_execution_hashes,
            "ablation_losses": self.ablation.ablation_losses,
            "ablation_invalid_reasons": tuple(
                item.hmm_evaluation.invalid_reason for item in self.ablation.one_feature_results
            ),
        }


@dataclass(frozen=True, slots=True)
class _FamilyStageTask:
    family: str
    feature_names: tuple[str, ...]
    column_indices: tuple[int, ...]
    matrix_path: str
    matrix_shape: tuple[int, int]
    matrix_dtype: str
    contract: FeatureRoleContract
    profile: FeatureSelectionProfile


@dataclass(frozen=True, slots=True)
class _FamilyStageResult:
    family: str
    reduction: FamilyNearDuplicateResult
    artifact: FamilyPCAArtifact | None
    generated_values: tuple[tuple[str, tuple[float, ...]], ...]
    statistically_invalid: bool


def _run_family_stage(task: _FamilyStageTask) -> _FamilyStageResult:
    mapped = np.memmap(
        task.matrix_path,
        dtype=np.dtype(task.matrix_dtype),
        mode="r",
        shape=task.matrix_shape,
    )
    feature_values = {
        name: tuple(float(value) for value in mapped[:, column])
        for name, column in zip(task.feature_names, task.column_indices, strict=True)
    }
    del mapped
    reduction = prune_family_near_duplicates(
        feature_values,
        task.contract,
        profile=task.profile,
    )
    retained_names = reduction.retained_features
    matrix = tuple(
        tuple(feature_values[name][row] for name in retained_names)
        for row in range(len(feature_values[retained_names[0]]))
    )
    try:
        artifact = fit_family_pca(
            matrix,
            retained_names,
            task.contract,
            profile=task.profile,
        )
    except FamilyPCAStatisticalInvalid:
        return _FamilyStageResult(task.family, reduction, None, (), True)
    transformed = artifact.transform(matrix)
    generated = tuple(
        (
            name,
            tuple(float(row[column]) for row in transformed),
        )
        for column, name in enumerate(artifact.generated_feature_names)
    )
    return _FamilyStageResult(task.family, reduction, artifact, generated, False)


def run_canonical_feature_selection(
    feature_values: Mapping[str, Sequence[float | None]],
    contract: FeatureRoleContract,
    *,
    quality_eligible_features: Sequence[str],
    evaluate_subset: SubsetEvaluator,
    evaluate_hmm_subset: HMMSubsetEvaluator,
    hmm_selector_contract_hash: str,
    max_sffs_features: int | None = None,
    profile: FeatureSelectionProfile | None = None,
    max_workers: int | None = None,
    metadata_store: FeatureSelectionMetadataStore | None = None,
    metadata_fold_id: str | None = None,
    metadata_source_build_id: str | None = None,
    metadata_state_count: int | None = None,
    evaluate_gaussian_subset_by_k: KSubsetScore | None = None,
    state_counts: tuple[int, ...] = LEGAL_K,
    metadata_fold_feature_stats: tuple[FoldFeatureStat, ...] | None = None,
    metadata_fold_commit_hook: Callable[[FeatureSelectionMetadataStore, str], None] | None = None,
) -> FeatureSelectionPipelineResult:
    """Run all currently implemented selection stages on one TRAIN snapshot.

    ``quality_eligible_features`` is an explicit boundary: finite/coverage/
    variance validation must have happened upstream on TRAIN rows.  This
    function never sees TEST data and does not infer eligibility from values.
    """

    eligible = tuple(quality_eligible_features)
    contract.validate_stage_features(FeatureStage.QUALITY, eligible)
    if not eligible or any(name not in feature_values for name in eligible):
        raise ValueError("quality-eligible features must have supplied TRAIN vectors")
    resolved_profile = contract.profile if profile is None else profile
    if resolved_profile.profile_hash != contract.profile.profile_hash:
        raise ValueError("pipeline profile must match the role contract profile")
    if metadata_store is not None and (
        metadata_fold_id is None or metadata_source_build_id is None or metadata_state_count is None
    ):
        raise ValueError(
            "metadata_fold_id, metadata_source_build_id and metadata_state_count are required"
        )
    values = {name: _complete_vector(name, feature_values[name]) for name in eligible}
    family_values = {
        name: values[name] for name in eligible if contract.assignment(name).family is not None
    }
    pca_artifacts: list[FamilyPCAArtifact] = []
    invalid_families: list[str] = []
    generated_values: dict[str, tuple[float, ...]] = {}
    if family_values:
        family_names_by_family: dict[str, tuple[str, ...]] = {}
        families = tuple(
            sorted(
                {
                    family
                    for name in family_values
                    if (family := contract.assignment(name).family) is not None
                }
            )
        )
        for family in families:
            family_names_by_family[family] = tuple(
                assignment.feature_name
                for assignment in contract.assignments
                if assignment.feature_name in family_values and assignment.family == family
            )
        all_family_names = tuple(
            name for family in family_names_by_family.values() for name in family
        )
        matrix = np.asarray(
            tuple(
                tuple(values[name][row] for name in all_family_names)
                for row in range(len(values[all_family_names[0]]))
            ),
            dtype=np.float64,
        )
        with (
            TemporaryDirectory(prefix="regime-family-stage-") as matrix_directory,
            ReadOnlyMatrix.create(matrix, matrix_directory) as shared,
        ):
            column_lookup = {name: index for index, name in enumerate(all_family_names)}
            tasks = tuple(
                _FamilyStageTask(
                    family=family,
                    feature_names=family_names,
                    column_indices=tuple(column_lookup[name] for name in family_names),
                    matrix_path=str(shared.path),
                    matrix_shape=matrix.shape,
                    matrix_dtype=matrix.dtype.str,
                    contract=contract,
                    profile=resolved_profile,
                )
                for family, family_names in sorted(family_names_by_family.items())
            )
            plan = ParallelExecutionPlan.create(
                len(tasks),
                requested_workers=max_workers,
                shared_matrix_identity=shared.identity,
            )
            executor: FoldParallelExecutor[_FamilyStageTask, _FamilyStageResult] = (
                FoldParallelExecutor(plan, max_pending=len(tasks))
            )
            with executor:
                family_results = executor.map_ordered(_run_family_stage, tasks)
        retained = tuple(
            name
            for name in eligible
            if name
            in {item for result in family_results for item in result.reduction.retained_features}
        )
        removed = tuple(
            name
            for name in eligible
            if name
            in {item for result in family_results for item in result.reduction.removed_features}
        )
        evidence = tuple(item for result in family_results for item in result.reduction.evidence)
        family_reduction = FamilyNearDuplicateResult(
            retained,
            removed,
            tuple(sorted(evidence, key=lambda item: (item.family, item.leader, item.duplicate))),
            resolved_profile.profile_hash,
        )
        for result in family_results:
            if result.statistically_invalid:
                invalid_families.append(result.family)
            if result.artifact is not None:
                pca_artifacts.append(result.artifact)
            generated_values.update(dict(result.generated_values))
        pca_artifacts.sort(key=lambda artifact: artifact.family)
    else:
        family_reduction = _empty_family_reduction(resolved_profile.profile_hash)

    core_names = tuple(
        name for name in eligible if contract.assignment(name).role is FeatureRole.CORE
    )
    candidate_values = {name: values[name] for name in core_names}
    candidate_values.update(generated_values)
    _bind_feature_values(evaluate_subset, candidate_values)
    _bind_feature_values(evaluate_hmm_subset, candidate_values)
    _bind_feature_values(evaluate_gaussian_subset_by_k, candidate_values)
    global_reduction = prune_global_correlated_features(
        candidate_values,
        contract,
        profile=resolved_profile,
    )
    sffs_max_features = (
        resolved_profile.sffs_max_features if max_sffs_features is None else max_sffs_features
    )
    selection_evaluator = (
        evaluate_gaussian_subset_by_k
        if evaluate_gaussian_subset_by_k is not None
        else evaluate_subset
    )
    can_share_frontier = (
        max_workers != 1
        and is_pickleable(selection_evaluator)
        and is_pickleable(evaluate_hmm_subset)
    )
    frontier_context: Any = (
        SharedTaskFrontier(max_workers) if can_share_frontier else nullcontext(None)
    )
    with frontier_context as frontier:
        if evaluate_gaussian_subset_by_k is None:
            sffs = select_sffs(
                global_reduction.representatives,
                evaluate_subset,
                max_features=sffs_max_features,
                max_workers=max_workers,
                frontier=frontier,
            )
            k_sffs: tuple[KSlotSFFSResult, ...] = ()
        else:
            k_sffs = select_k_slot_sffs(
                global_reduction.representatives,
                evaluate_gaussian_subset_by_k,
                state_counts=state_counts,
                max_features=sffs_max_features,
                max_workers=max_workers,
                frontier=frontier,
            )
            sffs = k_sffs[0].sffs
        if metadata_store is not None:
            assert metadata_fold_id is not None
            assert metadata_source_build_id is not None
            assert metadata_state_count is not None
            records = (
                tuple((item.state_count, item.sffs) for item in k_sffs)
                if k_sffs
                else ((metadata_state_count, sffs),)
            )
            for state_count, slot_sffs in records:
                metadata_store.commit_sffs_steps(
                    sffs_step_records(
                        slot_sffs,
                        fold_id=metadata_fold_id,
                        profile_hash=resolved_profile.profile_hash,
                        source_build_id=metadata_source_build_id,
                        state_count=state_count,
                    )
                )
        ablation = run_one_feature_hmm_ablation(
            sffs.selected_features,
            evaluate_hmm_subset,
            selector_contract_hash=hmm_selector_contract_hash,
            max_workers=max_workers,
            frontier=frontier,
        )
    if metadata_fold_feature_stats is not None and metadata_store is None:
        raise ValueError("metadata_store is required for feature-stat persistence")
    if metadata_store is not None:
        assert metadata_fold_id is not None
        assert metadata_source_build_id is not None
        pca_loadings = tuple(
            loading
            for artifact in pca_artifacts
            for loading in artifact.pca_loadings(metadata_fold_id, metadata_source_build_id)
        )
        metadata_store.commit_pca_loadings(pca_loadings)
        if metadata_fold_feature_stats is not None:
            metadata_store.commit_fold_feature_stats(
                apply_pca_credit_to_feature_stats(
                    apply_ablation_to_feature_stats(metadata_fold_feature_stats, ablation),
                    pca_loadings,
                    ablation,
                )
            )
        if metadata_fold_commit_hook is not None:
            metadata_fold_commit_hook(metadata_store, metadata_fold_id)
    return FeatureSelectionPipelineResult(
        quality_eligible_features=eligible,
        family_reduction=family_reduction,
        family_pca=tuple(pca_artifacts),
        global_reduction=global_reduction,
        sffs=sffs,
        ablation=ablation,
        profile_hash=resolved_profile.profile_hash,
        role_contract_hash=contract.contract_hash,
        invalid_families=tuple(invalid_families),
        k_sffs=k_sffs,
    )


__all__ = ["FeatureSelectionPipelineResult", "run_canonical_feature_selection"]
