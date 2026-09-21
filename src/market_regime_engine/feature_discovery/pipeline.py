"""Composition of the canonical TRAIN-only feature-selection stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from market_regime_engine.feature_discovery.ablation import (
    AblationResult,
    SubsetEvaluator,
    run_one_feature_ablation,
)
from market_regime_engine.feature_discovery.family_pca import (
    FamilyPCAArtifact,
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
from market_regime_engine.feature_discovery.sffs import SFFSResult, select_sffs


def _empty_family_reduction(profile_hash: str) -> FamilyNearDuplicateResult:
    return FamilyNearDuplicateResult((), (), (), profile_hash)


def _complete_vector(name: str, values: Sequence[float | None]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values if value is not None)
    if len(result) != len(values) or any(not isfinite(value) for value in result):
        raise ValueError(
            f"quality-eligible feature {name!r} must have finite complete TRAIN values"
        )
    return result


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

    @property
    def selected_features(self) -> tuple[str, ...]:
        return self.sffs.selected_features

    @property
    def evidence_metadata(self) -> dict[str, object]:
        return {
            "feature_selection_profile_hash": self.profile_hash,
            "feature_role_contract_hash": self.role_contract_hash,
            "family_reduction_hash": self.family_reduction.result_hash,
            "family_pca_fit_hashes": tuple(item.fit_hash for item in self.family_pca),
            "global_reduction_hash": self.global_reduction.result_hash,
            "sffs_selected_features": self.sffs.selected_features,
            "ablation_feature_count": len(self.ablation.one_feature_results),
        }


def run_canonical_feature_selection(
    feature_values: Mapping[str, Sequence[float | None]],
    contract: FeatureRoleContract,
    *,
    quality_eligible_features: Sequence[str],
    evaluate_subset: SubsetEvaluator,
    max_sffs_features: int | None = None,
    profile: FeatureSelectionProfile | None = None,
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
    values = {name: _complete_vector(name, feature_values[name]) for name in eligible}
    family_values = {
        name: values[name] for name in eligible if contract.assignment(name).family is not None
    }
    family_reduction = (
        prune_family_near_duplicates(family_values, contract, profile=resolved_profile)
        if family_values
        else _empty_family_reduction(resolved_profile.profile_hash)
    )

    pca_artifacts: list[FamilyPCAArtifact] = []
    generated_values: dict[str, tuple[float, ...]] = {}
    families = tuple(
        sorted(
            {
                family
                for name in family_reduction.retained_features
                if (family := contract.assignment(name).family) is not None
            }
        )
    )
    for family in families:
        family_names = tuple(
            name
            for name in family_reduction.retained_features
            if contract.assignment(name).family == family
        )
        matrix = tuple(
            tuple(values[name][row] for name in family_names)
            for row in range(len(values[family_names[0]]))
        )
        artifact = fit_family_pca(
            matrix,
            family_names,
            contract,
            profile=resolved_profile,
        )
        pca_artifacts.append(artifact)
        transformed = artifact.transform(matrix)
        for column, name in enumerate(artifact.generated_feature_names):
            generated_values[name] = tuple(float(row[column]) for row in transformed)

    core_names = tuple(
        name for name in eligible if contract.assignment(name).role is FeatureRole.CORE
    )
    candidate_values = {name: values[name] for name in core_names}
    candidate_values.update(generated_values)
    global_reduction = prune_global_correlated_features(
        candidate_values,
        contract,
        profile=resolved_profile,
    )
    sffs = select_sffs(
        global_reduction.representatives,
        evaluate_subset,
        max_features=(
            resolved_profile.sffs_max_features if max_sffs_features is None else max_sffs_features
        ),
    )
    ablation = run_one_feature_ablation(sffs.selected_features, evaluate_subset)
    return FeatureSelectionPipelineResult(
        quality_eligible_features=eligible,
        family_reduction=family_reduction,
        family_pca=tuple(pca_artifacts),
        global_reduction=global_reduction,
        sffs=sffs,
        ablation=ablation,
        profile_hash=resolved_profile.profile_hash,
        role_contract_hash=contract.contract_hash,
    )


__all__ = ["FeatureSelectionPipelineResult", "run_canonical_feature_selection"]
