from __future__ import annotations

import time
from hashlib import sha256

import numpy as np
import pytest

import market_regime_engine.feature_discovery.feature_roles as roles_module
import market_regime_engine.feature_discovery.pipeline as pipeline_module
from market_regime_engine.feature_discovery.ablation import HMMSubsetEvaluation
from market_regime_engine.feature_discovery.feature_roles import (
    CORE_FEATURES,
    TEMPORAL_KEY,
    TRANSFORMATION_FAMILIES,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.pipeline import (
    _FamilyStageResult,
    _FamilyStageTask,
    run_canonical_feature_selection,
)
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore

_ORIGINAL_FAMILY_STAGE = pipeline_module._run_family_stage


def delayed_family_stage(task: _FamilyStageTask) -> _FamilyStageResult:
    time.sleep((int(task.family[-2:]) % 7) * 0.002)
    return _ORIGINAL_FAMILY_STAGE(task)


def test_all_canonical_families_have_parallel_serial_hash_parity() -> None:
    transformation_names = tuple(f"{family}_delta_1obs" for family in TRANSFORMATION_FAMILIES)
    core_names = CORE_FEATURES[:3]
    names = (*core_names, *transformation_names)
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    rng = np.random.default_rng(518)
    matrix = rng.normal(size=(45, len(names)))
    values = {
        name: tuple(float(value) for value in matrix[:, index]) for index, name in enumerate(names)
    }

    def evaluate(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, float(len(features)))

    def evaluate_hmm(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        return HMMSubsetEvaluation(
            FeatureSubsetScore(features, float(len(features))),
            "gaussian_hmm",
            2,
            "a" * 64,
            sha256("|".join(features).encode()).hexdigest(),
        )

    serial = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=evaluate,
        evaluate_hmm_subset=evaluate_hmm,
        hmm_selector_contract_hash="a" * 64,
        max_sffs_features=3,
        max_workers=1,
    )
    parallel = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=evaluate,
        evaluate_hmm_subset=evaluate_hmm,
        hmm_selector_contract_hash="a" * 64,
        max_sffs_features=3,
        max_workers=8,
    )

    assert parallel.family_reduction.result_hash == serial.family_reduction.result_hash
    assert tuple(item.fit_hash for item in parallel.family_pca) == tuple(
        item.fit_hash for item in serial.family_pca
    )
    assert parallel.selected_features == serial.selected_features
    assert parallel.evidence_metadata == serial.evidence_metadata


def test_thirty_two_independent_family_tasks_are_submitted_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "_run_family_stage", delayed_family_stage)
    families = tuple(f"synthetic_family_{index:02d}" for index in range(32))
    monkeypatch.setattr(roles_module, "TRANSFORMATION_FAMILIES", families)
    names = tuple(f"{family}_delta_1obs" for family in families)
    assignments = tuple(
        roles_module.FeatureRoleAssignment(name, roles_module.FeatureRole.TRANSFORMATION, family)
        for name, family in zip(names, families, strict=True)
    )
    contract = roles_module.FeatureRoleContract(assignments)
    matrix = np.random.default_rng(519).normal(size=(40, len(names)))
    values = {
        name: tuple(float(value) for value in matrix[:, index]) for index, name in enumerate(names)
    }

    def evaluate(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, float(len(features)))

    def evaluate_hmm(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        return HMMSubsetEvaluation(
            FeatureSubsetScore(features, float(len(features))),
            "gaussian_hmm",
            2,
            "b" * 64,
            sha256("|".join(features).encode()).hexdigest(),
        )

    result = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=evaluate,
        evaluate_hmm_subset=evaluate_hmm,
        hmm_selector_contract_hash="b" * 64,
        max_sffs_features=3,
        max_workers=8,
    )
    assert len(result.family_pca) == 32
    assert tuple(item.family for item in result.family_pca) == families
