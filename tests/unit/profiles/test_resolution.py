from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from market_regime_engine.feature_selection.contracts import FeatureBlock, FeatureSelectionPolicy
from market_regime_engine.feature_selection.freeze import freeze_first_train_features
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import (
    resolve_selected_feature_profile,
    validate_candidate_comparison_inputs,
)

FEATURE_POLICY_CONFIG = Path("configs/feature_selection/xetra_semantic_medoid_v1.yaml")
PROFILE_CONFIG = Path("configs/profiles/xetra_v1.yaml")
V2_FEATURE_POLICY_CONFIG = Path("configs/feature_selection/xetra_semantic_medoid_v2.yaml")
V2_PROFILE_CONFIG = Path("configs/profiles/xetra_v2.yaml")


def load_policy(path: Path = FEATURE_POLICY_CONFIG) -> FeatureSelectionPolicy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw_blocks = raw["blocks"]
    assert isinstance(raw_blocks, list)
    blocks = tuple(
        FeatureBlock(
            block_id=str(item["block_id"]),
            features=tuple(str(value) for value in item["features"]),
        )
        for item in raw_blocks
    )
    return FeatureSelectionPolicy(
        policy_id=str(raw["policy_id"]),
        blocks=blocks,
        within_block_method=str(raw["within_block_method"]),
        cross_block_method=str(raw["cross_block_method"]),
        minimum_feature_coverage=float(raw["minimum_feature_coverage"]),
        minimum_nonzero_variance=float(raw["minimum_nonzero_variance"]),
        minimum_block_complete_observations=int(raw["minimum_block_complete_observations"]),
        maximum_cross_block_abs_spearman=float(raw["maximum_cross_block_abs_spearman"]),
        numeric_tie_abs_tolerance=float(raw["numeric_tie_abs_tolerance"]),
    )


def make_selection(policy: FeatureSelectionPolicy):
    rng = np.random.default_rng(20260824)
    frame = pd.DataFrame(
        {
            feature: rng.normal(size=600) + position * 1e-6
            for position, feature in enumerate(policy.feature_universe)
        }
    )
    return freeze_first_train_features(
        frame,
        policy,
        source_build_id="build-1",
        data_sha256="a" * 64,
        evaluation_plan_hash="b" * 64,
    )


def test_resolution_shares_exact_frozen_contract_across_k2_k3_k4_k5() -> None:
    policy = load_policy()
    selection = make_selection(policy)
    resolved = resolve_selected_feature_profile(
        load_profile(PROFILE_CONFIG),
        policy,
        selection,
        source_build_id="build-1",
    )
    assert tuple(candidate.state_count for candidate in resolved.candidates) == (2, 3, 4, 5)
    assert tuple(candidate.candidate_id for candidate in resolved.candidates) == (
        "gaussian_hmm_k2_full",
        "gaussian_hmm_k3_full",
        "gaussian_hmm_k4_full",
        "gaussian_hmm_k5_full",
    )
    assert all(
        candidate.feature_order == selection.final_features for candidate in resolved.candidates
    )
    assert all(
        candidate.feature_dimension == len(selection.final_features)
        for candidate in resolved.candidates
    )
    assert all(candidate.source_build_id == "build-1" for candidate in resolved.candidates)
    assert all(
        candidate.feature_selection_definition_hash == selection.feature_selection_definition_hash
        for candidate in resolved.candidates
    )
    assert all(
        candidate.feature_selection_execution_hash == selection.feature_selection_execution_hash
        for candidate in resolved.candidates
    )


def test_resolution_retains_original_universe_and_preliminary_medoids_separately() -> None:
    policy = load_policy()
    selection = make_selection(policy)
    resolved = resolve_selected_feature_profile(
        load_profile(PROFILE_CONFIG),
        policy,
        selection,
        source_build_id="build-1",
    )
    assert resolved.original_feature_universe == policy.feature_universe
    assert len(resolved.original_feature_universe) == 48
    assert resolved.preliminary_medoids == selection.evidence.preliminary_medoids
    assert len(resolved.preliminary_medoids) == 8
    assert resolved.final_features == selection.final_features
    assert set(resolved.final_features) <= set(resolved.preliminary_medoids)


def test_xetra_v2_resolution_adds_gmm_hmm_k2_through_k5_with_two_mixtures() -> None:
    policy = load_policy(V2_FEATURE_POLICY_CONFIG)
    selection = make_selection(policy)
    resolved = resolve_selected_feature_profile(
        load_profile(V2_PROFILE_CONFIG),
        policy,
        selection,
        source_build_id="build-1",
    )
    gmm_candidates = resolved.candidates[-8:-4]
    assert tuple(candidate.candidate_id for candidate in gmm_candidates) == (
        "gmm_hmm_k2_m2_full",
        "gmm_hmm_k3_m2_full",
        "gmm_hmm_k4_m2_full",
        "gmm_hmm_k5_m2_full",
    )
    assert tuple(
        (candidate.state_count, candidate.mixture_count) for candidate in gmm_candidates
    ) == ((2, 2), (3, 2), (4, 2), (5, 2))
    assert all(candidate.feature_order == selection.final_features for candidate in gmm_candidates)
    student_candidates = resolved.candidates[-4:]
    assert tuple(candidate.candidate_id for candidate in student_candidates) == (
        "student_t_hmm_k2_full",
        "student_t_hmm_k3_full",
        "student_t_hmm_k4_full",
        "student_t_hmm_k5_full",
    )
    assert all(candidate.model_family == "student_t_hmm" for candidate in student_candidates)
    assert all(
        candidate.feature_order == selection.final_features for candidate in student_candidates
    )


def test_candidate_comparison_validation_fails_before_mismatched_feature_contract() -> None:
    policy = load_policy()
    selection = make_selection(policy)
    resolved = resolve_selected_feature_profile(
        load_profile(PROFILE_CONFIG),
        policy,
        selection,
        source_build_id="build-1",
    )
    mismatched = replace(
        resolved.candidates[1],
        source_build_id="build-2",
    )
    with pytest.raises(ValueError, match="must share exact feature order"):
        validate_candidate_comparison_inputs(
            (
                resolved.candidates[0],
                mismatched,
                resolved.candidates[2],
                resolved.candidates[3],
            )
        )


def test_resolution_rejects_policy_and_source_identity_mismatches() -> None:
    policy = load_policy()
    selection = make_selection(policy)
    profile = load_profile(PROFILE_CONFIG)
    with pytest.raises(ValueError, match="source_build_id"):
        resolve_selected_feature_profile(profile, policy, selection, source_build_id=" build-1")

    static_feature_config = replace(
        profile.feature_selection,
        policy_id=None,
        static_features=("vix_level",),
    )
    mismatched_profile = replace(profile, feature_selection=static_feature_config)
    with pytest.raises(ValueError, match="policy_id mismatch"):
        resolve_selected_feature_profile(
            mismatched_profile,
            policy,
            selection,
            source_build_id="build-1",
        )
