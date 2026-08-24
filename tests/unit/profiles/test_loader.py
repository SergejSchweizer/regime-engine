from __future__ import annotations

from copy import deepcopy

import pytest

from market_regime_engine.profiles import assert_xetra_v1_pins, load_profile_mapping


def xetra_mapping() -> dict[str, object]:
    return {
        "profile_id": "xetra",
        "profile_config_version": 1,
        "registered_model": "regime-xetra",
        "production_alias": "champion",
        "challenger_alias": "challenger",
        "feature_selection": {
            "policy_id": "xetra_semantic_medoid_v1",
            "static_features": [],
            "within_block_method": "absolute_spearman_medoid",
            "cross_block_method": "absolute_spearman_prune",
            "minimum_feature_coverage": 0.90,
            "minimum_nonzero_variance": 1e-12,
            "minimum_block_complete_observations": 504,
            "maximum_cross_block_abs_spearman": 0.85,
            "numeric_tie_abs_tolerance": 1e-12,
        },
        "walk_forward": {
            "minimum_train_source_observations": 1260,
            "test_source_observations": 63,
            "step_source_observations": 63,
            "allow_partial_final_test": False,
            "minimum_model_train_observations": 504,
            "minimum_model_test_observations": 42,
            "ranking_abs_tolerance": 1e-12,
        },
        "gaussian_hmm": {
            "candidate_states": [2, 3, 4],
            "backend": "hmmlearn==0.3.3",
            "covariance_type": "full",
            "implementation": "log",
            "seeds": [11, 23, 37, 53, 71, 89, 107, 131],
            "minimum_valid_starts": 6,
            "minimum_multistart_success_rate": 0.75,
            "n_iter": 1000,
            "tol": 1e-4,
            "min_covar": 1e-6,
            "startprob_prior": 1.0,
            "transmat_prior": 1.0,
            "means_prior": 0.0,
            "means_weight": 0.0,
            "covars_prior": 0.01,
            "covars_weight": 1.0,
            "params": "stmc",
            "init_params": "stmc",
        },
        "gates": {
            "minimum_train_hard_occupancy": 0.03,
            "minimum_train_soft_occupancy": 0.05,
            "candidate_minimum_valid_fold_rate": 0.80,
            "low_confidence_threshold": 0.60,
            "state_alignment_ambiguity_abs_tolerance": 1e-10,
            "covariance_asymmetry_abs_tolerance": 1e-10,
            "probability_normalization_abs_tolerance": 1e-10,
            "minimum_covariance_diagonal_variance": 1e-12,
        },
    }


def test_exact_xetra_profile_is_valid_and_hash_is_deterministic() -> None:
    first = load_profile_mapping(xetra_mapping())
    second = load_profile_mapping(deepcopy(xetra_mapping()))
    assert_xetra_v1_pins(first)
    assert first.profile_hash == second.profile_hash
    assert len(first.profile_hash) == 64


def test_unknown_key_fails_closed() -> None:
    raw = xetra_mapping()
    raw["mystery"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        load_profile_mapping(raw)


def test_reduced_covariance_fails_closed() -> None:
    raw = xetra_mapping()
    hmm = raw["gaussian_hmm"]
    assert isinstance(hmm, dict)
    hmm["covariance_type"] = "diag"
    with pytest.raises(ValueError, match="full"):
        load_profile_mapping(raw)


def test_feature_source_modes_are_mutually_exclusive() -> None:
    raw = xetra_mapping()
    feature_selection = raw["feature_selection"]
    assert isinstance(feature_selection, dict)
    feature_selection["static_features"] = ["feature_a"]
    with pytest.raises(ValueError, match="exactly one"):
        load_profile_mapping(raw)


def test_duplicate_static_features_fail() -> None:
    raw = xetra_mapping()
    feature_selection = raw["feature_selection"]
    assert isinstance(feature_selection, dict)
    feature_selection["policy_id"] = None
    feature_selection["static_features"] = ["feature_a", "feature_a"]
    with pytest.raises(ValueError, match="duplicates"):
        load_profile_mapping(raw)


def test_xetra_pin_audit_rejects_agent_selected_constant() -> None:
    raw = xetra_mapping()
    walk_forward = raw["walk_forward"]
    assert isinstance(walk_forward, dict)
    walk_forward["test_source_observations"] = 64
    profile = load_profile_mapping(raw)
    with pytest.raises(ValueError, match="pinned evaluation contract"):
        assert_xetra_v1_pins(profile)
