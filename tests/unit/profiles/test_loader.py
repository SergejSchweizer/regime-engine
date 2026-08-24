from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from market_regime_engine.profiles import (
    assert_xetra_v1_pins,
    load_profile,
    load_profile_mapping,
)


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


def _section(raw: dict[str, object], name: str) -> dict[str, object]:
    section = raw[name]
    assert isinstance(section, dict)
    return section


def test_exact_xetra_profile_is_valid_and_hash_is_deterministic() -> None:
    first = load_profile_mapping(xetra_mapping())
    second = load_profile_mapping(deepcopy(xetra_mapping()))
    assert_xetra_v1_pins(first)
    assert first.profile_hash == second.profile_hash
    assert len(first.profile_hash) == 64
    assert first.canonical_dict()["profile_id"] == "xetra"


def test_unknown_and_missing_keys_fail_closed() -> None:
    raw = xetra_mapping()
    raw["mystery"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        load_profile_mapping(raw)

    raw = xetra_mapping()
    del raw["registered_model"]
    with pytest.raises(ValueError, match="missing keys"):
        load_profile_mapping(raw)


def test_nested_section_must_be_mapping() -> None:
    raw = xetra_mapping()
    raw["gates"] = []
    with pytest.raises(ValueError, match="gates must be a mapping"):
        load_profile_mapping(raw)


def test_load_profile_file_and_invalid_yaml_root(tmp_path: Path) -> None:
    valid = tmp_path / "valid.yaml"
    valid.write_text(yaml.safe_dump(xetra_mapping()), encoding="utf-8")
    assert load_profile(valid).profile_id == "xetra"

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_profile(invalid)


def test_reduced_covariance_fails_closed() -> None:
    raw = xetra_mapping()
    _section(raw, "gaussian_hmm")["covariance_type"] = "diag"
    with pytest.raises(ValueError, match="full"):
        load_profile_mapping(raw)


def test_feature_source_modes_are_mutually_exclusive() -> None:
    raw = xetra_mapping()
    _section(raw, "feature_selection")["static_features"] = ["feature_a"]
    with pytest.raises(ValueError, match="exactly one"):
        load_profile_mapping(raw)


def test_duplicate_static_features_fail() -> None:
    raw = xetra_mapping()
    feature_selection = _section(raw, "feature_selection")
    feature_selection["policy_id"] = None
    feature_selection["static_features"] = ["feature_a", "feature_a"]
    with pytest.raises(ValueError, match="duplicates"):
        load_profile_mapping(raw)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("within_block_method", "pearson", "within-block"),
        ("cross_block_method", "none", "cross-block"),
        ("minimum_feature_coverage", 0.0, "coverage"),
        ("minimum_nonzero_variance", 0.0, "variance"),
        ("minimum_block_complete_observations", 0, "complete_observations"),
        ("maximum_cross_block_abs_spearman", 1.0, "spearman"),
        ("numeric_tie_abs_tolerance", 0.0, "tie_abs_tolerance"),
    ],
)
def test_feature_selection_numeric_and_method_guards(key: str, value: object, match: str) -> None:
    raw = xetra_mapping()
    _section(raw, "feature_selection")[key] = value
    with pytest.raises(ValueError, match=match):
        load_profile_mapping(raw)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("minimum_train_source_observations", 0, "observation counts"),
        ("minimum_model_test_observations", 0, "observation counts"),
        ("ranking_abs_tolerance", 0.0, "ranking_abs_tolerance"),
    ],
)
def test_walk_forward_guards(key: str, value: object, match: str) -> None:
    raw = xetra_mapping()
    _section(raw, "walk_forward")[key] = value
    with pytest.raises(ValueError, match=match):
        load_profile_mapping(raw)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("candidate_states", [2, 3], "K=2,3,4"),
        ("backend", "other", "backend"),
        ("implementation", "scaling", "implementation"),
        ("seeds", [11, 11], "unique"),
        ("minimum_valid_starts", 9, "minimum_valid_starts"),
        ("minimum_multistart_success_rate", 0.0, "success_rate"),
        ("n_iter", 0, "positive"),
        ("tol", 0.0, "positive"),
        ("min_covar", 0.0, "positive"),
        ("params", "tmc", "params/init_params"),
        ("init_params", "tmc", "params/init_params"),
    ],
)
def test_hmm_guards(key: str, value: object, match: str) -> None:
    raw = xetra_mapping()
    _section(raw, "gaussian_hmm")[key] = value
    with pytest.raises(ValueError, match=match):
        load_profile_mapping(raw)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("minimum_train_hard_occupancy", 0.0, "gate rates"),
        ("low_confidence_threshold", 1.1, "gate rates"),
        ("state_alignment_ambiguity_abs_tolerance", 0.0, "tolerances"),
        ("minimum_covariance_diagonal_variance", 0.0, "tolerances"),
    ],
)
def test_evaluation_gate_guards(key: str, value: object, match: str) -> None:
    raw = xetra_mapping()
    _section(raw, "gates")[key] = value
    with pytest.raises(ValueError, match=match):
        load_profile_mapping(raw)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("profile_id", "", "profile_id"),
        ("registered_model", " regime-xetra", "registered_model"),
        ("profile_config_version", 0, "profile_config_version"),
        ("production_alias", "current", "aliases"),
        ("challenger_alias", "candidate", "aliases"),
    ],
)
def test_profile_identity_guards(key: str, value: object, match: str) -> None:
    raw = xetra_mapping()
    raw[key] = value
    with pytest.raises(ValueError, match=match):
        load_profile_mapping(raw)


def test_xetra_pin_audit_rejects_agent_selected_constant() -> None:
    raw = xetra_mapping()
    _section(raw, "walk_forward")["test_source_observations"] = 64
    profile = load_profile_mapping(raw)
    with pytest.raises(ValueError, match="pinned evaluation contract"):
        assert_xetra_v1_pins(profile)
