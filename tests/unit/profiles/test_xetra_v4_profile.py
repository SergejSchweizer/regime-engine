from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from market_regime_engine.feature_discovery.contracts import (
    FINAL_CANDIDATE_IDS,
    V4_STATISTICAL_CONSTANTS,
)
from market_regime_engine.profiles.loader import load_profile, load_profile_mapping

PROFILE = Path("configs/profiles/xetra_v4.yaml")


def _raw() -> dict[str, object]:
    value = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _discovery(raw: dict[str, object]) -> dict[str, object]:
    value = raw["feature_discovery"]
    assert isinstance(value, dict)
    return value


def test_v4_uses_a_dedicated_explicit_discovery_contract() -> None:
    profile = load_profile(PROFILE)
    assert profile.profile_id == "xetra"
    assert profile.profile_config_version == 4
    assert profile.feature_selection is None
    assert profile.feature_discovery is not None
    assert profile.feature_discovery.final_candidate_ids == FINAL_CANDIDATE_IDS
    assert profile.feature_discovery.maximum_prefix_length == 8
    assert profile.feature_discovery.cluster_count_max == 12
    assert profile.gaussian_hmm.candidate_states == (2, 3, 4, 5)
    assert tuple((item.state_count, item.mixture_count) for item in profile.gmm_hmms) == (
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
    )
    assert profile.student_t_hmm is not None


def test_v4_explicit_fields_match_the_shared_contract_inventory() -> None:
    discovery = load_profile(PROFILE).feature_discovery
    assert discovery is not None
    for field_name, expected in V4_STATISTICAL_CONSTANTS.items():
        if field_name == "distance":
            field_name = "distance_method"
        elif field_name == "clustering":
            field_name = "clustering_method"
        elif field_name == "temporary_prototype":
            field_name = "temporary_prototype_method"
        elif field_name == "feature_score":
            field_name = "feature_regime_score"
        elif field_name == "feature_score_diagnostic":
            field_name = "feature_score_diagnostic"
        elif field_name == "outer_walk_forward":
            continue
        elif field_name == "final_candidate_universe":
            field_name = "final_candidate_ids"
        elif field_name in {"outer_fold_state_identity", "production_state_identity"}:
            field_name = field_name.replace("_fold_", "_")
        elif field_name == "minimum_nonzero_population_variance":
            field_name = "minimum_feature_variance"
        elif field_name == "feature_score_bin_count":
            field_name = "feature_score_bin_count"
        elif field_name == "minimum_shared_teacher_support":
            field_name = "minimum_teacher_shared_support"
        elif field_name == "prefix_nmi_tie_tolerance":
            field_name = "prefix_nmi_tie_tolerance"
        if hasattr(discovery, field_name):
            actual = getattr(discovery, field_name)
            if field_name == "outer_fold_state_identity":
                actual = discovery.outer_state_identity
            assert actual == expected


def test_v4_one_field_mutation_and_legacy_v1_v3_separation_fail_closed() -> None:
    raw = _raw()
    _discovery(raw)["maximum_prefix_length"] = 9
    with pytest.raises(ValueError, match="maximum_prefix_length"):
        load_profile_mapping(raw)

    raw = _raw()
    legacy_raw = yaml.safe_load(Path("configs/profiles/xetra_v1.yaml").read_text(encoding="utf-8"))
    assert isinstance(legacy_raw, dict)
    raw["feature_selection"] = legacy_raw["feature_selection"]
    with pytest.raises(ValueError, match="exactly one"):
        load_profile_mapping(raw)

    raw = _raw()
    del raw["feature_discovery"]
    with pytest.raises(ValueError, match="exactly one"):
        load_profile_mapping(raw)

    for version in (1, 2, 3):
        legacy = load_profile(Path(f"configs/profiles/xetra_v{version}.yaml"))
        assert legacy.feature_selection is not None
        assert legacy.feature_discovery is None


def test_feature_discovery_config_is_frozen_and_rejects_non_contract_values() -> None:
    discovery = load_profile(PROFILE).feature_discovery
    assert discovery is not None
    with pytest.raises(AttributeError):
        discovery.maximum_prefix_length = 7  # type: ignore[misc]
    with pytest.raises(ValueError, match="feature_regime_score"):
        replace(discovery, feature_regime_score="eta_squared")
