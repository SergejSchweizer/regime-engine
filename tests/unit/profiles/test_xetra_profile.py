from __future__ import annotations

from pathlib import Path

import pytest

from market_regime_engine.profiles.config import assert_xetra_v1_pins
from market_regime_engine.profiles.loader import load_profile

PROFILE = Path("configs/profiles/xetra_v1.yaml")


def test_xetra_v1_loads_and_matches_every_pinned_contract_value() -> None:
    profile = load_profile(PROFILE)
    assert_xetra_v1_pins(profile)
    assert profile.profile_id == "xetra"
    assert profile.profile_config_version == 1
    assert profile.registered_model == "regime-xetra"
    assert profile.production_alias == "champion"
    assert profile.challenger_alias == "challenger"
    assert profile.feature_selection.policy_id == "xetra_semantic_medoid_v1"
    assert profile.feature_selection.static_features == ()
    assert profile.gaussian_hmm.candidate_states == (2, 3, 4)
    assert profile.gaussian_hmm.covariance_type == "full"
    assert profile.walk_forward.ranking_abs_tolerance == 1e-12
    assert len(profile.profile_hash) == 64


def test_xetra_profile_is_deterministic() -> None:
    first = load_profile(PROFILE)
    second = load_profile(PROFILE)
    assert first == second
    assert first.profile_hash == second.profile_hash


def test_xetra_pins_reject_mutated_profile() -> None:
    profile = load_profile(PROFILE)
    object.__setattr__(profile.walk_forward, "ranking_abs_tolerance", 1e-9)
    with pytest.raises(ValueError, match="pinned evaluation contract"):
        assert_xetra_v1_pins(profile)
