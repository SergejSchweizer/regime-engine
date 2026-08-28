from __future__ import annotations

from pathlib import Path

from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import expected_candidate_ids

PROFILE = Path("configs/profiles/xetra_v3.yaml")
V2 = Path("configs/profiles/xetra_v2.yaml")

EXPECTED_CANDIDATES = (
    "gaussian_hmm_k2_full",
    "gaussian_hmm_k3_full",
    "gaussian_hmm_k4_full",
    "gaussian_hmm_k5_full",
    "gmm_hmm_k2_m2_full",
    "gmm_hmm_k3_m2_full",
    "gmm_hmm_k4_m2_full",
    "gmm_hmm_k5_m2_full",
    "student_t_hmm_k2_full",
    "student_t_hmm_k3_full",
    "student_t_hmm_k4_full",
    "student_t_hmm_k5_full",
)


def test_xetra_v3_loads_with_versioned_policy_and_exact_candidate_universe() -> None:
    profile = load_profile(PROFILE)
    assert profile.profile_id == "xetra"
    assert profile.profile_config_version == 3
    assert profile.feature_selection.policy_id == "xetra_semantic_medoid_v3"
    assert profile.feature_selection.static_features == ()
    assert profile.gaussian_hmm.candidate_states == (2, 3, 4, 5)
    assert tuple((item.state_count, item.mixture_count) for item in profile.gmm_hmms) == (
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
    )
    assert profile.student_t_hmm is not None
    assert profile.student_t_hmm.candidate_states == (2, 3, 4, 5)
    assert expected_candidate_ids(3) == EXPECTED_CANDIDATES


def test_xetra_v3_evaluation_settings_equal_v2_except_versioned_policy_identity() -> None:
    v2 = load_profile(V2)
    v3 = load_profile(PROFILE)
    assert v3.walk_forward == v2.walk_forward
    assert v3.gaussian_hmm == v2.gaussian_hmm
    assert v3.gmm_hmms == v2.gmm_hmms
    assert v3.student_t_hmm == v2.student_t_hmm
    assert v3.gates == v2.gates
    assert v3.registered_model == v2.registered_model
    assert v3.production_alias == v2.production_alias
    assert v3.challenger_alias == v2.challenger_alias
    assert v3.profile_hash != v2.profile_hash
