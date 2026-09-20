from __future__ import annotations

from dataclasses import replace

import pytest

from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import (
    ResolvedCandidateProfile,
    expected_candidate_ids,
    resolve_global_feature_profile,
    validate_candidate_comparison_inputs,
)


def _resolved_inputs() -> dict[str, object]:
    return {
        "source_build_id": "build-1",
        "original_feature_universe": ("f0", "f1", "f2"),
        "final_features": ("f0", "f1"),
        "feature_discovery_definition_hash": "a" * 64,
        "feature_discovery_execution_hash": "b" * 64,
    }


def test_resolve_global_profile_builds_all_candidate_families() -> None:
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    resolved = resolve_global_feature_profile(profile, **_resolved_inputs())

    assert tuple(candidate.candidate_id for candidate in resolved.candidates) == (
        *expected_candidate_ids(),
    )
    assert resolved.final_features == ("f0", "f1")
    assert resolved.candidates[4].model_family == "gmm_hmm"
    assert resolved.candidates[8].model_family == "student_t_hmm"


def test_resolution_rejects_invalid_profile_and_feature_contract() -> None:
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    with pytest.raises(ValueError, match="only the Xetra public profile"):
        resolve_global_feature_profile(replace(profile, profile_id="other"), **_resolved_inputs())
    with pytest.raises(ValueError, match="non-empty and duplicate-free"):
        resolve_global_feature_profile(
            profile,
            **{**_resolved_inputs(), "original_feature_universe": ("f0", "f0")},
        )
    with pytest.raises(ValueError, match=r"belong to the (dynamic )?universe"):
        resolve_global_feature_profile(
            profile,
            **{**_resolved_inputs(), "final_features": ("missing",)},
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        resolve_global_feature_profile(
            profile,
            **{**_resolved_inputs(), "feature_discovery_definition_hash": "A" * 64},
        )


def test_candidate_contract_validates_order_and_shared_lineage() -> None:
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    resolved = resolve_global_feature_profile(profile, **_resolved_inputs())
    candidates = resolved.candidates
    validate_candidate_comparison_inputs(candidates)

    with pytest.raises(ValueError, match="IDs/order"):
        validate_candidate_comparison_inputs(tuple(reversed(candidates)))
    with pytest.raises(ValueError, match="exact v4 feature contract"):
        validate_candidate_comparison_inputs(
            (candidates[0], replace(candidates[1], source_build_id="other"), *candidates[2:])
        )
    with pytest.raises(ValueError, match="configuration version"):
        expected_candidate_ids(3)


def test_candidate_dataclass_rejects_identity_drift() -> None:
    values = {
        "candidate_id": "gaussian_hmm_k2_full",
        "state_count": 2,
        "covariance_type": "full",
        "feature_order": ("f0",),
        "feature_dimension": 1,
        "source_build_id": "build-1",
        "feature_selection_definition_hash": "a" * 64,
        "feature_selection_execution_hash": "b" * 64,
        "original_feature_universe": ("f0",),
    }
    ResolvedCandidateProfile(**values)
    for field, value, message in (
        ("candidate_id", "wrong", "candidate_id"),
        ("covariance_type", "diag", "covariance_type"),
        ("feature_dimension", 2, "feature_dimension"),
        ("source_build_id", " build-1", "source_build_id"),
    ):
        with pytest.raises(ValueError, match=message):
            ResolvedCandidateProfile(**{**values, field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_contract_version", 3, "contract version"),
        ("state_count", 6, "state count"),
        ("mixture_count", 2, "Gaussian HMM"),
        ("model_family", "unknown", "candidate_id"),
        ("original_feature_universe", ("f0", "f0"), "dynamic feature universe"),
        ("feature_order", ("missing",), "belong to the dynamic universe"),
        ("feature_selection_execution_hash", "A" * 64, "lowercase SHA-256"),
    ],
)
def test_candidate_dataclass_rejects_family_and_universe_contracts(
    field: str, value: object, message: str
) -> None:
    values = {
        "candidate_id": "gaussian_hmm_k2_full",
        "state_count": 2,
        "covariance_type": "full",
        "feature_order": ("f0",),
        "feature_dimension": 1,
        "source_build_id": "build-1",
        "feature_selection_definition_hash": "a" * 64,
        "feature_selection_execution_hash": "b" * 64,
        "original_feature_universe": ("f0",),
    }
    with pytest.raises(ValueError, match=message):
        ResolvedCandidateProfile(**{**values, field: value})
