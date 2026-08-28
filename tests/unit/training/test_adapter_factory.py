from __future__ import annotations

from pathlib import Path

import pytest

from market_regime_engine.evaluations import CandidateSpec
from market_regime_engine.models.student_t_hmm import StudentTHMMAdapter
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.training.adapter_factory import adapter_factory


def test_factory_accepts_structural_student_t_candidate_and_copies_profile_settings() -> None:
    profile = load_profile(Path("configs/profiles/xetra_v3.yaml"))
    candidate = CandidateSpec("student_t_hmm_k2_full", "student_t_hmm", 2, 1, ("f0",))
    adapter = adapter_factory(profile, candidate)()
    assert isinstance(adapter, StudentTHMMAdapter)
    assert profile.student_t_hmm is not None
    assert adapter.settings.initial_nu == profile.student_t_hmm.initial_nu


def test_factory_rejects_invalid_structural_mixture_before_fit() -> None:
    profile = load_profile(Path("configs/profiles/xetra_v3.yaml"))
    candidate = CandidateSpec("gmm_hmm_k2_m2_full", "gmm_hmm", 2, 2, ("f0",))
    object.__setattr__(candidate, "mixture_count", 1)
    with pytest.raises(ValueError, match="family/mixture"):
        adapter_factory(profile, candidate)
