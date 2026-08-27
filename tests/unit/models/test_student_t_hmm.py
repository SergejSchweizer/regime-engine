from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from market_regime_engine.models.student_t_hmm import (
    StudentTHMMAdapter,
    StudentTHMMSettings,
)


def sample() -> np.ndarray:
    rng = np.random.default_rng(113)
    first = rng.standard_t(df=4.0, size=(180, 2)) + np.asarray((-2.0, 0.5))
    second = rng.standard_t(df=12.0, size=(180, 2)) + np.asarray((2.0, -0.5))
    return np.vstack((first, second))


def test_student_t_em_estimates_state_specific_nu_and_filters_causally() -> None:
    adapter = StudentTHMMAdapter(
        ("a", "b"),
        StudentTHMMSettings(n_iter=80, tol=1e-3),
    )
    result = adapter.fit(sample(), state_count=2, seed=11)
    assert result.artifact.model_family == "student_t_hmm"
    assert result.artifact.degrees_of_freedom is not None
    assert len(result.artifact.degrees_of_freedom) == 2
    assert all(2.0 < value <= 200.0 for value in result.artifact.degrees_of_freedom)
    assert np.isfinite(result.train_log_likelihood)
    filtered = adapter.causal_filter(sample()[:8])
    assert np.allclose(filtered.filtered_probabilities.sum(axis=1), 1.0)


def test_student_t_adapter_reconstructs_and_rejects_invalid_contracts() -> None:
    adapter = StudentTHMMAdapter(("a", "b"), StudentTHMMSettings(n_iter=5))
    result = adapter.fit(sample(), state_count=2, seed=23)
    restored = StudentTHMMAdapter(("a", "b"))
    restored.reconstruct(result.artifact)
    assert restored.extract() == result.artifact
    with pytest.raises(ValueError, match="Student-t artifact"):
        restored.reconstruct(
            replace(result.artifact, model_family="gaussian_hmm", degrees_of_freedom=None)
        )
    with pytest.raises(ValueError, match="K=2,3,4,5"):
        adapter.fit(sample(), state_count=6, seed=11)
