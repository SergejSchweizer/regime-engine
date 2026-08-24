from __future__ import annotations

import platform

import numpy as np
from hmmlearn.hmm import GaussianHMM


def test_exact_python_runtime() -> None:
    assert platform.python_version() == "3.14.7"


def test_hmmlearn_full_covariance_k2_smoke() -> None:
    rng = np.random.default_rng(20260824)
    first = rng.multivariate_normal(
        mean=np.array([-1.0, 0.5]),
        cov=np.array([[0.25, 0.08], [0.08, 0.36]]),
        size=160,
    )
    second = rng.multivariate_normal(
        mean=np.array([1.2, -0.7]),
        cov=np.array([[0.30, -0.06], [-0.06, 0.20]]),
        size=160,
    )
    observations = np.vstack([first, second])

    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        implementation="log",
        n_iter=250,
        tol=1e-4,
        min_covar=1e-6,
        random_state=11,
    )
    model.fit(observations)

    assert model.covariance_type == "full"
    assert model.covars_.shape == (2, 2, 2)
    assert np.isfinite(model.score(observations))
    assert np.all(np.isfinite(model.covars_))
    assert any(abs(cov[0, 1]) > 1e-8 for cov in model.covars_)
