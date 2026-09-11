from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).parents[2] / "scripts" / "verify_xetra_v4_math.py"
    spec = importlib.util.spec_from_file_location("verify_xetra_v4_math", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent math verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_feature_score_and_soft_nmi_are_recomputable() -> None:
    module = _module()
    timestamps = tuple(
        datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(10)
    )
    probabilities = tuple((1.0, 0.0) if index < 5 else (0.0, 1.0) for index in range(10))
    information_ratio, eta_squared = module.independent_feature_score(
        np.arange(10, dtype=float),
        np.asarray(timestamps, dtype=object),
        timestamps,
        probabilities,
    )

    assert information_ratio == pytest.approx(1.0)
    assert eta_squared == pytest.approx(0.7575757575757576)
    assert module.independent_soft_nmi(timestamps, probabilities, timestamps, probabilities) == 1.0


def test_independent_gaussian_likelihood_uses_forward_scaling() -> None:
    module = _module()
    actual = module.independent_gaussian_log_likelihood(
        np.asarray([[0.0], [1.0]], dtype=float),
        np.asarray([1.0]),
        np.asarray([[1.0]]),
        np.asarray([[0.0]]),
        np.asarray([[[1.0]]]),
    )

    expected = -0.5 * (2.0 * np.log(2.0 * np.pi) + 1.0)
    assert actual == pytest.approx(expected)
