from __future__ import annotations

import numpy as np
import pytest

from market_regime_engine.preprocessing import StandardScalerArtifact, fit_standard_scaler


def test_scaler_fits_only_supplied_train_rows_and_round_trips() -> None:
    train = np.array([[1.0, 10.0], [2.0, 12.0], [4.0, 18.0]], dtype=float)
    scaler = fit_standard_scaler(train, ("a", "b"))
    transformed = scaler.transform(train)
    assert np.allclose(np.mean(transformed, axis=0), 0.0)
    assert np.allclose(np.var(transformed, axis=0, ddof=0), 1.0)
    restored = StandardScalerArtifact.from_canonical_json(scaler.to_canonical_json())
    assert restored == scaler


def test_future_rows_cannot_change_fitted_parameters() -> None:
    train = np.array([[1.0], [2.0], [5.0]], dtype=float)
    first = fit_standard_scaler(train, ("x",))
    _ = first.transform(np.array([[1000.0], [-1000.0]], dtype=float))
    second = fit_standard_scaler(train, ("x",))
    assert first == second


def test_zero_or_tiny_variance_fails_closed() -> None:
    with pytest.raises(ValueError, match="greater than 1e-12"):
        fit_standard_scaler(np.array([[1.0], [1.0], [1.0]]), ("x",))
    tiny = np.array([[1.0], [1.0 + 1e-7], [1.0 - 1e-7]])
    with pytest.raises(ValueError, match="greater than 1e-12"):
        fit_standard_scaler(tiny, ("x",))


def test_nonfinite_and_wrong_order_dimension_fail() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_standard_scaler(np.array([[1.0, np.nan], [2.0, 3.0]]), ("a", "b"))
    scaler = fit_standard_scaler(np.array([[1.0], [2.0], [3.0]]), ("x",))
    with pytest.raises(ValueError, match="exact feature order"):
        scaler.transform(np.array([[1.0, 2.0]]))
