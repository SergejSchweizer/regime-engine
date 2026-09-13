from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from market_regime_engine.preprocessing import PCAArtifact, fit_pca_transformer


def _train() -> np.ndarray:
    rng = np.random.default_rng(303)
    first = rng.normal(size=256)
    second = 2.0 * first + 0.02 * rng.normal(size=256)
    third = rng.normal(size=256)
    return np.column_stack((first, second, third))


def test_pca_is_train_only_deterministic_and_selects_minimal_90_percent_prefix() -> None:
    train = _train()
    artifact = fit_pca_transformer(train, ("a", "b", "c"))
    repeated = fit_pca_transformer(train, ("a", "b", "c"))

    assert artifact == repeated
    assert artifact.retained_component_count < 3
    assert artifact.cumulative_explained_variance >= 0.90
    assert sum(artifact.explained_variance_ratio[: artifact.retained_component_count - 1]) < 0.90
    assert artifact.generated_feature_names == tuple(
        f"pca_pc_{index:03d}" for index in range(1, artifact.retained_component_count + 1)
    )
    for component in artifact.components:
        pivot = max(range(len(component)), key=lambda index: abs(component[index]))
        assert component[pivot] >= 0.0

    changed_future = np.vstack((train, np.full((100, 3), 10_000.0)))
    assert artifact == fit_pca_transformer(train, ("a", "b", "c"))
    assert artifact != fit_pca_transformer(changed_future, ("a", "b", "c"))


def test_pca_artifact_round_trip_preserves_exact_parameters() -> None:
    artifact = fit_pca_transformer(_train(), ("a", "b", "c"))
    restored = PCAArtifact.from_canonical_json(artifact.to_canonical_json())

    assert restored == artifact
    assert np.array_equal(restored.transform(_train()), artifact.transform(_train()))


def test_pca_rejects_nonfinite_input_threshold_and_wrong_dimension() -> None:
    train = _train()
    with pytest.raises(ValueError, match="finite"):
        fit_pca_transformer(np.array([[1.0, np.nan, 2.0]]), ("a", "b", "c"))
    with pytest.raises(ValueError, match="variance_threshold"):
        fit_pca_transformer(train, ("a", "b", "c"), variance_threshold=0.0)
    artifact = fit_pca_transformer(train, ("a", "b", "c"))
    with pytest.raises(ValueError, match="exact feature order"):
        artifact.transform(np.ones((2, 2)))


def test_pca_unknown_artifact_schema_fails_closed(tmp_path: Path) -> None:
    artifact = fit_pca_transformer(_train(), ("a", "b", "c"))
    payload = artifact.to_canonical_json().replace("RegimeEnginePCA.v1", "RegimeEnginePCA.v2")
    path = tmp_path / "pca.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported PCA artifact schema"):
        PCAArtifact.from_canonical_json(path.read_text(encoding="utf-8"))
