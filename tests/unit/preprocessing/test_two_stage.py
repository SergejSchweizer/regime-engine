from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from market_regime_engine.feature_discovery.family_pca import fit_family_pca
from market_regime_engine.feature_discovery.feature_roles import build_feature_role_contract
from market_regime_engine.preprocessing import (
    PCATwoStageScalerArtifact,
    fit_pca_hmm_scaler,
)
from market_regime_engine.preprocessing.two_stage import (
    FamilyPCATwoStageScalerArtifact,
    fit_family_pca_hmm_scaler,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def _source(count: int = 120) -> tuple[tuple[datetime, ...], np.ndarray]:
    index = np.arange(count, dtype=np.float64)
    return (
        tuple(START + timedelta(days=int(value)) for value in index),
        np.column_stack(
            (
                np.sin(index / 5.0) + index / 100.0,
                np.cos(index / 7.0) - index / 120.0,
                np.sin(index / 11.0),
            )
        ),
    )


def test_two_stage_scaler_is_train_only_and_standardizes_augmented_train() -> None:
    timestamps, rows = _source()
    artifact = fit_pca_hmm_scaler(
        timestamps,
        rows,
        raw_feature_order=("a", "b", "c"),
        inner_fold_id="inner_fold_001",
        fit_start=START + timedelta(days=10),
        fit_end=START + timedelta(days=89),
    )
    future_rows = np.vstack((rows, np.full((10, 3), 1_000_000.0)))
    future_timestamps = timestamps + tuple(
        START + timedelta(days=index) for index in range(120, 130)
    )
    resumed = fit_pca_hmm_scaler(
        future_timestamps,
        future_rows,
        raw_feature_order=("a", "b", "c"),
        inner_fold_id="inner_fold_001",
        fit_start=START + timedelta(days=10),
        fit_end=START + timedelta(days=89),
    )

    assert artifact.fit_hash == resumed.fit_hash
    selected = rows[10:90]
    transformed = artifact.transform(selected)
    assert np.allclose(np.mean(transformed, axis=0), 0.0, atol=1.0e-12)
    assert np.allclose(np.var(transformed, axis=0), 1.0, atol=1.0e-12)
    assert artifact.model_feature_order == (
        "a",
        "b",
        "c",
        *artifact.pca_fit.artifact.generated_feature_names,
    )


def test_two_stage_scaler_round_trip_and_invalid_inputs_fail_closed() -> None:
    timestamps, rows = _source()
    artifact = fit_pca_hmm_scaler(
        timestamps,
        rows,
        raw_feature_order=("a", "b", "c"),
        inner_fold_id="inner_fold_001",
        fit_start=START,
        fit_end=START + timedelta(days=80),
    )
    restored = PCATwoStageScalerArtifact.from_canonical_json(artifact.to_canonical_json())
    assert restored == artifact
    assert np.array_equal(restored.transform(rows[:10]), artifact.transform(rows[:10]))
    with pytest.raises(ValueError, match="raw feature order"):
        artifact.transform(np.ones((2, 2)))


def test_two_stage_can_select_raw_and_generated_columns_in_candidate_order() -> None:
    timestamps, rows = _source()
    probe = fit_pca_hmm_scaler(
        timestamps,
        rows,
        raw_feature_order=("a", "b", "c"),
        inner_fold_id="inner_fold_001",
        fit_start=START,
        fit_end=START + timedelta(days=80),
    )
    component = probe.pca_fit.artifact.generated_feature_names[0]
    selected = fit_pca_hmm_scaler(
        timestamps,
        rows,
        raw_feature_order=("a", "b", "c"),
        inner_fold_id="inner_fold_001",
        fit_start=START,
        fit_end=START + timedelta(days=80),
        model_feature_order=("b", component),
    )

    assert selected.model_feature_order == ("b", component)
    assert selected.hmm_scaler.feature_order == ("b", component)
    assert selected.transform(rows[:5]).shape == (5, 2)
    assert PCATwoStageScalerArtifact.from_canonical_json(selected.to_canonical_json()) == selected


def test_family_pca_two_stage_round_trip_materializes_selected_family_pc() -> None:
    index = np.arange(120, dtype=np.float64)
    rows = np.column_stack(
        (
            np.sin(index / 5.0) + index / 100.0,
            np.cos(index / 7.0) - index / 120.0,
        )
    )
    feature_order = ("vix_delta_1d", "vix_delta_2d")
    contract = build_feature_role_contract(feature_order)
    family = fit_family_pca(rows, feature_order, contract)
    generated = family.generated_feature_names[0]
    artifact = fit_family_pca_hmm_scaler(
        rows,
        raw_feature_order=feature_order,
        family_pca=(family,),
        model_feature_order=("vix_delta_1d", generated),
    )

    restored = FamilyPCATwoStageScalerArtifact.from_canonical_json(artifact.to_canonical_json())
    assert restored == artifact
    transformed = restored.transform(rows[:10])
    assert transformed.shape == (10, 2)
    assert np.all(np.isfinite(transformed))
