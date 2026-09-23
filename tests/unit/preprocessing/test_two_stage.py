from __future__ import annotations

import numpy as np

from market_regime_engine.feature_discovery.family_pca import fit_family_pca
from market_regime_engine.feature_discovery.feature_roles import build_feature_role_contract
from market_regime_engine.preprocessing.two_stage import (
    FamilyPCATwoStageScalerArtifact,
    fit_family_pca_hmm_scaler,
)


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
