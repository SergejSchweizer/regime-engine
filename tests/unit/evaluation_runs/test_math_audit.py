from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation_runs.math_audit import build_math_expectations
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact


def test_math_audit_serializes_independent_primitives_and_likelihoods() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=index) for index in range(4))
    rows = pd.DataFrame(
        {
            "timestamp_m1": timestamps,
            "f0": (0.0, 1.0, 0.5, 1.5),
            "f1": (1.0, 2.0, 1.5, 2.5),
        }
    )
    artifact = GaussianHMMArtifact(
        state_count=2,
        feature_order=("f0",),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.8, 0.2), (0.2, 0.8)),
        means=((-1.0,), (1.0,)),
        full_covariances=(((1.0,),), ((1.0,),)),
    )
    fold = SimpleNamespace(
        fold_index=1,
        valid=True,
        oos_timestamps=timestamps[2:],
        oos_filtered_probabilities=((0.5, 0.5),) * 2,
        model_artifact=artifact,
        scaler_artifact=StandardScalerArtifact(
            feature_order=("f0",),
            means=(0.5,),
            variances=(0.25,),
            scales=(0.5,),
        ),
        train_log_likelihood=-3.0,
        oos_predictive_log_likelihood=-2.0,
    )
    selection = SimpleNamespace(
        distance=SimpleNamespace(
            feature_order=("f0", "f1"),
            distances=((0.0, 0.25), (0.25, 0.0)),
        ),
        clusters=SimpleNamespace(
            candidate_memberships=((2, (("cluster_000", ("f0",)), ("cluster_001", ("f1",)))),),
            silhouette_curve=((2, 0.5),),
        ),
        teacher_reference=SimpleNamespace(
            timestamps=timestamps,
            filtered_probabilities=((0.5, 0.5),) * 4,
        ),
        prefix_search=SimpleNamespace(
            evaluations=(
                SimpleNamespace(
                    prefix_length=2,
                    candidate_id="gaussian_hmm_k2_full",
                    valid=True,
                    soft_regime_nmi=0.5,
                ),
            ),
        ),
        feature_scores=(
            SimpleNamespace(
                feature_name="f0",
                eligible=True,
                state_information_ratio=0.4,
                eta_squared=0.3,
            ),
        ),
        final_candidate=SimpleNamespace(candidate_id="gaussian_hmm_k2_full"),
        final_grid=SimpleNamespace(
            grid=SimpleNamespace(
                evaluations=(
                    SimpleNamespace(
                        candidate_id="gaussian_hmm_k2_full",
                        feature_order=("f0",),
                        folds=(fold,),
                    ),
                ),
            ),
        ),
        final_grid_plan=SimpleNamespace(
            folds=(SimpleNamespace(train_source_observations=2, test_source_observations=2),),
        ),
    )
    result = SimpleNamespace(
        outer_folds=(SimpleNamespace(fold_index=1, train_end=timestamps[1]),),
    )

    expectations = build_math_expectations(
        rows,
        cast(Any, result),
        {1: cast(Any, selection)},
        prefix_evaluations={
            1: {(2, "gaussian_hmm_k2_full"): cast(Any, SimpleNamespace(valid_folds=(fold,)))}
        },
    )

    assert expectations["feature_order"] == ["f0", "f1"]
    assert expectations["silhouette_clusters"]
    assert expectations["feature_scores"]
    likelihoods = cast(list[dict[str, object]], expectations["likelihoods"])
    assert len(likelihoods) == 2
    assert {item["scope"] for item in likelihoods} == {"TRAIN", "OOS"}

    prefix_nmi = cast(list[dict[str, object]], expectations["prefix_nmi"])
    assert len(prefix_nmi) == 1
    assert prefix_nmi[0]["candidate_id"] == "gaussian_hmm_k2_full"
