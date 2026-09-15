from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation_runs.math_audit import (
    _likelihood_item,
    build_math_expectations,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.preprocessing.two_stage import fit_pca_hmm_scaler


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
    pca_scaler = fit_pca_hmm_scaler(
        timestamps,
        rows[["f0"]].to_numpy(),
        raw_feature_order=("f0",),
        inner_fold_id="fold_001",
        fit_start=timestamps[0],
        fit_end=timestamps[-1],
        model_feature_order=("f0",),
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
        scaler_artifact=pca_scaler.hmm_scaler,
        pca_scaler_artifact=pca_scaler,
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
    outer_fold = SimpleNamespace(
        fold_index=1,
        train_end=timestamps[1],
        valid=True,
        result_hash="a" * 64,
        oos_timestamps=timestamps[2:],
        oos_filtered_probabilities=((0.5, 0.5),) * 2,
        teacher_oos_timestamps=timestamps,
        teacher_oos_filtered_probabilities=((0.5, 0.5),) * 4,
        outer_teacher_final_soft_nmi=0.5,
        outer_shared_timestamp_count=2,
    )
    result = SimpleNamespace(outer_folds=(outer_fold,))

    expectations = build_math_expectations(
        rows,
        cast(Any, result),
        {1: cast(Any, selection)},
        prefix_evaluations={
            1: {(2, "gaussian_hmm_k2_full"): cast(Any, SimpleNamespace(valid_folds=(fold,)))}
        },
    )

    assert expectations["feature_order"] == ["f0", "f1"]
    assert expectations["final_feature_order"] == ["f0"]
    assert expectations["silhouette_clusters"]
    assert expectations["feature_scores"]
    likelihoods = cast(list[dict[str, object]], expectations["likelihoods"])
    assert len(likelihoods) == 2
    assert {item["scope"] for item in likelihoods} == {"TRAIN", "OOS"}

    prefix_nmi = cast(list[dict[str, object]], expectations["prefix_nmi"])
    assert len(prefix_nmi) == 1
    assert prefix_nmi[0]["candidate_id"] == "gaussian_hmm_k2_full"
    assert expectations["schema_version"] == 2
    assert expectations["audit_outer_fold_indices"] == [1]
    assert len(cast(list[dict[str, object]], expectations["fold_audits"])) == 1
    outer_agreements = cast(list[dict[str, object]], expectations["outer_fold_agreements"])
    assert outer_agreements[0]["outer_fold_index"] == 1
    assert outer_agreements[0]["shared_timestamp_count"] == 2

    five_fold_result = SimpleNamespace(
        outer_folds=tuple(
            SimpleNamespace(
                fold_index=index,
                train_end=timestamps[1],
                valid=True,
                result_hash=f"{index:064x}",
                oos_timestamps=timestamps[2:],
                oos_filtered_probabilities=((0.5, 0.5),) * 2,
                teacher_oos_timestamps=timestamps,
                teacher_oos_filtered_probabilities=((0.5, 0.5),) * 4,
                outer_teacher_final_soft_nmi=0.5,
                outer_shared_timestamp_count=2,
            )
            for index in range(1, 6)
        ),
    )
    five_fold_expectations = build_math_expectations(
        rows,
        cast(Any, five_fold_result),
        {index: cast(Any, selection) for index in range(1, 6)},
        prefix_evaluations={
            index: {(2, "gaussian_hmm_k2_full"): cast(Any, SimpleNamespace(valid_folds=(fold,)))}
            for index in range(1, 6)
        },
        max_workers=2,
    )
    assert five_fold_expectations["audit_outer_fold_indices"] == [1, 3, 5]
    assert [
        item["outer_fold_index"]
        for item in cast(list[dict[str, object]], five_fold_expectations["fold_audits"])
    ] == [1, 3, 5]


def test_math_audit_transforms_raw_rows_with_fold_local_pca_artifact() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=index) for index in range(120))
    rows = pd.DataFrame(
        {
            "timestamp_m1": timestamps,
            "f0": tuple(float(index) for index in range(120)),
            "f1": tuple(float((index % 7) - 3) for index in range(120)),
        }
    )
    pca_scaler = fit_pca_hmm_scaler(
        timestamps[:90],
        rows.loc[:89, ["f0", "f1"]].to_numpy(),
        raw_feature_order=("f0", "f1"),
        inner_fold_id="fold_001",
        fit_start=timestamps[0],
        fit_end=timestamps[89],
        variance_threshold=0.5,
        component_count=1,
        model_feature_order=("f0", "pca_pc_001"),
    )
    artifact = GaussianHMMArtifact(
        state_count=2,
        feature_order=("f0", "pca_pc_001"),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.8, 0.2), (0.2, 0.8)),
        means=((-1.0, -1.0), (1.0, 1.0)),
        full_covariances=(
            ((1.0, 0.0), (0.0, 1.0)),
            ((1.0, 0.0), (0.0, 1.0)),
        ),
    )
    fold = SimpleNamespace(
        fold_index=1,
        valid=True,
        model_artifact=artifact,
        scaler_artifact=pca_scaler.hmm_scaler,
        pca_scaler_artifact=pca_scaler,
        train_log_likelihood=-3.0,
        oos_predictive_log_likelihood=-2.0,
    )

    items = _likelihood_item(
        rows,
        ("f0", "pca_pc_001"),
        SimpleNamespace(train_source_observations=90, test_source_observations=30),
        fold,
    )

    assert all(item["feature_order"] == ["f0", "pca_pc_001"] for item in items)
    assert all(len(cast(list[list[float]], item["observations"])[0]) == 2 for item in items)
