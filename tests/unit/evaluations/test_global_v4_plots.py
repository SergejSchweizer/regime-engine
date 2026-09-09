from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from market_regime_engine.evaluations.plots import render_global_v4_diagnostics
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    FinalSelectedConfiguration,
    OuterFoldResult,
)

HASH = "a" * 64
START = datetime(2024, 1, 1, tzinfo=UTC)


def _configuration(index: int) -> FinalSelectedConfiguration:
    return FinalSelectedConfiguration(
        feature_order=("feature_a", "feature_b"),
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        model_family="gaussian_hmm",
        selected_prefix_length=2,
        feature_discovery_hash=HASH,
        source_build_id="build-1",
        catalog_hash=HASH,
        selection_definition_hash=HASH,
        selection_execution_hash=HASH,
    )


def _result() -> AdaptiveEvaluationResult:
    folds = []
    for index in range(1, 4):
        start = START + timedelta(days=(index - 1) * 10)
        folds.append(
            OuterFoldResult(
                fold_index=index,
                train_start=start,
                train_end=start + timedelta(days=4),
                test_start=start + timedelta(days=5),
                test_end=start + timedelta(days=9),
                final_configuration=_configuration(index),
                oos_predictive_loglik_per_observation=-1.0 - index / 10.0,
                oos_timestamps=(start + timedelta(days=5),),
                oos_filtered_probabilities=((0.5, 0.5),),
                teacher_reference_hash=HASH,
                outer_teacher_final_soft_nmi=0.5 + index / 10.0,
                outer_shared_timestamp_count=63,
            )
        )
    return AdaptiveEvaluationResult(
        source_build_id="build-1",
        catalog_hash=HASH,
        validation_evaluation_cutoff=folds[-1].test_end,
        outer_folds=tuple(folds),
        valid_fold_count=3,
        valid_fold_rate=1.0,
        soft_nmi_mean=0.7,
        soft_nmi_population_std=0.0816496580927726,
        soft_nmi_worst=0.6,
        latest_complete_fold_valid=True,
        production_eligible=True,
        policy_hash=HASH,
    )


def _selection(index: int) -> SimpleNamespace:
    aggregate_ids = (
        *(f"gaussian_hmm_k{count}_full" for count in (2, 3, 4, 5)),
        *(f"gmm_hmm_k{count}_m2_full" for count in (2, 3, 4, 5)),
        *(f"student_t_hmm_k{count}_full" for count in (2, 3, 4, 5)),
    )
    aggregates = tuple(
        SimpleNamespace(
            candidate_id=candidate_id,
            state_count=2 + position % 4,
            planned_fold_count=4,
            valid_fold_count=4,
            invalid_fold_count=0,
            valid_fold_rate=1.0,
            passes_valid_fold_rate_gate=True,
            oos_predictive_loglik_mean=-2.0 + position / 20.0,
            oos_predictive_loglik_std=0.1,
            oos_predictive_loglik_worst_fold=-2.2,
            oos_predictive_loglik_best_fold=-1.8,
            bic_mean=10.0 + position,
            aic_mean=9.0 + position,
        )
        for position, candidate_id in enumerate(aggregate_ids)
    )
    feature_scores = tuple(
        SimpleNamespace(
            feature_name=name,
            canonical_ordinal=position + 1,
            state_information_ratio=0.2 + position / 10.0,
            eta_squared=0.3 + position / 10.0,
            coverage=1.0,
            observation_count=1000,
            eligible=True,
            exclusion_reason=None,
        )
        for position, name in enumerate(("feature_a", "feature_b", "feature_c"))
    )
    return SimpleNamespace(
        source_build_id="build-1",
        catalog_hash=HASH,
        quality=SimpleNamespace(
            result_hash=HASH,
            train_source_observation_count=1260,
            eligible_features=("feature_a", "feature_b", "feature_c"),
            features=tuple(
                SimpleNamespace(
                    feature_name=name,
                    canonical_ordinal=position + 1,
                    finite_observation_count=1260,
                    coverage=1.0,
                    population_variance=1.0,
                    eligible=True,
                    rejection_reason=None,
                )
                for position, name in enumerate(("feature_a", "feature_b", "feature_c"))
            ),
        ),
        distance=SimpleNamespace(
            feature_order=("feature_a", "feature_b", "feature_c"),
            matrix_hash=HASH,
            minimum_pairwise_observations=504,
            distances=((0.0, 0.2, 0.3), (0.2, 0.0, 0.4), (0.3, 0.4, 0.0)),
            pairwise_support=((1000, 1000, 1000),) * 3,
            spearman_correlations=((1.0, 0.8, 0.7), (0.8, 1.0, 0.6), (0.7, 0.6, 1.0)),
        ),
        clusters=SimpleNamespace(
            solution_hash=HASH,
            candidate_count=3,
            selected_count=2,
            silhouette_curve=((2, 0.4),),
            selected_silhouette=0.4,
            singleton_count=1,
            memberships=(
                ("cluster_000", ("feature_a", "feature_b")),
                ("cluster_001", ("feature_c",)),
            ),
        ),
        prototypes=SimpleNamespace(
            temporary=True,
            cluster_ids=("cluster_000", "cluster_001"),
            prototypes=("feature_a", "feature_c"),
            mean_distances=(("feature_a", 0.1), ("feature_c", 0.0)),
        ),
        teacher_evaluation=SimpleNamespace(
            provisional_candidate_id="gaussian_hmm_k2_full",
            provisional_state_count=2,
            no_selection_reason=None,
        ),
        teacher_reference=SimpleNamespace(
            candidate_id="gaussian_hmm_k2_full",
            state_count=2,
            reference_hash=HASH,
            prototype_features=("feature_a", "feature_c"),
            valid_inner_fold_ids=("fold_001",),
            inner_plan_hash=HASH,
        ),
        feature_scores=feature_scores,
        winner_selection=SimpleNamespace(
            ranked_features=("feature_b", "feature_c"),
            winners=(
                SimpleNamespace(
                    cluster_id="cluster_000",
                    winner_feature="feature_b",
                    member_features=("feature_a", "feature_b"),
                ),
                SimpleNamespace(
                    cluster_id="cluster_001",
                    winner_feature="feature_c",
                    member_features=("feature_c",),
                ),
            ),
        ),
        prefix_search=SimpleNamespace(
            ranked_features=("feature_b", "feature_c"),
            selected_prefix_length=2,
            selected_candidate_id="gaussian_hmm_k2_full",
            evaluations=(
                SimpleNamespace(
                    prefix_length=2,
                    feature_order=("feature_b", "feature_c"),
                    candidate_id="gaussian_hmm_k2_full",
                    shared_timestamp_count=126,
                    shared_teacher_coverage=1.0,
                    soft_regime_nmi=0.75,
                    valid=True,
                    invalid_reason=None,
                ),
            ),
        ),
        final_grid=SimpleNamespace(
            candidate_grid=SimpleNamespace(
                feature_order=("feature_a", "feature_b"),
                evaluation_plan_hash=HASH,
                aggregates=aggregates,
            ),
            selection=SimpleNamespace(
                champion_candidate_id="gaussian_hmm_k2_full",
                champion_state_count=2,
                ranked_candidate_ids=aggregate_ids,
            ),
        ),
        final_candidate=SimpleNamespace(
            candidate_id="gaussian_hmm_k2_full",
            feature_order=("feature_a", "feature_b"),
        ),
        feature_discovery_hash=HASH,
    )


def test_global_v4_diagnostics_are_png_only_and_source_hash_deterministic(tmp_path) -> None:
    result = _result()
    selections = {fold.fold_index: _selection(fold.fold_index) for fold in result.outer_folds}

    first = render_global_v4_diagnostics(result, selections, tmp_path / "first")
    second = render_global_v4_diagnostics(result, selections, tmp_path / "second")

    expected = {
        "quality_eligibility",
        "silhouette_curve",
        "cluster_size",
        "state_information_rank",
        "prefix_soft_nmi_history",
        "final_12_model_same_vector_comparison",
        "outer_soft_nmi_history",
        "selected_m_l_history",
        "feature_selection_frequency",
        "adjacent_fold_cluster_stability",
    }
    assert expected <= {entry.plot_type for entry in first}
    assert all(entry.png_path.endswith(".png") for entry in first)
    assert all(
        (tmp_path / "first" / "plots" / entry.png_path.split("/plots/", 1)[1]).is_file()
        for entry in first
    )
    assert tuple(entry.source_artifact_hash for entry in first) == tuple(
        entry.source_artifact_hash for entry in second
    )
    assert not tuple(tmp_path.rglob("*.svg"))
    assert all(
        "oos_predictive_loglik" not in entry.source_metric_keys
        for entry in first
        if entry.plot_type != "final_12_model_same_vector_comparison"
    )
