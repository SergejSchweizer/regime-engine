from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from market_regime_engine.feature_discovery.contracts import (
    FINAL_CANDIDATE_IDS,
    MIN_ELIGIBLE_FEATURES,
    AdaptiveEvaluationResult,
    CandidateEvaluation,
    ClusterSolution,
    ClusterWinner,
    DeploymentSelection,
    DiscoveryStatus,
    DistanceMatrixResult,
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureQuality,
    FeatureRegimeScore,
    FinalSelectedConfiguration,
    ModelClockFold,
    ModelClockPreflight,
    OuterFoldResult,
    PrefixEvaluation,
    PrefixSearchResult,
    PrototypeSet,
    ProvisionalTeacherReference,
    QualityFilterResult,
    RankedWinnerSet,
    SameFeatureEvaluationSet,
    canonical_json,
    content_hash,
    final_prefix_upper_bound,
    gaussian_parameter_count,
    gmm_parameter_count,
)


def q(name: str, ordinal: int, eligible: bool = True) -> FeatureQuality:
    return FeatureQuality(
        name,
        ordinal,
        10,
        10 if eligible else 8,
        1.0 if eligible else 0.8,
        1.0,
        eligible,
        None if eligible else "coverage_below_threshold",
    )


def score(name: str = "f0", ordinal: int = 1) -> FeatureRegimeScore:
    joint = tuple((0.09, 0.01) if index % 2 == 0 else (0.01, 0.09) for index in range(10))
    return FeatureRegimeScore(
        feature_name=name,
        canonical_ordinal=ordinal,
        coverage=1.0,
        observation_count=126,
        bin_counts=(13, 13, 13, 13, 13, 13, 12, 12, 12, 12),
        joint_bin_state_masses=joint,
        bin_masses=(0.1,) * 10,
        state_masses=(0.5, 0.5),
        mutual_information=0.1,
        state_entropy=0.6931471805599453,
        state_information_ratio=0.5,
        state_weights=(0.5, 0.5),
        state_means=(1.0, -1.0),
        state_variances=(0.5, 0.5),
        between_variance=1.0,
        within_variance=0.5,
        eta_squared=2 / 3,
        eligible=True,
    )


def candidate(
    candidate_id: str = FINAL_CANDIDATE_IDS[0], feature_order: tuple[str, ...] = ("f0", "f1")
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id,
        feature_order,
        "build",
        "b" * 64,
        3,
        3,
        -1.0,
        0.2,
        -2.0,
        10.0,
        9.0,
    )


def final_config(scope: str = "outer_fold_local") -> FinalSelectedConfiguration:
    return FinalSelectedConfiguration(
        ("f0", "f1"),
        FINAL_CANDIDATE_IDS[0],
        2,
        "gaussian_hmm",
        2,
        "c" * 64,
        "build",
        "a" * 64,
        "d" * 64,
        "e" * 64,
        scope,
    )


def outer_fold(valid: bool = True) -> OuterFoldResult:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return OuterFoldResult(
        1,
        start,
        start + timedelta(days=1),
        start + timedelta(days=2),
        start + timedelta(days=3),
        final_config(),
        -1.0,
        (start + timedelta(days=2),) if valid else (),
        ((0.25, 0.75),) if valid else (),
        "outer_fold_local",
        "f" * 64,
        0.5 if valid else None,
        1 if valid else 0,
        valid,
        None if valid else "selection_failed",
    )


def test_parameter_count_bounds_are_independently_reproducible() -> None:
    assert gaussian_parameter_count(5, 12) == 474
    assert gaussian_parameter_count(5, 13) == 544
    assert gmm_parameter_count(5, 2, 8) == 469
    assert gmm_parameter_count(5, 2, 9) == 569
    assert final_prefix_upper_bound(12) == 8
    with pytest.raises(ValueError):
        gaussian_parameter_count(0, 2)
    with pytest.raises(ValueError):
        gmm_parameter_count(2, 0, 2)
    with pytest.raises(ValueError):
        final_prefix_upper_bound(1)


def test_canonical_json_and_hash_are_order_independent_and_finite_only() -> None:
    left = {"z": (2, 1), "a": {"right": 2, "left": 1}}
    right = {"a": {"left": 1, "right": 2}, "z": (2, 1)}
    assert canonical_json(left) == canonical_json(right)
    assert content_hash(left) == content_hash(right)
    with pytest.raises(ValueError):
        canonical_json({"metric": float("nan")})
    with pytest.raises(ValueError, match="statistical decision"):
        canonical_json({"semantic_group": "rates"})
    for field in ("economic_label", "portfolio_target", "raw_rows", "password", "dsn", "secret"):
        with pytest.raises(ValueError):
            canonical_json({field: "forbidden"})
    with pytest.raises(ValueError, match="mapping keys"):
        canonical_json({1: "not canonical"})
    with pytest.raises(ValueError, match="unsupported"):
        canonical_json({"value": {"not-json"}})
    with pytest.raises(ValueError, match="timezone"):
        canonical_json({"timestamp": datetime(2026, 1, 1)})


def test_catalog_quality_clock_and_distance_contracts_are_complete() -> None:
    catalog = FeatureCatalogSnapshot(
        "build",
        "schema-v4",
        "features-v4",
        "timestamp_m1",
        tuple(FeatureCatalogEntry(f"f{index}", index + 1) for index in range(3)),
    )
    assert len(catalog.catalog_hash) == 64
    with pytest.raises(ValueError):
        FeatureCatalogEntry("Bad-Name", 1)
    with pytest.raises(ValueError):
        FeatureCatalogEntry("f0", 1, "NUMERIC")
    with pytest.raises(ValueError):
        FeatureCatalogSnapshot("build", "schema", "version", "timestamp", catalog.features)
    with pytest.raises(ValueError):
        FeatureCatalogSnapshot(
            "build",
            "schema",
            "version",
            "timestamp_m1",
            (FeatureCatalogEntry("f1", 2), FeatureCatalogEntry("f0", 1)),
        )

    fold = ModelClockFold(
        "inner-1",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
        datetime(2026, 1, 4, tzinfo=UTC),
        504,
        42,
        True,
    )
    preflight = ModelClockPreflight(
        ("f0", "f1"),
        "b" * 64,
        504,
        (("f0", 1.0), ("f1", 2.0)),
        (fold,),
        1.0,
        DiscoveryStatus.VALID,
    )
    assert preflight.preflight_hash
    with pytest.raises(ValueError):
        replace(fold, structurally_valid=False)
    with pytest.raises(ValueError):
        replace(preflight, structural_valid_fold_rate=0.0)

    distance = DistanceMatrixResult(
        ("f0", "f1", "f2"),
        ((0.0, 0.1, 1.0), (0.1, 0.0, 0.9), (1.0, 0.9, 0.0)),
        ((504, 504, 504), (504, 504, 504), (504, 504, 504)),
        spearman_correlations=((1.0, 0.9, -1.0), (0.9, 1.0, 0.8), (-1.0, 0.8, 1.0)),
    )
    assert distance.matrix_hash
    invalid_distances = (
        ((0.0,),),
        ((0.0, 1.1, 1.0), (1.1, 0.0, 0.9), (1.0, 0.9, 0.0)),
    )
    for distances in invalid_distances:
        with pytest.raises(ValueError):
            DistanceMatrixResult(
                ("f0", "f1", "f2"),
                distances,
                ((504, 504, 504), (504, 504, 504), (504, 504, 504)),
            )
    with pytest.raises(ValueError):
        DistanceMatrixResult(
            ("f0", "f1", "f2"),
            distance.distances,
            ((503, 504, 504), (504, 504, 504), (504, 504, 504)),
        )
    with pytest.raises(ValueError):
        DistanceMatrixResult(
            ("f0", "f1", "f2"),
            distance.distances,
            distance.pairwise_support,
            spearman_correlations=(
                (1.0, 0.9, -1.0),
                (0.9, 1.0, 0.8),
                (-1.0, 0.7, 1.0),
            ),
        )


def test_discovery_selection_contracts_cover_full_synthetic_chain() -> None:
    quality_result = QualityFilterResult(
        "build",
        10,
        "a" * 64,
        (q("f0", 1), q("f1", 2), q("f2", 3)),
        ("f0", "f1", "f2"),
    )
    cluster = ClusterSolution(
        3,
        2,
        ((2, 0.25),),
        (("cluster_000", ("f0", "f1")), ("cluster_001", ("f2",))),
        0.25,
        1,
        feature_ordinals=(("f0", 1), ("f1", 2), ("f2", 3)),
    )
    prototypes = PrototypeSet(
        ("cluster_000", "cluster_001"),
        ("f0", "f2"),
        (("f0", 0.05), ("f1", 0.05), ("f2", 0.0)),
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    teacher = ProvisionalTeacherReference(
        "gaussian_hmm_k2_full",
        2,
        (start, start + timedelta(days=1)),
        ((0.8, 0.2), (0.3, 0.7)),
        (0, 1),
        ("inner-1",),
        "build",
        "b" * 64,
        ("f0", "f2"),
    )
    score_0 = score()
    score_2 = score("f2", 3)
    winners = RankedWinnerSet(
        (
            ClusterWinner("cluster_000", ("f0", "f1"), "f0", score_0),
            ClusterWinner("cluster_001", ("f2",), "f2", score_2),
        ),
        ("f0", "f2"),
    )
    same_features = SameFeatureEvaluationSet(
        ("f0", "f2"),
        "build",
        "b" * 64,
        (
            candidate("gaussian_hmm_k2_full", ("f0", "f2")),
            candidate("gaussian_hmm_k3_full", ("f0", "f2")),
        ),
    )
    prefix = PrefixSearchResult(
        ("f0", "f2"),
        (
            PrefixEvaluation(
                2,
                ("f0", "f2"),
                "gaussian_hmm_k2_full",
                126,
                1.0,
                0.8,
                same_features.candidates,
            ),
        ),
        2,
        "gaussian_hmm_k2_full",
    )
    fold = outer_fold()
    adaptive = AdaptiveEvaluationResult(
        "build",
        "a" * 64,
        start + timedelta(days=3),
        (fold,),
        1,
        1.0,
        0.5,
        0.0,
        0.5,
        True,
        False,
        "f" * 64,
    )
    deployment = DeploymentSelection(
        "build",
        "a" * 64,
        start + timedelta(days=3),
        start + timedelta(days=4),
        final_config("model_version_local"),
        "c" * 64,
    )
    evidence = (
        quality_result,
        cluster,
        prototypes,
        teacher,
        score_0,
        winners,
        prefix,
        fold,
        adaptive,
        deployment,
    )
    assert all(canonical_json(value) for value in evidence)
    assert all(content_hash(value) == content_hash(value) for value in evidence)
    assert len(quality_result.eligible_features) == MIN_ELIGIBLE_FEATURES


def test_contract_mutations_fail_closed() -> None:
    teacher = ProvisionalTeacherReference(
        "gaussian_hmm_k2_full",
        2,
        (datetime(2026, 1, 1, tzinfo=UTC),),
        ((0.5, 0.5),),
        (0,),
        ("inner-1",),
        "build",
        "b" * 64,
    )
    with pytest.raises(ValueError):
        replace(teacher, filtered_probabilities=((0.5, 0.4),))
    with pytest.raises(ValueError):
        replace(teacher, dominant_states=(2,))
    with pytest.raises(ValueError):
        replace(teacher, timestamps=(datetime(2026, 1, 1),))

    with pytest.raises(ValueError):
        ClusterSolution(
            3, 2, ((2, 0.2),), (("cluster_000", ("f0",)), ("cluster_001", ("f1",))), 0.2, 1
        )
    with pytest.raises(ValueError):
        PrototypeSet(("cluster_001",), ("f0",), (("f0", 0.0),))
    with pytest.raises(ValueError):
        replace(score(), state_information_ratio=None)
    with pytest.raises(ValueError):
        SameFeatureEvaluationSet(
            ("f0", "f1"),
            "build",
            "b" * 64,
            (
                candidate(),
                replace(candidate(), feature_order=("f0",)),
            ),
        )
    with pytest.raises(ValueError):
        PrefixSearchResult(
            ("f0", "f2", "f3"),
            (PrefixEvaluation(2, ("f0", "f2"), "gaussian_hmm_k2_full", 126, 1.0, 0.5),),
            2,
            "gaussian_hmm_k2_full",
        )
    with pytest.raises(ValueError):
        replace(outer_fold(), state_identity="global")
    with pytest.raises(ValueError):
        replace(
            AdaptiveEvaluationResult(
                "build",
                "a" * 64,
                datetime(2026, 1, 4, tzinfo=UTC),
                (outer_fold(),),
                1,
                1.0,
                0.5,
                0.0,
                0.5,
                True,
                False,
                "f" * 64,
            ),
            production_eligible=True,
        )


def test_all_contract_boundary_errors_are_explicit() -> None:
    valid_quality = q("f0", 1)
    reject(
        lambda: replace(valid_quality, feature_name=""),
        lambda: replace(valid_quality, source_position=-1),
        lambda: replace(
            valid_quality, train_observation_count=0, finite_observation_count=0, coverage=0.0
        ),
        lambda: replace(valid_quality, finite_observation_count=11, coverage=1.1),
        lambda: replace(valid_quality, coverage=0.9),
        lambda: replace(valid_quality, population_variance=-1.0),
        lambda: replace(valid_quality, eligible=False, rejection_reason=None),
        lambda: replace(valid_quality, eligible=True, rejection_reason="reason"),
    )


def reject(*factories: Callable[[], object]) -> None:
    for factory in factories:
        with pytest.raises(ValueError):
            factory()


def test_score_and_ranking_contract_boundaries_are_explicit() -> None:
    valid = score()
    reject(
        lambda: replace(valid, bin_counts=(1,)),
        lambda: replace(valid, bin_counts=(-1, *valid.bin_counts[1:])),
        lambda: replace(valid, bin_counts=(13,) * 10),
        lambda: replace(valid, joint_bin_state_masses=((0.5, 0.5),)),
        lambda: replace(valid, state_masses=()),
        lambda: replace(valid, bin_masses=(0.1,) * 9),
        lambda: replace(valid, joint_bin_state_masses=tuple((0.1, 0.1) for _ in range(10))),
        lambda: replace(valid, bin_masses=(-0.1,) + (0.1,) * 9),
        lambda: replace(valid, state_masses=(-0.1, 1.1)),
        lambda: replace(
            valid,
            joint_bin_state_masses=((float("nan"), 0.01), *valid.joint_bin_state_masses[1:]),
        ),
        lambda: replace(valid, bin_masses=(0.2,) * 10),
        lambda: replace(valid, state_masses=(0.4, 0.4)),
        lambda: replace(valid, mutual_information=-1.0),
        lambda: replace(valid, state_entropy=-1.0),
        lambda: replace(valid, state_information_ratio=None),
        lambda: replace(valid, state_weights=(0.5,)),
        lambda: replace(valid, state_weights=(0.4, 0.6)),
        lambda: replace(valid, state_weights=(float("nan"), 0.5)),
        lambda: replace(valid, state_means=(float("nan"), -1.0)),
        lambda: replace(valid, state_variances=(-1.0, 0.5)),
        lambda: replace(valid, coverage=float("nan")),
        lambda: replace(valid, coverage=1.1),
        lambda: replace(valid, eta_squared=1.1),
        lambda: replace(valid, observation_count=125),
        lambda: replace(valid, eligible=False, exclusion_reason=None),
        lambda: replace(valid, eligible=True, exclusion_reason="reason"),
    )
    reject(
        lambda: ClusterWinner("cluster_000", (), "f0", valid),
        lambda: ClusterWinner("bad", ("f0",), "f0", valid),
        lambda: ClusterWinner("cluster_000", ("f0",), "f1", valid),
        lambda: ClusterWinner("cluster_000", ("f0",), "f0", replace(valid, feature_name="f1")),
        lambda: RankedWinnerSet((), ()),
        lambda: RankedWinnerSet((ClusterWinner("cluster_000", ("f0",), "f0", valid),), ("f1",)),
        lambda: RankedWinnerSet(
            (
                ClusterWinner("cluster_000", ("f0",), "f0", valid),
                ClusterWinner("cluster_000", ("f2",), "f2", score("f2", 3)),
            ),
            ("f0", "f2"),
        ),
    )


def test_candidate_prefix_outer_and_deployment_boundaries_are_explicit() -> None:
    valid_candidate = candidate()
    reject(
        lambda: replace(valid_candidate, candidate_id="unknown"),
        lambda: replace(valid_candidate, feature_order=()),
        lambda: replace(valid_candidate, source_build_id=""),
        lambda: replace(valid_candidate, plan_hash="x"),
        lambda: replace(valid_candidate, valid_fold_count=-1),
        lambda: replace(valid_candidate, total_fold_count=0),
        lambda: replace(valid_candidate, valid_fold_count=4),
        lambda: replace(valid_candidate, oos_predictive_loglik_mean=float("nan")),
        lambda: replace(valid_candidate, valid=False),
        lambda: replace(valid_candidate, oos_predictive_loglik_mean=None),
    )
    valid_set = SameFeatureEvaluationSet(("f0", "f1"), "build", "b" * 64, (valid_candidate,))
    assert valid_set.raw_likelihood_comparison_permitted
    reject(
        lambda: SameFeatureEvaluationSet((), "build", "b" * 64, (valid_candidate,)),
        lambda: SameFeatureEvaluationSet(("f0", "f1"), "", "b" * 64, (valid_candidate,)),
        lambda: SameFeatureEvaluationSet(("f0", "f1"), "build", "x", (valid_candidate,)),
        lambda: SameFeatureEvaluationSet(("f0", "f1"), "build", "b" * 64, ()),
        lambda: SameFeatureEvaluationSet(
            ("f0", "f1"), "build", "b" * 64, (valid_candidate, valid_candidate)
        ),
        lambda: SameFeatureEvaluationSet(
            ("f0", "f1"), "build", "b" * 64, (replace(valid_candidate, feature_order=("f0",)),)
        ),
    )
    valid_prefix = PrefixEvaluation(2, ("f0", "f1"), "gaussian_hmm_k2_full", 100, 1.0, 0.5)
    reject(
        lambda: replace(valid_prefix, prefix_length=1, feature_order=("f0",)),
        lambda: replace(valid_prefix, prefix_length=9, feature_order=("f0",) * 9),
        lambda: replace(valid_prefix, candidate_id="student_t_hmm_k2_full"),
        lambda: replace(valid_prefix, shared_timestamp_count=-1),
        lambda: replace(valid_prefix, shared_teacher_coverage=1.1),
        lambda: replace(valid_prefix, soft_regime_nmi=float("nan")),
        lambda: replace(
            valid_prefix,
            candidate_evaluations=(replace(valid_candidate, feature_order=("f0",)),),
        ),
    )
    valid_prefix_3 = PrefixEvaluation(3, ("f0", "f1", "f2"), "gaussian_hmm_k2_full", 100, 1.0, 0.5)
    valid_result = PrefixSearchResult(
        ("f0", "f1", "f2"), (valid_prefix, valid_prefix_3), 2, "gaussian_hmm_k2_full"
    )
    reject(
        lambda: PrefixSearchResult((), (), 2, "gaussian_hmm_k2_full"),
        lambda: PrefixSearchResult(("f0", "f1"), (valid_prefix,), 3, "gaussian_hmm_k2_full"),
        lambda: replace(valid_result, selected_candidate_id="gaussian_hmm_k3_full"),
    )
    reject(
        lambda: replace(final_config(), feature_order=("f0",)),
        lambda: replace(final_config(), selected_prefix_length=9, feature_order=("f0",) * 9),
        lambda: replace(final_config(), candidate_id="unknown"),
        lambda: replace(final_config(), state_count=3),
        lambda: replace(final_config(), model_family="student_t_hmm"),
        lambda: replace(final_config(), source_build_id=""),
        lambda: replace(final_config(), catalog_hash="x"),
        lambda: replace(final_config(), state_identity_scope="global"),
    )
    valid_outer = outer_fold()
    reject(
        lambda: replace(valid_outer, fold_index=0),
        lambda: replace(valid_outer, train_start=datetime(2026, 1, 1)),
        lambda: replace(valid_outer, test_start=valid_outer.train_end),
        lambda: replace(valid_outer, state_identity="global"),
        lambda: replace(valid_outer, final_configuration=final_config("model_version_local")),
        lambda: replace(valid_outer, oos_predictive_loglik_per_observation=float("nan")),
        lambda: replace(valid_outer, oos_timestamps=()),
        lambda: replace(
            valid_outer,
            oos_timestamps=(valid_outer.test_start, valid_outer.test_end),
            oos_filtered_probabilities=((0.5, 0.5),),
        ),
        lambda: replace(
            valid_outer,
            oos_timestamps=(valid_outer.test_start, valid_outer.test_start),
            oos_filtered_probabilities=((0.5, 0.5), (0.5, 0.5)),
        ),
        lambda: replace(valid_outer, oos_filtered_probabilities=((1.0,),)),
        lambda: replace(valid_outer, oos_filtered_probabilities=((float("nan"), 1.0),)),
        lambda: replace(valid_outer, teacher_reference_hash="x"),
        lambda: replace(valid_outer, outer_shared_timestamp_count=-1),
        lambda: replace(valid_outer, outer_teacher_final_soft_nmi=1.1),
        lambda: replace(valid_outer, valid=False),
    )
    assert valid_outer.result_hash
    valid_adaptive = AdaptiveEvaluationResult(
        "build",
        "a" * 64,
        valid_outer.test_end,
        (valid_outer,),
        1,
        1.0,
        0.5,
        0.0,
        0.5,
        True,
        False,
        "f" * 64,
    )
    reject(
        lambda: replace(valid_adaptive, source_build_id=""),
        lambda: replace(valid_adaptive, catalog_hash="x"),
        lambda: replace(valid_adaptive, validation_evaluation_cutoff=datetime(2026, 1, 1)),
        lambda: replace(valid_adaptive, outer_folds=()),
        lambda: replace(valid_adaptive, valid_fold_count=0),
        lambda: replace(valid_adaptive, valid_fold_rate=0.0),
        lambda: replace(valid_adaptive, soft_nmi_mean=1.1),
        lambda: replace(valid_adaptive, production_eligible=True),
        lambda: replace(valid_adaptive, policy_hash="x"),
        lambda: replace(valid_adaptive, failure_reason="bad", production_eligible=True),
    )
    deployment = DeploymentSelection(
        "build",
        "a" * 64,
        valid_outer.test_end,
        valid_outer.test_end + timedelta(days=1),
        final_config("model_version_local"),
        "c" * 64,
    )
    reject(
        lambda: replace(deployment, source_build_id=""),
        lambda: replace(deployment, source_catalog_hash="x"),
        lambda: replace(deployment, validation_evaluation_cutoff=datetime(2026, 1, 1)),
        lambda: replace(
            deployment, deployment_selection_cutoff=deployment.validation_evaluation_cutoff
        ),
        lambda: replace(deployment, configuration=final_config()),
        lambda: replace(
            deployment,
            configuration=replace(final_config("model_version_local"), source_build_id="other"),
        ),
        lambda: replace(deployment, discovery_hash="x"),
    )


def test_remaining_contract_boundaries_fail_closed() -> None:
    catalog_features = tuple(FeatureCatalogEntry(f"f{index}", index + 1) for index in range(3))
    reject(
        lambda: FeatureCatalogEntry("f0", 0),
        lambda: FeatureCatalogSnapshot("", "schema", "version", "timestamp_m1", catalog_features),
        lambda: FeatureCatalogSnapshot("build", "", "version", "timestamp_m1", catalog_features),
        lambda: FeatureCatalogSnapshot("build", "schema", "", "timestamp_m1", catalog_features),
        lambda: FeatureCatalogSnapshot("build", "schema", "version", "timestamp_m1", ()),
        lambda: FeatureCatalogSnapshot(
            "build", "schema", "version", "timestamp_m1", (catalog_features[0], catalog_features[0])
        ),
        lambda: FeatureCatalogSnapshot(
            "build",
            "schema",
            "version",
            "timestamp_m1",
            (catalog_features[0], FeatureCatalogEntry("f0", 2), catalog_features[2]),
        ),
    )

    quality = q("f0", 1)
    quality_features = (quality, q("f1", 2), q("f2", 3))
    quality_result = QualityFilterResult(
        "build", 10, "a" * 64, quality_features, ("f0", "f1", "f2")
    )
    assert quality.canonical_ordinal == 1
    assert quality_result.result_hash
    reject(
        lambda: replace(quality, source_position=-1),
        lambda: replace(
            quality, train_observation_count=0, finite_observation_count=0, coverage=0.0
        ),
        lambda: replace(quality, finite_observation_count=11, coverage=1.1),
        lambda: replace(quality, coverage=0.9),
        lambda: replace(quality, population_variance=float("nan")),
        lambda: replace(quality, eligible=False),
        lambda: replace(quality, eligible=True, rejection_reason="bad"),
        lambda: replace(quality_result, catalog_hash="x"),
        lambda: replace(quality_result, train_source_observation_count=0),
        lambda: replace(quality_result, source_build_id=""),
        lambda: replace(quality_result, eligible_features=("missing",)),
        lambda: replace(quality_result, eligible_features=("f0",)),
        lambda: replace(
            quality_result,
            features=(
                replace(quality, train_observation_count=9, coverage=1.0),
                *quality_features[1:],
            ),
        ),
        lambda: replace(
            quality_result, features=quality_features[:2], eligible_features=("f0", "f1")
        ),
    )

    start = datetime(2026, 1, 1, tzinfo=UTC)
    clock_fold = ModelClockFold(
        "clock-1",
        start,
        start + timedelta(days=1),
        start + timedelta(days=2),
        start + timedelta(days=3),
        504,
        42,
        True,
    )
    invalid_clock_fold = ModelClockFold(
        "clock-2",
        start,
        start + timedelta(days=1),
        start + timedelta(days=2),
        start + timedelta(days=3),
        504,
        42,
        False,
        "missing",
    )
    preflight = ModelClockPreflight(
        ("f0", "f1"),
        "b" * 64,
        504,
        (("f0", 1.0), ("f1", 2.0)),
        (clock_fold,),
        1.0,
        DiscoveryStatus.VALID,
    )
    assert preflight.preflight_hash
    reject(
        lambda: replace(clock_fold, fold_id=""),
        lambda: replace(clock_fold, test_start=clock_fold.train_end),
        lambda: replace(clock_fold, train_complete_observations=-1),
        lambda: replace(clock_fold, structurally_valid=True, invalid_reason="bad"),
        lambda: replace(invalid_clock_fold, structurally_valid=False, invalid_reason=None),
        lambda: replace(preflight, plan_hash="x"),
        lambda: replace(preflight, status=cast(DiscoveryStatus, "valid")),
        lambda: replace(preflight, first_train_complete_observations=-1),
        lambda: replace(preflight, first_train_feature_variances=(("f1", 1.0), ("f0", 2.0))),
        lambda: replace(preflight, first_train_feature_variances=(("f0", -1.0), ("f1", 2.0))),
        lambda: replace(preflight, folds=()),
        lambda: replace(preflight, structural_valid_fold_rate=float("nan")),
        lambda: replace(preflight, structural_valid_fold_rate=0.0),
        lambda: replace(preflight, status=DiscoveryStatus.VALID, invalid_reason="bad"),
        lambda: replace(preflight, status=DiscoveryStatus.INVALID, invalid_reason=None),
    )

    distances = DistanceMatrixResult(
        ("f0", "f1", "f2"),
        ((0.0, 0.1, 1.0), (0.1, 0.0, 0.9), (1.0, 0.9, 0.0)),
        ((504, 504, 504), (504, 504, 504), (504, 504, 504)),
        spearman_correlations=((1.0, 0.9, -1.0), (0.9, 1.0, 0.8), (-1.0, 0.8, 1.0)),
    )
    reject(
        lambda: replace(distances, minimum_pairwise_observations=0),
        lambda: replace(distances, distances=((0.0,),)),
        lambda: replace(distances, pairwise_support=((504,),)),
        lambda: replace(distances, spearman_correlations=((1.0,),)),
        lambda: replace(distances, distances=((0.0, 0.2, 1.0), (0.1, 0.0, 0.9), (1.0, 0.9, 0.0))),
        lambda: replace(distances, distances=((0.1, 0.1, 1.0), (0.1, 0.0, 0.9), (1.0, 0.9, 0.0))),
        lambda: replace(
            distances,
            distances=((0.0, float("nan"), 1.0), (float("nan"), 0.0, 0.9), (1.0, 0.9, 0.0)),
        ),
        lambda: replace(
            distances, pairwise_support=((True, 504, 504), (504, 504, 504), (504, 504, 504))
        ),
        lambda: replace(
            distances, pairwise_support=((504, 505, 504), (504, 504, 504), (504, 504, 504))
        ),
        lambda: replace(
            distances, spearman_correlations=((1.0, 1.1, -1.0), (1.1, 1.0, 0.8), (-1.0, 0.8, 1.0))
        ),
        lambda: replace(
            distances, spearman_correlations=((1.0, 0.9, -1.0), (0.9, 1.0, 0.7), (-1.0, 0.8, 1.0))
        ),
    )

    cluster = ClusterSolution(
        4,
        2,
        ((2, 0.4), (3, 0.3)),
        (("cluster_000", ("f0", "f1")), ("cluster_001", ("f2", "f3"))),
        0.4,
        0,
        merge_tree=((0, 1, 0.1, 2), (2, 3, 0.2, 2), (4, 5, 0.3, 4)),
        candidate_memberships=((2, ()), (3, ())),
        feature_ordinals=(("f0", 1), ("f1", 2), ("f2", 3), ("f3", 4)),
    )
    assert cluster.solution_hash
    reject(
        lambda: replace(cluster, candidate_count=2),
        lambda: replace(cluster, selected_count=3),
        lambda: replace(cluster, silhouette_curve=((2, 0.4),)),
        lambda: replace(cluster, silhouette_curve=((2, 1.1), (3, 0.3))),
        lambda: replace(cluster, selected_silhouette=0.0),
        lambda: replace(cluster, selected_silhouette=0.2),
        lambda: replace(cluster, memberships=(("cluster_000", ("f0",)),)),
        lambda: replace(cluster, singleton_count=1),
        lambda: replace(
            cluster, memberships=(("cluster_000", ("f0", "f0")), ("cluster_001", ("f2",)))
        ),
        lambda: replace(
            cluster, memberships=(("cluster_001", ("f0", "f1")), ("cluster_002", ("f2", "f3")))
        ),
        lambda: replace(cluster, feature_ordinals=(("f1", 2), ("f0", 1), ("f2", 3), ("f3", 4))),
        lambda: replace(cluster, feature_ordinals=(("f0", 1), ("f1", 1), ("f2", 3), ("f3", 4))),
        lambda: replace(cluster, merge_tree=((0, 1, 0.1, 2),)),
        lambda: replace(cluster, merge_tree=((0, 1, -1.0, 2), (2, 3, 0.2, 2), (4, 5, 0.3, 4))),
        lambda: replace(cluster, candidate_memberships=((2, ()),)),
    )

    prototypes = PrototypeSet(
        ("cluster_000", "cluster_001"), ("f0", "f2"), (("f0", 0.0), ("f1", 0.1), ("f2", 0.0))
    )
    assert prototypes.mean_distances
    reject(
        lambda: replace(prototypes, temporary=False),
        lambda: replace(prototypes, cluster_ids=()),
        lambda: replace(prototypes, cluster_ids=("bad", "cluster_001")),
        lambda: replace(prototypes, prototypes=("f0",)),
        lambda: replace(prototypes, mean_distances=()),
        lambda: replace(
            prototypes, mean_distances=(("f0", float("nan")), ("f1", 0.1), ("f2", 0.0))
        ),
        lambda: replace(prototypes, mean_distances=(("f1", 0.1), ("f2", 0.0))),
    )

    teacher = ProvisionalTeacherReference(
        "gaussian_hmm_k2_full",
        2,
        (start, start + timedelta(days=1)),
        ((0.5, 0.5), (0.2, 0.8)),
        (0, 1),
        ("fold-1",),
        "build",
        "b" * 64,
    )
    assert teacher.reference_hash
    reject(
        lambda: replace(teacher, candidate_id="gaussian_hmm_k3_full"),
        lambda: replace(teacher, prototype_features=("f0", "f0")),
        lambda: replace(teacher, timestamps=(start,)),
        lambda: replace(teacher, timestamps=(start + timedelta(days=1), start)),
        lambda: replace(teacher, filtered_probabilities=((0.5, 0.5), (0.2, 0.7))),
        lambda: replace(teacher, dominant_states=(0, 2)),
        lambda: replace(teacher, timestamps=(datetime(2026, 1, 1), datetime(2026, 1, 2))),
    )

    valid_score = score()
    assert valid_score.information_ratio == valid_score.state_information_ratio
    reject(
        lambda: replace(valid_score, canonical_ordinal=0),
        lambda: replace(valid_score, observation_count=-1),
        lambda: replace(valid_score, bin_counts=(0,) * 10),
        lambda: replace(valid_score, bin_counts=(True, *valid_score.bin_counts[1:])),
        lambda: replace(valid_score, joint_bin_state_masses=((0.1,),) * 10),
        lambda: replace(valid_score, bin_masses=(-0.1,) + (0.1,) * 9),
        lambda: replace(valid_score, joint_bin_state_masses=((0.2, 0.0),) * 10),
        lambda: replace(valid_score, state_masses=(1.0, 0.0)),
        lambda: replace(valid_score, mutual_information=float("nan")),
        lambda: replace(valid_score, state_weights=(0.4, 0.6)),
        lambda: replace(valid_score, state_means=(float("nan"), -1.0)),
        lambda: replace(valid_score, between_variance=-1.0),
        lambda: replace(valid_score, eligible=True, state_entropy=1.0e-13),
        lambda: replace(valid_score, eligible=True, coverage=0.8),
        lambda: replace(
            valid_score, eligible=False, state_information_ratio=None, exclusion_reason=None
        ),
        lambda: replace(valid_score, eta_squared=float("nan")),
    )
