from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.feature_discovery.contracts import (
    FINAL_CANDIDATE_IDS,
    ClusterSolution,
    ClusterWinner,
    DistanceMatrixResult,
    FeatureQuality,
    FeatureRegimeScore,
    FinalSelectedConfiguration,
    OuterFoldResult,
    PrefixEvaluation,
    PrefixSearchResult,
    PrototypeSet,
    ProvisionalTeacherReference,
    QualityFilterResult,
    RankedWinnerSet,
    canonical_json,
    content_hash,
)


def quality(name: str, position: int, eligible: bool = True) -> FeatureQuality:
    return FeatureQuality(
        name,
        position,
        10,
        10 if eligible else 8,
        1.0 if eligible else 0.8,
        1.0,
        eligible,
        None if eligible else "coverage",
    )


def test_quality_result_is_canonical_and_order_preserving() -> None:
    result = QualityFilterResult(
        "build",
        10,
        "a" * 64,
        (quality("f0", 0), quality("f1", 1), quality("f2", 2)),
        ("f0", "f1", "f2"),
    )
    assert result.result_hash == result.result_hash
    assert canonical_json(result) == canonical_json(result)


def test_canonical_json_is_independent_of_mapping_insertion_order() -> None:
    left = {"z": (2, 1), "a": {"right": 2, "left": 1}}
    right = {"a": {"left": 1, "right": 2}, "z": (2, 1)}
    assert canonical_json(left) == canonical_json(right)
    assert content_hash(left) == content_hash(right)


def test_nonfinite_and_semantic_decision_fields_fail_closed() -> None:
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
        canonical_json({"unsupported": {"a"}})
    with pytest.raises(ValueError, match="timezone"):
        canonical_json({"created_at": datetime(2026, 1, 1)})


def test_quality_contract_rejects_invalid_counts_and_order() -> None:
    with pytest.raises(ValueError):
        FeatureQuality("", 0, 10, 10, 1.0, 1.0, True)
    with pytest.raises(ValueError):
        FeatureQuality("f", -1, 10, 10, 1.0, 1.0, True)
    with pytest.raises(ValueError):
        FeatureQuality("f", 0, 0, 0, 0.0, 1.0, False, "empty")
    with pytest.raises(ValueError):
        FeatureQuality("f", 0, 10, 11, 1.1, 1.0, True)
    with pytest.raises(ValueError):
        FeatureQuality("f", 0, 10, 8, 1.0, 1.0, False, "coverage")
    with pytest.raises(ValueError):
        FeatureQuality("f", 0, 10, 10, 0.8, 1.0, True)
    with pytest.raises(ValueError):
        FeatureQuality("f", 0, 10, 10, 1.0, -1.0, True)
    with pytest.raises(ValueError):
        FeatureQuality("f", 0, 10, 10, 1.0, 1.0, True, "reason")
    with pytest.raises(ValueError):
        QualityFilterResult("", 10, "a" * 64, (), ())
    with pytest.raises(ValueError):
        QualityFilterResult("build", 0, "a" * 64, (), ())
    with pytest.raises(ValueError):
        QualityFilterResult("build", 10, "A" * 64, (), ())
    valid = (quality("f0", 0), quality("f1", 1), quality("f2", 2))
    with pytest.raises(ValueError):
        QualityFilterResult("build", 10, "a" * 64, (valid[0], valid[0], valid[2]), ("f0", "f2"))
    with pytest.raises(ValueError):
        QualityFilterResult("build", 10, "a" * 64, valid, ("missing", "f1", "f2"))
    with pytest.raises(ValueError):
        QualityFilterResult("build", 10, "a" * 64, valid, ("f1", "f0", "f2"))
    with pytest.raises(ValueError):
        QualityFilterResult(
            "build",
            10,
            "a" * 64,
            (quality("f0", 0), quality("f1", 1, False), quality("f2", 2)),
            ("f0", "f2"),
        )
    wrong_denominator = replace(
        valid[0], train_observation_count=9, finite_observation_count=9, coverage=1.0
    )
    with pytest.raises(ValueError):
        QualityFilterResult(
            "build", 10, "a" * 64, (wrong_denominator, valid[1], valid[2]), ("f0", "f1", "f2")
        )


def valid_distance() -> DistanceMatrixResult:
    return DistanceMatrixResult(
        ("f0", "f1", "f2"),
        ((0.0, 0.1, 0.8), (0.1, 0.0, 0.7), (0.8, 0.7, 0.0)),
        ((504, 504, 504), (504, 504, 504), (504, 504, 504)),
    )


def test_distance_contract_rejects_non_square_or_unsupported_matrices() -> None:
    with pytest.raises(ValueError):
        DistanceMatrixResult(("f0", "f1"), ((0.0,),), ((504,),))
    with pytest.raises(ValueError):
        DistanceMatrixResult(("f0", "f1"), ((0.0, 0.1), (0.1, 0.0)), ((504, 504), (504, 504)), 0)
    with pytest.raises(ValueError):
        DistanceMatrixResult(("f0", "f1"), ((0.0, 1.1), (1.1, 0.0)), ((504, 504), (504, 504)))
    with pytest.raises(ValueError):
        DistanceMatrixResult(("f0", "f1"), ((0.0, 0.1), (0.2, 0.0)), ((504, 504), (504, 504)))
    with pytest.raises(ValueError):
        DistanceMatrixResult(("f0", "f1"), ((0.1, 0.1), (0.1, 0.0)), ((504, 504), (504, 504)))
    with pytest.raises(ValueError):
        DistanceMatrixResult(("f0", "f1"), ((0.0, 0.1), (0.1, 0.0)), ((503, 504), (504, 504)))
    with pytest.raises(ValueError):
        DistanceMatrixResult(("f0", "f1"), ((0.0, 0.1), (0.1, 0.0)), ((504, 504), (504, True)))
    with pytest.raises(ValueError):
        DistanceMatrixResult(("f0", "f1"), ((0.0, 0.1), (0.1, 0.0)), ((504, 503), (504, 504)))
    assert valid_distance().matrix_hash


def test_cluster_and_prototype_contracts_reject_inconsistent_evidence() -> None:
    with pytest.raises(ValueError):
        ClusterSolution(
            2, 2, ((2, 0.2),), (("cluster_000", ("f0",)), ("cluster_001", ("f1",))), 0.2, 2
        )
    with pytest.raises(ValueError):
        ClusterSolution(3, 1, ((2, 0.2),), (("cluster_000", ("f0", "f1", "f2")),), 0.2, 0)
    with pytest.raises(ValueError):
        ClusterSolution(
            3, 2, ((2, 0.2),), (("cluster_000", ("f0",)), ("cluster_001", ("f1",))), 0.2, 1
        )
    with pytest.raises(ValueError):
        ClusterSolution(
            3, 2, ((2, -0.2),), (("cluster_000", ("f0", "f1")), ("cluster_001", ("f2",))), 0.2, 1
        )
    with pytest.raises(ValueError):
        ClusterSolution(
            3, 2, ((2, 0.2),), (("cluster_000", ("f0", "f1")), ("cluster_001", ("f2",))), 0.0, 1
        )
    with pytest.raises(ValueError):
        ClusterSolution(
            3, 2, ((2, 0.2),), (("cluster_000", ("f0", "f1")), ("cluster_001", ("f2",))), 0.3, 1
        )
    with pytest.raises(ValueError):
        ClusterSolution(
            3, 2, ((2, 0.2),), (("cluster_000", ("f0", "f1")), ("cluster_001", ("f2",))), 0.2, 0
        )
    with pytest.raises(ValueError):
        ClusterSolution(
            3, 2, ((2, 0.2),), (("bad", ("f0", "f1")), ("cluster_001", ("f2",))), 0.2, 1
        )
    with pytest.raises(ValueError):
        PrototypeSet(("cluster_000",), ("f0",), (("f0", 0.0),), False)
    with pytest.raises(ValueError):
        PrototypeSet((), (), ())
    with pytest.raises(ValueError):
        PrototypeSet(("cluster_000",), ("f0", "f1"), (("f0", 0.0),))
    with pytest.raises(ValueError):
        PrototypeSet(("cluster_000",), ("f0",), (), True)
    with pytest.raises(ValueError):
        PrototypeSet(("cluster_000",), ("f0",), (("f1", 0.0),))
    with pytest.raises(ValueError):
        PrototypeSet(("cluster_001",), ("f0",), (("f0", 0.0),))
    with pytest.raises(ValueError):
        PrototypeSet(("cluster_000",), ("f0",), (("f0", float("nan")),))


def test_eta_score_requires_support_for_eligible_result() -> None:
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 0.89, 126, (1.0,), (0.0,), (1.0,), 1.0, 1.0, 0.5, True)


def valid_teacher() -> ProvisionalTeacherReference:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return ProvisionalTeacherReference(
        FINAL_CANDIDATE_IDS[0],
        2,
        (start, start + timedelta(days=1)),
        ((0.8, 0.2), (0.3, 0.7)),
        (0, 1),
        ("inner_001",),
        "build",
        "b" * 64,
    )


def valid_score(name: str = "f0") -> FeatureRegimeScore:
    return FeatureRegimeScore(
        name,
        1.0,
        126,
        (60.0, 66.0),
        (1.0, -1.0),
        (0.5, 0.5),
        1.0,
        0.5,
        2 / 3,
        True,
    )


def test_teacher_and_feature_score_contracts_reject_invalid_evidence() -> None:
    teacher = valid_teacher()
    start = teacher.timestamps[0]
    for kwargs in (
        {"state_count": 1},
        {"candidate_id": ""},
        {"inner_plan_hash": "x" * 64},
        {"timestamps": (start,)},
        {"timestamps": (start + timedelta(days=1), start)},
        {"filtered_probabilities": ((1.0,), (0.0,))},
        {"dominant_states": (0, 2)},
        {"filtered_probabilities": ((0.8, 0.1), (0.3, 0.7))},
    ):
        with pytest.raises(ValueError):
            replace(teacher, **kwargs)
    with pytest.raises(ValueError):
        replace(teacher, filtered_probabilities=((float("nan"), 1.0), (0.3, 0.7)))

    with pytest.raises(ValueError):
        FeatureRegimeScore("", 1.0, 126, (1.0,), (0.0,), (1.0,), 1.0, 1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", -1.0, 126, (1.0,), (0.0,), (1.0,), 1.0, 1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 126, (1.0, 1.0), (0.0,), (1.0,), 1.0, 1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 126, (-1.0,), (0.0,), (1.0,), 1.0, 1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 126, (float("nan"),), (0.0,), (1.0,), 1.0, 1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 126, (1.0,), (float("nan"),), (1.0,), 1.0, 1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 126, (1.0,), (0.0,), (-1.0,), 1.0, 1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 126, (1.0,), (0.0,), (1.0,), -1.0, 1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 126, (1.0,), (0.0,), (1.0,), 1.0, -1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 126, (1.0,), (0.0,), (1.0,), 1.0, 1.0, None, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 126, (1.0,), (0.0,), (1.0,), 1.0, 1.0, 1.1, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 0.89, 126, (1.0,), (0.0,), (1.0,), 1.0, 1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 125, (1.0,), (0.0,), (1.0,), 1.0, 1.0, 0.5, True)
    with pytest.raises(ValueError):
        FeatureRegimeScore("f", 1.0, 126, (1.0,), (0.0,), (1.0,), 1.0, 1.0, 0.5, False)
    assert valid_score().eta_squared == pytest.approx(2 / 3)


def test_winner_prefix_and_final_contracts_reject_invalid_identity() -> None:
    score = valid_score()
    winner = ClusterWinner("cluster_000", ("f0", "f1"), "f0", score)
    with pytest.raises(ValueError):
        ClusterWinner("bad", ("f0", "f1"), "f0", score)
    with pytest.raises(ValueError):
        ClusterWinner("cluster_000", ("f0", "f1"), "f1", score)
    with pytest.raises(ValueError):
        ClusterWinner("cluster_000", ("f0", "f1"), "f0", valid_score("f1"))
    with pytest.raises(ValueError):
        RankedWinnerSet((), ())
    with pytest.raises(ValueError):
        RankedWinnerSet((winner,), ("f1",))
    with pytest.raises(ValueError):
        RankedWinnerSet((winner, winner), ("f0", "f0"))
    duplicate_cluster = ClusterWinner("cluster_000", ("f2",), "f2", valid_score("f2"))
    with pytest.raises(ValueError):
        RankedWinnerSet((winner, duplicate_cluster), ("f0", "f2"))
    with pytest.raises(ValueError):
        PrefixEvaluation(1, ("f0",), FINAL_CANDIDATE_IDS[0], 1, 1.0, 0.5)
    with pytest.raises(ValueError):
        PrefixEvaluation(2, ("f0", "f1"), "student_t_hmm_k2_full", 1, 1.0, 0.5)
    with pytest.raises(ValueError):
        PrefixEvaluation(2, ("f0", "f1"), FINAL_CANDIDATE_IDS[0], -1, 1.0, 0.5)
    with pytest.raises(ValueError):
        PrefixEvaluation(2, ("f0", "f1"), FINAL_CANDIDATE_IDS[0], 1, 1.1, 0.5)
    with pytest.raises(ValueError):
        PrefixEvaluation(2, ("f0", "f1"), FINAL_CANDIDATE_IDS[0], 1, 1.0, -0.1)
    evaluation = PrefixEvaluation(2, ("f0", "f1"), FINAL_CANDIDATE_IDS[0], 1, 1.0, 0.5)
    with pytest.raises(ValueError):
        PrefixSearchResult(("f0", "f1", "f2"), (evaluation,), 3, FINAL_CANDIDATE_IDS[0])
    with pytest.raises(ValueError):
        PrefixSearchResult(("f0", "f1"), (), 2, FINAL_CANDIDATE_IDS[0])
    with pytest.raises(ValueError):
        PrefixSearchResult(("f0", "f1"), (evaluation,), 2, "gaussian_hmm_k3_full")
    with pytest.raises(ValueError):
        FinalSelectedConfiguration(("f0",), FINAL_CANDIDATE_IDS[0], 2, "gaussian_hmm", 1, "a" * 64)
    with pytest.raises(ValueError):
        FinalSelectedConfiguration(("f0", "f1"), "not-a-candidate", 2, "gaussian_hmm", 2, "a" * 64)
    with pytest.raises(ValueError):
        FinalSelectedConfiguration(
            ("f0", "f1"), FINAL_CANDIDATE_IDS[0], 2, "student_t_hmm", 2, "a" * 64
        )
    with pytest.raises(ValueError):
        FinalSelectedConfiguration(
            ("f0", "f1"), FINAL_CANDIDATE_IDS[0], 2, "gaussian_hmm", 2, "A" * 64
        )


def test_prefix_result_is_exact_nested_ranked_prefix() -> None:
    evaluations = tuple(
        PrefixEvaluation(
            index, tuple(f"f{j}" for j in range(index)), FINAL_CANDIDATE_IDS[0], 100, 1.0, 0.5
        )
        for index in (2, 3)
    )
    result = PrefixSearchResult(("f0", "f1", "f2"), evaluations, 2, FINAL_CANDIDATE_IDS[0])
    assert result.selected_prefix_length == 2
    with pytest.raises(ValueError):
        PrefixSearchResult(("f0", "f2", "f1"), evaluations, 2, FINAL_CANDIDATE_IDS[0])


def test_final_configuration_and_outer_fold_pin_state_identity() -> None:
    config = FinalSelectedConfiguration(
        ("f0", "f1"), FINAL_CANDIDATE_IDS[0], 2, "gaussian_hmm", 2, "b" * 64
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    fold = OuterFoldResult(
        1,
        start,
        start + timedelta(days=1),
        start + timedelta(days=2),
        start + timedelta(days=3),
        config,
        -1.0,
        (start + timedelta(days=2),),
        ((0.25, 0.75),),
    )
    assert fold.result_hash
    with pytest.raises(ValueError):
        OuterFoldResult(
            1,
            start,
            start + timedelta(days=1),
            start + timedelta(days=2),
            start + timedelta(days=3),
            config,
            -1.0,
            (start + timedelta(days=2),),
            ((0.25, 0.75),),
            "global",
        )


def test_complete_discovery_evidence_chain_is_serializable() -> None:
    quality_result = QualityFilterResult(
        "build",
        10,
        "a" * 64,
        (quality("f0", 0), quality("f1", 1), quality("f2", 2)),
        ("f0", "f1", "f2"),
    )
    distance = DistanceMatrixResult(
        ("f0", "f1", "f2"),
        ((0.0, 0.1, 0.8), (0.1, 0.0, 0.7), (0.8, 0.7, 0.0)),
        ((504, 504, 504), (504, 504, 504), (504, 504, 504)),
    )
    clusters = ClusterSolution(
        3,
        2,
        ((2, 0.25),),
        (("cluster_000", ("f0", "f1")), ("cluster_001", ("f2",))),
        0.25,
        1,
    )
    prototypes = PrototypeSet(
        ("cluster_000", "cluster_001"),
        ("f0", "f2"),
        (("f0", 0.05), ("f1", 0.05), ("f2", 0.0)),
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    teacher = ProvisionalTeacherReference(
        FINAL_CANDIDATE_IDS[0],
        2,
        (start, start + timedelta(days=1)),
        ((0.8, 0.2), (0.3, 0.7)),
        (0, 1),
        ("inner_001",),
        "build",
        "b" * 64,
    )
    score_0 = FeatureRegimeScore(
        "f0", 1.0, 126, (60.0, 66.0), (1.0, -1.0), (0.5, 0.5), 1.0, 0.5, 2 / 3, True
    )
    score_2 = FeatureRegimeScore(
        "f2", 1.0, 126, (60.0, 66.0), (0.5, -0.5), (0.5, 0.5), 0.25, 0.5, 1 / 3, True
    )
    winners = RankedWinnerSet(
        (
            ClusterWinner("cluster_000", ("f0", "f1"), "f0", score_0),
            ClusterWinner("cluster_001", ("f2",), "f2", score_2),
        ),
        ("f0", "f2"),
    )
    prefix = PrefixSearchResult(
        ("f0", "f2"),
        (PrefixEvaluation(2, ("f0", "f2"), FINAL_CANDIDATE_IDS[0], 126, 1.0, 1.0),),
        2,
        FINAL_CANDIDATE_IDS[0],
    )
    final = FinalSelectedConfiguration(
        ("f0", "f2"), FINAL_CANDIDATE_IDS[0], 2, "gaussian_hmm", 2, "c" * 64
    )
    fold = OuterFoldResult(
        1,
        start,
        start + timedelta(days=1),
        start + timedelta(days=2),
        start + timedelta(days=3),
        final,
        -1.0,
        (start + timedelta(days=2),),
        ((0.25, 0.75),),
    )
    evidence = (
        quality_result,
        distance,
        clusters,
        prototypes,
        teacher,
        score_0,
        winners,
        prefix,
        fold,
    )
    assert all(canonical_json(item) for item in evidence)
    assert all(content_hash(item) == content_hash(item) for item in evidence)
