"""Hermetic full-computation and independent-math proof for global v4."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
from collections.abc import Mapping
from datetime import datetime
from math import fsum, log
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples

import market_regime_engine.evaluations.global_regime_v4 as global_v4
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation_runs.math_audit import build_math_expectations
from market_regime_engine.evaluation_statistics.contracts import GlobalV4Evidence
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.teacher_reference import refit_frozen_teacher
from market_regime_engine.feature_discovery.contracts import (
    FINAL_CANDIDATE_IDS,
    V4_PROVISIONAL_STATE_COUNTS,
    AdaptiveEvaluationResult,
    ClusterSolution,
    FeatureRegimeScore,
    ProvisionalTeacherReference,
    canonical_json,
    content_hash,
)
from market_regime_engine.mlflow_support.evaluation_tracking import track_global_v4_evaluation
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort
from market_regime_engine.profiles.loader import load_profile
from tests.fixtures.global_regime_v4.synthetic import (
    FEATURE_COUNT,
    SOURCE_BUILD_ID,
    SyntheticGlobalV4,
    build_synthetic_global_v4,
    canonical_snapshot_bytes,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

PR231_GOLDEN_SNAPSHOT_HASH = "d6dd33bd7ff133d7d32ddc68971243008b4c3d6149cca303b3183cb3f4caca65"

_GOLDEN_TOP_LEVEL_KEYS = frozenset(
    {
        "source_row_count",
        "feature_count",
        "outer_fold_count",
        "folds",
        "validity",
        "stability",
        "hashes",
    }
)
_GOLDEN_FOLD_KEYS = frozenset(
    {
        "fold_index",
        "source_observations",
        "m_star",
        "cluster_memberships",
        "cluster_candidates",
        "cluster_solution_hash",
        "prototypes",
        "teacher",
        "feature_scores",
        "winners",
        "l_star",
        "prefix_candidate_id",
        "prefix_hashes",
        "candidate",
        "final_grid",
        "selection_hash",
        "outer_result_hash",
        "outer",
    }
)
_GOLDEN_MODEL_FAMILIES = frozenset({"gaussian_hmm", "gmm_hmm", "student_t_hmm"})


def _independent_soft_nmi(
    left_timestamps: tuple[datetime, ...],
    left_probabilities: tuple[tuple[float, ...], ...],
    right_timestamps: tuple[datetime, ...],
    right_probabilities: tuple[tuple[float, ...], ...],
) -> float:
    left = dict(zip(left_timestamps, left_probabilities, strict=True))
    right = dict(zip(right_timestamps, right_probabilities, strict=True))
    shared = tuple(sorted(set(left) & set(right)))
    assert shared
    matrix = np.asarray(
        [
            [
                fmean(
                    left[timestamp][left_state] * right[timestamp][right_state]
                    for timestamp in shared
                )
                for right_state in range(len(right[shared[0]]))
            ]
            for left_state in range(len(left[shared[0]]))
        ],
        dtype=np.float64,
    )
    left_marginal = matrix.sum(axis=1)
    right_marginal = matrix.sum(axis=0)
    mutual_information = fsum(
        float(value)
        * log(float(value) / float(left_marginal[left_state] * right_marginal[right_state]))
        for left_state, row in enumerate(matrix)
        for right_state, value in enumerate(row)
        if value > 0.0
    )
    left_entropy = -fsum(float(value) * log(float(value)) for value in left_marginal if value > 0.0)
    right_entropy = -fsum(
        float(value) * log(float(value)) for value in right_marginal if value > 0.0
    )
    return 2.0 * mutual_information / (left_entropy + right_entropy)


def _independent_feature_score(
    rows: pd.DataFrame,
    score: FeatureRegimeScore,
    teacher: ProvisionalTeacherReference,
) -> tuple[float, float]:
    values = dict(zip(rows["timestamp_m1"], rows[score.feature_name], strict=True))
    supported = tuple(
        (float(values[timestamp]), probability)
        for timestamp, probability in zip(
            teacher.timestamps, teacher.filtered_probabilities, strict=True
        )
        if pd.notna(values[timestamp])
    )
    ranked = rankdata(tuple(value for value, _probability in supported), method="average")
    bins = tuple(min(9, int(10 * (rank - 1.0) / len(supported))) for rank in ranked)
    state_count = teacher.state_count
    joint = np.zeros((10, state_count), dtype=np.float64)
    for bin_index, (_value, probability) in zip(bins, supported, strict=True):
        joint[bin_index] += np.asarray(probability, dtype=np.float64) / len(supported)
    bin_marginal = joint.sum(axis=1)
    state_marginal = joint.sum(axis=0)
    mutual_information = fsum(
        float(value) * log(float(value) / float(bin_marginal[b] * state_marginal[k]))
        for b, row in enumerate(joint)
        for k, value in enumerate(row)
        if value > 0.0
    )
    entropy = -fsum(float(value) * log(float(value)) for value in state_marginal if value > 0.0)
    eta_values: list[float] = []
    numeric = np.asarray(tuple(value for value, _probability in supported), dtype=np.float64)
    overall_mean = float(numeric.mean())
    for state in range(state_count):
        weights = np.asarray(
            [probability[state] for _value, probability in supported], dtype=np.float64
        )
        denominator = float(weights.sum())
        mean = float(np.dot(weights, numeric) / denominator)
        variance = float(np.dot(weights, (numeric - mean) ** 2) / denominator)
        eta_values.append((denominator / len(supported)) * (mean - overall_mean) ** 2)
        if state == 0:
            between = 0.0
            within = 0.0
        between += eta_values[-1]
        within += (denominator / len(supported)) * variance
    eta = between / (between + within)
    return mutual_information / entropy, eta


def _independent_cluster_solution(
    distance: object,
    selected: ClusterSolution,
) -> None:
    matrix = np.asarray(distance.distances, dtype=np.float64)
    feature_order = distance.feature_order
    ordinals = dict(selected.feature_ordinals)
    for cluster_count, expected in selected.candidate_memberships:
        model = AgglomerativeClustering(
            n_clusters=cluster_count,
            metric="precomputed",
            linkage="average",
        ).fit(matrix)
        clusters: list[tuple[str, ...]] = []
        for label in sorted(set(model.labels_.tolist())):
            members = tuple(
                sorted(
                    (
                        feature_order[index]
                        for index, value in enumerate(model.labels_)
                        if value == label
                    ),
                    key=ordinals.__getitem__,
                )
            )
            clusters.append(members)
        actual = tuple(sorted(clusters, key=lambda members: ordinals[members[0]]))
        expected_members = tuple(members for _cluster, members in expected)
        assert actual == expected_members
        labels = np.asarray(
            [
                next(index for index, members in enumerate(actual) if feature in members)
                for feature in feature_order
            ]
        )
        samples = silhouette_samples(matrix, labels, metric="precomputed")
        samples = np.asarray(
            [
                0.0 if len(actual[label]) == 1 else value
                for label, value in zip(labels, samples, strict=True)
            ]
        )
        assert float(samples.mean()) == pytest.approx(
            dict(selected.silhouette_curve)[cluster_count]
        )


def _independent_cluster_jaccard(previous: ClusterSolution, current: ClusterSolution) -> float:
    """Recompute adjacent-fold cluster stability from membership primitives."""

    previous_sets = tuple({*members} for _cluster, members in previous.memberships)
    current_sets = tuple({*members} for _cluster, members in current.memberships)
    if not previous_sets or not current_sets:
        raise AssertionError("cluster stability requires non-empty memberships")
    return float(
        fmean(
            max(
                len(current_members & previous_members) / len(current_members | previous_members)
                for previous_members in previous_sets
            )
            for current_members in current_sets
        )
    )


def _independent_distance(rows: pd.DataFrame, feature_order: tuple[str, ...]) -> np.ndarray:
    matrix = np.zeros((len(feature_order), len(feature_order)), dtype=np.float64)
    for left_index, left_name in enumerate(feature_order):
        for right_index in range(left_index):
            right_name = feature_order[right_index]
            complete = rows[[left_name, right_name]].dropna()
            left_rank = rankdata(complete[left_name].to_numpy(), method="average")
            right_rank = rankdata(complete[right_name].to_numpy(), method="average")
            correlation = float(np.corrcoef(left_rank, right_rank)[0, 1])
            matrix[left_index, right_index] = matrix[right_index, left_index] = 1.0 - abs(
                correlation
            )
    return matrix


def _evidence(
    fixture: SyntheticGlobalV4,
    result: AdaptiveEvaluationResult,
    selections: Mapping[int, object],
) -> GlobalV4Evidence:
    ordered = tuple(selections[index] for index in sorted(selections))
    adjacent_stability = [
        {
            "outer_fold_pair": [
                result.outer_folds[index].fold_index,
                result.outer_folds[index + 1].fold_index,
            ],
            "mean_best_cluster_jaccard": _independent_cluster_jaccard(
                ordered[index].clusters,
                ordered[index + 1].clusters,
            ),
        }
        for index in range(len(ordered) - 1)
    ]
    groups = {
        "identity": {"policy_id": "xetra_global_regime_v4", "schema_version": 1},
        "lineage": {
            "source_build_id": SOURCE_BUILD_ID,
            "source_data_hash": fixture.source_data_hash,
            "catalog_hash": fixture.catalog.catalog_hash,
            "profile_hash": load_profile("configs/profiles/xetra_v4.yaml").profile_hash,
            "repository_hash": "a" * 64,
            "outer_plan_hash": content_hash(
                tuple(fold.test_end.isoformat() for fold in result.outer_folds)
            ),
        },
        "input": {
            "feature_count": FEATURE_COUNT,
            "feature_order": list(fixture.catalog.feature_names),
        },
        "quality": {"fold_hashes": [selection.quality.result_hash for selection in ordered]},
        "distance": {"fold_hashes": [selection.distance.matrix_hash for selection in ordered]},
        "clustering": {"fold_hashes": [selection.clusters.solution_hash for selection in ordered]},
        "prototypes": {
            "fold_features": [list(selection.prototypes.prototypes) for selection in ordered]
        },
        "teacher": {
            "fold_hashes": [selection.teacher_reference.reference_hash for selection in ordered]
        },
        "feature_scores": {"fold_counts": [len(selection.feature_scores) for selection in ordered]},
        "prefix_search": {
            "fold_selected_l": [
                selection.prefix_search.selected_prefix_length for selection in ordered
            ]
        },
        "final_grid": {
            "fold_candidate_counts": [
                len(selection.final_grid.grid.aggregates) for selection in ordered
            ]
        },
        "outer_folds": [
            {"fold_index": fold.fold_index, "result_hash": fold.result_hash, "valid": fold.valid}
            for fold in result.outer_folds
        ],
        "agreement": {
            "soft_nmi": [fold.outer_teacher_final_soft_nmi for fold in result.outer_folds]
        },
        "validity": {
            "valid_fold_count": result.valid_fold_count,
            "valid_fold_rate": result.valid_fold_rate,
            "production_eligible": result.production_eligible,
        },
        "stability": {
            "selection_sink": "outer_fold_train_only",
            "adjacent_fold_cluster_membership_jaccard": adjacent_stability,
        },
    }
    return GlobalV4Evidence(
        SOURCE_BUILD_ID,
        fixture.source_data_hash,
        fixture.catalog.catalog_hash,
        load_profile("configs/profiles/xetra_v4.yaml").profile_hash,
        "a" * 64,
        groups["lineage"]["outer_plan_hash"],
        groups,
    )


def _golden_snapshot(
    fixture: SyntheticGlobalV4,
    result: AdaptiveEvaluationResult,
    selections: Mapping[int, object],
    evidence: GlobalV4Evidence,
) -> dict[str, object]:
    """Capture every statistical selection as one fixed, reviewable digest."""

    ordered = tuple(selections[index] for index in sorted(selections))
    return {
        "source_row_count": len(fixture.rows),
        "feature_count": len(fixture.catalog.feature_names),
        "outer_fold_count": len(result.outer_folds),
        "folds": [
            {
                "fold_index": fold.fold_index,
                "source_observations": {
                    "train": 1260 + (fold.fold_index - 1) * 63,
                    "test": 63,
                },
                "m_star": selection.clusters.selected_count,
                "cluster_memberships": [
                    {"cluster_id": cluster_id, "features": list(features)}
                    for cluster_id, features in selection.clusters.memberships
                ],
                "cluster_candidates": [
                    {
                        "m": cluster_count,
                        "memberships": [
                            {"cluster_id": cluster_id, "features": list(features)}
                            for cluster_id, features in memberships
                        ],
                    }
                    for cluster_count, memberships in selection.clusters.candidate_memberships
                ],
                "cluster_solution_hash": selection.clusters.solution_hash,
                "prototypes": {
                    "features": list(selection.prototypes.prototypes),
                    "mean_distance_hash": content_hash(selection.prototypes.mean_distances),
                    "hash": content_hash(selection.prototypes),
                },
                "teacher": {
                    "candidate_id": selection.teacher_reference.candidate_id,
                    "state_count": selection.teacher_reference.state_count,
                    "prototype_features": list(selection.teacher_reference.prototype_features),
                    "inner_plan_hash": selection.teacher_reference.inner_plan_hash,
                    "reference_hash": selection.teacher_reference.reference_hash,
                    "candidate_aggregate_hashes": [
                        {
                            "candidate_id": aggregate.candidate_id,
                            "hash": content_hash(aggregate),
                        }
                        for aggregate in selection.teacher_evaluation.candidate_aggregates
                    ],
                    "selection_hash": content_hash(selection.teacher_evaluation.selection),
                },
                "feature_scores": [
                    {
                        "feature_name": score.feature_name,
                        "state_information_ratio": score.state_information_ratio,
                        "eta_squared": score.eta_squared,
                        "hash": content_hash(score),
                    }
                    for score in selection.feature_scores
                ],
                "winners": {
                    "ranked_features": list(selection.winner_selection.ranked_features),
                    "hash": content_hash(selection.winner_selection),
                },
                "l_star": selection.prefix_search.selected_prefix_length,
                "prefix_candidate_id": selection.prefix_search.selected_candidate_id,
                "prefix_hashes": [
                    content_hash(prefix) for prefix in selection.prefix_search.evaluations
                ],
                "candidate": {
                    "candidate_id": selection.final_candidate.candidate_id,
                    "feature_order": list(selection.final_candidate.feature_order),
                    "state_count": selection.final_candidate.state_count,
                    "model_family": selection.final_candidate.model_family,
                    "hash": content_hash(selection.final_candidate),
                },
                "final_grid": {
                    "candidate_ids": [
                        candidate.candidate_id
                        for candidate in selection.final_grid.grid.evaluations
                    ],
                    "aggregate_hashes": [
                        content_hash(aggregate)
                        for aggregate in selection.final_grid.grid.aggregates
                    ],
                    "selection_hash": content_hash(selection.final_grid.selection),
                },
                "selection_hash": content_hash(selection),
                "outer_result_hash": fold.result_hash,
                "outer": {
                    "soft_nmi": fold.outer_teacher_final_soft_nmi,
                    "shared_timestamp_count": fold.outer_shared_timestamp_count,
                    "valid": fold.valid,
                    "failure_reason": fold.failure_reason,
                },
            }
            for fold, selection in zip(result.outer_folds, ordered, strict=True)
        ],
        "validity": {
            "valid_fold_count": result.valid_fold_count,
            "valid_fold_rate": result.valid_fold_rate,
            "latest_complete_fold_valid": result.latest_complete_fold_valid,
            "production_eligible": result.production_eligible,
            "failure_reason": result.failure_reason,
        },
        "stability": evidence.evidence["stability"],
        "hashes": {
            "policy_hash": result.policy_hash,
            "result_hash": result.result_hash,
            "evidence_hash": evidence.evidence_hash,
            "catalog_hash": fixture.catalog.catalog_hash,
            "source_data_hash": fixture.source_data_hash,
        },
    }


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"golden {name} must be an object")
    return cast(Mapping[str, object], value)


def _assert_complete_golden_snapshot(
    snapshot: Mapping[str, object],
    fixture: SyntheticGlobalV4,
    result: AdaptiveEvaluationResult,
    selections: Mapping[int, object],
    evidence: GlobalV4Evidence,
) -> None:
    """Fail closed when a golden snapshot silently omits a proof primitive."""

    assert frozenset(snapshot) == _GOLDEN_TOP_LEVEL_KEYS
    assert snapshot["source_row_count"] == len(fixture.rows)
    assert snapshot["feature_count"] == len(fixture.catalog.feature_names)
    assert snapshot["outer_fold_count"] == len(result.outer_folds)

    raw_folds = snapshot["folds"]
    assert isinstance(raw_folds, list)
    assert len(raw_folds) == len(result.outer_folds) == len(selections)
    ordered = tuple(selections[index] for index in sorted(selections))
    expected_train_counts = tuple(1260 + index * 63 for index in range(len(ordered)))
    assert tuple(sorted(selections)) == expected_train_counts

    for fold, selection, raw_fold in zip(result.outer_folds, ordered, raw_folds, strict=True):
        golden_fold = _as_mapping(raw_fold, "fold")
        assert frozenset(golden_fold) == _GOLDEN_FOLD_KEYS
        train_count = 1260 + (fold.fold_index - 1) * 63
        assert golden_fold["fold_index"] == fold.fold_index
        assert golden_fold["source_observations"] == {"train": train_count, "test": 63}
        assert golden_fold["m_star"] == selection.clusters.selected_count

        memberships = [
            {"cluster_id": cluster_id, "features": list(features)}
            for cluster_id, features in selection.clusters.memberships
        ]
        assert golden_fold["cluster_memberships"] == memberships
        cluster_candidates = [
            {
                "m": cluster_count,
                "memberships": [
                    {"cluster_id": cluster_id, "features": list(features)}
                    for cluster_id, features in candidate_memberships
                ],
            }
            for cluster_count, candidate_memberships in selection.clusters.candidate_memberships
        ]
        assert golden_fold["cluster_candidates"] == cluster_candidates
        assert golden_fold["cluster_solution_hash"] == selection.clusters.solution_hash

        prototype = _as_mapping(golden_fold["prototypes"], "prototypes")
        assert frozenset(prototype) == {"features", "mean_distance_hash", "hash"}
        assert prototype["features"] == list(selection.prototypes.prototypes)
        assert prototype["mean_distance_hash"] == content_hash(selection.prototypes.mean_distances)
        assert prototype["hash"] == content_hash(selection.prototypes)

        teacher = _as_mapping(golden_fold["teacher"], "teacher")
        assert frozenset(teacher) == {
            "candidate_id",
            "state_count",
            "prototype_features",
            "inner_plan_hash",
            "reference_hash",
            "candidate_aggregate_hashes",
            "selection_hash",
        }
        assert teacher["candidate_id"] == selection.teacher_reference.candidate_id
        assert teacher["state_count"] == selection.teacher_reference.state_count
        assert teacher["prototype_features"] == list(selection.teacher_reference.prototype_features)
        assert teacher["inner_plan_hash"] == selection.teacher_reference.inner_plan_hash
        assert teacher["reference_hash"] == selection.teacher_reference.reference_hash
        teacher_aggregates = teacher["candidate_aggregate_hashes"]
        assert isinstance(teacher_aggregates, list)
        teacher_aggregate_mappings = tuple(
            _as_mapping(value, "teacher aggregate") for value in teacher_aggregates
        )
        assert tuple(item["candidate_id"] for item in teacher_aggregate_mappings) == tuple(
            f"gaussian_hmm_k{state_count}_full" for state_count in V4_PROVISIONAL_STATE_COUNTS
        )
        assert tuple(item["hash"] for item in teacher_aggregate_mappings) == tuple(
            content_hash(aggregate)
            for aggregate in selection.teacher_evaluation.candidate_aggregates
        )
        assert teacher["selection_hash"] == content_hash(selection.teacher_evaluation.selection)

        feature_scores = golden_fold["feature_scores"]
        assert isinstance(feature_scores, list)
        assert len(feature_scores) == len(selection.feature_scores)
        for score_payload, score in zip(feature_scores, selection.feature_scores, strict=True):
            score_mapping = _as_mapping(score_payload, "feature score")
            assert frozenset(score_mapping) == {
                "feature_name",
                "state_information_ratio",
                "eta_squared",
                "hash",
            }
            assert score_mapping["feature_name"] == score.feature_name
            assert score_mapping["state_information_ratio"] == score.state_information_ratio
            assert score_mapping["eta_squared"] == score.eta_squared
            assert score_mapping["hash"] == content_hash(score)

        winners = _as_mapping(golden_fold["winners"], "winners")
        assert frozenset(winners) == {"ranked_features", "hash"}
        assert winners["ranked_features"] == list(selection.winner_selection.ranked_features)
        assert winners["hash"] == content_hash(selection.winner_selection)
        assert golden_fold["l_star"] == selection.prefix_search.selected_prefix_length
        assert golden_fold["prefix_candidate_id"] == selection.prefix_search.selected_candidate_id
        assert golden_fold["prefix_hashes"] == [
            content_hash(prefix) for prefix in selection.prefix_search.evaluations
        ]

        candidate = _as_mapping(golden_fold["candidate"], "candidate")
        assert frozenset(candidate) == {
            "candidate_id",
            "feature_order",
            "state_count",
            "model_family",
            "hash",
        }
        assert candidate["candidate_id"] == selection.final_candidate.candidate_id
        assert candidate["feature_order"] == list(selection.final_candidate.feature_order)
        assert candidate["state_count"] == selection.final_candidate.state_count
        assert candidate["model_family"] == selection.final_candidate.model_family
        assert candidate["hash"] == content_hash(selection.final_candidate)
        assert candidate["model_family"] in _GOLDEN_MODEL_FAMILIES
        assert len(candidate["feature_order"]) == golden_fold["l_star"]

        final_grid = _as_mapping(golden_fold["final_grid"], "final grid")
        assert frozenset(final_grid) == {"candidate_ids", "aggregate_hashes", "selection_hash"}
        candidate_ids = tuple(final_grid["candidate_ids"])
        assert candidate_ids == FINAL_CANDIDATE_IDS
        assert len(final_grid["aggregate_hashes"]) == len(FINAL_CANDIDATE_IDS)
        assert final_grid["aggregate_hashes"] == [
            content_hash(aggregate) for aggregate in selection.final_grid.grid.aggregates
        ]
        assert final_grid["selection_hash"] == content_hash(selection.final_grid.selection)
        assert golden_fold["selection_hash"] == content_hash(selection)
        assert golden_fold["outer_result_hash"] == fold.result_hash
        assert golden_fold["outer"] == {
            "soft_nmi": fold.outer_teacher_final_soft_nmi,
            "shared_timestamp_count": fold.outer_shared_timestamp_count,
            "valid": fold.valid,
            "failure_reason": fold.failure_reason,
        }

    assert snapshot["validity"] == {
        "valid_fold_count": result.valid_fold_count,
        "valid_fold_rate": result.valid_fold_rate,
        "latest_complete_fold_valid": result.latest_complete_fold_valid,
        "production_eligible": result.production_eligible,
        "failure_reason": result.failure_reason,
    }
    assert snapshot["stability"] == evidence.evidence["stability"]
    assert snapshot["hashes"] == {
        "policy_hash": result.policy_hash,
        "result_hash": result.result_hash,
        "evidence_hash": evidence.evidence_hash,
        "catalog_hash": fixture.catalog.catalog_hash,
        "source_data_hash": fixture.source_data_hash,
    }


def _evaluate_with_selection_capture(
    rows: pd.DataFrame,
    catalog: object,
    profile: object,
) -> tuple[AdaptiveEvaluationResult, dict[int, object]]:
    captured: dict[int, object] = {}

    def capture_selection(fold_index: int, selection: object) -> None:
        captured[1260 + (fold_index - 1) * 63] = selection

    result = global_v4.evaluate_global_regime_v4(
        rows,
        catalog=catalog,
        profile=profile,
        max_workers=None,
        selection_sink=capture_selection,
    )
    return result, captured


def _per_fold_evidence_payload(evidence: GlobalV4Evidence, index: int) -> dict[str, object]:
    groups = evidence.evidence
    return {
        "quality_hash": groups["quality"]["fold_hashes"][index],
        "distance_hash": groups["distance"]["fold_hashes"][index],
        "clustering_hash": groups["clustering"]["fold_hashes"][index],
        "prototype_features": groups["prototypes"]["fold_features"][index],
        "teacher_hash": groups["teacher"]["fold_hashes"][index],
        "feature_score_count": groups["feature_scores"]["fold_counts"][index],
        "selected_prefix_length": groups["prefix_search"]["fold_selected_l"][index],
        "final_candidate_count": groups["final_grid"]["fold_candidate_counts"][index],
        "outer_fold": groups["outer_folds"][index],
        "soft_nmi": groups["agreement"]["soft_nmi"][index],
    }


def _independent_math_verifier() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "verify_xetra_v4_math.py"
    spec = importlib.util.spec_from_file_location("verify_xetra_v4_math", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent math verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_selected_likelihood_parity(
    fixture: SyntheticGlobalV4,
    result: AdaptiveEvaluationResult,
    selections: Mapping[int, object],
) -> int:
    """Independently recompute selected TRAIN/OOS likelihood evidence."""

    by_outer_fold = {
        fold.fold_index: selections[1260 + (fold.fold_index - 1) * 63]
        for fold in result.outer_folds
    }
    expectations = build_math_expectations(
        fixture.rows,
        result,
        cast(Any, by_outer_fold),
    )
    dossiers = cast(list[dict[str, object]], expectations["fold_audits"])
    independent = _independent_math_verifier()
    likelihood_count = 0
    for dossier in dossiers:
        likelihoods = cast(list[dict[str, object]], dossier["likelihoods"])
        assert len(likelihoods) >= 2
        assert len(likelihoods) % 2 == 0
        candidate_id = str(dossier["final_candidate_id"])
        expected_family = (
            "gmm_hmm" if candidate_id.startswith("gmm_hmm_") else candidate_id.rsplit("_k", 1)[0]
        )
        assert expected_family in _GOLDEN_MODEL_FAMILIES
        assert {item["scope"] for item in likelihoods} == {"TRAIN", "OOS"}
        assert all(item["model_family"] == expected_family for item in likelihoods)
        fold_scope_pairs = {(item["fold_index"], item["scope"]) for item in likelihoods}
        fold_indices = {item["fold_index"] for item in likelihoods}
        assert len(likelihoods) == 2 * len(fold_indices)
        assert fold_scope_pairs == {
            (fold_index, scope) for fold_index in fold_indices for scope in ("TRAIN", "OOS")
        }
        for item in likelihoods:
            assert isinstance(item["fold_index"], int) and item["fold_index"] >= 1
            actual = independent.independent_hmm_log_likelihood(item)
            assert actual == pytest.approx(float(item["log_likelihood"]), abs=1.0e-10)
            likelihood_count += 1
    assert likelihood_count >= 2 * len(dossiers)
    return likelihood_count


def _assert_canonical_process_rerun(
    result: AdaptiveEvaluationResult,
    evidence: GlobalV4Evidence,
    result_path: Path,
    evidence_path: Path,
) -> tuple[bytes, bytes]:
    """Require a spawned rerun to emit the exact canonical bytes, not just equal hashes."""

    assert result_path.is_file()
    assert evidence_path.is_file()
    result_bytes = result_path.read_bytes()
    evidence_bytes = evidence_path.read_bytes()
    expected_result_bytes = canonical_json(result)
    expected_evidence_bytes = evidence.canonical_json()
    assert result_bytes == expected_result_bytes
    assert evidence_bytes == expected_evidence_bytes
    assert hashlib.sha256(result_bytes).hexdigest() == content_hash(result)
    assert hashlib.sha256(evidence_bytes).hexdigest() == evidence.evidence_hash
    return result_bytes, evidence_bytes


def _randomized_semantic_labels(fixture: SyntheticGlobalV4) -> dict[str, str]:
    """Return deterministic labels with exactly the fixture's feature identity set."""

    features = tuple(fixture.semantic_labels)
    labels = {feature: f"randomized_{index}" for index, feature in enumerate(reversed(features))}
    if set(labels) != set(features) or labels == fixture.semantic_labels:
        raise AssertionError(
            "semantic-label mutation is not a true identity-preserving randomization"
        )
    return labels


def _independent_process_worker(
    snapshot_path: str,
    labels_path: str,
    result_path: str,
    evidence_path: str,
) -> None:
    fixture = build_synthetic_global_v4()
    snapshot_bytes = Path(snapshot_path).read_bytes()
    generated_snapshot = fixture.canonical_snapshot_bytes
    if snapshot_bytes != generated_snapshot:
        raise AssertionError("independent process did not use the pinned snapshot bytes")
    if hashlib.sha256(snapshot_bytes).hexdigest() != fixture.source_data_hash:
        raise AssertionError("pinned snapshot hash does not match fixture lineage")
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    if not isinstance(labels, dict):
        raise TypeError("randomized semantic labels must be a JSON object")
    if set(labels) != set(fixture.semantic_labels) or any(
        not isinstance(value, str) for value in labels.values()
    ):
        raise AssertionError("randomized semantic labels must preserve the feature identity set")
    fixture.semantic_labels.clear()
    fixture.semantic_labels.update(labels)
    result, captured = _evaluate_with_selection_capture(
        fixture.rows, fixture.catalog, load_profile("configs/profiles/xetra_v4.yaml")
    )
    evidence = _evidence(fixture, result, captured)
    Path(result_path).write_bytes(canonical_json(result))
    Path(evidence_path).write_bytes(evidence.canonical_json())


def _lineage(fixture: SyntheticGlobalV4) -> SourceLineage:
    source = fixture.catalog.lineage
    return SourceLineage(
        source_dataset=source.source_dataset,
        source_build_id=source.source_build_id,
        data_sha256=fixture.source_data_hash,
        schema_version=source.schema_version,
        feature_version=source.feature_version,
        source_table=source.source_table,
        synced_at_utc=source.synced_at_utc,
        row_count=source.row_count,
        min_timestamp=source.min_timestamp,
        max_timestamp=source.max_timestamp,
    )


def test_global_v4_full_compute_and_independent_math_proof(
    tmp_path: Path,
) -> None:
    fixture = build_synthetic_global_v4()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    captured: dict[int, object] = {}
    captured_prefix_evaluations: dict[tuple[int, int], dict[str, object]] = {}

    def capture_selection(fold_index: int, selection: object) -> None:
        captured[1260 + (fold_index - 1) * 63] = selection

    def capture_prefix_evaluations(
        fold_index: int,
        prefix_length: int,
        candidate_id: str,
        evaluation: object,
    ) -> None:
        train_count = 1260 + (fold_index - 1) * 63
        captured_prefix_evaluations.setdefault((train_count, prefix_length), {})[candidate_id] = (
            evaluation
        )

    result = global_v4.evaluate_global_regime_v4(
        fixture.rows,
        catalog=fixture.catalog,
        profile=profile,
        max_workers=None,
        selection_sink=capture_selection,
        prefix_evaluation_sink=capture_prefix_evaluations,
    )

    assert len(result.outer_folds) >= 3
    assert len(captured) == len(result.outer_folds)
    assert all(len(selection.quality.eligible_features) >= 50 for selection in captured.values())
    assert all(selection.clusters.singleton_count >= 1 for selection in captured.values())
    assert all(selection.final_grid.selection is not None for selection in captured.values())
    assert any(
        prototype not in selection.winner_selection.ranked_features
        for selection in captured.values()
        for prototype in selection.prototypes.prototypes
    )
    assert all(
        prefix.prefix_length == len(prefix.feature_order)
        for selection in captured.values()
        for prefix in selection.prefix_search.evaluations
    )
    assert all(
        prefix.shared_timestamp_count > 0
        for selection in captured.values()
        for prefix in selection.prefix_search.evaluations
    )

    first = captured[1260]
    first_train = fixture.rows.iloc[:1260]
    independent_distance = _independent_distance(first_train, first.distance.feature_order)
    assert np.asarray(first.distance.distances) == pytest.approx(independent_distance, abs=1.0e-10)
    _independent_cluster_solution(first.distance, first.clusters)

    for score in first.feature_scores:
        expected_sir, expected_eta = _independent_feature_score(
            first_train,
            score,
            first.teacher_reference,
        )
        assert score.state_information_ratio == pytest.approx(expected_sir, abs=1.0e-10)
        assert score.eta_squared == pytest.approx(expected_eta, abs=1.0e-10)

    for train_count, selection in captured.items():
        teacher = selection.teacher_reference
        for prefix in selection.prefix_search.evaluations:
            if not prefix.valid:
                continue
            candidate = captured_prefix_evaluations[(train_count, prefix.prefix_length)][
                prefix.candidate_id
            ]
            candidate_timestamps = tuple(
                timestamp
                for fold in candidate.folds
                if fold.valid
                for timestamp in fold.oos_timestamps
            )
            candidate_probabilities = tuple(
                probability
                for fold in candidate.folds
                if fold.valid
                for probability in fold.oos_filtered_probabilities
            )
            assert prefix.soft_regime_nmi == pytest.approx(
                _independent_soft_nmi(
                    candidate_timestamps,
                    candidate_probabilities,
                    teacher.timestamps,
                    teacher.filtered_probabilities,
                ),
                abs=1.0e-10,
            )

        for grid_evaluation, aggregate in zip(
            selection.final_grid.grid.evaluations,
            selection.final_grid.grid.aggregates,
            strict=True,
        ):
            valid = tuple(fold for fold in grid_evaluation.folds if fold.valid)
            if not valid:
                assert aggregate.valid_fold_count == 0
                assert aggregate.oos_predictive_loglik_mean is None
                assert aggregate.oos_predictive_loglik_std is None
                assert aggregate.oos_predictive_loglik_worst_fold is None
                assert aggregate.bic_mean is None
                assert aggregate.aic_mean is None
                continue
            oos = tuple(fold.oos_predictive_log_likelihood_per_observation for fold in valid)
            assert aggregate.oos_predictive_loglik_mean == pytest.approx(fmean(oos))
            assert aggregate.oos_predictive_loglik_std == pytest.approx(pstdev(oos))
            assert aggregate.oos_predictive_loglik_worst_fold == pytest.approx(min(oos))
            assert aggregate.bic_mean == pytest.approx(fmean(fold.bic for fold in valid))
            assert aggregate.aic_mean == pytest.approx(fmean(fold.aic for fold in valid))

    for fold in result.outer_folds:
        selection = captured[1260 + (fold.fold_index - 1) * 63]
        train = fixture.rows.iloc[: 1260 + (fold.fold_index - 1) * 63]
        test = fixture.rows.iloc[1260 + (fold.fold_index - 1) * 63 : 1260 + fold.fold_index * 63]
        teacher = refit_frozen_teacher(
            train,
            test,
            reference=selection.teacher_reference,
            profile=profile,
        )
        assert fold.outer_teacher_final_soft_nmi == pytest.approx(
            _independent_soft_nmi(
                fold.oos_timestamps,
                fold.oos_filtered_probabilities,
                teacher.test_timestamps,
                teacher.test_filtered_probabilities,
            ),
            abs=1.0e-10,
        )

    selected_likelihood_count = _assert_selected_likelihood_parity(
        fixture,
        result,
        captured,
    )

    evidence = _evidence(fixture, result, captured)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
    tracked_selections = {
        fold.fold_index: captured[1260 + (fold.fold_index - 1) * 63] for fold in result.outer_folds
    }
    tracked = track_global_v4_evaluation(
        FileMlflowTrackingPort(tracking_uri, experiment_name="global-v4-full-proof"),
        StatisticsWriter(tmp_path / "statistics"),
        evidence=evidence,
        result=result,
        selections=tracked_selections,
    )
    assert tracked.global_evidence_hash == evidence.evidence_hash
    assert Path(tracked.plot_manifest_path).is_file()

    golden_snapshot = _golden_snapshot(fixture, result, captured, evidence)
    golden_hash = content_hash(golden_snapshot)
    _assert_complete_golden_snapshot(golden_snapshot, fixture, result, captured, evidence)
    assert golden_hash == PR231_GOLDEN_SNAPSHOT_HASH

    snapshot_path = tmp_path / "pr231-pinned-snapshot.csv"
    snapshot_bytes = fixture.canonical_snapshot_bytes
    snapshot_path.write_bytes(snapshot_bytes)
    randomized_labels = _randomized_semantic_labels(fixture)
    labels_path = tmp_path / "pr231-randomized-labels.json"
    labels_path.write_text(json.dumps(randomized_labels, sort_keys=True) + "\n", encoding="utf-8")
    independent_result_path = tmp_path / "pr231-independent-result.json"
    independent_evidence_path = tmp_path / "pr231-independent-evidence.json"
    context = multiprocessing.get_context("spawn")
    independent = context.Process(
        target=_independent_process_worker,
        args=(
            str(snapshot_path),
            str(labels_path),
            str(independent_result_path),
            str(independent_evidence_path),
        ),
    )
    independent.start()
    independent.join(timeout=3_600)
    if independent.is_alive():
        independent.terminate()
        independent.join()
        pytest.fail("independent PR-231 process rerun exceeded one hour")
    assert independent.exitcode == 0
    independent_result_bytes, independent_evidence_bytes = _assert_canonical_process_rerun(
        result,
        evidence,
        independent_result_path,
        independent_evidence_path,
    )

    last_fold = result.outer_folds[-1]
    mutated_feature = last_fold.final_configuration.feature_order[0]
    last_test_start = 1260 + (last_fold.fold_index - 1) * 63
    mutation_index = next(
        index
        for index in range(last_test_start, len(fixture.rows))
        if pd.notna(fixture.rows.iloc[index][mutated_feature])
        and np.isfinite(float(fixture.rows.iloc[index][mutated_feature]))
    )
    mutated_rows = fixture.rows.copy()
    mutated_rows.loc[mutation_index, mutated_feature] += 10_000.0
    mutated_snapshot_bytes = canonical_snapshot_bytes(mutated_rows)
    mutated_fixture = SyntheticGlobalV4(
        mutated_rows,
        fixture.catalog,
        dict(fixture.semantic_labels),
        hashlib.sha256(mutated_snapshot_bytes).hexdigest(),
    )
    assert hashlib.sha256(mutated_snapshot_bytes).hexdigest() != fixture.source_data_hash
    mutated_result, mutated_captured = _evaluate_with_selection_capture(
        mutated_fixture.rows, mutated_fixture.catalog, profile
    )
    mutated_evidence = _evidence(mutated_fixture, mutated_result, mutated_captured)
    assert mutated_result.result_hash != result.result_hash
    assert mutated_evidence.evidence_hash != evidence.evidence_hash
    assert mutated_result.outer_folds[-1].result_hash != last_fold.result_hash
    for index, fold in enumerate(result.outer_folds[:-1]):
        assert canonical_json(mutated_result.outer_folds[index]) == canonical_json(fold)
        train_count = 1260 + index * 63
        assert canonical_json(mutated_captured[train_count]) == canonical_json(
            captured[train_count]
        )
        assert canonical_json(
            _per_fold_evidence_payload(mutated_evidence, index)
        ) == canonical_json(_per_fold_evidence_payload(evidence, index))

    proof_output = os.environ.get("PR231_PROOF_OUTPUT")
    if proof_output is not None:
        proof = {
            "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            "baseline_result_hash": result.result_hash,
            "baseline_evidence_hash": evidence.evidence_hash,
            "independent_process_result_hash": hashlib.sha256(independent_result_bytes).hexdigest(),
            "independent_process_evidence_hash": hashlib.sha256(
                independent_evidence_bytes
            ).hexdigest(),
            "independent_canonical_result_bytes_equal": independent_result_bytes
            == canonical_json(result),
            "independent_canonical_evidence_bytes_equal": independent_evidence_bytes
            == evidence.canonical_json(),
            "randomized_semantic_labels_recomputed": True,
            "selected_likelihood_records_recomputed": selected_likelihood_count,
            "mutation": {
                "row_index": mutation_index,
                "feature_name": mutated_feature,
                "mutated_result_hash": mutated_result.result_hash,
                "mutated_evidence_hash": mutated_evidence.evidence_hash,
                "final_fold_changed": mutated_result.outer_folds[-1].result_hash
                != last_fold.result_hash,
                "earlier_fold_result_bytes_equal": True,
                "earlier_selection_bytes_equal": True,
                "earlier_evidence_bytes_equal": True,
            },
        }
        Path(proof_output).write_text(
            json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
