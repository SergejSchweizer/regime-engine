"""Hermetic full-computation and independent-math proof for global v4."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from math import fsum, log
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples

import market_regime_engine.evaluations.global_regime_v4 as global_v4
import market_regime_engine.feature_discovery.prefix_search as prefix_search
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation.walk_forward import run_walk_forward_candidate
from market_regime_engine.evaluation_statistics.contracts import GlobalV4Evidence
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.teacher_reference import refit_frozen_teacher
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    ClusterSolution,
    FeatureRegimeScore,
    ProvisionalTeacherReference,
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
)

pytestmark = pytest.mark.integration


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
        "stability": {"adjacent_cluster_stability": []},
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_synthetic_global_v4()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    captured: dict[int, object] = {}
    captured_prefix_evaluations: dict[tuple[int, int], dict[str, object]] = {}
    original_selection = global_v4.select_v4_configuration
    original_prefix_evaluate_candidates = prefix_search._evaluate_candidates

    def parallel_real_runner(frame, plan, shared_profile, candidate, candidate_adapter_factory):
        return run_walk_forward_candidate(
            frame,
            plan=plan,
            profile=shared_profile,
            candidate=candidate,
            adapter_factory=candidate_adapter_factory,
            max_workers=2,
        )

    def capture_selection(train_rows: pd.DataFrame, **kwargs: object) -> object:
        selection = original_selection(
            train_rows,
            teacher_runner=parallel_real_runner,
            prefix_runner=parallel_real_runner,
            grid_runner=parallel_real_runner,
            **kwargs,
        )
        captured[len(train_rows)] = selection
        return selection

    def capture_prefix_evaluations(
        source_rows,
        plan,
        shared_profile,
        candidates,
        runner,
        max_workers,
    ):
        evaluations = original_prefix_evaluate_candidates(
            source_rows,
            plan,
            shared_profile,
            candidates,
            runner,
            max_workers,
        )
        captured_prefix_evaluations[(len(source_rows), len(candidates[0].feature_order))] = (
            evaluations
        )
        return evaluations

    monkeypatch.setattr(global_v4, "select_v4_configuration", capture_selection)
    monkeypatch.setattr(prefix_search, "_evaluate_candidates", capture_prefix_evaluations)
    result = global_v4.evaluate_global_regime_v4(
        fixture.rows,
        catalog=fixture.catalog,
        profile=profile,
        outer_runner=parallel_real_runner,
        max_workers=4,
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
    assert any(
        not prefix.valid
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

    evidence = _evidence(fixture, result, captured)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    tracked = track_global_v4_evaluation(
        FileMlflowTrackingPort(tracking_uri, experiment_name="global-v4-full-proof"),
        StatisticsWriter(tmp_path / "statistics"),
        evidence=evidence,
        result=result,
        selections=captured,
    )
    assert tracked.global_evidence_hash == evidence.evidence_hash
    assert Path(tracked.plot_manifest_path).is_file()

    canonical_result_hash = result.result_hash
    mutated = fixture.rows.copy()
    mutated.loc[0, fixture.catalog.feature_names[0]] += 10_000.0
    randomized_labels = dict(reversed(tuple(fixture.semantic_labels.items())))
    assert randomized_labels != fixture.semantic_labels
    assert result.result_hash == canonical_result_hash
    assert result.result_hash == content_hash(result)

    rerun_evidence = _evidence(fixture, result, captured)
    assert rerun_evidence.canonical_json() == evidence.canonical_json()
    assert rerun_evidence.evidence_hash == evidence.evidence_hash
