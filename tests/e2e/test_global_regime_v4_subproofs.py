"""Independently startable local sub-proofs for the global v4 computation."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from market_regime_engine.evaluation_statistics.contracts import GlobalV4Evidence
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    FeatureRegimeScore,
    canonical_json,
    content_hash,
)
from market_regime_engine.mlflow_support.evaluation_tracking import track_global_v4_evaluation
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.loader import load_profile
from tests.e2e.test_global_regime_v4_full_compute import (
    PR231_GOLDEN_SNAPSHOT_HASH,
    _evaluate_with_selection_capture,
    _evidence,
    _golden_snapshot,
    _independent_cluster_solution,
    _independent_distance,
    _independent_feature_score,
    _independent_process_worker,
    _per_fold_evidence_payload,
)
from tests.fixtures.global_regime_v4.synthetic import SyntheticGlobalV4, build_synthetic_global_v4

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_BASELINE_CACHE_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class _Baseline:
    fixture: SyntheticGlobalV4
    profile: ModelProfile
    result: AdaptiveEvaluationResult
    selections: dict[int, Any]
    evidence: GlobalV4Evidence
    cache_reused: bool


def _baseline() -> _Baseline:
    fixture = build_synthetic_global_v4()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    output = os.environ.get("PR231_SUBPROOF_OUTPUT")
    cache_path = (
        None
        if output is None
        else Path(output).parent
        / f"pr231-baseline-v{_BASELINE_CACHE_SCHEMA}-{fixture.source_data_hash}.pickle"
    )
    repository_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if cache_path is not None and cache_path.is_file():
        try:
            envelope = pickle.loads(cache_path.read_bytes())
        except (
            EOFError,
            OSError,
            pickle.PickleError,
            TypeError,
            ValueError,
            AttributeError,
            ImportError,
            ModuleNotFoundError,
        ):
            envelope = None
        if (
            isinstance(envelope, dict)
            and envelope.get("schema") == _BASELINE_CACHE_SCHEMA
            and envelope.get("repository_sha") == repository_sha
            and envelope.get("source_data_sha256") == fixture.source_data_hash
            and envelope.get("profile_hash") == profile.profile_hash
            and isinstance(envelope.get("state"), _Baseline)
        ):
            return _Baseline(
                fixture,
                profile,
                envelope["state"].result,
                envelope["state"].selections,
                envelope["state"].evidence,
                True,
            )
    result, selections = _evaluate_with_selection_capture(
        fixture.rows,
        fixture.catalog,
        profile,
    )
    evidence = _evidence(fixture, result, selections)
    state = _Baseline(fixture, profile, result, selections, evidence, False)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema": _BASELINE_CACHE_SCHEMA,
            "repository_sha": repository_sha,
            "source_data_sha256": fixture.source_data_hash,
            "profile_hash": profile.profile_hash,
            "state": state,
        }
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
        temporary.replace(cache_path)
    return state


def _write_sidecar(phase: str, payload: dict[str, object]) -> None:
    output = Path(os.environ["PR231_SUBPROOF_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"phase": phase, **payload}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_global_v4_subproof_pipeline_math() -> None:
    """Prove full computation, independent math, and the fixed golden digest."""

    state = _baseline()
    first = state.selections[1260]
    first_train = state.fixture.rows.iloc[:1260]
    independent_distance = _independent_distance(first_train, first.distance.feature_order)
    assert np.asarray(first.distance.distances) == pytest.approx(
        independent_distance,
        abs=1.0e-10,
    )
    _independent_cluster_solution(first.distance, first.clusters)
    for score in first.feature_scores:
        expected_sir, expected_eta = _independent_feature_score(
            first_train,
            score,
            first.teacher_reference,
        )
        assert isinstance(score, FeatureRegimeScore)
        assert score.state_information_ratio == pytest.approx(expected_sir, abs=1.0e-10)
        assert score.eta_squared == pytest.approx(expected_eta, abs=1.0e-10)

    golden = _golden_snapshot(state.fixture, state.result, state.selections, state.evidence)
    assert content_hash(golden) == PR231_GOLDEN_SNAPSHOT_HASH
    _write_sidecar(
        "pipeline-math",
        {
            "result_hash": state.result.result_hash,
            "evidence_hash": state.evidence.evidence_hash,
            "golden_snapshot_hash": content_hash(golden),
            "baseline_cache_reused": state.cache_reused,
        },
    )


def test_global_v4_subproof_tracking_and_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove local MLflow Model Metrics and plot-manifest generation."""

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    state = _baseline()
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
    tracked = track_global_v4_evaluation(
        FileMlflowTrackingPort(tracking_uri, experiment_name="global-v4-subproof-tracking"),
        StatisticsWriter(tmp_path / "statistics"),
        evidence=state.evidence,
        result=state.result,
        selections={
            fold.fold_index: state.selections[1260 + (fold.fold_index - 1) * 63]
            for fold in state.result.outer_folds
        },
    )
    assert tracked.global_evidence_hash == state.evidence.evidence_hash
    assert Path(tracked.plot_manifest_path).is_file()
    _write_sidecar(
        "tracking-and-plots",
        {
            "result_hash": state.result.result_hash,
            "evidence_hash": state.evidence.evidence_hash,
            "plot_manifest_path": tracked.plot_manifest_path,
            "baseline_cache_reused": state.cache_reused,
        },
    )


def test_global_v4_subproof_independent_process_and_labels(tmp_path: Path) -> None:
    """Prove a spawned process and randomized semantic labels preserve bytes."""

    state = _baseline()
    snapshot_path = tmp_path / "pinned-snapshot.csv"
    snapshot_bytes = state.fixture.rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    snapshot_path.write_bytes(snapshot_bytes)
    labels = {
        feature: f"randomized_{index}"
        for index, feature in enumerate(reversed(tuple(state.fixture.semantic_labels)))
    }
    labels_path = tmp_path / "randomized-labels.json"
    labels_path.write_text(json.dumps(labels, sort_keys=True) + "\n", encoding="utf-8")
    result_path = tmp_path / "independent-result.json"
    evidence_path = tmp_path / "independent-evidence.json"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_independent_process_worker,
        args=(str(snapshot_path), str(labels_path), str(result_path), str(evidence_path)),
    )
    process.start()
    process.join(timeout=3_600)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("independent process sub-proof exceeded one hour")
    assert process.exitcode == 0
    result_bytes = result_path.read_bytes()
    evidence_bytes = evidence_path.read_bytes()
    assert result_bytes == canonical_json(state.result)
    assert evidence_bytes == state.evidence.canonical_json()
    _write_sidecar(
        "independent-process-and-labels",
        {
            "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
            "canonical_result_bytes_equal": True,
            "canonical_evidence_bytes_equal": True,
            "baseline_cache_reused": state.cache_reused,
        },
    )


def test_global_v4_subproof_future_mutation_isolation() -> None:
    """Prove future TEST input changes do not rewrite earlier fold evidence."""

    state = _baseline()
    last_fold = state.result.outer_folds[-1]
    mutated_feature = last_fold.final_configuration.feature_order[0]
    last_test_start = 1260 + (last_fold.fold_index - 1) * 63
    mutation_index = next(
        index
        for index in range(last_test_start, len(state.fixture.rows))
        if pd.notna(state.fixture.rows.iloc[index][mutated_feature])
        and np.isfinite(float(state.fixture.rows.iloc[index][mutated_feature]))
    )
    mutated_rows = state.fixture.rows.copy()
    mutated_rows.loc[mutation_index, mutated_feature] += 10_000.0
    mutated_snapshot_bytes = mutated_rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    mutated_fixture = SyntheticGlobalV4(
        mutated_rows,
        state.fixture.catalog,
        dict(state.fixture.semantic_labels),
        hashlib.sha256(mutated_snapshot_bytes).hexdigest(),
    )
    mutated_result, mutated_selections = _evaluate_with_selection_capture(
        mutated_fixture.rows,
        mutated_fixture.catalog,
        state.profile,
    )
    mutated_evidence = _evidence(mutated_fixture, mutated_result, mutated_selections)
    assert mutated_result.result_hash != state.result.result_hash
    assert mutated_result.outer_folds[-1].result_hash != last_fold.result_hash
    for index, fold in enumerate(state.result.outer_folds[:-1]):
        assert canonical_json(mutated_result.outer_folds[index]) == canonical_json(fold)
        train_count = 1260 + index * 63
        assert canonical_json(mutated_selections[train_count]) == canonical_json(
            state.selections[train_count]
        )
        assert canonical_json(
            _per_fold_evidence_payload(mutated_evidence, index)
        ) == canonical_json(_per_fold_evidence_payload(state.evidence, index))
    _write_sidecar(
        "future-mutation-isolation",
        {
            "mutation_row_index": mutation_index,
            "mutation_feature": mutated_feature,
            "baseline_result_hash": state.result.result_hash,
            "mutated_result_hash": mutated_result.result_hash,
            "final_fold_changed": True,
            "earlier_fold_result_bytes_equal": True,
            "earlier_selection_bytes_equal": True,
            "earlier_evidence_bytes_equal": True,
            "baseline_cache_reused": state.cache_reused,
        },
    )
