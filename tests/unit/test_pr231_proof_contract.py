"""Fast, hermetic contract checks for the PR-231 proof helpers.

These tests deliberately exercise the proof assertions with tiny contract-shaped
objects.  They do not fit an HMM or run the full global-v4 evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest

from market_regime_engine.feature_discovery.contracts import canonical_json
from tests.fixtures.global_regime_v4.synthetic import canonical_snapshot_bytes


def _proof_module() -> Any:
    path = Path(__file__).parents[1] / "e2e" / "test_global_regime_v4_full_compute.py"
    spec = importlib.util.spec_from_file_location("pr231_full_compute_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PR-231 proof helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Value:
    value: str

    @property
    def candidate_id(self) -> str:
        return self.value


@dataclass(frozen=True)
class _Clusters:
    selected_count: int
    memberships: tuple[tuple[str, tuple[str, ...]], ...]
    candidate_memberships: tuple[tuple[int, tuple[tuple[str, tuple[str, ...]], ...]], ...]
    solution_hash: str


@dataclass(frozen=True)
class _Prototypes:
    prototypes: tuple[str, ...]
    mean_distances: tuple[float, ...]


@dataclass(frozen=True)
class _TeacherReference:
    candidate_id: str
    state_count: int
    prototype_features: tuple[str, ...]
    inner_plan_hash: str
    reference_hash: str


@dataclass(frozen=True)
class _TeacherEvaluation:
    candidate_aggregates: tuple[_Value, ...]
    selection: _Value


@dataclass(frozen=True)
class _FeatureScore:
    feature_name: str
    state_information_ratio: float
    eta_squared: float


@dataclass(frozen=True)
class _Winners:
    ranked_features: tuple[str, ...]


@dataclass(frozen=True)
class _Prefix:
    selected_prefix_length: int
    selected_candidate_id: str
    evaluations: tuple[_Value, ...]


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    feature_order: tuple[str, ...]
    state_count: int
    model_family: str


@dataclass(frozen=True)
class _Grid:
    evaluations: tuple[_Candidate, ...]
    aggregates: tuple[_Value, ...]


@dataclass(frozen=True)
class _FinalGrid:
    grid: _Grid
    selection: _Value


@dataclass(frozen=True)
class _Selection:
    clusters: _Clusters
    prototypes: _Prototypes
    teacher_reference: _TeacherReference
    teacher_evaluation: _TeacherEvaluation
    feature_scores: tuple[_FeatureScore, ...]
    winner_selection: _Winners
    prefix_search: _Prefix
    final_candidate: _Candidate
    final_grid: _FinalGrid


@dataclass(frozen=True)
class _Fold:
    fold_index: int
    outer_teacher_final_soft_nmi: float
    outer_shared_timestamp_count: int
    valid: bool
    failure_reason: str | None
    result_hash: str


@dataclass(frozen=True)
class _Result:
    outer_folds: tuple[_Fold, ...]
    valid_fold_count: int
    valid_fold_rate: float
    latest_complete_fold_valid: bool
    production_eligible: bool
    failure_reason: str | None
    policy_hash: str

    @property
    def result_hash(self) -> str:
        return hashlib.sha256(canonical_json(self)).hexdigest()


@dataclass(frozen=True)
class _RerunResult:
    result: str


@dataclass(frozen=True)
class _RerunEvidence:
    evidence: str

    def canonical_json(self) -> bytes:
        return canonical_json(self)

    @property
    def evidence_hash(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


def _golden_inputs() -> tuple[Any, _Result, dict[int, _Selection], Any]:
    proof = _proof_module()
    candidate_ids = proof.FINAL_CANDIDATE_IDS
    candidates = tuple(
        _Candidate(candidate_id, ("feature_a", "feature_b"), 2, "gaussian_hmm")
        for candidate_id in candidate_ids
    )
    selection = _Selection(
        clusters=_Clusters(
            selected_count=2,
            memberships=(("cluster_1", ("feature_a",)), ("cluster_2", ("feature_b",))),
            candidate_memberships=(
                (2, (("cluster_1", ("feature_a",)), ("cluster_2", ("feature_b",)))),
            ),
            solution_hash="cluster-solution",
        ),
        prototypes=_Prototypes(("feature_a", "feature_b"), (0.1, 0.2)),
        teacher_reference=_TeacherReference(
            "gaussian_hmm_k2_full",
            2,
            ("feature_a", "feature_b"),
            "inner-plan",
            "teacher-reference",
        ),
        teacher_evaluation=_TeacherEvaluation(
            tuple(
                _Value(candidate_id)
                for candidate_id in (
                    "gaussian_hmm_k2_full",
                    "gaussian_hmm_k3_full",
                    "gaussian_hmm_k4_full",
                    "gaussian_hmm_k5_full",
                )
            ),
            _Value("teacher-selection"),
        ),
        feature_scores=(_FeatureScore("feature_a", 0.8, 0.7),),
        winner_selection=_Winners(("feature_a", "feature_b")),
        prefix_search=_Prefix(2, "prefix-2", (_Value("prefix-2"),)),
        final_candidate=candidates[0],
        final_grid=_FinalGrid(
            _Grid(candidates, tuple(_Value(f"grid-{i}") for i in range(12))),
            _Value("grid-selection"),
        ),
    )
    fold = _Fold(1, 0.5, 2, True, None, "result-hash")
    result = _Result((fold,), 1, 1.0, True, False, None, "policy-hash")
    fixture = SimpleNamespace(
        rows=pd.DataFrame({"timestamp_m1": ["2020-01-01"], "feature_a": [1.0]}),
        catalog=SimpleNamespace(
            feature_names=("feature_a", "feature_b"), catalog_hash="catalog-hash"
        ),
        source_data_hash="source-hash",
    )
    evidence = SimpleNamespace(
        evidence={"stability": {"contract": True}}, evidence_hash="evidence-hash"
    )
    return fixture, result, {1260: selection}, evidence


def test_canonical_snapshot_bytes_are_stable_and_identity_sensitive() -> None:
    rows = pd.DataFrame({"timestamp_m1": ["2020-01-01", "2020-01-02"], "feature_a": [1.0, 2.0]})
    first = canonical_snapshot_bytes(rows)
    second = canonical_snapshot_bytes(rows.copy())
    changed = canonical_snapshot_bytes(rows.assign(feature_a=[1.0, 3.0]))

    assert first == second
    assert first.endswith(b"\n")
    assert hashlib.sha256(first).hexdigest() != hashlib.sha256(changed).hexdigest()


def test_golden_snapshot_assertion_accepts_and_cross_links_canonical_hashes() -> None:
    proof = _proof_module()
    fixture, result, selections, evidence = _golden_inputs()

    snapshot = proof._golden_snapshot(fixture, result, selections, evidence)

    assert proof.content_hash(snapshot) == proof.content_hash(snapshot)
    proof._assert_complete_golden_snapshot(snapshot, fixture, result, selections, evidence)


def test_golden_snapshot_rejects_mutated_cross_link() -> None:
    proof = _proof_module()
    fixture, result, selections, evidence = _golden_inputs()
    snapshot = proof._golden_snapshot(fixture, result, selections, evidence)
    snapshot["folds"][0]["selection_hash"] = "tampered"

    with pytest.raises(AssertionError):
        proof._assert_complete_golden_snapshot(snapshot, fixture, result, selections, evidence)


def test_canonical_rerun_requires_exact_result_and_evidence_bytes(tmp_path: Path) -> None:
    proof = _proof_module()
    result = _RerunResult("deterministic-result")
    evidence = _RerunEvidence("deterministic-evidence")
    result_path = tmp_path / "result.json"
    evidence_path = tmp_path / "evidence.json"
    result_path.write_bytes(canonical_json(result))
    evidence_path.write_bytes(evidence.canonical_json())

    result_bytes, evidence_bytes = proof._assert_canonical_process_rerun(
        result, evidence, result_path, evidence_path
    )

    assert result_bytes == canonical_json(result)
    assert evidence_bytes == evidence.canonical_json()

    result_path.write_bytes(json.dumps({"result": "deterministic-result"}).encode() + b"\n")
    with pytest.raises(AssertionError):
        proof._assert_canonical_process_rerun(result, evidence, result_path, evidence_path)


def test_selected_likelihood_contract_requires_independent_train_and_oos_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _proof_module()
    likelihoods = [
        {
            "fold_index": 1,
            "scope": scope,
            "model_family": "gaussian_hmm",
            "log_likelihood": -1.25 if scope == "TRAIN" else -0.75,
        }
        for scope in ("TRAIN", "OOS")
    ]
    monkeypatch.setattr(
        proof,
        "build_math_expectations",
        lambda *_args, **_kwargs: {
            "fold_audits": [
                {"final_candidate_id": "gaussian_hmm_k2_full", "likelihoods": likelihoods}
            ]
        },
    )
    monkeypatch.setattr(
        proof,
        "_independent_math_verifier",
        lambda: SimpleNamespace(independent_hmm_log_likelihood=lambda item: item["log_likelihood"]),
    )

    count = proof._assert_selected_likelihood_parity(
        SimpleNamespace(rows=pd.DataFrame()),
        SimpleNamespace(outer_folds=(SimpleNamespace(fold_index=1),)),
        {1260: object()},
    )

    assert count == 2


def test_selected_likelihood_contract_rejects_missing_oos_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _proof_module()
    likelihoods = [
        {
            "fold_index": 1,
            "scope": "TRAIN",
            "model_family": "gaussian_hmm",
            "log_likelihood": -1.25,
        }
    ]
    monkeypatch.setattr(
        proof,
        "build_math_expectations",
        lambda *_args, **_kwargs: {
            "fold_audits": [
                {"final_candidate_id": "gaussian_hmm_k2_full", "likelihoods": likelihoods}
            ]
        },
    )

    with pytest.raises(AssertionError):
        proof._assert_selected_likelihood_parity(
            SimpleNamespace(rows=pd.DataFrame()),
            SimpleNamespace(outer_folds=(SimpleNamespace(fold_index=1),)),
            {1260: object()},
        )
