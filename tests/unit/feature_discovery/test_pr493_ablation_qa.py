from __future__ import annotations

import json
import time
from dataclasses import asdict
from hashlib import sha256

import pytest

from market_regime_engine.feature_discovery.ablation import (
    AblationResult,
    HMMSubsetEvaluation,
    run_one_feature_hmm_ablation,
)
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore

SELECTOR_HASH = "a" * 64
PLAN_HASH = "b" * 64
SEED_HASH = "c" * 64


def _oracle_evaluation(features: tuple[str, ...]) -> HMMSubsetEvaluation:
    values = {
        ("a", "b", "c", "d"): 10.0,
        ("b", "c", "d"): 9.0,
        ("a", "c", "d"): 10.0,
        ("a", "b", "d"): 11.0,
        ("a", "b", "c"): 12.0,
    }
    return HMMSubsetEvaluation(
        FeatureSubsetScore(features, values[features]),
        "gaussian_hmm",
        3,
        SELECTOR_HASH,
        sha256("/".join(features).encode()).hexdigest(),
        PLAN_HASH,
        SEED_HASH,
    )


def _delayed_oracle_evaluation(features: tuple[str, ...]) -> HMMSubsetEvaluation:
    # Deliberately reverse completion order while preserving the same immutable evidence.
    time.sleep((ord(features[0]) - ord("a")) * 0.01)
    return _oracle_evaluation(features)


def _canonical_hash(result: AblationResult) -> str:
    encoded = json.dumps(asdict(result), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def test_independent_oracle_reproduces_exactly_n_signed_losses() -> None:
    selected = ("a", "b", "c", "d")
    result = run_one_feature_hmm_ablation(
        selected, _oracle_evaluation, selector_contract_hash=SELECTOR_HASH, max_workers=1
    )
    assert len(result.one_feature_results) == len(selected)
    assert tuple(item.ablation_loss for item in result.one_feature_results) == (
        1.0,
        0.0,
        -1.0,
        -2.0,
    )
    baseline = _oracle_evaluation(selected).score
    assert baseline is not None
    for observation in result.one_feature_results:
        removed = _oracle_evaluation(observation.remaining_features).score
        assert removed is not None
        assert observation.ablation_loss == baseline.value - removed.value


def test_reversed_completion_order_preserves_rows_and_hashes() -> None:
    selected = ("a", "b", "c", "d")
    serial = run_one_feature_hmm_ablation(
        selected, _oracle_evaluation, selector_contract_hash=SELECTOR_HASH, max_workers=1
    )
    parallel = run_one_feature_hmm_ablation(
        selected,
        _delayed_oracle_evaluation,
        selector_contract_hash=SELECTOR_HASH,
        max_workers=8,
    )
    assert parallel == serial
    assert _canonical_hash(parallel) == _canonical_hash(serial)


@pytest.mark.parametrize("worker_count", [1, 8, 32, 64, None])
def test_worker_counts_preserve_ablation_tuple_and_hash(worker_count: int | None) -> None:
    selected = ("a", "b", "c", "d")
    reference = run_one_feature_hmm_ablation(
        selected, _oracle_evaluation, selector_contract_hash=SELECTOR_HASH, max_workers=1
    )
    result = run_one_feature_hmm_ablation(
        selected,
        _delayed_oracle_evaluation,
        selector_contract_hash=SELECTOR_HASH,
        max_workers=worker_count,
    )
    assert result == reference
    assert _canonical_hash(result) == _canonical_hash(reference)


def test_outer_test_mutation_cannot_change_ablation_results() -> None:
    outer_test = {"target": 100.0}
    before = run_one_feature_hmm_ablation(
        ("a", "b", "c", "d"),
        _oracle_evaluation,
        selector_contract_hash=SELECTOR_HASH,
        max_workers=1,
    )
    outer_test["target"] = -100.0
    after = run_one_feature_hmm_ablation(
        ("a", "b", "c", "d"),
        _oracle_evaluation,
        selector_contract_hash=SELECTOR_HASH,
        max_workers=1,
    )
    assert before == after
