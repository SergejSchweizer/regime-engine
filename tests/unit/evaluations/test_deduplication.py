from __future__ import annotations

from market_regime_engine.evaluations.deduplication import EvaluationClaim, EvaluationDeduplicator


def test_evaluation_claim_is_deduplicated_by_code_and_dataset_hash(tmp_path) -> None:
    deduplicator = EvaluationDeduplicator(tmp_path, "a" * 40, "b" * 64)
    assert deduplicator.claim() is EvaluationClaim.CLAIMED
    assert deduplicator.claim() is EvaluationClaim.RUNNING
    deduplicator.complete()
    assert deduplicator.claim() is EvaluationClaim.COMPLETED


def test_failed_evaluation_releases_its_claim(tmp_path) -> None:
    deduplicator = EvaluationDeduplicator(tmp_path, "a" * 40, "b" * 64)
    assert deduplicator.claim() is EvaluationClaim.CLAIMED
    deduplicator.abort()
    assert deduplicator.claim() is EvaluationClaim.CLAIMED
