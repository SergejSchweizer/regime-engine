from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.evaluations.state_classification_metrics import (
    ClassificationContract,
    build_state_classification_evidence,
)
from market_regime_engine.mlflow_support.classification_projection import (
    classification_model_metric_points,
)


def _contract() -> ClassificationContract:
    return ClassificationContract("macro-regime-v1", ("bear", "bull"))


def test_classification_metrics_are_permutation_invariant_and_explicit() -> None:
    timestamps = tuple(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(4))
    evidence = build_state_classification_evidence(
        timestamps=timestamps,
        filtered_probabilities=((0.9, 0.1), (0.8, 0.2), (0.2, 0.8), (0.1, 0.9)),
        labels=("bear", "bear", "bull", "bull"),
        contract=_contract(),
    )
    permuted = build_state_classification_evidence(
        timestamps=timestamps,
        filtered_probabilities=((0.1, 0.9), (0.2, 0.8), (0.8, 0.2), (0.9, 0.1)),
        labels=("bear", "bear", "bull", "bull"),
        contract=_contract(),
    )
    assert evidence.hard_ari == pytest.approx(1.0)
    assert evidence.hard_nmi == pytest.approx(1.0)
    assert evidence.hard_accuracy == pytest.approx(1.0)
    assert evidence.hard_purity == pytest.approx(1.0)
    assert evidence.soft_nmi == pytest.approx(permuted.soft_nmi)
    assert evidence.confusion_matrix != permuted.confusion_matrix
    assert classification_model_metric_points(evidence)


def test_no_labels_is_not_available_without_numeric_placeholders() -> None:
    timestamps = (datetime(2026, 1, 1, tzinfo=UTC),)
    evidence = build_state_classification_evidence(
        timestamps=timestamps,
        filtered_probabilities=((1.0, 0.0),),
        labels=None,
        contract=_contract(),
    )
    assert evidence.status == "not_available"
    assert evidence.metric_points == ()
    assert evidence.unavailable_reason is not None


def test_unknown_labels_follow_declared_policy() -> None:
    timestamps = tuple(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(2))
    evidence = build_state_classification_evidence(
        timestamps=timestamps,
        filtered_probabilities=((1.0, 0.0), (0.0, 1.0)),
        labels=("unknown", "bull"),
        contract=_contract(),
    )
    assert evidence.status == "available"
    assert evidence.omitted_unknown_count == 1
    strict = ClassificationContract(
        "macro-regime-v1", ("bear", "bull"), unknown_label_policy="fail_closed"
    )
    with pytest.raises(ValueError, match="unknown classification label"):
        build_state_classification_evidence(
            timestamps=timestamps,
            filtered_probabilities=((1.0, 0.0), (0.0, 1.0)),
            labels=("unknown", "bull"),
            contract=strict,
        )
