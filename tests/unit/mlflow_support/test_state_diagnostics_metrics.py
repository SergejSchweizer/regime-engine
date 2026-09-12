from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from market_regime_engine.mlflow_support.state_diagnostics_metrics import (
    build_state_diagnostic_evidence,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.states.alignment import align_first_fold


def _artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("a", "b"),
        start_probabilities=(0.6, 0.4),
        transition_matrix=((0.8, 0.2), (0.1, 0.9)),
        means=((0.0, 0.0), (1.0, 1.0)),
        full_covariances=(
            ((1.0, 0.1), (0.1, 1.2)),
            ((1.5, 0.0), (0.0, 1.1)),
        ),
    )


def test_state_diagnostics_align_histories_and_emit_complete_metric_families() -> None:
    artifact = _artifact()
    alignment = align_first_fold(artifact)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=index) for index in range(4))
    raw = np.asarray(((0.9, 0.1), (0.8, 0.2), (0.2, 0.8), (0.1, 0.9)))

    evidence = build_state_diagnostic_evidence(
        fold_id="fold_001",
        artifact=artifact,
        alignment=alignment,
        oos_timestamps=timestamps,
        oos_filtered_probabilities=raw,
        train_hard_occupancy=(0.5, 0.5),
        train_soft_occupancy=(0.55, 0.45),
    )
    points = evidence.metric_points()
    identities = {(point.key, point.step) for point in points}

    assert len(identities) == len(points)
    assert evidence.viterbi_states == (0, 0, 1, 1)
    assert evidence.oos_hard_occupancy == (0.5, 0.5)
    assert evidence.oos_soft_occupancy == pytest.approx((0.5, 0.5))
    assert evidence.expected_duration == (2.0, 2.0)
    assert evidence.transition_matrix == ((0.8, 0.2), (0.1, 0.9))
    assert evidence.mapping_artifact["state_identity_scope"] == "model_version_local"
    assert len(evidence.mapping_artifact_sha256) == 64
    assert any(point.key == "state_diag_posterior_probability_state_0" for point in points)
    assert any(point.key == "state_diag_covariance_eigenvalue_state_0_0" for point in points)


def test_state_diagnostics_recomputes_occupancy_from_posterior_rows() -> None:
    artifact = _artifact()
    evidence = build_state_diagnostic_evidence(
        fold_id="fold_001",
        artifact=artifact,
        alignment=align_first_fold(artifact),
        oos_timestamps=(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
        oos_filtered_probabilities=((0.75, 0.25), (0.75, 0.25)),
        train_hard_occupancy=(1.0, 0.0),
        train_soft_occupancy=(0.75, 0.25),
        oos_soft_occupancy=(0.5, 0.5),
    )
    assert evidence.oos_soft_occupancy == pytest.approx((0.75, 0.25))


def test_state_diagnostics_reject_non_normalized_posterior() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="sum to one"):
        build_state_diagnostic_evidence(
            fold_id="fold_001",
            artifact=artifact,
            alignment=align_first_fold(artifact),
            oos_timestamps=(datetime(2026, 1, 1, tzinfo=UTC),),
            oos_filtered_probabilities=((0.8, 0.3),),
            train_hard_occupancy=(1.0, 0.0),
            train_soft_occupancy=(0.8, 0.2),
        )
