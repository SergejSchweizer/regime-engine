from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from market_regime_engine.evaluation_statistics import (
    GLOBAL_V4_EVALUATION_ID,
    GlobalV4Evidence,
    RunStatistics,
    RunType,
    StatisticsWriter,
    Status,
)
from market_regime_engine.evaluation_statistics.render import render_statistics

HASH = "a" * 64
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def evidence_groups() -> dict[str, object]:
    return {
        "identity": {"policy_id": "xetra_global_regime_v4"},
        "lineage": {
            "source_build_id": "build-1",
            "source_data_hash": HASH,
            "catalog_hash": HASH,
            "profile_hash": HASH,
            "repository_hash": HASH,
            "outer_plan_hash": HASH,
        },
        "input": {"feature_order": ["f0", "f1"], "outer_plan": "1260/63/63"},
        "quality": {"eligible_features": ["f0", "f1", "f2"]},
        "distance": {"matrix_hash": HASH, "minimum_pairwise_observations": 504},
        "clustering": {"selected_m": 3, "silhouette_curve": [[2, 0.1], [3, 0.2]]},
        "prototypes": {"features": ["f0", "f1", "f2"]},
        "teacher": {"candidate_id": "gaussian_hmm_k2_full", "shared_support": 63},
        "feature_scores": {"scores": [{"feature": "f0", "eta_squared": 0.2}]},
        "prefix_search": {"selected_l": 2, "nmi_by_l": [[2, 0.5]]},
        "final_grid": {"candidate_ids": ["gaussian_hmm_k2_full"]},
        "outer_folds": [{"fold_id": "fold_001", "valid": True}],
        "agreement": {"soft_nmi": [0.5]},
        "validity": {"valid_fold_rate": 1.0, "production_eligible": True},
        "stability": {"adjacent_cluster_stability": [1.0]},
        "deployment_selection": {"deployment_cutoff": NOW.isoformat()},
    }


def global_evidence() -> GlobalV4Evidence:
    return GlobalV4Evidence(
        source_build_id="build-1",
        source_data_hash=HASH,
        catalog_hash=HASH,
        profile_hash=HASH,
        repository_hash=HASH,
        outer_plan_hash=HASH,
        evidence=evidence_groups(),
    )


def test_global_v4_evidence_is_exactly_reproducible_and_finite() -> None:
    first = global_evidence()
    second = GlobalV4Evidence(
        source_build_id=first.source_build_id,
        source_data_hash=first.source_data_hash,
        catalog_hash=first.catalog_hash,
        profile_hash=first.profile_hash,
        repository_hash=first.repository_hash,
        outer_plan_hash=first.outer_plan_hash,
        evidence=dict(reversed(tuple(first.evidence.items()))),
    )

    assert first.canonical_json() == second.canonical_json()
    assert first.evidence_hash == sha256(first.canonical_json()).hexdigest()
    assert b"source_rows" not in first.canonical_json()
    assert b"model_binary" not in first.canonical_json()


def test_global_v4_evidence_requires_complete_known_groups_and_safe_values() -> None:
    complete = evidence_groups()
    complete.pop("stability")
    with pytest.raises(ValueError, match="missing global v4 evidence groups"):
        GlobalV4Evidence("build-1", HASH, HASH, HASH, HASH, HASH, complete)

    complete = evidence_groups()
    complete["unknown"] = {}
    with pytest.raises(ValueError, match="unknown global v4 evidence groups"):
        GlobalV4Evidence("build-1", HASH, HASH, HASH, HASH, HASH, complete)

    complete = evidence_groups()
    complete["quality"] = {"score": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        GlobalV4Evidence("build-1", HASH, HASH, HASH, HASH, HASH, complete)

    complete = evidence_groups()
    complete["input"] = {"source_rows": [1, 2, 3]}
    with pytest.raises(ValueError, match="forbidden"):
        GlobalV4Evidence("build-1", HASH, HASH, HASH, HASH, HASH, complete)


def test_global_v4_evidence_can_be_written_as_immutable_statistics_dossier(tmp_path: Path) -> None:
    payload = global_evidence()
    running = RunStatistics(
        evaluation_id=GLOBAL_V4_EVALUATION_ID,
        mlflow_run_id="run-v4",
        run_type=RunType.PARENT,
        run_name="global-v4",
        status=Status.RUNNING,
        started_at=NOW,
        evidence=payload.evidence,
    )
    writer = StatisticsWriter(tmp_path)
    directory = writer.start(running)
    final = RunStatistics(
        evaluation_id=GLOBAL_V4_EVALUATION_ID,
        mlflow_run_id="run-v4",
        run_type=RunType.PARENT,
        run_name="global-v4",
        status=Status.FINISHED,
        started_at=NOW,
        ended_at=NOW,
        evidence=payload.evidence,
    )
    digest = writer.finalize(final)

    assert directory == tmp_path / "evaluations" / GLOBAL_V4_EVALUATION_ID / "run-v4"
    assert digest == sha256((directory / "statistics.json").read_bytes()).hexdigest()
    assert '"evaluation_id":"global_regime_v4"' in (directory / "statistics.json").read_text(
        encoding="utf-8"
    )


def test_global_v4_markdown_renders_evidence_and_recomputable_formulas() -> None:
    statistics = RunStatistics(
        evaluation_id=GLOBAL_V4_EVALUATION_ID,
        mlflow_run_id="run-v4",
        run_type=RunType.PARENT,
        run_name="global-v4",
        status=Status.FINISHED,
        started_at=NOW,
        ended_at=NOW,
        evidence=global_evidence().evidence,
    )

    rendered = render_statistics(statistics)

    assert "## Evidence" in rendered
    assert "quality" in rendered
    assert "selected_l" in rendered
    assert "d_ij = 1 - |rho_ij|" in rendered
    assert "SIR = I(X; Z) / H(Z)" in rendered
    assert "exact shared timestamps" in rendered
