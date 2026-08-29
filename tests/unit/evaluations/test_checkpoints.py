from __future__ import annotations

from pathlib import Path

from market_regime_engine.evaluations.checkpoints import EvaluationCheckpointStore


def test_checkpoint_reuses_only_identical_model_scope(tmp_path: Path) -> None:
    store = EvaluationCheckpointStore(tmp_path, fingerprint="code-and-data")
    calls = 0

    def compute() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"calls": calls}

    arguments = {
        "evaluation_id": "medoid_univariate",
        "feature_order": ("vix_level",),
        "candidate_id": "gaussian_hmm_k2_full",
    }
    assert store.load_or_compute(**arguments, compute=compute) == {"calls": 1}
    assert store.load_or_compute(**arguments, compute=compute) == {"calls": 1}
    assert store.load_or_compute(
        **(arguments | {"feature_order": ("move_level",)}), compute=compute
    ) == {"calls": 2}


def test_checkpoint_fingerprint_isolates_code_and_dataset_versions(tmp_path: Path) -> None:
    arguments = {
        "evaluation_id": "medoid_multivariate",
        "feature_order": ("vix_level",),
        "candidate_id": "gaussian_hmm_k2_full",
    }
    assert (
        EvaluationCheckpointStore(tmp_path, fingerprint="one").load_or_compute(
            **arguments, compute=lambda: "one"
        )
        == "one"
    )
    assert (
        EvaluationCheckpointStore(tmp_path, fingerprint="two").load_or_compute(
            **arguments, compute=lambda: "two"
        )
        == "two"
    )
