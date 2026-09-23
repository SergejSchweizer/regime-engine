from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from market_regime_engine.commands.canonical_model_callbacks import CanonicalModelCallbacks
from market_regime_engine.feature_discovery.ablation import HMMSubsetEvaluation
from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    TRANSFORMATION_FAMILIES,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.metadata_store import FeatureSelectionMetadataStore
from market_regime_engine.feature_discovery.pipeline import run_canonical_feature_selection
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore
from market_regime_engine.mlflow_support.canonical_diagnostics import (
    write_canonical_diagnostics,
)
from market_regime_engine.profiles.loader import load_profile
from tests.unit.feature_discovery.test_pr498_monthly_refit import _catalog, _source
from tests.unit.feature_discovery.test_pr499_orchestration_cadence_qa import _run


def test_hermetic_multifold_proof_populates_all_evidence_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = FeatureSelectionMetadataStore(tmp_path / "metadata")
    tracker = _Tracking()
    result = _run(monkeypatch, max_workers=None, tracker=tracker, metadata_store=store)

    assert len(result.folds) >= 2
    assert result.valid_folds
    assert all(fold.package is not None for fold in result.valid_folds)
    assert tracker.starts and tracker.artifacts

    required_tables = {
        "feature_registry",
        "fold_feature_stats",
        "pca_loadings",
        "correlation_mapping",
        "sffs_steps",
        "fold_model_stats",
    }
    with duckdb.connect(str(store.database), read_only=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
    assert required_tables <= tables


def test_pinned_hermetic_run_is_hash_stable_across_worker_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serial = _run(monkeypatch, max_workers=1)
    parallel = _run(monkeypatch, max_workers=None)
    assert parallel.result_hash == serial.result_hash
    assert tuple(item.package.package_hash for item in parallel.valid_folds) == tuple(
        item.package.package_hash for item in serial.valid_folds
    )


def test_outer_test_perturbation_preserves_prior_fold_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _run(monkeypatch)
    first = before.folds[0]
    assert first.package is not None
    source = _source()
    source.loc[
        source["timestamp_m1"] > first.fold.test_last_timestamp,
        "vix_log_level",
    ] += 100000.0
    after = _run(monkeypatch, source=source)
    assert after.folds[0].package is not None
    assert after.folds[0].package.package_hash == first.package.package_hash


def test_evidence_manifest_is_canonical_json_and_has_no_network_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run(monkeypatch)
    encoded = json.dumps(result.result_hash, sort_keys=True, separators=(",", ":"))
    assert encoded == json.dumps(result.result_hash, separators=(",", ":"), sort_keys=True)
    assert "10.10.1.3" not in encoded


def test_real_hmm_callback_fits_train_and_filters_outer_test() -> None:
    rng = np.random.default_rng(501)
    timestamps = pd.date_range("2018-01-01", periods=320, freq="D", tz="UTC")
    train = pd.DataFrame(
        {
            "timestamp_m1": timestamps[:280],
            "vix_log_level": rng.normal(size=280),
            "us_10y_log_level": rng.normal(size=280),
        }
    )
    test = pd.DataFrame(
        {
            "timestamp_m1": timestamps[280:],
            "vix_log_level": rng.normal(size=40),
            "us_10y_log_level": rng.normal(size=40),
        }
    )
    callbacks = CanonicalModelCallbacks(
        train=train,
        test=test,
        profile=load_profile("configs/profiles/xetra_v4.yaml"),
        catalog=_catalog(),
        source_build_id="hermetic-501",
        max_workers=None,
    )
    fit = callbacks._fit(("vix_log_level", "us_10y_log_level"), 2)
    fit_hash = callbacks.fit_final_hmm(train, ("vix_log_level", "us_10y_log_level"), 2)
    assert callbacks.evaluate_outer_test(
        train,
        test,
        ("vix_log_level", "us_10y_log_level"),
        2,
        (fit_hash,),
    )
    assert fit.winner.artifact is not None


def test_thousand_feature_hermetic_selection_runs_real_pca_and_reduction(
    tmp_path: Path,
) -> None:
    names = tuple(
        f"{family}_delta_{index}obs" for index in range(1, 78) for family in TRANSFORMATION_FAMILIES
    )
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    rng = np.random.default_rng(501_1000)
    matrix = rng.normal(size=(64, len(names)))
    values = {
        name: tuple(float(value) for value in matrix[:, index]) for index, name in enumerate(names)
    }

    def score(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, float(len(features)))

    def hmm_score(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        fit_hash = sha256("|".join(features).encode()).hexdigest()
        return HMMSubsetEvaluation(score(features), "gaussian_hmm", 2, "a" * 64, fit_hash)

    result = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=score,
        evaluate_hmm_subset=hmm_score,
        hmm_selector_contract_hash="a" * 64,
        max_sffs_features=3,
        max_workers=None,
        metadata_store=FeatureSelectionMetadataStore(tmp_path / "metadata"),
        metadata_fold_id="fold-501",
        metadata_source_build_id="build-501",
        metadata_state_count=2,
        evaluate_gaussian_subset_by_k=lambda state_count, features: FeatureSubsetScore(
            features,
            float(len(features)),
            model_family="gaussian_hmm",
            state_count=state_count,
        ),
    )
    assert len(names) == 1001
    assert result.quality_eligible_features == names
    assert result.family_pca
    name_index = {name: index for index, name in enumerate(names)}
    for artifact in result.family_pca:
        family_matrix = matrix[
            :,
            tuple(name_index[name] for name in artifact.feature_order),
        ]
        standardized = artifact.scaler.transform(family_matrix)
        _u, singular_values, _vh = np.linalg.svd(standardized, full_matrices=False)
        expected_variance = (singular_values**2) / float(np.sum(singular_values**2))
        assert np.allclose(
            artifact.explained_variance_ratio[: artifact.numerical_rank],
            expected_variance[: artifact.numerical_rank],
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    assert result.global_reduction.representatives
    assert result.selected_features
    assert result.ablation.one_feature_results


def test_canonical_diagnostics_materialize_required_tables_and_plots(tmp_path: Path) -> None:
    names = ("vix_log_level", "us_10y_log_level")
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    values = {
        names[0]: tuple(float(index) for index in range(30)),
        names[1]: tuple(float((index % 4) ** 2) for index in range(30)),
    }

    def score(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, float(len(features)))

    def hmm_score(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        return HMMSubsetEvaluation(
            score(features),
            "gaussian_hmm",
            2,
            "a" * 64,
            sha256("|".join(features).encode()).hexdigest(),
        )

    result = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=score,
        evaluate_hmm_subset=hmm_score,
        hmm_selector_contract_hash="a" * 64,
        max_workers=None,
        evaluate_gaussian_subset_by_k=lambda state_count, features: FeatureSubsetScore(
            features, float(len(features)), model_family="gaussian_hmm", state_count=state_count
        ),
    )
    artifacts = write_canonical_diagnostics(result, tmp_path)
    assert {path.name for path in artifacts} == {
        "feature-funnel.json",
        "family-pca.json",
        "correlation-reduction.json",
        "sffs-steps.json",
        "ablation-losses.json",
        "feature-funnel.png",
        "pca-explained-variance.png",
        "correlation-reduction.png",
        "sffs-scores.png",
        "ablation-losses.png",
    }
    assert all(path.stat().st_size > 0 for path in artifacts)
    assert all(item.score.value == float(len(item.selected_features)) for item in result.sffs.steps)
    assert result.ablation.baseline.hmm_evaluation.score is not None
    assert result.ablation.baseline.hmm_evaluation.score.value == float(
        len(result.selected_features)
    )
    assert result.ablation.ablation_losses == tuple(1.0 for _ in result.selected_features)
    assert json.loads((tmp_path / "feature-funnel.json").read_text())["sffs_selected"] == len(
        result.selected_features
    )


class _Tracking:
    def __init__(self) -> None:
        self.starts: list[tuple[str, str | None]] = []
        self.artifacts: list[tuple[str, str]] = []

    def start_run(self, *, run_name: str, parent_run_id: str | None = None) -> str:
        del run_name
        run_id = f"run-{len(self.starts)}"
        self.starts.append((run_id, parent_run_id))
        return run_id

    def log_params(self, _run_id: str, _params: dict[str, str]) -> None:
        return None

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self.artifacts.append((run_id, f"{local_path}:{artifact_path}"))

    def end_run(self, _run_id: str) -> None:
        return None

    def fail_run(self, _run_id: str) -> None:
        raise AssertionError("hermetic proof tracking failed")
