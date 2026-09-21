from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from mlflow.tracking import MlflowClient

from market_regime_engine.feature_discovery.ablation import HMMSubsetEvaluation
from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.pipeline import run_canonical_feature_selection
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore
from market_regime_engine.mlflow_support.feature_selection_evidence import (
    log_feature_selection_evidence,
    render_feature_selection_evidence,
    verify_feature_selection_evidence_bundle,
)
from market_regime_engine.mlflow_support.ports import TrackingPort
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort


def test_feature_selection_evidence_bundle_is_complete_and_hash_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    names = ("vix_log_level", "us_10y_log_level", "vix_delta_1obs")
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    rows = tuple(float(index) for index in range(30))
    values = {
        names[0]: rows,
        names[1]: tuple(float((index % 7) ** 2) for index in range(30)),
        names[2]: rows,
    }

    def evaluate(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, float(len(features)))

    def evaluate_hmm(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        return HMMSubsetEvaluation(
            FeatureSubsetScore(features, float(len(features))),
            "gaussian_hmm",
            2,
            "a" * 64,
            sha256("|".join(features).encode()).hexdigest(),
        )

    result = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=evaluate,
        evaluate_hmm_subset=evaluate_hmm,
        hmm_selector_contract_hash="a" * 64,
        max_sffs_features=2,
    )
    first = render_feature_selection_evidence(
        result,
        contract,
        discovered_feature_names=names,
        source_build_id="source-build",
        fold_id="fold-001",
        output_dir=tmp_path / "first",
        feature_values=values,
    )
    second = render_feature_selection_evidence(
        result,
        contract,
        discovered_feature_names=names,
        source_build_id="source-build",
        fold_id="fold-001",
        output_dir=tmp_path / "second",
        feature_values=values,
    )

    assert {path.name for path in first} == {path.name for path in second}
    first_hashes = sorted(sha256(path.read_bytes()).hexdigest() for path in first)
    second_hashes = sorted(sha256(path.read_bytes()).hexdigest() for path in second)
    assert first_hashes == second_hashes
    assert any(path.name == "feature_funnel.png" for path in first)
    assert any(path.name == "family_survival.png" for path in first)
    assert any(path.name == "manifest.json" for path in first)
    verified = verify_feature_selection_evidence_bundle(tmp_path / "first")
    assert isinstance(verified["identity"], dict)
    assert verified["identity"]["fold_id"] == "fold-001"

    tracking_uri = (tmp_path / "mlruns").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="feature-selection-evidence")
    run_id = port.start_run(run_name="feature-selection-fixture")
    assert log_feature_selection_evidence(port, run_id, first)
    logged = MlflowClient(tracking_uri=tracking_uri).list_artifacts(run_id, "feature_selection")
    assert logged

    class FailingPort:
        def log_artifact(self, _run_id: str, _path: str, _artifact_path: str) -> None:
            raise OSError("offline")

    selected_before_failure = result.selected_features
    assert not log_feature_selection_evidence(cast(TrackingPort, FailingPort()), run_id, first)
    assert result.selected_features == selected_before_failure
    first[0].unlink()
    with pytest.raises(ValueError, match=r"artifact hash mismatch|required"):
        verify_feature_selection_evidence_bundle(tmp_path / "first")
