from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from market_regime_engine.feature_discovery.ablation import HMMSubsetEvaluation
from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.pipeline import run_canonical_feature_selection
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore
from market_regime_engine.mlflow_support.feature_selection_evidence import (
    render_feature_selection_evidence,
)


def test_feature_selection_evidence_bundle_is_complete_and_hash_stable(tmp_path: Path) -> None:
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
    )
    second = render_feature_selection_evidence(
        result,
        contract,
        discovered_feature_names=names,
        source_build_id="source-build",
        fold_id="fold-001",
        output_dir=tmp_path / "second",
    )

    assert {path.name for path in first} == {path.name for path in second}
    first_hashes = sorted(sha256(path.read_bytes()).hexdigest() for path in first)
    second_hashes = sorted(sha256(path.read_bytes()).hexdigest() for path in second)
    assert first_hashes == second_hashes
    assert any(path.name == "feature_funnel.png" for path in first)
    assert any(path.name == "family_survival.png" for path in first)
    assert any(path.name == "manifest.json" for path in first)
