from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
DOCUMENTATION = (
    ROOT / "README.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "EVALUATION.md",
    ROOT / "DATA_SOURCE.md",
    ROOT / "docs" / "regime_evaluations.md",
    ROOT / "docs" / "model_lifecycle_operations.md",
    ROOT / "docs" / "qa" / "xetra_v4_full_compute.md",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_v4_documentation_states_the_deployment_identity_contract() -> None:
    evaluation = _text(ROOT / "EVALUATION.md")
    lifecycle = _text(ROOT / "docs" / "model_lifecycle_operations.md")
    package_docs = _text(ROOT / "README.md")

    for required in (
        "source.max_timestamp",
        "validation_evaluation_cutoff",
        "deployment_selection_cutoff",
        "outer-fold configuration",
        "model_version_local",
    ):
        assert required in evaluation

    for required in (
        "source_build_id",
        "immutable saved source snapshot",
        "source build",
        "challenger",
        "champion",
        "RegimeEngineProductionModel.v4",
        "unknown, missing, or cross-version fields",
    ):
        assert required in lifecycle or required in package_docs


def test_v4_documentation_covers_the_statistical_and_operator_contracts() -> None:
    evaluation = _text(ROOT / "EVALUATION.md")
    full_compute = _text(ROOT / "docs" / "qa" / "xetra_v4_full_compute.md")
    lifecycle = _text(ROOT / "docs" / "model_lifecycle_operations.md")

    for required in (
        "p_GMM(K,M,d)",
        "p_GMM(5,2,8)",
        "state_information_ratio",
        "eta-squared",
        "soft_regime_nmi",
        "never pooled",
        "current-vintage",
    ):
        assert required in evaluation
    for required in (
        "scripts/run_xetra_v4_cron.sh",
        "first/middle/last",
        "source identity change fails",
        "restarted from the beginning",
    ):
        assert required in full_compute
    assert "does **not** move `champion`" in lifecycle


def test_relative_documentation_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^]]+\]\(([^)#]+)(?:#[^)]*)?\)")
    for document in DOCUMENTATION:
        assert document.is_file()
        for target in link_pattern.findall(_text(document)):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve()
            assert resolved.is_file(), f"broken documentation link: {document} -> {target}"
