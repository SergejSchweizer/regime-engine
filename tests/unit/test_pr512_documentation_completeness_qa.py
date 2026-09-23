from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
DOCS = (
    "README.md",
    "ARCHITECTURE.md",
    "EVALUATION.md",
    "OPERATIONS.md",
    "CONTRIBUTING.md",
    "BACKLOG.md",
)
OWNER_MARKERS = {
    "onboarding",
    "architecture",
    "statistical-evaluation",
    "source-operations",
    "contributing",
    "backlog",
}


def test_root_contract_inventory_and_owner_markers_are_exact() -> None:
    assert tuple(sorted(path.name for path in ROOT.glob("*.md"))) == tuple(sorted(DOCS))
    markers = {
        match.group(1)
        for document in DOCS
        for match in [re.search(r"<!-- owner: ([^ ]+) -->", (ROOT / document).read_text())]
        if match
    }
    assert markers == OWNER_MARKERS
    assert all((ROOT / document).read_text().count("<!-- owner:") == 1 for document in DOCS)


def test_retired_contract_sidecars_are_absent_and_migration_map_exists() -> None:
    assert all(
        not (ROOT / name).exists()
        for name in ("DATA_SOURCE.md", "EVALUATION_EXECUTION.md", "PLOT_STYLE.md")
    )
    assert (ROOT / "docs/qa/pr511_migration_map.md").is_file()


def test_internal_documentation_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^]]+\]\(([^)#]+)(?:#[^)]*)?\)")
    for document in DOCS:
        path = ROOT / document
        for target in link_pattern.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (path.parent / target).resolve().is_file(), (document, target)


def test_active_docs_have_only_current_source_identity() -> None:
    active = "\n".join((ROOT / document).read_text(encoding="utf-8") for document in DOCS[:-1])
    assert "macro_loader.macro_features_daily" not in active
    assert "PCA-only-prefix" not in active
    assert "macro_loader.macro_features" in active


def test_mermaid_blocks_use_valid_top_level_syntax() -> None:
    diagram_start = re.compile(
        r"^(flowchart|graph|sequenceDiagram|stateDiagram(?:-v2)?|classDiagram|"
        r"erDiagram|journey|timeline)\b"
    )
    for document in DOCS:
        text = (ROOT / document).read_text(encoding="utf-8")
        for block in re.findall(r"```mermaid\n(.*?)```", text, re.DOTALL):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            assert lines and diagram_start.match(lines[0]), (document, lines[:1])
            assert block.count("[") == block.count("]")
            assert block.count("(") == block.count(")")


def test_readme_walkthrough_marks_external_commands() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    operations = (ROOT / "OPERATIONS.md").read_text(encoding="utf-8")
    assert "./scripts/bootstrap.sh" in readme
    assert "./scripts/run_xetra_v4_cron.sh" in readme
    assert "external" in operations.lower()
