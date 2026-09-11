#!/usr/bin/env python3
"""Audit the active repository for removed v1-v3 compatibility surfaces."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_SCAN_PATHS = (
    "src",
    "configs",
    "tests",
    "scripts",
    "docs",
    "README.md",
    "ARCHITECTURE.md",
    "DATA_SOURCE.md",
    "EVALUATION_EXECUTION.md",
    "PLOT_STYLE.md",
    "pyproject.toml",
    ".github",
)
_HISTORICAL_DOCS = {"BACKLOG.md", "LEGACY_REMOVAL.md", "EVALUATION.md"}
_FORBIDDEN = (
    re.compile(r"semantic[_ -]?medoid", re.IGNORECASE),
    re.compile(r"preliminary_medoid", re.IGNORECASE),
    re.compile(r"medoid_(?:multi|uni)variate", re.IGNORECASE),
    re.compile(r"delta1_univariate", re.IGNORECASE),
    re.compile(r"\bxetra_v[123]\b", re.IGNORECASE),
    re.compile(r"profile_config_version\s*==\s*[123]\b"),
    re.compile(r"production package v[123]", re.IGNORECASE),
    re.compile(r"\b(?:legacy|compat|deprecated|old)_(?:[a-z0-9_]+)\b", re.IGNORECASE),
    re.compile(r"\b(?:backward compatibility|compatibility-only)\b", re.IGNORECASE),
)
_FORBIDDEN_IMPORTS = (
    "feature_selection",
    "medoid_multivariate",
    "medoid_univariate",
    "delta1_univariate",
    "univariate_grid",
    "registry_compat",
)


@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    line: int
    kind: str
    detail: str


def _files(root: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    for relative in _SCAN_PATHS:
        path = root / relative
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(item for item in path.rglob("*") if item.is_file())
    return tuple(sorted(set(result)))


def _text(path: Path) -> str | None:
    if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".parquet", ".db"}:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError, OSError:
        return None


def _content_violations(root: Path, path: Path) -> list[Violation]:
    relative = path.relative_to(root).as_posix()
    if path.name in _HISTORICAL_DOCS or relative == "scripts/verify_zero_legacy.py":
        return []
    content = _text(path)
    if content is None:
        return []
    violations: list[Violation] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for pattern in _FORBIDDEN:
            match = pattern.search(line)
            if match is not None:
                violations.append(Violation(relative, line_number, "content", match.group(0)))
    return violations


def _import_violations(root: Path, path: Path) -> list[Violation]:
    if path.suffix != ".py":
        return []
    content = _text(path)
    if content is None:
        return []
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        return [Violation(path.relative_to(root).as_posix(), exc.lineno or 1, "syntax", str(exc))]
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names = (node.module or "",)
        else:
            continue
        for name in names:
            if any(token in name for token in _FORBIDDEN_IMPORTS):
                violations.append(
                    Violation(
                        path.relative_to(root).as_posix(),
                        node.lineno,
                        "import",
                        name,
                    )
                )
    return violations


def audit(root: Path) -> dict[str, object]:
    files = _files(root)
    violations = [
        violation
        for path in files
        for violation in (*_content_violations(root, path), *_import_violations(root, path))
    ]
    return {
        "root": str(root),
        "scanned_file_count": len(files),
        "historical_allowlist": sorted(_HISTORICAL_DOCS),
        "violations": [asdict(item) for item in violations],
        "status": "failed" if violations else "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["violations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
