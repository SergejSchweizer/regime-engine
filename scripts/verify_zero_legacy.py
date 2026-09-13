#!/usr/bin/env python3
"""Audit the repository for removed v1-v3 compatibility surfaces.

The audit is deliberately static. It does not import the application, load a
model, connect to a service, or execute an evaluation. A clean result means
that the checked-in repository contains no denied legacy identifiers and that
the Python modules that are present have no denied local import edge.
"""

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
    "EVALUATION.md",
    "LEGACY_REMOVAL.md",
    "BACKLOG.md",
    "config.example.yaml",
    "pyproject.toml",
    ".gitignore",
    ".github",
)

# These are exact repository-relative paths, not filename or basename
# patterns. The verifier's own policy source is handled separately below.
_HISTORICAL_TOKEN_ALLOWLIST: dict[str, frozenset[str]] = {
    "BACKLOG.md": frozenset(
        {
            "backward compatibility",
            "compatibility-only",
            "delta1_univariate",
            "legacy package",
            "legacy_package",
            "medoid_multivariate",
            "medoid_univariate",
            "preliminary_medoid",
            "production package v1",
            "production package v2",
            "production package v3",
            "profile_config_version == 1",
            "profile_config_version == 2",
            "profile_config_version == 3",
            "semantic medoid",
            "semantic-medoid",
            "semantic_medoid",
            "xetra_v1",
            "xetra_v2",
            "xetra_v3",
        }
    ),
    "EVALUATION.md": frozenset(),
    "LEGACY_REMOVAL.md": frozenset(
        {
            "backward compatibility",
            "compatibility-only",
            "delta1_univariate",
            "legacy package",
            "legacy_package",
            "medoid_multivariate",
            "medoid_univariate",
            "preliminary_medoid",
            "production package v1",
            "production package v2",
            "production package v3",
            "profile_config_version == 1",
            "profile_config_version == 2",
            "profile_config_version == 3",
            "semantic medoid",
            "semantic-medoid",
            "semantic_medoid",
            "xetra_v1",
            "xetra_v2",
            "xetra_v3",
        }
    ),
}
_POLICY_SOURCE = "scripts/verify_zero_legacy.py"
_IGNORED_SCAN_DIRS = frozenset({"__pycache__"})

# Keep this denylist explicit. In particular, do not deny the generic word
# ``legacy`` in all prose: cleanup may describe the one-time deletion of
# retired objects. Removed module names and contracts are denied by exact
# token/segment instead.
_FORBIDDEN = (
    re.compile(r"\bsemantic[_ -]?medoid\b", re.IGNORECASE),
    re.compile(r"\bpreliminary_medoid\b", re.IGNORECASE),
    re.compile(r"\bmedoid_(?:multi|uni)variate\b", re.IGNORECASE),
    re.compile(r"\bdelta1_univariate\b", re.IGNORECASE),
    re.compile(r"\bxetra_v[123]\b", re.IGNORECASE),
    re.compile(r"\bprofile_config_version\s*(?:==|=|:)\s*[123]\b", re.IGNORECASE),
    re.compile(r"\bproduction package v[123]\b", re.IGNORECASE),
    re.compile(r"\blegacy[ _-]package\b", re.IGNORECASE),
    re.compile(r"\bcompatibility-only\b", re.IGNORECASE),
    re.compile(r"\bbackward compatibility\b", re.IGNORECASE),
)
_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "feature_selection",
        "medoid_multivariate",
        "medoid_univariate",
        "delta1_univariate",
        "univariate_grid",
        "registry_compat",
        "legacy",
        "legacy_package",
        "compatibility_only",
    }
)
_FORBIDDEN_PATH_PARTS = _FORBIDDEN_IMPORT_MODULES | frozenset(
    {
        "xetra_v1",
        "xetra_v2",
        "xetra_v3",
    }
)
_FORBIDDEN_NAMED_PATH = re.compile(
    r"(?:legacy|compat|deprecated|old)_[a-z0-9_]+",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    line: int
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class _Module:
    name: str
    path: Path
    is_package: bool
    tree: ast.Module


@dataclass(frozen=True, slots=True)
class _ImportReference:
    name: str | None
    line: int
    kind: str


def _files(root: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    for relative in _SCAN_PATHS:
        path = root / relative
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and not any(part in _IGNORED_SCAN_DIRS for part in item.parts)
            )
    return tuple(sorted(set(result)))


def _text(path: Path) -> str | None:
    if path.suffix.lower() in {".pyc", ".png", ".jpg", ".jpeg", ".parquet", ".db"}:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError, OSError:
        return None


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _historical_token_is_allowed(relative: str, token: str) -> bool:
    allowed = _HISTORICAL_TOKEN_ALLOWLIST.get(relative)
    return allowed is not None and token.casefold() in allowed


def _path_violations(root: Path, path: Path) -> list[Violation]:
    relative = _relative(root, path)
    if any(part in _IGNORED_SCAN_DIRS for part in path.parts):
        return []
    if relative in _HISTORICAL_TOKEN_ALLOWLIST:
        return []
    violations: list[Violation] = []
    for component in relative.split("/"):
        stem = Path(component).stem
        folded = stem.casefold()
        if folded in _FORBIDDEN_PATH_PARTS or _FORBIDDEN_NAMED_PATH.fullmatch(stem):
            violations.append(
                Violation(
                    relative,
                    1,
                    "path",
                    f"denied legacy path component: {component}",
                )
            )
    return violations


def _content_violations(root: Path, path: Path) -> list[Violation]:
    relative = _relative(root, path)
    # The denylist literals are the implementation of this policy. This is an
    # exact implementation-file exclusion, never a historical-document
    # allowlist and never a wildcard over scripts/source.
    if relative == _POLICY_SOURCE:
        return []
    content = _text(path)
    if content is None:
        return []
    violations: list[Violation] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for pattern in _FORBIDDEN:
            for match in pattern.finditer(line):
                token = match.group(0)
                if not _historical_token_is_allowed(relative, token):
                    violations.append(Violation(relative, line_number, "content", token))
    return violations


def _module_name(root: Path, path: Path) -> str | None:
    if path.suffix != ".py":
        return None
    for source_root_name in ("src", "scripts", "tests"):
        source_root = root / source_root_name
        try:
            relative = path.relative_to(source_root)
        except ValueError:
            continue
        parts = list(relative.with_suffix("").parts)
        if not parts:
            return None
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        if source_root_name != "src":
            parts.insert(0, source_root_name)
        return ".".join(parts) or source_root_name
    return None


def _parse_module(root: Path, path: Path) -> _Module | Violation | None:
    name = _module_name(root, path)
    if name is None:
        return None
    content = _text(path)
    if content is None:
        return Violation(_relative(root, path), 1, "encoding", "cannot read Python source as UTF-8")
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        return Violation(_relative(root, path), exc.lineno or 1, "syntax", str(exc))
    return _Module(name, path, path.name == "__init__.py", tree)


def _module_package(module: _Module) -> tuple[str, ...]:
    parts = module.name.split(".")
    if module.is_package:
        return tuple(parts)
    return tuple(parts[:-1])


def _absolute_import_name(module: _Module, imported: str | None, level: int) -> str | None:
    if level == 0:
        return imported
    package = _module_package(module)
    prefix_length = len(package) - level + 1
    if prefix_length <= 0:
        return None
    prefix = package[:prefix_length]
    suffix = tuple(imported.split(".")) if imported else ()
    return ".".join((*prefix, *suffix))


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _import_references(module: _Module) -> tuple[_ImportReference, ...]:
    references: list[_ImportReference] = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            references.extend(
                _ImportReference(alias.name, node.lineno, "import") for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import_name(module, node.module, node.level)
            if base is None:
                references.append(_ImportReference(None, node.lineno, "invalid-relative-import"))
                continue
            references.append(_ImportReference(base, node.lineno, "import"))
            for alias in node.names:
                if alias.name != "*":
                    references.append(
                        _ImportReference(f"{base}.{alias.name}", node.lineno, "import")
                    )
        elif isinstance(node, ast.Call) and _call_name(node) in {"import_module", "__import__"}:
            name = _literal_string(node.args[0] if node.args else None)
            references.append(
                _ImportReference(name, node.lineno, "dynamic-import" if name is None else "import")
            )
    return tuple(references)


def _is_forbidden_import(name: str) -> bool:
    return any(part.casefold() in _FORBIDDEN_IMPORT_MODULES for part in name.split("."))


def _resolve_local_module(name: str, modules: dict[str, _Module]) -> str | None:
    candidate = name
    while candidate:
        if candidate in modules:
            return candidate
        if "." not in candidate:
            break
        candidate = candidate.rpartition(".")[0]
    return None


def _graph_violations(
    root: Path, modules: tuple[_Module, ...]
) -> tuple[list[Violation], tuple[dict[str, object], ...]]:
    index = {module.name: module for module in modules}
    violations: list[Violation] = []
    edges: set[tuple[str, str, int]] = set()
    for module in modules:
        relative = _relative(root, module.path)
        for reference in _import_references(module):
            if reference.name is None:
                violations.append(
                    Violation(
                        relative,
                        reference.line,
                        "dynamic-import",
                        "module name must be a string literal for static graph verification",
                    )
                )
                continue
            if _is_forbidden_import(reference.name):
                violations.append(Violation(relative, reference.line, "import", reference.name))
            target = _resolve_local_module(reference.name, index)
            if target is not None:
                edges.add((module.name, target, reference.line))
    rendered_edges = tuple(
        {"source": source, "target": target, "line": line} for source, target, line in sorted(edges)
    )
    return violations, rendered_edges


def _import_violations(root: Path, path: Path) -> list[Violation]:
    parsed = _parse_module(root, path)
    if parsed is None:
        return []
    if isinstance(parsed, Violation):
        return [parsed]
    violations, _ = _graph_violations(root, (parsed,))
    return violations


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    files = _files(root)
    violations = [violation for path in files for violation in _path_violations(root, path)]
    violations.extend(violation for path in files for violation in _content_violations(root, path))

    modules: list[_Module] = []
    for path in files:
        parsed = _parse_module(root, path)
        if isinstance(parsed, Violation):
            violations.append(parsed)
        elif parsed is not None:
            modules.append(parsed)
    graph_violations, graph = _graph_violations(root, tuple(modules))
    violations.extend(graph_violations)

    unique_violations = tuple(
        sorted(
            set(violations),
            key=lambda item: (item.path, item.line, item.kind, item.detail),
        )
    )
    return {
        "root": str(root),
        "scanned_file_count": len(files),
        "python_module_count": len(modules),
        "import_edge_count": len(graph),
        "import_graph": list(graph),
        "denied_import_modules": sorted(_FORBIDDEN_IMPORT_MODULES),
        "historical_allowlist": sorted(_HISTORICAL_TOKEN_ALLOWLIST),
        "historical_token_allowlist": {
            path: sorted(tokens) for path, tokens in sorted(_HISTORICAL_TOKEN_ALLOWLIST.items())
        },
        "violations": [asdict(item) for item in unique_violations],
        "status": "failed" if unique_violations else "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["violations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
