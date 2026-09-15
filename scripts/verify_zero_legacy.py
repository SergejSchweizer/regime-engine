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
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any

PRODUCTION_MLFLOW_URI = "http://10.10.1.3:5000"
EVALUATION_EXPERIMENT_NAME = "macro-regime-evaluation"
REGISTERED_MODEL_NAME = "regime-xetra"
PACKAGE_SCHEMA_VERSION = "RegimeEngineProductionModel.v4"
GLOBAL_V4_EVALUATION_ID = "global_regime_v4"
_SUPPORTED_COMMANDS = frozenset({"evaluate", "final-refit", "publish-oos", "register", "status"})

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


def _runtime_violation(kind: str, detail: str) -> dict[str, str]:
    return {"path": "runtime", "line": "0", "kind": kind, "detail": detail}


def _runtime_audit(root: Path) -> dict[str, object]:
    """Verify local v4 runtime boundaries without running an evaluation."""

    violations: list[dict[str, str]] = []
    checks: dict[str, object] = {}
    source_root = str(root / "src")
    original_path = sys.path[:]
    sys.path.insert(0, source_root)
    try:
        try:
            from market_regime_engine import cli
            from market_regime_engine.commands.lifecycle import LifecycleOperatorService
            from market_regime_engine.commands.v4_backend import V4LifecycleBackend
            from market_regime_engine.feature_discovery.contracts import FINAL_CANDIDATE_IDS
            from market_regime_engine.mlflow_support.model_package import (
                load_production_package,
                production_artifact_from_json,
            )
            from market_regime_engine.profiles.loader import load_profile
            from market_regime_engine.profiles.resolution import expected_candidate_ids
            from market_regime_engine.serving.profile_registry import ProfileRegistry
        except Exception as exc:  # pragma: no cover - reported as a contract failure
            violations.append(_runtime_violation("import", f"runtime audit imports failed: {exc}"))
            return {"status": "failed", "checks": checks, "violations": violations}

        subparser_actions = tuple(
            action
            for action in cli.build_parser()._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        commands = (
            frozenset(subparser_actions[0].choices) if len(subparser_actions) == 1 else frozenset()
        )
        checks["cli_commands"] = sorted(commands)
        if commands != _SUPPORTED_COMMANDS:
            violations.append(
                _runtime_violation(
                    "cli",
                    f"public CLI commands differ from the v4 contract: {sorted(commands)}",
                )
            )
        cli_errors = StringIO()
        cli_return_code = cli.main(
            ["status", "--profile", "removed-profile"],
            service=LifecycleOperatorService(object()),
            stdout=StringIO(),
            stderr=cli_errors,
        )
        checks["cli_removed_profile"] = cli_errors.getvalue().strip()
        if cli_return_code != 2 or "unknown_profile" not in cli_errors.getvalue():
            violations.append(
                _runtime_violation("cli", "CLI accepted a removed public profile identifier")
            )

        profile_path = root / "configs" / "profiles" / "xetra_v4.yaml"
        profile = load_profile(profile_path)
        profile_identity = {
            "profile_id": profile.profile_id,
            "profile_config_version": profile.profile_config_version,
            "registered_model": profile.registered_model,
            "production_alias": profile.production_alias,
            "challenger_alias": profile.challenger_alias,
        }
        checks["profile_identity"] = profile_identity
        if profile_identity != {
            "profile_id": "xetra",
            "profile_config_version": 4,
            "registered_model": REGISTERED_MODEL_NAME,
            "production_alias": "champion",
            "challenger_alias": "challenger",
        }:
            violations.append(
                _runtime_violation("profile", "active profile identity is not v4 Xetra")
            )

        registry = ProfileRegistry()
        targets = registry.targets()
        checks["public_profiles"] = [target.profile_id for target in targets]
        if len(targets) != 1 or targets[0].profile_id != "xetra":
            violations.append(
                _runtime_violation("profile", "public profile registry is not Xetra-only")
            )
        for version in (1, 2, 3):
            for operation, callback in (
                ("profile resolution", lambda version=version: registry.resolve("xetra", version)),
                (
                    "candidate resolution",
                    lambda version=version: expected_candidate_ids(version),
                ),
            ):
                try:
                    callback()
                except KeyError, ValueError:
                    continue
                except Exception as exc:  # pragma: no cover - defensive contract reporting
                    violations.append(
                        _runtime_violation(
                            "profile", f"{operation} raised an unexpected error: {exc}"
                        )
                    )
                else:
                    violations.append(
                        _runtime_violation(
                            "profile",
                            f"{operation} accepted removed configuration version {version}",
                        )
                    )
        if tuple(expected_candidate_ids()) != tuple(FINAL_CANDIDATE_IDS):
            violations.append(
                _runtime_violation("evaluation", "candidate universe is not the pinned v4 set")
            )

        # These guards execute only their identity checks; they never construct
        # a backend, access a source, or start a model computation.
        backend = object.__new__(V4LifecycleBackend)
        for method_name in ("final_refit", "publish_oos"):
            try:
                getattr(backend, method_name)("xetra", "removed-evaluation-id")
            except ValueError:
                continue
            except Exception as exc:  # pragma: no cover - defensive contract reporting
                violations.append(
                    _runtime_violation(
                        "evaluation", f"{method_name} raised an unexpected error: {exc}"
                    )
                )
            else:
                violations.append(
                    _runtime_violation(
                        "evaluation", f"{method_name} accepted a non-v4 evaluation identifier"
                    )
                )

        with tempfile.TemporaryDirectory(prefix="zero-legacy-package-") as temporary:
            package = Path(temporary)
            (package / "MLmodel").write_text(
                json.dumps(
                    {
                        "flavors": {
                            "regime_engine": {
                                "data": "production_model.json",
                                "schema_version": PACKAGE_SCHEMA_VERSION,
                            }
                        },
                        "mlflow_version": "3.15.1",
                        "model_name": REGISTERED_MODEL_NAME,
                        "python_version": "3.14.7",
                    }
                ),
                encoding="utf-8",
            )
            legacy_payload = {
                "schema_version": "RegimeEngineProductionModel.v3",
                "opaque_legacy_payload": {"must_not": "be decoded"},
            }
            (package / "production_model.json").write_text(
                json.dumps(legacy_payload), encoding="utf-8"
            )
            for loader, label in (
                (lambda: production_artifact_from_json(json.dumps(legacy_payload)), "JSON loader"),
                (lambda: load_production_package(package), "filesystem package loader"),
            ):
                try:
                    loader()
                except ValueError as exc:
                    if "unsupported production package schema" not in str(exc):
                        violations.append(
                            _runtime_violation(
                                "package", f"{label} rejected an old package unclearly"
                            )
                        )
                except Exception as exc:  # pragma: no cover - defensive contract reporting
                    violations.append(
                        _runtime_violation("package", f"{label} raised an unexpected error: {exc}")
                    )
                else:
                    violations.append(
                        _runtime_violation("package", f"{label} accepted an old package schema")
                    )
        checks["old_package_rejection"] = "verified"
    finally:
        sys.path[:] = original_path
    return {
        "status": "failed" if violations else "verified",
        "checks": checks,
        "violations": violations,
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _entity_values(entity: object) -> dict[str, str]:
    values: dict[str, str] = {}
    for candidate in (entity, getattr(entity, "info", None), getattr(entity, "data", None)):
        if candidate is None:
            continue
        for field in ("tags", "params"):
            for key, value in _mapping(getattr(candidate, field, None)).items():
                values[str(key)] = str(value)
    return values


def _identifier(entity: object, field: str, fallback: str = "") -> str:
    value = getattr(entity, field, None)
    if value is None:
        value = getattr(getattr(entity, "info", None), field, None)
    return str(value) if value is not None else fallback


def _paged_search(
    search: Callable[..., object],
    **kwargs: object,
) -> tuple[object, ...]:
    results: list[object] = []
    page_token: str | None = None
    while True:
        page_kwargs = dict(kwargs)
        page_kwargs["max_results"] = 1000
        if page_token is not None:
            page_kwargs["page_token"] = page_token
        page = search(**page_kwargs)
        results.extend(page if isinstance(page, list | tuple) else tuple(page))  # type: ignore[arg-type]
        next_token = getattr(page, "token", None)
        if not next_token or next_token == page_token:
            return tuple(results)
        page_token = str(next_token)


def _external_missing(error: Exception) -> bool:
    code = str(getattr(error, "error_code", ""))
    message = str(error).lower()
    return (
        code in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"}
        or "not found" in message
        or "does not exist" in message
    )


def audit_external_mlflow(client: Any, *, tracking_uri: str) -> dict[str, object]:
    """Read-only audit of the exact regime-engine MLflow namespaces."""

    violations: list[dict[str, str]] = []
    experiment_report: dict[str, object] = {
        "name": EVALUATION_EXPERIMENT_NAME,
        "experiment_id": None,
        "run_count": 0,
        "logged_model_count": 0,
        "v4_run_ids": [],
        "legacy_run_ids": [],
        "legacy_logged_model_ids": [],
    }
    try:
        experiment = client.get_experiment_by_name(EVALUATION_EXPERIMENT_NAME)
    except Exception as exc:
        if not _external_missing(exc):
            raise
        experiment = None
    if experiment is not None:
        experiment_id = str(experiment.experiment_id)
        experiment_report["experiment_id"] = experiment_id
        runs = _paged_search(
            client.search_runs,
            experiment_ids=[experiment_id],
            run_view_type=3,
        )
        models = _paged_search(client.search_logged_models, experiment_ids=[experiment_id])
        experiment_report["run_count"] = len(runs)
        experiment_report["logged_model_count"] = len(models)
        run_by_id = {_identifier(run, "run_id"): run for run in runs if _identifier(run, "run_id")}
        accepted_run_ids: set[str] = set()
        changed = True
        while changed:
            changed = False
            for run_id, run in run_by_id.items():
                values = _entity_values(run)
                profile = values.get("profile_id")
                version = values.get("profile_config_version")
                evaluation_id = values.get("evaluation_id")
                parent_id = values.get("mlflow.parentRunId") or values.get("parent_run_id")
                explicit_v4 = (
                    profile == "xetra"
                    and version == "4"
                    and evaluation_id in {None, GLOBAL_V4_EVALUATION_ID}
                )
                inherited_v4 = (
                    profile is None
                    and version is None
                    and evaluation_id in {None, GLOBAL_V4_EVALUATION_ID}
                    and parent_id in accepted_run_ids
                )
                if (explicit_v4 or inherited_v4) and run_id not in accepted_run_ids:
                    accepted_run_ids.add(run_id)
                    changed = True
        legacy_run_ids = sorted(set(run_by_id) - accepted_run_ids)
        experiment_report["v4_run_ids"] = sorted(accepted_run_ids)
        experiment_report["legacy_run_ids"] = legacy_run_ids
        for run_id in legacy_run_ids:
            violations.append(
                _runtime_violation("external-run", f"legacy or unclassified run: {run_id}")
            )

        legacy_model_ids: list[str] = []
        for model in models:
            model_id = _identifier(model, "model_id")
            tags = _mapping(getattr(model, "tags", None))
            if not (
                str(tags.get("regime_engine.profile_id", "")) == "xetra"
                and str(tags.get("regime_engine.profile_config_version", "")) == "4"
            ):
                legacy_model_ids.append(model_id)
                violations.append(
                    _runtime_violation(
                        "external-logged-model", f"legacy or unclassified LoggedModel: {model_id}"
                    )
                )
        experiment_report["legacy_logged_model_ids"] = sorted(legacy_model_ids)

    registry_report: dict[str, object] = {
        "model_name": REGISTERED_MODEL_NAME,
        "version_count": 0,
        "v4_version_ids": [],
        "legacy_version_ids": [],
        "aliases": {},
        "legacy_aliases": [],
    }
    try:
        versions = _paged_search(
            client.search_model_versions,
            filter_string=f"name='{REGISTERED_MODEL_NAME}'",
        )
    except Exception as exc:
        if _external_missing(exc):
            versions = ()
        else:
            raise
    registry_report["version_count"] = len(versions)
    v4_versions: set[str] = set()
    legacy_versions: list[str] = []
    for version in versions:
        version_id = _identifier(version, "version")
        tags = _mapping(getattr(version, "tags", None))
        is_v4 = (
            str(tags.get("regime_engine.package_schema", "")) == PACKAGE_SCHEMA_VERSION
            and str(tags.get("regime_engine.profile_config_version", "")) == "4"
            and str(tags.get("regime_engine.profile_id", "xetra")) == "xetra"
        )
        if is_v4:
            v4_versions.add(version_id)
        else:
            legacy_versions.append(version_id)
            violations.append(
                _runtime_violation(
                    "external-model-version", f"legacy or unclassified model version: {version_id}"
                )
            )
    registry_report["v4_version_ids"] = sorted(v4_versions)
    registry_report["legacy_version_ids"] = sorted(legacy_versions)

    try:
        registered = client.get_registered_model(REGISTERED_MODEL_NAME)
    except Exception as exc:
        if _external_missing(exc):
            registered = None
        else:
            raise
    if registered is not None:
        aliases = {
            str(alias): str(target)
            for alias, target in dict(getattr(registered, "aliases", {}) or {}).items()
        }
        registry_report["aliases"] = dict(sorted(aliases.items()))
        legacy_aliases: list[str] = []
        for alias, target in sorted(aliases.items()):
            if alias not in {"champion", "challenger"} or target not in v4_versions:
                legacy_aliases.append(f"{alias}->{target}")
                violations.append(
                    _runtime_violation(
                        "external-alias", f"legacy or unclassified alias: {alias}->{target}"
                    )
                )
        registry_report["legacy_aliases"] = legacy_aliases

    return {
        "tracking_uri": tracking_uri,
        "evaluation": experiment_report,
        "registry": registry_report,
        "status": "failed" if violations else "verified",
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also verify local CLI/profile/evaluation/package fail-closed contracts",
    )
    parser.add_argument(
        "--external-mlflow",
        action="store_true",
        help="opt in to a read-only zero-legacy audit of the configured MLflow service",
    )
    parser.add_argument("--tracking-uri", default=PRODUCTION_MLFLOW_URI)
    args = parser.parse_args()
    report = audit(args.root)
    all_violations = list(report["violations"])
    if args.runtime:
        runtime_report = _runtime_audit(args.root.resolve())
        report["runtime"] = runtime_report
        all_violations.extend(runtime_report["violations"])
    if args.external_mlflow:
        try:
            from mlflow.tracking import MlflowClient

            external_report = audit_external_mlflow(
                MlflowClient(tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri),
                tracking_uri=args.tracking_uri,
            )
        except Exception as exc:
            external_report = {
                "status": "failed",
                "violations": [
                    _runtime_violation("external-mlflow", f"read-only audit query failed: {exc}")
                ],
            }
        report["external_mlflow"] = external_report
        all_violations.extend(external_report["violations"])
    report["violations"] = all_violations
    report["status"] = "failed" if all_violations else "verified"
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["violations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
