from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SOURCE_ROOT = ROOT / "src" / "market_regime_engine"
PACKAGE = "market_regime_engine"


def _modules() -> dict[str, Path]:
    modules = {}
    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(SOURCE_ROOT).with_suffix("").as_posix().replace("/", ".")
        modules[f"{PACKAGE}.{relative}".removesuffix(".__init__")] = path
    return modules


def _edges() -> dict[str, set[str]]:
    modules = _modules()
    edges: dict[str, set[str]] = {name: set() for name in modules}

    class ImportVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: set[str] = set()
            self._type_checking = 0

        def visit_If(self, node: ast.If) -> None:
            is_type_checking = (
                isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
            )
            if is_type_checking:
                self._type_checking += 1
            for child in node.body:
                self.visit(child)
            if is_type_checking:
                self._type_checking -= 1
            for child in node.orelse:
                self.visit(child)

        def visit_Import(self, node: ast.Import) -> None:
            if not self._type_checking:
                self.names.update(alias.name for alias in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if not self._type_checking and not node.level and node.module:
                self.names.add(node.module)

    for module, path in modules.items():
        visitor = ImportVisitor()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
                if node.test.id == "TYPE_CHECKING":
                    continue
                visitor.visit(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                visitor.visit(node)
        edges[module].update(name for name in visitor.names if name in modules)
    return edges


def test_production_import_graph_is_acyclic() -> None:
    edges = _edges()
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str, trail: tuple[str, ...] = ()) -> None:
        if module in visiting:
            cycle = " -> ".join((*trail, module))
            pytest.fail(f"production import cycle: {cycle}")
        if module in visited:
            return
        visiting.add(module)
        for dependency in sorted(edges[module]):
            visit(dependency, (*trail, module))
        visiting.remove(module)
        visited.add(module)

    for module in sorted(edges):
        visit(module)


def test_layer_direction_has_no_forbidden_reverse_edges() -> None:
    edges = _edges()
    forbidden = {
        "market_regime_engine.features": ("market_regime_engine.serving",),
        "market_regime_engine.preprocessing": ("market_regime_engine.serving",),
        "market_regime_engine.feature_discovery": ("market_regime_engine.serving",),
        "market_regime_engine.evaluation": ("market_regime_engine.serving",),
    }
    violations = [
        (source, dependency)
        for source, dependencies in edges.items()
        for dependency in dependencies
        for prefix, consumers in forbidden.items()
        if source.startswith(prefix) and any(dependency.startswith(item) for item in consumers)
    ]
    assert not violations


def test_public_exports_resolve_and_removed_exports_are_absent() -> None:
    modules = (
        "market_regime_engine.features",
        "market_regime_engine.preprocessing",
        "market_regime_engine.mlflow_support",
    )
    for name in modules:
        module = importlib.import_module(name)
        for exported in getattr(module, "__all__", ()):
            assert hasattr(module, exported), (name, exported)
    assert not hasattr(
        importlib.import_module("market_regime_engine.features"), "DynamicFeatureSource"
    )


def test_removed_runtime_switches_and_forwarding_names_are_absent() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8") for path in SOURCE_ROOT.rglob("*.py")
    )
    forbidden = (
        "REGIME_EVALUATION_CHECKPOINT_ROOT",
        "REGIME_FEATURE_PGPASSWORD_SECRET_FILE",
        "class PCATwoStageScalerArtifact",
        "def fit_pca_hmm_scaler",
        "RegimeEnginePCATwoStageScaler",
        "DynamicFeatureSource",
    )
    assert all(token not in production for token in forbidden)


def test_canonical_lifecycle_wiring_is_single_source() -> None:
    backend = (SOURCE_ROOT / "commands/v4_backend.py").read_text(encoding="utf-8")
    refit = (SOURCE_ROOT / "training/final_refit.py").read_text(encoding="utf-8")
    latest = (SOURCE_ROOT / "serving/latest_handler.py").read_text(encoding="utf-8")
    replay = (SOURCE_ROOT / "serving/replay_handler.py").read_text(encoding="utf-8")
    assert "run_canonical_xetra_evaluation" in backend
    assert "fit_and_materialize_pca_source" in backend
    assert "fit_family_pca_hmm_scaler" in refit
    assert "ModelResolver" in latest and "FeatureSource" in latest
    assert "ModelResolver" in replay and "FeatureSource" in replay
