from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module() -> object:
    path = Path(__file__).parents[2] / "scripts" / "verify_zero_legacy.py"
    spec = importlib.util.spec_from_file_location("verify_zero_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load zero-legacy verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_active_repository_has_no_removed_legacy_surface() -> None:
    module = _module()
    report = module.audit(Path(__file__).parents[2])  # type: ignore[attr-defined]
    assert report["status"] == "verified"
    assert report["violations"] == []
    assert report["python_module_count"] > 0
    assert report["import_edge_count"] == len(report["import_graph"])


def test_static_graph_rejects_removed_relative_import(tmp_path: Path) -> None:
    module = _module()
    package = tmp_path / "src" / "market_regime_engine"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    removed = "feature" + "_selection"
    (package / "current.py").write_text(
        f"from .{removed} import selector\n",
        encoding="utf-8",
    )

    report = module.audit(tmp_path)  # type: ignore[attr-defined]

    assert report["status"] == "failed"
    assert any(
        violation["kind"] == "import" and removed in violation["detail"]
        for violation in report["violations"]
    )


def test_static_audit_rejects_dynamic_module_names(tmp_path: Path) -> None:
    module = _module()
    package = tmp_path / "src" / "market_regime_engine"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "current.py").write_text(
        "from importlib import import_module\n"
        "module_name = 'market_regime_engine.' + 'safe'\n"
        "import_module(module_name)\n",
        encoding="utf-8",
    )

    report = module.audit(tmp_path)  # type: ignore[attr-defined]

    assert report["status"] == "failed"
    assert any(violation["kind"] == "dynamic-import" for violation in report["violations"])


def test_historical_allowlist_is_exact_path_and_token(tmp_path: Path) -> None:
    module = _module()
    token = "legacy" + " package"
    (tmp_path / "BACKLOG.md").write_text(token + "\n", encoding="utf-8")
    nested = tmp_path / "docs"
    nested.mkdir()
    (nested / "BACKLOG.md").write_text(token + "\n", encoding="utf-8")

    report = module.audit(tmp_path)  # type: ignore[attr-defined]

    assert report["status"] == "failed"
    assert any(violation["path"] == "docs/BACKLOG.md" for violation in report["violations"])
    assert not any(violation["path"] == "BACKLOG.md" for violation in report["violations"])
