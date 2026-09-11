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
