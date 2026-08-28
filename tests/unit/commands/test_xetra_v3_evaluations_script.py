from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/run_xetra_v3_evaluations.py")


def _module():
    spec = importlib.util.spec_from_file_location("xetra_v3_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_runner_uses_canonical_policy_and_has_no_production_path() -> None:
    module = _module()
    policy = module._policy(Path("."))
    source = SCRIPT.read_text(encoding="utf-8")
    assert policy.policy_id == "xetra_semantic_medoid_v3"
    assert len(policy.feature_universe) == 61
    assert "final_production_refit" not in source
    assert "publish_walk_forward_oos" not in source
    assert "MlflowModelRegistry" not in source
