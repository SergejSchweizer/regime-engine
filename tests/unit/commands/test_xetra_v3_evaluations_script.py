from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

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


def test_v3_runner_uses_common_feature_history_without_hiding_later_missingness() -> None:
    module = _module()
    policy = module._policy(Path("."))
    timestamps = tuple(
        datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(5)
    )
    rows = pd.DataFrame(
        {"timestamp_m1": timestamps}
        | {
            feature: np.array([np.nan, np.nan, 1.0, np.nan, 2.0])
            if feature == policy.feature_universe[-1]
            else np.ones(len(timestamps))
            for feature in policy.feature_universe
        }
    )
    bounded = module._common_feature_history(rows, policy.feature_universe)
    assert tuple(bounded["timestamp_m1"]) == timestamps[2:]
    assert pd.isna(bounded[policy.feature_universe[-1]].iloc[1])
