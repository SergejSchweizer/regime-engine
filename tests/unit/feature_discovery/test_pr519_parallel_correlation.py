from __future__ import annotations

from collections import OrderedDict

import numpy as np

from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.global_reduction import (
    prune_global_correlated_features,
)


def test_parallel_global_correlation_matches_serial_hash() -> None:
    rows = np.arange(120, dtype=np.float64)
    values = OrderedDict(
        (
            ("vix_log_level", tuple(rows)),
            ("us_10y_log_level", tuple(-rows)),
            ("us_2y_log_level", tuple(rows + 0.1)),
            ("family_pc_vix_1", tuple(rows * 0.5 + 2.0)),
            ("family_pc_vix_2", tuple(np.sin(rows))),
        )
    )
    contract = build_feature_role_contract(
        (TEMPORAL_KEY, "vix_log_level", "us_10y_log_level", "us_2y_log_level")
    )

    serial = prune_global_correlated_features(values, contract, max_workers=1)
    parallel = prune_global_correlated_features(values, contract, max_workers=4)

    assert parallel == serial
    assert parallel.result_hash == serial.result_hash
