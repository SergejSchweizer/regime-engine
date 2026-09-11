"""Deterministic high-dimensional source fixture for the global v4 proof."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.features.ports import FeatureCatalogEntry, FeatureCatalogSnapshot

ROW_COUNT = 1_449
FEATURE_COUNT = 52
SOURCE_BUILD_ID = "synthetic-global-v4-proof-build"
START = datetime(2010, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class SyntheticGlobalV4:
    rows: pd.DataFrame
    catalog: FeatureCatalogSnapshot
    semantic_labels: dict[str, str]
    source_data_hash: str


def build_synthetic_global_v4() -> SyntheticGlobalV4:
    """Build 1,449 rows and 52 features covering the PR-231 edge cases.

    The generator has no model-specific shortcuts.  It creates actual source
    columns: exact positive/negative redundancy, exact ties, a feature with
    changing variance but stable mean, a late-activating feature that makes one
    inner prefix infeasible, missing observations, and an independent feature
    intended to form a singleton cluster.
    """

    rng = np.random.default_rng(231042)
    index = np.arange(ROW_COUNT, dtype=np.float64)
    regime = np.where((index // 21).astype(np.int64) % 3 == 0, -1.0, 1.0)
    common = regime + 0.10 * rng.normal(size=ROW_COUNT)
    positive = common + 0.02 * rng.normal(size=ROW_COUNT)
    negative = -positive
    tie = 0.7 * common + 0.03 * rng.normal(size=ROW_COUNT)
    variance_regime = rng.normal(size=ROW_COUNT) * np.where(regime < 0.0, 0.20, 2.00)
    late = np.zeros(ROW_COUNT, dtype=np.float64)
    late[756:] = regime[756:] + 0.05 * rng.normal(size=ROW_COUNT - 756)
    missing = common + 0.05 * rng.normal(size=ROW_COUNT)
    missing[::19] = np.nan
    singleton = rng.normal(size=ROW_COUNT)

    values: dict[str, np.ndarray] = {
        "pair_positive": positive,
        "pair_negative": negative,
        "tie_left": tie,
        "tie_right": tie.copy(),
        "variance_only": variance_regime,
        "late_signal": late,
        "missing_feature": missing,
        "singleton_signal": singleton,
    }
    for position in range(FEATURE_COUNT - len(values)):
        phase = (position + 1) / 7.0
        independent = rng.normal(size=ROW_COUNT)
        values[f"feature_{position:02d}"] = (
            0.05 * np.sin(index / (5.0 + position) + phase) + 0.20 * independent + 1.50 * regime
        )

    assert len(values) == FEATURE_COUNT
    timestamps = tuple(START + timedelta(days=int(day)) for day in index)
    rows = pd.DataFrame({"timestamp_m1": timestamps, **values})
    lineage = SourceLineage(
        source_dataset="synthetic_global_regime_v4",
        source_build_id=SOURCE_BUILD_ID,
        data_sha256="0" * 64,
        schema_version=4,
        feature_version=4,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=START,
        row_count=ROW_COUNT,
        min_timestamp=timestamps[0],
        max_timestamp=timestamps[-1],
    )
    entries = tuple(
        FeatureCatalogEntry(name, ordinal) for ordinal, name in enumerate(values, start=1)
    )
    catalog = FeatureCatalogSnapshot.from_entries(lineage, "timestamp_m1", entries)
    csv_bytes = rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    source_data_hash = sha256(csv_bytes).hexdigest()
    semantic_labels = {
        name: ("redundancy" if name.startswith("pair_") else "synthetic") for name in values
    }
    return SyntheticGlobalV4(rows, catalog, semantic_labels, source_data_hash)


__all__ = [
    "FEATURE_COUNT",
    "ROW_COUNT",
    "SOURCE_BUILD_ID",
    "SyntheticGlobalV4",
    "build_synthetic_global_v4",
]
