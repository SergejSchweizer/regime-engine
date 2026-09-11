"""Durable, dataset-pinned execution primitives for v4 evaluations."""

from market_regime_engine.evaluation_runs.contracts import (
    DatasetSnapshotIdentity,
    EvaluationRunIdentity,
    WorkUnitIdentity,
)
from market_regime_engine.evaluation_runs.snapshot import ArrowDatasetSnapshotStore
from market_regime_engine.evaluation_runs.stages import StageCheckpoint
from market_regime_engine.evaluation_runs.store import (
    SQLiteEvaluationRunStore,
    WorkUnitStatus,
)

__all__ = [
    "ArrowDatasetSnapshotStore",
    "DatasetSnapshotIdentity",
    "EvaluationRunIdentity",
    "SQLiteEvaluationRunStore",
    "StageCheckpoint",
    "WorkUnitIdentity",
    "WorkUnitStatus",
]
