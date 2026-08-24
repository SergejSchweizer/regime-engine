"""Feature-selection contracts and deterministic policy components."""

from market_regime_engine.feature_selection.contracts import (
    BlockSelectionEvidence,
    FeatureBlock,
    FeatureScore,
    FeatureSelectionEvidence,
    FeatureSelectionPolicy,
    FeatureSelectionResult,
    Stage2ConflictEvidence,
    canonical_json,
    definition_hash,
    execution_hash,
)

__all__ = [
    "BlockSelectionEvidence",
    "FeatureBlock",
    "FeatureScore",
    "FeatureSelectionEvidence",
    "FeatureSelectionPolicy",
    "FeatureSelectionResult",
    "Stage2ConflictEvidence",
    "canonical_json",
    "definition_hash",
    "execution_hash",
]
