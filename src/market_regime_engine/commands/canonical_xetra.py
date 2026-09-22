"""Public canonical Xetra lifecycle boundary.

The command layer must not know how individual feature-selection stages are
implemented.  This module is the only runtime boundary used by the Xetra
backend: it owns the closed-month orchestration and requires one explicit
set of CPU/model callbacks for every stage.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.calendar_clock import CalendarMonthFold
from market_regime_engine.feature_discovery.ablation import HMMSubsetEvaluation
from market_regime_engine.feature_discovery.metadata_store import FeatureSelectionMetadataStore
from market_regime_engine.feature_discovery.monthly_refit import (
    MonthlyPackageIdentity,
    MonthlyRefitResult,
    StageCallbacks,
    run_monthly_outer_refit,
)
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore
from market_regime_engine.features.ports import FeatureCatalogSnapshot
from market_regime_engine.mlflow_support.ports import TrackingPort
from market_regime_engine.profiles.config import ModelProfile


@dataclass(frozen=True, slots=True)
class CanonicalStageCallbacks:
    """Pickle-safe model seams consumed by the canonical selection runner."""

    evaluate_subset: Callable[[tuple[str, ...]], FeatureSubsetScore | None]
    evaluate_hmm_subset: Callable[[tuple[str, ...]], HMMSubsetEvaluation | None]
    hmm_selector_contract_hash: str
    fit_final_hmm: Callable[[pd.DataFrame, tuple[str, ...], int], str | Sequence[str]]
    evaluate_gaussian_subset_by_k: Callable[[int, tuple[str, ...]], FeatureSubsetScore | None]
    evaluate_outer_test: Callable[
        [pd.DataFrame, pd.DataFrame, tuple[str, ...], int, tuple[str, ...]], str
    ]


StageCallbackFactory = Callable[[pd.DataFrame, pd.DataFrame, CalendarMonthFold], StageCallbacks]


@dataclass(frozen=True, slots=True)
class CanonicalXetraEvaluation:
    """Immutable result of one complete canonical closed-month evaluation."""

    monthly: MonthlyRefitResult
    source_build_id: str
    feature_selection_profile_hash: str
    selected_features_by_fold: tuple[tuple[str, ...], ...]

    @property
    def result_hash(self) -> str:
        return self.monthly.result_hash

    @property
    def production_eligible(self) -> bool:
        valid = self.monthly.valid_folds
        return len(valid) >= 3 and len(valid) / len(self.monthly.folds) >= 0.80

    @property
    def latest_package(self) -> MonthlyPackageIdentity:
        valid = self.monthly.valid_folds
        if not valid:
            raise ValueError("canonical evaluation produced no valid monthly package")
        package = valid[-1].package
        if package is None:
            raise ValueError("latest canonical fold is missing its package")
        return package


def run_canonical_xetra_evaluation(
    source_rows: pd.DataFrame,
    *,
    catalog: FeatureCatalogSnapshot,
    profile: ModelProfile,
    callbacks: CanonicalStageCallbacks,
    stage_callback_factory: StageCallbackFactory | None = None,
    metadata_store: FeatureSelectionMetadataStore,
    tracking: TrackingPort | None = None,
    max_workers: int | None = None,
    evaluation_cutoff: datetime | None = None,
) -> CanonicalXetraEvaluation:
    """Run the sole supported Xetra evaluation/refit-selection pipeline.

    The function deliberately exposes no selector/source override and never
    delegates to a historical evaluation policy.  The metadata store is
    mandatory for every new evaluation; inference from an already frozen
    package remains outside this boundary.
    """

    result = run_monthly_outer_refit(
        source_rows,
        catalog=catalog,
        profile=profile,
        evaluate_subset=callbacks.evaluate_subset,
        evaluate_hmm_subset=callbacks.evaluate_hmm_subset,
        hmm_selector_contract_hash=callbacks.hmm_selector_contract_hash,
        fit_final_hmm=callbacks.fit_final_hmm,
        evaluate_gaussian_subset_by_k=callbacks.evaluate_gaussian_subset_by_k,
        evaluate_outer_test=callbacks.evaluate_outer_test,
        metadata_store=metadata_store,
        tracking=tracking,
        max_workers=max_workers,
        evaluation_cutoff=evaluation_cutoff,
        stage_callback_factory=stage_callback_factory,
    )
    if result.source_build_id != catalog.lineage.source_build_id:
        raise ValueError("canonical result source build differs from its catalog")
    valid_packages = tuple(item.package for item in result.valid_folds if item.package is not None)
    if not valid_packages:
        raise ValueError("canonical evaluation produced no valid monthly package")
    return CanonicalXetraEvaluation(
        monthly=result,
        source_build_id=result.source_build_id,
        feature_selection_profile_hash=valid_packages[0].feature_selection_profile_hash,
        selected_features_by_fold=tuple(item.selected_features for item in valid_packages),
    )


__all__ = [
    "CanonicalStageCallbacks",
    "CanonicalXetraEvaluation",
    "run_canonical_xetra_evaluation",
]
