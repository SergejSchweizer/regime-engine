from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import pytest

import market_regime_engine.evaluation.walk_forward as walk_forward
import market_regime_engine.evaluations.global_regime_v4 as global_v4
import market_regime_engine.evaluations.teacher_reference as teacher_reference
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluations.teacher_reference import refit_frozen_teacher
from market_regime_engine.feature_discovery.contracts import ProvisionalTeacherReference
from market_regime_engine.feature_discovery.prefix_search import run_prefix_gaussian_candidate
from market_regime_engine.features.ports import FeatureCatalogEntry, FeatureCatalogSnapshot
from market_regime_engine.models.protocols import FitResult, GaussianHMMAdapter
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    MultistartResult,
    StartDiagnostic,
)

pytestmark = pytest.mark.integration

START = datetime(2020, 1, 1, tzinfo=UTC)
HASH = "b" * 64


def _rows(count: int = 1449) -> pd.DataFrame:
    index = np.arange(count, dtype=np.float64)
    regime = (index.astype(np.int64) // 35) % 2
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(START + timedelta(days=int(value)) for value in index),
            "f0": np.where(regime == 0, -1.0, 1.0) + 0.12 * np.sin(index / 5.0),
            "f1": np.where(regime == 0, 1.0, -1.0) + 0.12 * np.cos(index / 7.0),
            "f2": np.sin(index / 13.0),
        }
    )


def _catalog() -> FeatureCatalogSnapshot:
    lineage = SourceLineage(
        source_dataset="xetra_gold",
        source_build_id="synthetic-build",
        data_sha256=HASH,
        schema_version=2,
        feature_version=4,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=START,
        row_count=1449,
        min_timestamp=START,
        max_timestamp=START + timedelta(days=1448),
    )
    return FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        tuple(
            FeatureCatalogEntry(name, index + 1) for index, name in enumerate(("f0", "f1", "f2"))
        ),
    )


def _candidate(catalog: FeatureCatalogSnapshot) -> ResolvedCandidateProfile:
    return ResolvedCandidateProfile(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        covariance_type="full",
        feature_order=("f0", "f1"),
        feature_dimension=2,
        source_build_id=catalog.lineage.source_build_id,
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
        original_feature_universe=catalog.feature_names,
        feature_contract_version=4,
    )


def _teacher_reference() -> ProvisionalTeacherReference:
    return ProvisionalTeacherReference(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        timestamps=(START,),
        filtered_probabilities=((0.5, 0.5),),
        dominant_states=(0,),
        valid_inner_fold_ids=("fold_001",),
        source_build_id="synthetic-build",
        inner_plan_hash=HASH,
        prototype_features=("f0",),
    )


def _one_real_fit_multistart(
    train_rows: npt.ArrayLike,
    *,
    state_count: int,
    adapter_factory: Callable[[], GaussianHMMAdapter],
    **_kwargs: object,
) -> MultistartResult:
    """Use real backend math once per fit while preserving the eight-start contract."""

    result: FitResult = adapter_factory().fit(train_rows, state_count, MULTISTART_SEEDS[0])
    diagnostics = tuple(
        StartDiagnostic(
            seed=seed,
            success=True,
            converged=True,
            iterations=result.iterations,
            train_log_likelihood=result.train_log_likelihood,
            artifact=result.artifact,
            failure_reason=None,
        )
        for seed in MULTISTART_SEEDS
    )
    return MultistartResult(state_count=state_count, winner=result, diagnostics=diagnostics)


def test_three_outer_folds_execute_real_hmm_math(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows()
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    candidate = _candidate(catalog)

    monkeypatch.setattr(walk_forward, "run_multistart", _one_real_fit_multistart)
    monkeypatch.setattr(teacher_reference, "run_multistart", _one_real_fit_multistart)
    monkeypatch.setattr(
        global_v4,
        "select_v4_configuration",
        lambda train_rows, **kwargs: SimpleNamespace(
            final_candidate=candidate,
            teacher_reference=_teacher_reference(),
            catalog_hash=catalog.catalog_hash,
            feature_discovery_hash=HASH,
        ),
    )

    result = global_v4.evaluate_global_regime_v4(
        rows,
        catalog=catalog,
        profile=profile,
        outer_runner=run_prefix_gaussian_candidate,
        teacher_refitter=refit_frozen_teacher,
        max_workers=1,
    )

    assert len(result.outer_folds) == 3
    assert result.valid_fold_count == 3
    assert result.production_eligible is True
    assert all(fold.outer_shared_timestamp_count >= 42 for fold in result.outer_folds)
    assert all(fold.outer_teacher_final_soft_nmi is not None for fold in result.outer_folds)
    assert all(
        fold.final_configuration.state_identity_scope == "outer_fold_local"
        for fold in result.outer_folds
    )
