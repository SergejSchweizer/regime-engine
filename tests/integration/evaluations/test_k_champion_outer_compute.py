from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

import market_regime_engine.evaluation.walk_forward as walk_forward
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.evaluation.walk_forward import run_walk_forward_candidate
from market_regime_engine.evaluation.walk_forward_splits import (
    WalkForwardFold,
    WalkForwardPlan,
)
from market_regime_engine.evaluations.k_champion_contract import (
    KChampionSelection,
    feature_order_hash,
)
from market_regime_engine.evaluations.k_champion_outer import (
    KChampionFoldEvaluation,
    run_k_champion_outer_policy,
)
from market_regime_engine.evaluations.k_feature_selection import run_real_k_feature_selection
from market_regime_engine.features.ports import FeatureCatalogEntry, FeatureCatalogSnapshot
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.training.adapter_factory import adapter_factory
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    MultistartResult,
    StartDiagnostic,
)

pytestmark = pytest.mark.integration

BASE = datetime(2026, 2, 1, tzinfo=UTC)
SOURCE_BUILD = "real-outer-k-build"


def _plan() -> WalkForwardPlan:
    folds = tuple(
        WalkForwardFold(
            fold_index=index,
            fold_id=f"fold_{index:03d}",
            train_start=BASE,
            train_end=BASE + timedelta(minutes=1260 + (index - 1) * 63 - 1),
            test_start=BASE + timedelta(minutes=1260 + (index - 1) * 63),
            test_end=BASE + timedelta(minutes=1260 + index * 63 - 1),
            train_source_observations=1260 + (index - 1) * 63,
            test_source_observations=63,
        )
        for index in (1, 2, 3)
    )
    return WalkForwardPlan(folds=folds, evaluation_cutoff=folds[-1].test_end, plan_hash="b" * 64)


def _rows() -> pd.DataFrame:
    count = 1449
    index = np.arange(count, dtype=np.float64)
    regime = np.random.default_rng(43).integers(0, 5, size=count)
    noise = np.random.default_rng(44)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(BASE + timedelta(minutes=int(value)) for value in index),
            "f0": np.asarray((-2.0, -1.0, 0.0, 1.0, 2.0))[regime]
            + noise.normal(scale=0.03, size=count),
            "f1": np.asarray((1.5, -1.5, 0.5, -0.5, 1.0))[regime]
            + noise.normal(scale=0.03, size=count),
            **{
                f"pca_pc_{component:03d}": np.sin(index / (13.0 + component))
                for component in range(1, 9)
            },
        }
    )


def _selection_rows() -> pd.DataFrame:
    count = 1449
    index = np.arange(count, dtype=np.float64)
    hidden = np.where((np.arange(count) // 47) % 2 == 0, -1.0, 1.0)
    noise = np.random.default_rng(9001)
    base = np.column_stack(
        (
            hidden + noise.normal(0.0, 0.15, count),
            0.75 * hidden + noise.normal(0.0, 0.15, count),
            np.sin(index / 19.0) + noise.normal(0.0, 0.05, count),
            *(noise.normal(0.0, 0.5, count) for _ in range(13)),
        )
    )
    pca_weights = np.random.default_rng(9002).normal(size=(16, 8))
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(BASE + timedelta(minutes=int(value)) for value in index),
            **{f"f{feature_index}": base[:, feature_index] for feature_index in range(16)},
            **{
                f"pca_pc_{component:03d}": (base @ pca_weights)[:, component - 1]
                for component in range(1, 9)
            },
        }
    )


def _catalog(rows: pd.DataFrame) -> FeatureCatalogSnapshot:
    names = tuple(column for column in rows.columns if column != "timestamp_m1")
    lineage = SourceLineage(
        source_dataset="macro_features_daily",
        source_build_id=SOURCE_BUILD,
        data_sha256="d" * 64,
        schema_version=6,
        feature_version=5,
        source_table="macro_loader.macro_features_daily",
        synced_at_utc=BASE,
        row_count=len(rows),
        min_timestamp=rows["timestamp_m1"].iloc[0],
        max_timestamp=rows["timestamp_m1"].iloc[-1],
    )
    entries = tuple(
        FeatureCatalogEntry(
            name,
            ordinal,
            schema_name="macro_loader" if not name.startswith("pca_pc_") else "regime_engine",
            relation_name=(
                "macro_features_daily"
                if not name.startswith("pca_pc_")
                else "pca_generated_features"
            ),
            relation_kind="BASE TABLE" if not name.startswith("pca_pc_") else "MATERIALIZED VIEW",
        )
        for ordinal, name in enumerate(names, 1)
    )
    return FeatureCatalogSnapshot.from_entries(lineage, "timestamp_m1", entries)


def _real_train_only_selector(train_rows, *, slot_id, fold):
    state_count = int(slot_id[1:])
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    selected = run_real_k_feature_selection(
        train_rows,
        catalog=_catalog(train_rows),
        profile=profile,
        source_snapshot_id="real-outer-snapshot",
        source_build_id=SOURCE_BUILD,
        validation_cutoff=fold.train_end,
        deployment_cutoff=fold.test_end,
        requested_state_counts=(state_count,),
        max_workers=1,
    )[0]
    if selected.rejection_reason and selected.rejection_reason.startswith(
        "RecoverableEvaluationInvalidity: "
    ):
        raise RecoverableEvaluationInvalidity(
            selected.rejection_reason.removeprefix("RecoverableEvaluationInvalidity: ")
        )
    return selected.selection


def _one_real_fit_multistart(train_rows, *, state_count, adapter_factory, **_kwargs):
    seed = 89
    result: FitResult = adapter_factory().fit(train_rows, state_count, seed)
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
    return MultistartResult(
        state_count=state_count,
        winner=result,
        diagnostics=diagnostics,
    )


def _real_selection(train_rows, *, slot_id, fold) -> KChampionSelection:
    del train_rows
    state_count = int(slot_id[1:])
    order = ("f0", "f1")
    return KChampionSelection(
        slot_id=slot_id,
        state_count=state_count,
        model_family="gaussian_hmm",
        candidate_identity=f"gaussian_hmm_k{state_count}_full",
        feature_order=order,
        feature_order_hash=feature_order_hash(order),
        source_snapshot_id="real-outer-snapshot",
        profile_id="xetra",
        profile_config_version=4,
        policy_id="k_champion_portfolio",
        policy_version="k_champion_portfolio.v1",
        validation_cutoff=fold.train_end,
        deployment_cutoff=fold.test_end,
        comparison_domain_id="k_slot_promotion.v1",
        promotion_score_version="k_slot_promotion.v1",
        reference_teacher_id=f"teacher-{state_count}",
        artifact_hash="a" * 64,
    )


def _real_outer_refit(
    train_rows,
    test_rows,
    *,
    selection: KChampionSelection,
    slot_id,
    fold,
) -> KChampionFoldEvaluation:
    del slot_id
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    source_rows = pd.concat((train_rows, test_rows), ignore_index=True)
    universe = tuple(column for column in source_rows.columns if column != "timestamp_m1")
    candidate = ResolvedCandidateProfile(
        candidate_id=selection.candidate_identity,
        state_count=selection.state_count,
        covariance_type="full",
        feature_order=selection.feature_order,
        feature_dimension=len(selection.feature_order),
        source_build_id=SOURCE_BUILD,
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        original_feature_universe=universe,
    )
    single_fold = replace(fold, fold_index=1, fold_id="fold_001")
    one_fold_plan = WalkForwardPlan(
        folds=(single_fold,),
        evaluation_cutoff=single_fold.test_end,
        plan_hash="c" * 64,
    )
    evaluation = run_walk_forward_candidate(
        source_rows,
        plan=one_fold_plan,
        profile=profile,
        candidate=candidate,
        adapter_factory=adapter_factory(profile, candidate),
        max_workers=1,
        pca_raw_feature_order=("f0", "f1"),
        pca_variance_threshold=profile.pca.variance_threshold,
    )
    fold_result = evaluation.folds[0]
    if not fold_result.valid:
        reason = fold_result.failure_reason or "real HMM outer refit failed"
        if reason.startswith("RecoverableEvaluationInvalidity: "):
            raise RecoverableEvaluationInvalidity(
                reason.removeprefix("RecoverableEvaluationInvalidity: ")
            )
        raise ValueError(reason)
    return KChampionFoldEvaluation(
        oos_timestamps=fold_result.oos_timestamps,
        oos_filtered_probabilities=fold_result.oos_filtered_probabilities,
        teacher_oos_timestamps=fold_result.oos_timestamps,
        teacher_oos_filtered_probabilities=fold_result.oos_filtered_probabilities,
        stability=1.0,
    )


def test_real_hmm_outer_slots_have_process_serial_parity_and_single_test_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(walk_forward, "run_multistart", _one_real_fit_multistart)
    rows = _rows()
    kwargs = dict(
        plan=_plan(),
        source_snapshot_id="real-outer-snapshot",
        profile_id="xetra",
        profile_config_version=4,
        validation_cutoff=_plan().evaluation_cutoff,
        selector=_real_selection,
        evaluator=_real_outer_refit,
    )

    parallel = run_k_champion_outer_policy(rows, **kwargs, max_workers=4)
    serial = run_k_champion_outer_policy(rows, **kwargs, max_workers=1)

    assert parallel == serial
    assert parallel.result_hash == serial.result_hash
    assert len(parallel.outer_folds) == 12
    assert {item.slot_id for item in parallel.slots} == {"k2", "k3", "k4", "k5"}
    assert all(len(item.fold_result_hashes) == 3 for item in parallel.slots)
    assert any(item.eligible for item in parallel.slots)
    assert all(
        item.valid_fold_count == 3 if item.eligible else item.valid_fold_count < 3
        for item in parallel.slots
    )
    assert all(item.selection is not None for item in parallel.outer_folds)
    assert all(
        item.shared_timestamp_count == 63
        and item.soft_regime_nmi is not None
        and 0.0 <= item.soft_regime_nmi <= 1.0
        for item in parallel.outer_folds
        if item.valid
    )


def test_outer_policy_invokes_the_real_fixed_k_selector_inside_each_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(walk_forward, "run_multistart", _one_real_fit_multistart)
    result = run_k_champion_outer_policy(
        _selection_rows(),
        plan=_plan(),
        source_snapshot_id="real-outer-snapshot",
        profile_id="xetra",
        profile_config_version=4,
        validation_cutoff=_plan().evaluation_cutoff,
        selector=_real_train_only_selector,
        evaluator=_real_outer_refit,
        max_workers=2,
    )

    assert len(result.outer_folds) == 12
    assert {item.slot_id for item in result.slots} == {"k2", "k3", "k4", "k5"}
    assert all(
        item.selection is None or item.selection.validation_cutoff == item.train_end
        for item in result.outer_folds
    )
    assert all(len(item.fold_result_hashes) == 3 for item in result.slots)
