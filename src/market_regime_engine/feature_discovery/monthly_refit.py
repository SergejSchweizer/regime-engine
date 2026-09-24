"""Canonical closed-month feature selection and package-freeze orchestration."""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.calendar_clock import (
    CalendarMonthFold,
    CalendarMonthPlan,
    plan_calendar_month,
)
from market_regime_engine.feature_discovery.ablation import (
    HMMSubsetEvaluation,
)
from market_regime_engine.feature_discovery.contracts import content_hash
from market_regime_engine.feature_discovery.feature_roles import (
    FeatureRoleContract,
    build_feature_role_contract_from_catalog,
)
from market_regime_engine.feature_discovery.metadata_store import (
    CorrelationMapping,
    FeatureRegistryRow,
    FeatureSelectionMetadataStore,
    FoldFeatureStat,
    FoldMetadataBundle,
    FoldModelStat,
    SFFSStepRecord,
    sffs_step_records,
)
from market_regime_engine.feature_discovery.pipeline import (
    FeatureSelectionPipelineResult,
    run_canonical_feature_selection,
)
from market_regime_engine.feature_discovery.provenance import build_feature_provenance
from market_regime_engine.feature_discovery.quality import (
    filter_outer_train_quality,
    quality_to_fold_feature_stats,
)
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore, SFFSResult
from market_regime_engine.features.ports import (
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)
from market_regime_engine.mlflow_support.canonical_diagnostics import (
    write_canonical_diagnostics,
)
from market_regime_engine.mlflow_support.ports import TrackingPort
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.runtime.cpu import cpu_worker_count

_TIMESTAMP = "timestamp_m1"
_FOLD_CHECKPOINT_SCHEMA_VERSION = 1
_FOLD_CHECKPOINT_ALGORITHM_VERSION = "pr-503-fold-checkpoint-v1"


def _sha(value: str, field: str) -> None:
    if (
        len(value) != 64
        or value != value.lower()
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")


def _utc(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


def _tuple_hash(values: Sequence[object]) -> str:
    return content_hash(tuple(values))


def _sffs_hash(result: object) -> str:
    """Hash SFFS evidence using only its canonical primitive fields."""

    selected = tuple(getattr(result, "selected_features", ()))
    best_singleton = str(getattr(result, "best_singleton", ""))
    steps: list[object] = []
    for step in getattr(result, "steps", ()):
        score = getattr(step, "score", None)
        score_payload = (
            None
            if score is None
            else (
                cast(Any, score).feature_names,
                cast(Any, score).value,
                cast(Any, score).metric,
                cast(Any, score).state_count,
            )
        )
        steps.append(
            (
                str(getattr(step, "action", "")),
                tuple(getattr(step, "selected_features", ())),
                score_payload,
            )
        )
    return content_hash((selected, best_singleton, steps))


@dataclass(frozen=True, slots=True)
class MonthlyPackageIdentity:
    """Identity of the package fitted through one closed-month TRAIN cutoff."""

    source_build_id: str
    source_catalog_hash: str
    train_cutoff: datetime
    available_for_test_month: str
    feature_selection_profile_hash: str
    feature_role_contract_hash: str
    provenance_hash: str
    family_pca_fit_hashes: tuple[str, ...]
    representative_hash: str
    selected_features: tuple[str, ...]
    state_count: int
    model_hashes: tuple[str, ...]
    outer_test_hash: str

    def __post_init__(self) -> None:
        if not self.source_build_id.strip():
            raise ValueError("package source build must be non-empty")
        for value, field in (
            (self.source_catalog_hash, "source_catalog_hash"),
            (self.feature_selection_profile_hash, "feature_selection_profile_hash"),
            (self.feature_role_contract_hash, "feature_role_contract_hash"),
            (self.provenance_hash, "provenance_hash"),
            (self.representative_hash, "representative_hash"),
            (self.outer_test_hash, "outer_test_hash"),
        ):
            _sha(value, field)
        for value in (*self.family_pca_fit_hashes, *self.model_hashes):
            _sha(value, "package component hash")
        _utc(self.train_cutoff, "train_cutoff")
        if not self.available_for_test_month or len(self.available_for_test_month) != 7:
            raise ValueError("package test month must use YYYY-MM")
        if not self.selected_features or len(set(self.selected_features)) != len(
            self.selected_features
        ):
            raise ValueError("package selected features must be non-empty and unique")
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("package state count must be 2, 3, 4 or 5")

    @property
    def package_hash(self) -> str:
        return content_hash(
            {
                "source_build_id": self.source_build_id,
                "source_catalog_hash": self.source_catalog_hash,
                "train_cutoff": self.train_cutoff,
                "available_for_test_month": self.available_for_test_month,
                "feature_selection_profile_hash": self.feature_selection_profile_hash,
                "feature_role_contract_hash": self.feature_role_contract_hash,
                "provenance_hash": self.provenance_hash,
                "family_pca_fit_hashes": self.family_pca_fit_hashes,
                "representative_hash": self.representative_hash,
                "selected_features": self.selected_features,
                "state_count": self.state_count,
                "model_hashes": self.model_hashes,
            }
        )


@dataclass(frozen=True, slots=True)
class MonthlyRefitFoldResult:
    fold: CalendarMonthFold
    package: MonthlyPackageIdentity | None
    pipeline: FeatureSelectionPipelineResult | None
    valid: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.valid != (self.package is not None and self.pipeline is not None):
            raise ValueError("monthly fold validity must match package and pipeline presence")
        if self.valid and self.failure_reason is not None:
            raise ValueError("valid monthly folds cannot carry a failure reason")
        if not self.valid and not self.failure_reason:
            raise ValueError("invalid monthly folds require a failure reason")


@dataclass(frozen=True, slots=True)
class MonthlyRefitResult:
    plan: CalendarMonthPlan
    source_build_id: str
    folds: tuple[MonthlyRefitFoldResult, ...]
    result_hash: str

    def __post_init__(self) -> None:
        if tuple(item.fold for item in self.folds) != self.plan.folds:
            raise ValueError("monthly refit results must preserve canonical plan order")
        if self.source_build_id.strip() == "":
            raise ValueError("monthly refit source build must be non-empty")
        _sha(self.result_hash, "result_hash")

    @property
    def valid_folds(self) -> tuple[MonthlyRefitFoldResult, ...]:
        return tuple(item for item in self.folds if item.valid)


SubsetEvaluator = Callable[[tuple[str, ...]], Any]
HMMSubsetEvaluator = Callable[[tuple[str, ...]], HMMSubsetEvaluation | None]
FinalHMMFitter = Callable[[pd.DataFrame, tuple[str, ...], int], str | Sequence[str]]
PerKSubsetEvaluator = Callable[[int, tuple[str, ...]], FeatureSubsetScore | None]
OuterTestEvaluator = Callable[
    [pd.DataFrame, pd.DataFrame, tuple[str, ...], int, tuple[str, ...]], str
]


class StageCallbacks(Protocol):
    evaluate_subset: SubsetEvaluator
    evaluate_hmm_subset: HMMSubsetEvaluator
    hmm_selector_contract_hash: str
    fit_final_hmm: FinalHMMFitter
    evaluate_gaussian_subset_by_k: PerKSubsetEvaluator
    evaluate_outer_test: OuterTestEvaluator


StageCallbackFactory = Callable[
    [pd.DataFrame, pd.DataFrame, CalendarMonthFold, int], StageCallbacks
]


def _snapshot(frame: pd.DataFrame, catalog: FeatureCatalogSnapshot) -> FeatureSnapshot:
    rows = tuple(
        FeatureRow(
            _utc(row[_TIMESTAMP], "source timestamp"),
            tuple(
                None if pd.isna(row[name]) else float(row[name]) for name in catalog.feature_names
            ),
        )
        for _, row in frame.iterrows()
    )
    if not rows:
        raise ValueError("monthly TRAIN frame cannot be empty")
    return FeatureSnapshot(catalog.lineage, catalog.feature_names, rows)


def _feature_values(
    snapshot: FeatureSnapshot, names: Sequence[str]
) -> dict[str, tuple[float | None, ...]]:
    positions = {name: index for index, name in enumerate(snapshot.feature_names)}
    return {name: tuple(row.values[positions[name]] for row in snapshot.rows) for name in names}


def _materialize_family_pca(
    frame: pd.DataFrame,
    contract: FeatureRoleContract,
    pipeline: FeatureSelectionPipelineResult,
) -> pd.DataFrame:
    """Apply TRAIN-fitted family PCA artifacts to one source frame."""

    output = frame.copy()
    for artifact in pipeline.family_pca:
        input_names = tuple(
            assignment.feature_name
            for assignment in contract.assignments
            if assignment.family == artifact.family
        )
        complete = output.loc[:, list(input_names)].notna().all(axis=1)
        generated: dict[str, list[float | None]] = {
            name: [None] * len(output) for name in artifact.generated_feature_names
        }
        if complete.any():
            matrix = output.loc[complete, list(input_names)].to_numpy(dtype=np.float64)
            transformed = artifact.transform(matrix)
            complete_positions = tuple(output.index[complete])
            for column, name in enumerate(artifact.generated_feature_names):
                for row_position, source_index in enumerate(complete_positions):
                    generated[name][output.index.get_loc(source_index)] = float(
                        transformed[row_position, column]
                    )
        for name, values in generated.items():
            output[name] = values
    return output


def _registry_rows(
    contract: FeatureRoleContract,
    catalog: FeatureCatalogSnapshot,
    first_seen: datetime,
) -> tuple[FeatureRegistryRow, ...]:
    rows = build_feature_provenance(contract)
    return tuple(
        FeatureRegistryRow(
            feature_identity=f"feature:{item.feature_identity}",
            source_dataset=catalog.lineage.source_dataset,
            source_build_id=catalog.lineage.source_build_id,
            source_catalog_hash=catalog.catalog_hash,
            feature_name=item.feature_name,
            role=item.role.value,
            family=item.family,
            transformation_name=item.transformation_name,
            transformation_parameters=dict(item.transformation_parameters),
            first_seen_utc=first_seen,
            lifecycle_status="ACTIVE",
        )
        for item in rows
    )


def _correlation_rows(
    result: FeatureSelectionPipelineResult,
    *,
    fold_id: str,
    profile_hash: str,
    source_build_id: str,
) -> tuple[CorrelationMapping, ...]:
    output: list[CorrelationMapping] = []
    for item in result.global_reduction.evidence:
        sub = tuple(item.subwindow_absolute_pearsons)
        supports = tuple(item.subwindow_support_counts)
        output.append(
            CorrelationMapping(
                fold_id,
                profile_hash,
                source_build_id,
                item.removed,
                item.leader,
                item.full_absolute_pearson,
                sub[0],
                sub[1],
                sub[2],
                item.full_support_count,
                supports[0],
                supports[1],
                supports[2],
                False,
                "stable redundant feature removed",
            )
        )
    return tuple(output)


def _metadata_bundle(
    *,
    fold: CalendarMonthFold,
    catalog: FeatureCatalogSnapshot,
    contract: FeatureRoleContract,
    quality_rows: tuple[FoldFeatureStat, ...],
    pipeline: FeatureSelectionPipelineResult,
    package: MonthlyPackageIdentity,
    model_family: str,
    model_run_id: str | None,
    sffs_result: SFFSResult,
    outer_test_hash: str,
) -> FoldMetadataBundle:
    selected_hash = _tuple_hash(sffs_result.selected_features)
    model_stat = FoldModelStat(
        fold_id=fold.fold_id,
        profile_hash=pipeline.profile_hash,
        source_build_id=catalog.lineage.source_build_id,
        selected_tuple_hash=selected_hash,
        state_count=package.state_count,
        model_family=model_family,
        valid=True,
        diagnostics={
            "package_hash": package.package_hash,
            "outer_test_hash": outer_test_hash,
            "train_cutoff": fold.train_cutoff_timestamp.isoformat(),
            "test_calendar_month": fold.test_calendar_month,
        },
        mlflow_run_id=model_run_id,
    )
    steps: tuple[SFFSStepRecord, ...] = sffs_step_records(
        sffs_result,
        fold_id=fold.fold_id,
        profile_hash=pipeline.profile_hash,
        source_build_id=catalog.lineage.source_build_id,
        state_count=package.state_count,
    )
    return FoldMetadataBundle(
        fold_id=fold.fold_id,
        profile_hash=pipeline.profile_hash,
        source_build_id=catalog.lineage.source_build_id,
        feature_registry=_registry_rows(contract, catalog, fold.train_first_timestamp),
        fold_feature_stats=quality_rows,
        pca_loadings=tuple(
            loading
            for artifact in pipeline.family_pca
            for loading in artifact.pca_loadings(fold.fold_id, catalog.lineage.source_build_id)
        ),
        correlation_mapping=_correlation_rows(
            pipeline,
            fold_id=fold.fold_id,
            profile_hash=pipeline.profile_hash,
            source_build_id=catalog.lineage.source_build_id,
        ),
        sffs_steps=steps,
        fold_model_stats=(model_stat,),
    )


def _track_stage(
    tracking: TrackingPort,
    parent_run_id: str,
    fold: CalendarMonthFold,
    stage: str,
    params: Mapping[str, object],
    diagnostics: FeatureSelectionPipelineResult | None = None,
) -> str:
    run_id = tracking.start_run(
        run_name=f"monthly-{fold.test_calendar_month}-{stage}",
        parent_run_id=parent_run_id,
    )
    try:
        with TemporaryDirectory(prefix="regime-monthly-stage-") as directory:
            manifest = Path(directory) / "stage-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "fold_id": fold.fold_id,
                        "stage": stage,
                        "test_calendar_month": fold.test_calendar_month,
                        "parameters": params,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            tracking.log_artifact(run_id, str(manifest), "stage")
            if diagnostics is not None:
                diagnostic_directory = Path(directory) / "diagnostics"
                for artifact in write_canonical_diagnostics(diagnostics, diagnostic_directory):
                    tracking.log_artifact(run_id, str(artifact), "diagnostics")
        tracking.log_params(
            run_id,
            {
                "fold_id": fold.fold_id,
                "test_calendar_month": fold.test_calendar_month,
                **{
                    key: json.dumps(value, sort_keys=True, default=str)
                    for key, value in params.items()
                },
            },
        )
        tracking.end_run(run_id)
    except Exception:
        tracking.fail_run(run_id)
        raise
    return run_id


@dataclass(frozen=True, slots=True)
class _ComputedMonthlyFold:
    fold: CalendarMonthFold
    source_data_sha256: str
    provenance: tuple[Any, ...]
    quality: Any
    pipeline: FeatureSelectionPipelineResult
    selected_sffs: SFFSResult
    selected_features: tuple[str, ...]
    model_hashes: tuple[str, ...]
    outer_test_hash: str
    package: MonthlyPackageIdentity


def _fold_checkpoint_path(
    metadata_store: FeatureSelectionMetadataStore, fold_id: str
) -> Path:
    root = metadata_store.database.parent / "fold-checkpoints"
    root.mkdir(mode=0o750, parents=True, exist_ok=True)
    return root / f"{fold_id}.pickle"


def _write_fold_checkpoint(
    metadata_store: FeatureSelectionMetadataStore,
    computed: _ComputedMonthlyFold,
    *,
    plan_hash: str,
    profile_hash: str,
    role_contract_hash: str,
) -> None:
    payload = {
        "schema_version": _FOLD_CHECKPOINT_SCHEMA_VERSION,
        "algorithm_version": _FOLD_CHECKPOINT_ALGORITHM_VERSION,
        "source_build_id": computed.package.source_build_id,
        "source_data_sha256": computed.source_data_sha256,
        "source_catalog_hash": computed.package.source_catalog_hash,
        "plan_hash": plan_hash,
        "fold_id": computed.fold.fold_id,
        "fold_clock_hash": computed.fold.month_clock_hash,
        "profile_hash": profile_hash,
        "role_contract_hash": role_contract_hash,
        "package_hash": computed.package.package_hash,
        "computed": computed,
    }
    path = _fold_checkpoint_path(metadata_store, computed.fold.fold_id)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def _read_fold_checkpoint(
    metadata_store: FeatureSelectionMetadataStore,
    *,
    fold: CalendarMonthFold,
    source_build_id: str,
    source_data_sha256: str,
    source_catalog_hash: str,
    plan_hash: str,
    profile_hash: str,
    role_contract_hash: str,
) -> _ComputedMonthlyFold | None:
    path = _fold_checkpoint_path(metadata_store, fold.fold_id)
    if not path.is_file():
        return None
    try:
        payload = pickle.loads(path.read_bytes())
        computed = payload["computed"]
        if (
            payload["schema_version"] != _FOLD_CHECKPOINT_SCHEMA_VERSION
            or payload["algorithm_version"] != _FOLD_CHECKPOINT_ALGORITHM_VERSION
            or payload["source_build_id"] != source_build_id
            or payload["source_data_sha256"] != source_data_sha256
            or payload["source_catalog_hash"] != source_catalog_hash
            or payload["plan_hash"] != plan_hash
            or payload["fold_id"] != fold.fold_id
            or payload["fold_clock_hash"] != fold.month_clock_hash
            or payload["profile_hash"] != profile_hash
            or payload["role_contract_hash"] != role_contract_hash
            or not isinstance(computed, _ComputedMonthlyFold)
            or computed.fold != fold
            or computed.package.package_hash != payload["package_hash"]
        ):
            return None
        return computed
    except (
        AttributeError,
        EOFError,
        IndexError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        pickle.PickleError,
    ):
        return None


def _fold_resource_partition(
    fold_count: int, requested_workers: int | None
) -> tuple[int, int]:
    if fold_count < 1:
        raise ValueError("fold_count must be positive")
    total = cpu_worker_count(requested_workers)
    if total == 1:
        return 1, 1
    configured_outer = int(os.environ.get("REGIME_OUTER_FOLD_WORKERS", "4"))
    if configured_outer < 1:
        raise ValueError("REGIME_OUTER_FOLD_WORKERS must be positive")
    outer = min(fold_count, total, configured_outer)
    return outer, max(1, total // outer)


def _compute_monthly_fold(
    source_rows: pd.DataFrame,
    *,
    catalog: FeatureCatalogSnapshot,
    profile: ModelProfile,
    contract: FeatureRoleContract,
    fold: CalendarMonthFold,
    evaluate_subset: SubsetEvaluator,
    evaluate_hmm_subset: HMMSubsetEvaluator,
    hmm_selector_contract_hash: str,
    fit_final_hmm: FinalHMMFitter,
    evaluate_gaussian_subset_by_k: PerKSubsetEvaluator,
    evaluate_outer_test: OuterTestEvaluator,
    stage_callback_factory: StageCallbackFactory | None,
    inner_workers: int,
    state_count: int,
) -> _ComputedMonthlyFold:
    train = source_rows.iloc[: fold.train_source_observations].copy()
    if train[_TIMESTAMP].iloc[-1] != fold.train_cutoff_timestamp:
        raise ValueError("monthly TRAIN prefix does not end at the calendar cutoff")
    provenance = build_feature_provenance(contract)
    snapshot = _snapshot(train, catalog)
    quality = filter_outer_train_quality(
        catalog,
        snapshot,
        fold.train_first_timestamp,
        fold.train_cutoff_timestamp,
        max_workers=inner_workers,
    )
    values = _feature_values(snapshot, quality.eligible_features)
    test = source_rows.iloc[
        fold.train_source_observations : fold.train_source_observations
        + fold.test_source_observations
    ].copy()
    selection_train = train.loc[
        train.loc[:, list(quality.eligible_features)].notna().all(axis=1)
    ].copy()
    if selection_train.empty:
        raise ValueError("quality-eligible features have no shared finite TRAIN rows")
    stage_callbacks = (
        stage_callback_factory(selection_train, test, fold, inner_workers)
        if stage_callback_factory is not None
        else None
    )
    current_evaluate_subset = (
        cast(Any, stage_callbacks.evaluate_subset)
        if stage_callbacks is not None
        else evaluate_subset
    )
    current_evaluate_hmm_subset = (
        cast(Any, stage_callbacks.evaluate_hmm_subset)
        if stage_callbacks is not None
        else evaluate_hmm_subset
    )
    current_selector_hash = (
        str(stage_callbacks.hmm_selector_contract_hash)
        if stage_callbacks is not None
        else hmm_selector_contract_hash
    )
    current_evaluate_gaussian_subset_by_k = (
        cast(Any, stage_callbacks.evaluate_gaussian_subset_by_k)
        if stage_callbacks is not None
        else evaluate_gaussian_subset_by_k
    )
    pipeline = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=quality.eligible_features,
        evaluate_subset=current_evaluate_subset,
        evaluate_hmm_subset=current_evaluate_hmm_subset,
        hmm_selector_contract_hash=current_selector_hash,
        evaluate_gaussian_subset_by_k=current_evaluate_gaussian_subset_by_k,
        state_counts=(2, 3, 4, 5),
        profile=contract.profile,
        max_workers=inner_workers,
    )
    selection_by_k = {item.state_count: item.sffs for item in pipeline.k_sffs}
    selected_sffs = selection_by_k.get(state_count, pipeline.sffs)
    selected_features = selected_sffs.selected_features
    fit_callbacks = (
        stage_callback_factory(
            _materialize_family_pca(train, contract, pipeline),
            _materialize_family_pca(test, contract, pipeline),
            fold,
            inner_workers,
        )
        if stage_callback_factory is not None
        else None
    )
    current_fit_final_hmm = (
        cast(Any, fit_callbacks.fit_final_hmm)
        if fit_callbacks is not None
        else fit_final_hmm
    )
    fitted_hashes = (
        cast(Any, fit_callbacks.fit_final_hmm)(
            _materialize_family_pca(train, contract, pipeline),
            selected_features,
            state_count,
        )
        if fit_callbacks is not None
        else current_fit_final_hmm(train, selected_features, 2)
    )
    model_hashes = (fitted_hashes,) if isinstance(fitted_hashes, str) else tuple(fitted_hashes)
    if not model_hashes:
        raise ValueError("final HMM fit returned no model hash")
    for model_hash in model_hashes:
        _sha(model_hash, "final HMM model hash")
    current_evaluate_outer_test = (
        cast(Any, fit_callbacks.evaluate_outer_test)
        if fit_callbacks is not None
        else evaluate_outer_test
    )
    outer_test_hash = current_evaluate_outer_test(
        train, test, selected_features, state_count, model_hashes
    )
    _sha(outer_test_hash, "outer TEST hash")
    package = MonthlyPackageIdentity(
        source_build_id=catalog.lineage.source_build_id,
        source_catalog_hash=catalog.catalog_hash,
        train_cutoff=fold.train_cutoff_timestamp,
        available_for_test_month=fold.test_calendar_month,
        feature_selection_profile_hash=pipeline.profile_hash,
        feature_role_contract_hash=pipeline.role_contract_hash,
        provenance_hash=_tuple_hash(tuple(item.canonical_dict for item in provenance)),
        family_pca_fit_hashes=tuple(item.fit_hash for item in pipeline.family_pca),
        representative_hash=content_hash(pipeline.global_reduction.representatives),
        selected_features=selected_features,
        state_count=state_count,
        model_hashes=model_hashes,
        outer_test_hash=outer_test_hash,
    )
    return _ComputedMonthlyFold(
        fold,
        catalog.lineage.data_sha256,
        provenance,
        quality,
        pipeline,
        selected_sffs,
        selected_features,
        model_hashes,
        outer_test_hash,
        package,
    )


def run_monthly_outer_refit(
    source_rows: pd.DataFrame,
    *,
    catalog: FeatureCatalogSnapshot,
    profile: ModelProfile,
    evaluate_subset: SubsetEvaluator,
    evaluate_hmm_subset: HMMSubsetEvaluator,
    hmm_selector_contract_hash: str,
    fit_final_hmm: FinalHMMFitter,
    evaluate_gaussian_subset_by_k: PerKSubsetEvaluator,
    evaluate_outer_test: OuterTestEvaluator,
    metadata_store: FeatureSelectionMetadataStore | None = None,
    tracking: TrackingPort | None = None,
    max_workers: int | None = None,
    evaluation_cutoff: datetime | None = None,
    state_count: int = 2,
    stage_callback_factory: StageCallbackFactory | None = None,
) -> MonthlyRefitResult:
    """Rerun canonical selection at every closed month and freeze each package.

    Selection receives only the expanding TRAIN prefix ending at the fold's
    closed-month cutoff.  The following calendar month is represented only by
    the package availability identity and is never passed to selection or fit.
    A fold is committed to DuckDB only after the final HMM fit and MLflow stage
    runs have completed successfully.
    """

    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("monthly refit source must be a pandas DataFrame")
    if _TIMESTAMP not in source_rows or tuple(catalog.feature_names) == ():
        raise ValueError("monthly refit source requires timestamp and catalog features")
    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("monthly refit requires canonical Xetra v4 profile")
    if not callable(evaluate_gaussian_subset_by_k) or not callable(evaluate_outer_test):
        raise TypeError("monthly refit requires per-K selection and Outer TEST evaluators")
    timestamps = tuple(_utc(value, "source timestamp") for value in source_rows[_TIMESTAMP])
    if any(right <= left for left, right in pairwise(timestamps)):
        raise ValueError("monthly refit timestamps must be strictly increasing")
    if len(timestamps) != len(source_rows):
        raise ValueError("monthly refit timestamps are malformed")
    expected = {_TIMESTAMP, *catalog.feature_names}
    if set(source_rows.columns) != expected:
        raise ValueError("monthly refit source columns must equal the canonical catalog")
    contract = build_feature_role_contract_from_catalog(catalog)
    plan = plan_calendar_month(
        timestamps,
        minimum_train_source_observations=profile.feature_discovery.outer_train_source_observations,
        evaluation_cutoff=evaluation_cutoff,
    )
    parent_run_id: str | None = None
    if tracking is not None:
        parent_run_id = tracking.start_run(
            run_name=f"monthly-outer-refit-{catalog.lineage.source_build_id}"
        )
        tracking.log_params(
            parent_run_id,
            {
                "source_build_id": catalog.lineage.source_build_id,
                "source_catalog_hash": catalog.catalog_hash,
                "feature_selection_profile_hash": contract.profile.profile_hash,
                "month_plan_hash": plan.plan_hash,
            },
        )
    results: list[MonthlyRefitFoldResult] = []
    try:
        if stage_callback_factory is None:
            # Caller-supplied callbacks may carry mutable test or application
            # state.  The production factory is immutable and fold-local, so
            # only that path is eligible for outer-fold concurrency.
            outer_workers, inner_workers = 1, max(1, cpu_worker_count(max_workers))
        else:
            outer_workers, inner_workers = _fold_resource_partition(
                len(plan.folds), max_workers
            )
        fold_futures = {}
        cached_folds: dict[str, _ComputedMonthlyFold] = {}
        checkpoint_store = (
            metadata_store
            if metadata_store is not None and hasattr(metadata_store, "database")
            else None
        )
        if checkpoint_store is not None:
            for fold in plan.folds:
                cached = _read_fold_checkpoint(
                    checkpoint_store,
                    fold=fold,
                    source_build_id=catalog.lineage.source_build_id,
                    source_data_sha256=catalog.lineage.data_sha256,
                    source_catalog_hash=catalog.catalog_hash,
                    plan_hash=plan.plan_hash,
                    profile_hash=contract.profile.profile_hash,
                    role_contract_hash=contract.contract_hash,
                )
                if cached is not None:
                    cached_folds[fold.fold_id] = cached
        with ThreadPoolExecutor(max_workers=outer_workers) as fold_executor:
            for fold in plan.folds:
                if fold.fold_id in cached_folds:
                    continue
                fold_futures[fold.fold_id] = fold_executor.submit(
                    _compute_monthly_fold,
                    source_rows,
                    catalog=catalog,
                    profile=profile,
                    contract=contract,
                    fold=fold,
                    evaluate_subset=evaluate_subset,
                    evaluate_hmm_subset=evaluate_hmm_subset,
                    hmm_selector_contract_hash=hmm_selector_contract_hash,
                    fit_final_hmm=fit_final_hmm,
                    evaluate_gaussian_subset_by_k=evaluate_gaussian_subset_by_k,
                    evaluate_outer_test=evaluate_outer_test,
                    stage_callback_factory=stage_callback_factory,
                    inner_workers=inner_workers,
                    state_count=state_count,
                )
            for fold in plan.folds:
                reused = fold.fold_id in cached_folds
                fold_computation = fold_futures.get(fold.fold_id)
                child_runs: list[str] = []
                try:
                    if reused:
                        computed = cached_folds[fold.fold_id]
                    else:
                        if fold_computation is None:
                            raise RuntimeError("missing monthly fold computation")
                        computed = fold_computation.result()
                    provenance = computed.provenance
                    quality = computed.quality
                    pipeline = computed.pipeline
                    selected_sffs = computed.selected_sffs
                    package = computed.package
                    if parent_run_id is not None and not reused:
                        assert tracking is not None
                        child_runs.append(
                            _track_stage(
                                tracking,
                                parent_run_id,
                                fold,
                                "provenance",
                                {
                                    "hash": _tuple_hash(
                                        tuple(item.canonical_dict for item in provenance)
                                    )
                                },
                            )
                        )
                        child_runs.append(
                            _track_stage(
                                tracking,
                                parent_run_id,
                                fold,
                                "quality",
                                {"quality_hash": quality.result_hash},
                            )
                        )
                        for stage, params in (
                            (
                                "family_pca",
                                {"hashes": tuple(item.fit_hash for item in pipeline.family_pca)},
                            ),
                            ("correlation", {"hash": pipeline.global_reduction.result_hash}),
                            ("sffs", {"hash": _sffs_hash(selected_sffs)}),
                            ("ablation", {"hashes": pipeline.ablation.fit_execution_hashes}),
                        ):
                            child_runs.append(
                                _track_stage(
                                    tracking,
                                    parent_run_id,
                                    fold,
                                    stage,
                                    params,
                                    diagnostics=(
                                        pipeline
                                        if stage == "family_pca"
                                        and isinstance(pipeline, FeatureSelectionPipelineResult)
                                        else None
                                    ),
                                )
                            )
                        model_run_id = _track_stage(
                            tracking,
                            parent_run_id,
                            fold,
                            "final_hmm",
                            {
                                "package_hash": package.package_hash,
                                "model_hashes": computed.model_hashes,
                            },
                        )
                        child_runs.append(model_run_id)
                        child_runs.append(
                            _track_stage(
                                tracking,
                                parent_run_id,
                                fold,
                                "outer_test",
                                {"hash": computed.outer_test_hash},
                            )
                        )
                    elif parent_run_id is not None:
                        assert tracking is not None
                        tracking.log_params(
                            parent_run_id,
                            {
                                f"{fold.fold_id}.checkpoint_reused": "true",
                                f"{fold.fold_id}.package_hash": package.package_hash,
                            },
                        )
                        model_run_id = None
                    else:
                        model_run_id = None
                    if metadata_store is not None and not reused:
                        metadata_store.commit_fold(
                            _metadata_bundle(
                                fold=fold,
                                catalog=catalog,
                                contract=contract,
                                quality_rows=quality_to_fold_feature_stats(
                                    quality,
                                    fold_id=fold.fold_id,
                                    profile_hash=pipeline.profile_hash,
                                    role_contract=contract,
                                ),
                                pipeline=pipeline,
                                package=package,
                                model_family="gaussian_hmm",
                                model_run_id=model_run_id,
                                sffs_result=selected_sffs,
                                outer_test_hash=computed.outer_test_hash,
                            )
                        )
                        if checkpoint_store is not None:
                            _write_fold_checkpoint(
                                checkpoint_store,
                                computed,
                                plan_hash=plan.plan_hash,
                                profile_hash=contract.profile.profile_hash,
                                role_contract_hash=pipeline.role_contract_hash,
                            )
                    results.append(MonthlyRefitFoldResult(fold, package, pipeline, True))
                except (ValueError, TypeError, KeyError, TimeoutError) as exc:
                    if parent_run_id is not None:
                        assert tracking is not None
                        tracking.log_params(
                            parent_run_id, {f"{fold.fold_id}.failure": str(exc)}
                        )
                    results.append(
                        MonthlyRefitFoldResult(
                            fold, None, None, False, f"{type(exc).__name__}: {exc}"
                        )
                    )
        result_hash = content_hash(
            (
                catalog.lineage.source_build_id,
                plan.plan_hash,
                tuple(
                    None
                    if not item.valid
                    else (item.package.package_hash if item.package is not None else None)
                    for item in results
                ),
            )
        )
        return MonthlyRefitResult(
            plan, catalog.lineage.source_build_id, tuple(results), result_hash
        )
    finally:
        if parent_run_id is not None:
            assert tracking is not None
            tracking.end_run(parent_run_id)


__all__ = [
    "MonthlyPackageIdentity",
    "MonthlyRefitFoldResult",
    "MonthlyRefitResult",
    "run_monthly_outer_refit",
]
