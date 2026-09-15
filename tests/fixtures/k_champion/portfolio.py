"""Real, hermetic inputs for the four-slot K-champion E2E proof.

The fixture deliberately keeps model fitting outside the assertion code.  All
fits are performed by the production Gaussian, GMM-HMM, and Student-t adapter
factories; the portfolio modules then receive only immutable, pickleable model
records.  This keeps the expensive numerical work process-parallel while the
outer policy and deployment orchestration remain the system under test.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from market_regime_engine.evaluation.walk_forward_splits import (
    WalkForwardFold,
    WalkForwardPlan,
    plan_walk_forward,
)
from market_regime_engine.evaluations.k_champion_contract import (
    KChampionSelection,
    feature_order_hash,
)
from market_regime_engine.evaluations.k_champion_outer import (
    KChampionFoldEvaluation,
    KChampionOuterPolicyResult,
    run_k_champion_outer_policy,
)
from market_regime_engine.evaluations.k_deployment_selection import (
    KDeploymentArtifact,
    KDeploymentSelectionResult,
    select_k_deployment_packages,
)
from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.mlflow_support.k_slot_metrics import (
    KCandidateMetricProjection,
    KSlotMetadata,
    KSlotMetricProjection,
    build_four_k_slot_projection,
    build_k_slot_projection,
)
from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.training.adapter_factory import adapter_factory

RAW_FEATURES = ("f0", "f1", "f2", "f3")
PCA_FEATURES = tuple(f"pca_pc_{index:03d}" for index in range(1, 9))
FEATURE_UNIVERSE = RAW_FEATURES + PCA_FEATURES
FAMILIES = ("gaussian_hmm", "gmm_hmm", "student_t_hmm")
K_VALUES = (2, 3, 4, 5)
FIT_SEEDS = (11, 23, 37, 53, 71, 89, 107, 131)
FEATURE_ORDER_BY_K = {
    2: ("f0", "f1"),
    3: ("f0", "f2"),
    4: ("f1", "f3"),
    5: ("f0", "f3"),
}
SOURCE_BUILD_ID = "hermetic-k-champion-e2e-v1"
SOURCE_SNAPSHOT_ID = "hermetic-k-champion-snapshot-v1"
PROFILE_ID = "xetra"
PROFILE_CONFIG_VERSION = 4
SEED = 11


def source_rows(*, row_count: int = 1512) -> pd.DataFrame:
    """Create deterministic raw plus generated-PCA-shaped source columns."""

    start = datetime(2020, 1, 1, tzinfo=UTC)
    index = np.arange(row_count, dtype=np.float64)
    regime = (np.arange(row_count) // 31) % 4
    centers = np.asarray(
        (
            (-1.5, 0.8, -0.4, 1.2),
            (-0.4, -1.2, 1.3, 0.1),
            (0.9, 1.4, -1.0, -1.1),
            (1.7, -0.3, 0.5, 0.6),
        ),
        dtype=np.float64,
    )
    noise = np.random.default_rng(90210)
    raw = centers[regime] + noise.normal(0.0, 0.16, size=(row_count, len(RAW_FEATURES)))
    frame = pd.DataFrame(
        raw,
        columns=RAW_FEATURES,
    )
    frame.insert(
        0,
        "timestamp_m1",
        tuple(start + timedelta(minutes=int(value)) for value in index),
    )
    for component in range(1, 9):
        frame[f"pca_pc_{component:03d}"] = np.sin(index / (13.0 + component))
    return frame


def _candidate(
    state_count: int,
    family: str,
    feature_order: tuple[str, ...],
) -> ResolvedCandidateProfile:
    mixture_count = 2 if family == "gmm_hmm" else 1
    candidate_id = (
        f"gmm_hmm_k{state_count}_m2_full"
        if family == "gmm_hmm"
        else f"{family}_k{state_count}_full"
    )
    return ResolvedCandidateProfile(
        candidate_id=candidate_id,
        state_count=state_count,
        covariance_type="full",
        feature_order=feature_order,
        feature_dimension=len(feature_order),
        source_build_id=SOURCE_BUILD_ID,
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        original_feature_universe=FEATURE_UNIVERSE,
        model_family=family,
        mixture_count=mixture_count,
    )


def _artifact_hash(artifact: GaussianHMMArtifact) -> str:
    payload = json.dumps(
        {
            "state_count": artifact.state_count,
            "feature_order": artifact.feature_order,
            "start_probabilities": artifact.start_probabilities,
            "transition_matrix": artifact.transition_matrix,
            "means": artifact.means,
            "full_covariances": artifact.full_covariances,
            "model_family": artifact.model_family,
            "mixture_weights": artifact.mixture_weights,
            "mixture_means": artifact.mixture_means,
            "mixture_full_covariances": artifact.mixture_full_covariances,
            "degrees_of_freedom": artifact.degrees_of_freedom,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelFitRecord:
    fold_id: str
    slot_id: str
    model_family: str
    feature_order: tuple[str, ...]
    artifact: GaussianHMMArtifact
    train_log_likelihood: float

    @property
    def state_count(self) -> int:
        return int(self.slot_id[1:])


@dataclass(frozen=True, slots=True)
class _FitTask:
    rows: pd.DataFrame
    fold_id: str
    slot_id: str
    model_family: str
    feature_order: tuple[str, ...]
    profile: ModelProfile


def _fit_task(task: _FitTask) -> ModelFitRecord:
    state_count = int(task.slot_id[1:])
    candidate = _candidate(state_count, task.model_family, task.feature_order)
    values = task.rows.loc[:, list(task.feature_order)].to_numpy(dtype=np.float64, copy=True)
    fit = _fit_real(task.profile, candidate, values, state_count)
    return ModelFitRecord(
        fold_id=task.fold_id,
        slot_id=task.slot_id,
        model_family=task.model_family,
        feature_order=task.feature_order,
        artifact=fit.artifact,
        train_log_likelihood=float(fit.train_log_likelihood),
    )


def _fit_real(
    profile: ModelProfile,
    candidate: ResolvedCandidateProfile,
    values: np.ndarray,
    state_count: int,
) -> object:
    """Run real adapter fits, retrying only with another pinned production seed."""

    fit = None
    for seed in FIT_SEEDS:
        try:
            fit = adapter_factory(profile, candidate)().fit(values, state_count, seed)
        except ValueError, np.linalg.LinAlgError:
            continue
        break
    if fit is None:
        raise ValueError(f"all pinned real starts failed for {candidate.candidate_id}")
    return fit


def _run_fit_tasks(
    tasks: Sequence[_FitTask], *, max_workers: int | None
) -> tuple[ModelFitRecord, ...]:
    ordered = tuple(tasks)
    if not ordered:
        raise ValueError("hermetic portfolio requires at least one fit task")
    if max_workers == 1:
        results = tuple(_fit_task(task) for task in ordered)
    else:
        with cpu_process_pool(max_workers) as executor:
            results = tuple(executor.map(_fit_task, ordered))
    return tuple(sorted(results, key=lambda item: (item.fold_id, item.slot_id, item.model_family)))


@dataclass(frozen=True, slots=True)
class HermeticPortfolioModels:
    candidates: tuple[ModelFitRecord, ...]
    refits: tuple[ModelFitRecord, ...]
    teachers: tuple[ModelFitRecord, ...]

    def candidates_for(self, fold_id: str, slot_id: str) -> tuple[ModelFitRecord, ...]:
        result = tuple(
            item for item in self.candidates if item.fold_id == fold_id and item.slot_id == slot_id
        )
        if tuple(item.model_family for item in result) != FAMILIES:
            raise AssertionError("every K/fold must retain all three family fits")
        if len({item.feature_order for item in result}) != 1:
            raise AssertionError("families in one K slot do not share one feature tuple")
        return result

    def refit_for(self, fold_id: str, slot_id: str) -> ModelFitRecord:
        return next(
            item for item in self.refits if item.fold_id == fold_id and item.slot_id == slot_id
        )

    def teacher_for(self, fold_id: str, slot_id: str) -> ModelFitRecord:
        return next(
            item for item in self.teachers if item.fold_id == fold_id and item.slot_id == slot_id
        )


def build_models(
    rows: pd.DataFrame,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    *,
    max_workers: int | None = None,
) -> HermeticPortfolioModels:
    """Fit all candidates, then one winner refit and one teacher per slot/fold."""

    candidate_tasks = tuple(
        _FitTask(
            rows=rows.iloc[: fold.train_source_observations].copy(deep=True),
            fold_id=fold.fold_id,
            slot_id=f"k{state_count}",
            model_family=family,
            feature_order=FEATURE_ORDER_BY_K[state_count],
            profile=profile,
        )
        for fold in plan.folds
        for state_count in K_VALUES
        for family in FAMILIES
    )
    candidates = _run_fit_tasks(candidate_tasks, max_workers=max_workers)
    by_fold_slot = {
        (fold_id, slot_id): tuple(
            item for item in candidates if item.fold_id == fold_id and item.slot_id == slot_id
        )
        for fold_id in tuple(fold.fold_id for fold in plan.folds)
        for slot_id in tuple(f"k{k}" for k in K_VALUES)
    }
    winners = tuple(
        max(
            items,
            key=lambda item: (
                item.train_log_likelihood,
                -FAMILIES.index(item.model_family),
            ),
        )
        for items in by_fold_slot.values()
    )
    refit_tasks = tuple(
        _FitTask(
            rows=rows.iloc[: fold.train_source_observations].copy(deep=True),
            fold_id=fold.fold_id,
            slot_id=winner.slot_id,
            model_family=winner.model_family,
            feature_order=winner.feature_order,
            profile=profile,
        )
        for fold in plan.folds
        for winner in winners
        if winner.fold_id == fold.fold_id
    )
    teacher_tasks = tuple(
        _FitTask(
            rows=rows.iloc[: fold.train_source_observations].copy(deep=True),
            fold_id=fold.fold_id,
            slot_id=f"k{state_count}",
            model_family="gaussian_hmm",
            feature_order=FEATURE_ORDER_BY_K[state_count],
            profile=profile,
        )
        for fold in plan.folds
        for state_count in K_VALUES
    )
    return HermeticPortfolioModels(
        candidates=candidates,
        refits=_run_fit_tasks(refit_tasks, max_workers=max_workers),
        teachers=_run_fit_tasks(teacher_tasks, max_workers=max_workers),
    )


def _selection_from_record(
    record: ModelFitRecord,
    *,
    validation_cutoff: datetime,
    deployment_cutoff: datetime,
) -> KChampionSelection:
    return KChampionSelection(
        slot_id=record.slot_id,
        state_count=record.state_count,
        model_family=record.model_family,
        candidate_identity=_candidate(
            record.state_count, record.model_family, record.feature_order
        ).candidate_id,
        feature_order=record.feature_order,
        feature_order_hash=feature_order_hash(record.feature_order),
        source_snapshot_id=SOURCE_SNAPSHOT_ID,
        profile_id=PROFILE_ID,
        profile_config_version=PROFILE_CONFIG_VERSION,
        policy_id="k_champion_portfolio",
        policy_version="k_champion_portfolio.v1",
        validation_cutoff=validation_cutoff,
        deployment_cutoff=deployment_cutoff,
        comparison_domain_id="k_slot_promotion.v1",
        promotion_score_version="k_slot_promotion.v1",
        reference_teacher_id=f"teacher-{record.slot_id}",
        artifact_hash=_artifact_hash(record.artifact),
    )


@dataclass(frozen=True, slots=True)
class RealFoldSelector:
    models: HermeticPortfolioModels
    validation_cutoff: datetime
    deployment_cutoff: datetime
    ineligible_slots: tuple[str, ...] = ()

    def __call__(
        self,
        train_rows: pd.DataFrame,
        *,
        slot_id: str,
        fold: WalkForwardFold,
    ) -> KChampionSelection | None:
        del train_rows
        candidates = self.models.candidates_for(fold.fold_id, slot_id)
        winner = max(
            candidates,
            key=lambda item: (item.train_log_likelihood, -FAMILIES.index(item.model_family)),
        )
        if slot_id in self.ineligible_slots:
            return None
        return _selection_from_record(
            winner,
            validation_cutoff=self.validation_cutoff,
            deployment_cutoff=self.deployment_cutoff,
        )


@dataclass(frozen=True, slots=True)
class RealFoldEvaluator:
    models: HermeticPortfolioModels
    profile: ModelProfile

    def __call__(
        self,
        train_rows: pd.DataFrame,
        test_rows: pd.DataFrame,
        *,
        selection: KChampionSelection,
        slot_id: str,
        fold: WalkForwardFold,
    ) -> KChampionFoldEvaluation:
        refit = self.models.refit_for(fold.fold_id, slot_id)
        teacher = self.models.teacher_for(fold.fold_id, slot_id)
        if refit.model_family != selection.model_family:
            raise ValueError("refit family differs from frozen K selection")
        candidate = _candidate(
            selection.state_count, selection.model_family, selection.feature_order
        )
        candidate_adapter = adapter_factory(self.profile, candidate)()
        candidate_adapter.reconstruct(refit.artifact)
        teacher_candidate = _candidate(
            selection.state_count, "gaussian_hmm", selection.feature_order
        )
        teacher_adapter = adapter_factory(self.profile, teacher_candidate)()
        teacher_adapter.reconstruct(teacher.artifact)
        train_values = train_rows.loc[:, list(selection.feature_order)].to_numpy(
            dtype=np.float64, copy=True
        )
        test_values = test_rows.loc[:, list(selection.feature_order)].to_numpy(
            dtype=np.float64, copy=True
        )
        candidate_train = candidate_adapter.causal_filter(train_values)
        teacher_train = teacher_adapter.causal_filter(train_values)
        candidate_test = candidate_adapter.causal_filter(
            test_values, candidate_train.terminal_probabilities
        )
        teacher_test = teacher_adapter.causal_filter(
            test_values, teacher_train.terminal_probabilities
        )
        stability = float(np.mean(np.max(candidate_test.filtered_probabilities, axis=1)))
        return KChampionFoldEvaluation(
            oos_timestamps=tuple(test_rows["timestamp_m1"]),
            oos_filtered_probabilities=tuple(
                tuple(float(value) for value in row)
                for row in candidate_test.filtered_probabilities
            ),
            teacher_oos_timestamps=tuple(test_rows["timestamp_m1"]),
            teacher_oos_filtered_probabilities=tuple(
                tuple(float(value) for value in row) for row in teacher_test.filtered_probabilities
            ),
            stability=stability,
        )


@dataclass(frozen=True, slots=True)
class RefitDeploymentSelector:
    selection_by_slot: Mapping[str, KChampionSelection]

    def __call__(
        self,
        source_rows: pd.DataFrame,
        *,
        slot_id: str,
        deployment_cutoff: datetime,
    ) -> KChampionSelection:
        if tuple(source_rows["timestamp_m1"])[-1] != deployment_cutoff:
            raise ValueError("deployment selector did not receive the full source maximum")
        return self.selection_by_slot[slot_id]


@dataclass(frozen=True, slots=True)
class RealDeploymentRefitter:
    profile: ModelProfile
    output_directory: Path

    def __call__(
        self,
        source_rows: pd.DataFrame,
        *,
        selection: KChampionSelection,
        slot_id: str,
    ) -> KDeploymentArtifact:
        candidate = _candidate(
            selection.state_count, selection.model_family, selection.feature_order
        )
        values = source_rows.loc[:, list(selection.feature_order)].to_numpy(
            dtype=np.float64, copy=True
        )
        fit = _fit_real(self.profile, candidate, values, selection.state_count)
        directory = self.output_directory / slot_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.json").write_text(
            json.dumps(
                {
                    "candidate_id": candidate.candidate_id,
                    "artifact_hash": _artifact_hash(fit.artifact),
                    "feature_order": list(selection.feature_order),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return KDeploymentArtifact(
            slot_id=slot_id,
            selection_hash=selection.selection_hash,
            artifact_hash=selection.artifact_hash,
            package_directory=str(directory),
            source_snapshot_id=selection.source_snapshot_id,
            deployment_cutoff=selection.deployment_cutoff,
        )


@dataclass(frozen=True, slots=True)
class HermeticPortfolio:
    rows: pd.DataFrame
    profile: ModelProfile
    plan: WalkForwardPlan
    models: HermeticPortfolioModels
    source_data_sha256: str
    validation_cutoff: datetime
    deployment_cutoff: datetime

    @property
    def selector(self) -> RealFoldSelector:
        return RealFoldSelector(self.models, self.validation_cutoff, self.deployment_cutoff)

    @property
    def evaluator(self) -> RealFoldEvaluator:
        return RealFoldEvaluator(self.models, self.profile)

    def outer(self, *, max_workers: int | None) -> KChampionOuterPolicyResult:
        return run_k_champion_outer_policy(
            self.rows,
            plan=self.plan,
            source_snapshot_id=SOURCE_SNAPSHOT_ID,
            profile_id=PROFILE_ID,
            profile_config_version=PROFILE_CONFIG_VERSION,
            validation_cutoff=self.validation_cutoff,
            selector=self.selector,
            evaluator=self.evaluator,
            max_workers=max_workers,
        )

    def deployment(
        self,
        validation: KChampionOuterPolicyResult,
        output_directory: Path,
        *,
        ineligible_slots: tuple[str, ...] = (),
        max_workers: int | None = None,
    ) -> KDeploymentSelectionResult:
        latest = {
            slot.slot_id: next(
                item
                for item in reversed(validation.outer_folds)
                if item.slot_id == slot.slot_id and item.selection is not None
            ).selection
            for slot in validation.slots
            if slot.eligible and slot.slot_id not in ineligible_slots
        }
        selector = RefitDeploymentSelector(latest)
        refitter = RealDeploymentRefitter(self.profile, output_directory)
        return select_k_deployment_packages(
            self.rows,
            validation=validation,
            deployment_cutoff=self.deployment_cutoff,
            selector=selector,
            refitter=refitter,
            max_workers=max_workers,
        )


def build_portfolio(*, max_workers: int | None = None) -> HermeticPortfolio:
    rows = source_rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"].iloc[:1449]), profile.walk_forward)
    models = build_models(rows, plan, profile, max_workers=max_workers)
    source_digest = sha256(
        pd.util.hash_pandas_object(rows, index=False).to_numpy().tobytes()
    ).hexdigest()
    assert plan.evaluation_cutoff is not None
    return HermeticPortfolio(
        rows=rows,
        profile=profile,
        plan=plan,
        models=models,
        source_data_sha256=source_digest,
        validation_cutoff=plan.evaluation_cutoff,
        deployment_cutoff=rows["timestamp_m1"].iloc[-1],
    )


def metric_slots(
    portfolio: HermeticPortfolio,
    validation: KChampionOuterPolicyResult,
    deployment: KDeploymentSelectionResult,
) -> tuple[KSlotMetricProjection, ...]:
    """Project existing fold evidence into the production K-slot metric contract."""

    selected_by_slot = {item.slot_id: item.selection for item in deployment.slots}
    result: list[KSlotMetricProjection] = []
    for state_count in K_VALUES:
        slot_id = f"k{state_count}"
        metadata = KSlotMetadata(
            slot_id=slot_id,
            state_count=state_count,
            feature_order=FEATURE_ORDER_BY_K[state_count],
            policy_version="k_champion_policy.v1",
            comparison_domain_id="k_specific_shared_feature_vector.v1",
            source_build_id=SOURCE_BUILD_ID,
            source_data_sha256=portfolio.source_data_sha256,
            evaluation_plan_hash=validation.outer_plan_hash,
        )
        points_by_family: list[KCandidateMetricProjection] = []
        family_records = {
            family: tuple(
                item
                for item in portfolio.models.candidates
                if item.slot_id == slot_id and item.model_family == family
            )
            for family in FAMILIES
        }
        for family in FAMILIES:
            records = tuple(sorted(family_records[family], key=lambda item: item.fold_id))
            points = (
                *tuple(
                    MetricPoint(
                        key="train_loglik_per_obs",
                        value=item.train_log_likelihood
                        / next(
                            fold.train_source_observations
                            for fold in portfolio.plan.folds
                            if fold.fold_id == item.fold_id
                        ),
                        step=index,
                        timestamp_ms=0,
                    )
                    for index, item in enumerate(records)
                ),
                MetricPoint(
                    key="valid_fold_rate",
                    value=next(
                        slot.valid_fold_rate for slot in validation.slots if slot.slot_id == slot_id
                    ),
                    step=0,
                    timestamp_ms=0,
                ),
                MetricPoint(
                    key="aic",
                    value=-2.0 * records[0].train_log_likelihood,
                    step=0,
                    timestamp_ms=0,
                ),
                MetricPoint(
                    key="bic",
                    value=-2.0 * records[0].train_log_likelihood
                    + float(np.log(portfolio.plan.folds[0].train_source_observations)),
                    step=0,
                    timestamp_ms=0,
                ),
            )
            points_by_family.append(
                KCandidateMetricProjection(
                    logged_model_id=f"{slot_id}:{family}:selected-candidate",
                    model_family=family,
                    metadata=metadata,
                    metric_points=points,
                )
            )
        selected = selected_by_slot[slot_id]
        eligible = selected is not None
        result.append(
            build_k_slot_projection(
                points_by_family,
                eligible=eligible,
                selected_logged_model_id=(
                    f"{slot_id}:{selected.model_family}:selected-candidate"
                    if selected is not None
                    else None
                ),
                selected_metric_points=(
                    next(
                        item.metric_points
                        for item in points_by_family
                        if selected is not None and item.model_family == selected.model_family
                    )
                    if selected is not None
                    else ()
                ),
                unavailable_reason=None if eligible else "slot is ineligible in hermetic QA",
            )
        )
    return build_four_k_slot_projection(result)


def canonical_hash_payload(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def independent_canonical_hash(payload: object) -> str:
    """Top-level process entry point used for independent canonicalization."""

    return canonical_hash_payload(payload)


def independent_portfolio_outer_hash() -> str:
    """Build and validate a fresh portfolio in an independent process."""

    portfolio = build_portfolio(max_workers=None)
    return portfolio.outer(max_workers=None).result_hash


__all__ = [
    "FAMILIES",
    "FEATURE_ORDER_BY_K",
    "K_VALUES",
    "HermeticPortfolio",
    "RealDeploymentRefitter",
    "build_portfolio",
    "canonical_hash_payload",
    "independent_canonical_hash",
    "independent_portfolio_outer_hash",
    "metric_slots",
    "source_rows",
]
