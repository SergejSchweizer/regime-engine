"""Process-safe HMM callbacks for the canonical Xetra selection pipeline."""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from math import tanh
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.commands.canonical_xetra import CanonicalStageCallbacks
from market_regime_engine.evaluation.calendar_clock import (
    CalendarMonthFold,
    CalendarMonthPlan,
    plan_calendar_month,
)
from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.feature_discovery.ablation import HMMSubsetEvaluation
from market_regime_engine.feature_discovery.feature_subset_score import (
    FeatureSubsetCandidate,
    FeatureSubsetFoldEvidence,
    score_feature_subset,
    to_sffs_score,
)
from market_regime_engine.feature_discovery.monthly_refit import StageCallbacks
from market_regime_engine.feature_discovery.sffs import (
    FeatureSubsetScore,
    FrontierFeatureSubsetEvaluator,
    FrontierFoldJob,
)
from market_regime_engine.features.ports import FeatureCatalogSnapshot
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.runtime.task_frontier import SharedTaskFrontier
from market_regime_engine.training.adapter_factory import CandidateAdapterFactory
from market_regime_engine.training.multistart import (
    MultistartBatchJob,
    MultistartResult,
    run_multistart,
)


def _hash(value: object) -> str:
    return sha256(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def _selector_hash(profile: ModelProfile) -> str:
    return sha256(
        pickle.dumps(
            ("canonical-gaussian-subset-selector", profile.profile_hash),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalModelCallbacks:
    """One fold-local callback set; all model work is TRAIN/TEST bound."""

    train: pd.DataFrame
    test: pd.DataFrame
    profile: ModelProfile
    catalog: FeatureCatalogSnapshot
    source_build_id: str
    max_workers: int | None = None
    _bound_feature_values: Mapping[str, Sequence[float]] | None = field(
        default=None, repr=False, compare=False
    )

    def bind_feature_values(self, values: Mapping[str, Sequence[float]]) -> None:
        object.__setattr__(self, "_bound_feature_values", dict(values))

    def _effective_train(self) -> pd.DataFrame:
        if self._bound_feature_values is None:
            return self.train
        frame = self.train.copy()
        for name, values in self._bound_feature_values.items():
            frame[name] = tuple(values)
        return frame

    @property
    def hmm_selector_contract_hash(self) -> str:
        return _selector_hash(self.profile)

    def _matrix(self, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray[Any, Any]:
        if not features or any(feature not in frame.columns for feature in features):
            raise ValueError("canonical HMM callback received unknown or empty features")
        selected = frame.loc[:, list(features)].dropna(axis=0, how="any")
        values = selected.to_numpy(dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError("canonical HMM callback requires finite TRAIN observations")
        return cast(np.ndarray[Any, Any], values)

    def _fit(self, features: tuple[str, ...], state_count: int) -> MultistartResult:
        matrix = self._matrix(self._effective_train(), features)
        # A canonical candidate may itself be evaluated inside the shared
        # process frontier.  Never create a child pool from that worker; the
        # frontier already supplies the CPU lane.  Direct parent calls retain
        # the configured multistart parallelism.
        effective_workers = (
            1 if os.environ.get("REGIME_CPU_PROCESS_WORKER") == "1" else self.max_workers
        )
        return run_multistart(
            matrix,
            state_count=state_count,
            adapter_factory=CandidateAdapterFactory("gaussian_hmm", features),
            max_workers=effective_workers,
        )

    @staticmethod
    def _score(
        features: tuple[str, ...],
        state_count: int,
        fit: MultistartResult,
    ) -> FeatureSubsetScore:
        per_observation = fit.winner.train_log_likelihood / max(1, fit.winner.artifact.state_count)
        # The value is the canonical dimension-independent selector score
        # seam.  The bounded diagnostic components are derived from the same
        # TRAIN-only fit and never inspect Outer TEST.
        bounded = (tanh(abs(per_observation)) + 1.0) / 2.0
        return FeatureSubsetScore(
            feature_names=features,
            value=per_observation,
            forecast_score=bounded,
            worst_fold_forecast_score=bounded,
            calibration_score=bounded,
            stability_score=fit.success_rate,
            robustness_score=fit.success_rate,
            model_family="gaussian_hmm",
            state_count=state_count,
        )

    def _inner_score(
        self, features: tuple[str, ...], state_count: int
    ) -> FeatureSubsetScore | None:
        source_train = self._effective_train()
        timestamps = tuple(source_train.loc[:, "timestamp_m1"].tolist())
        plan = plan_calendar_month(
            timestamps,
            minimum_train_source_observations=(
                self.profile.feature_discovery.inner_train_source_observations
            ),
        )
        evidence: list[FeatureSubsetFoldEvidence] = []
        for fold in plan.folds:
            train = source_train.iloc[: fold.train_source_observations].copy()
            test = source_train.iloc[
                fold.train_source_observations : fold.train_source_observations
                + fold.test_source_observations
            ].copy()
            fold_callbacks = CanonicalModelCallbacks(
                train=train,
                test=test,
                profile=self.profile,
                catalog=self.catalog,
                source_build_id=self.source_build_id,
                max_workers=self.max_workers,
            )
            fold_callbacks.bind_feature_values(
                {name: tuple(train[name].tolist()) for name in features}
            )
            try:
                fit = fold_callbacks._fit(features, state_count)
                adapter = CandidateAdapterFactory("gaussian_hmm", features)()
                adapter.reconstruct(fit.winner.artifact)
                filtered = adapter.causal_filter(fold_callbacks._matrix(test, features))
                target = filtered.log_likelihood / max(1, len(test))
                bounded = (tanh(abs(target)) + 1.0) / 2.0
                evidence.append(
                    FeatureSubsetFoldEvidence(
                        fold_id=fold.fold_id,
                        valid=True,
                        latest=fold is plan.folds[-1],
                        target_log_score=target,
                        baseline_target_log_score=0.0,
                        calibration_error=1.0 - bounded,
                        stability_score=fit.success_rate,
                        support_score=min(1.0, len(test) / 42.0),
                    )
                )
            except (RecoverableEvaluationInvalidity, ValueError, np.linalg.LinAlgError) as exc:
                evidence.append(
                    FeatureSubsetFoldEvidence(
                        fold_id=fold.fold_id,
                        valid=False,
                        latest=fold is plan.folds[-1],
                        invalid_reason=f"{type(exc).__name__}: {exc}",
                    )
                )
        candidate = FeatureSubsetCandidate(
            feature_names=features,
            feature_order_hash=_hash(features),
            source_build_id=self.source_build_id,
            evaluation_plan_hash=plan.plan_hash,
            folds=tuple(evidence),
        )
        score = to_sffs_score(
            score_feature_subset(candidate, latest_fold_id=plan.folds[-1].fold_id)
        )
        return (
            None
            if score is None
            else replace(score, model_family="gaussian_hmm", state_count=state_count)
        )

    def evaluate_subset(self, features: tuple[str, ...]) -> FeatureSubsetScore | None:
        try:
            return self._inner_score(features, 2)
        except RecoverableEvaluationInvalidity, ValueError, np.linalg.LinAlgError:
            return None

    def evaluate_gaussian_subset_by_k(
        self, state_count: int, features: tuple[str, ...]
    ) -> FeatureSubsetScore | None:
        try:
            return self._inner_score(features, state_count)
        except RecoverableEvaluationInvalidity, ValueError, np.linalg.LinAlgError:
            return None

    def evaluate_hmm_subset(self, features: tuple[str, ...]) -> HMMSubsetEvaluation | None:
        try:
            fit = self._fit(features, 2)
            return HMMSubsetEvaluation(
                score=self._inner_score(features, 2),
                model_family="gaussian_hmm",
                state_count=2,
                selector_contract_hash=self.hmm_selector_contract_hash,
                fit_execution_hash=_hash(fit),
            )
        except (RecoverableEvaluationInvalidity, ValueError, np.linalg.LinAlgError) as exc:
            return HMMSubsetEvaluation(
                score=None,
                model_family="gaussian_hmm",
                state_count=2,
                selector_contract_hash=self.hmm_selector_contract_hash,
                fit_execution_hash=_hash(("invalid", features, str(exc))),
                invalid_reason=f"{type(exc).__name__}: {exc}",
            )

    def fit_final_hmm(
        self, _train: pd.DataFrame, features: tuple[str, ...], state_count: int
    ) -> str | Sequence[str]:
        return _hash(self._fit(features, state_count))

    def evaluate_outer_test(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: tuple[str, ...],
        state_count: int,
        model_hashes: tuple[str, ...],
    ) -> str:
        del train
        fit = self._fit(features, state_count)
        if _hash(fit) not in model_hashes:
            raise ValueError("Outer TEST callback model hash differs from the final TRAIN fit")
        adapter = CandidateAdapterFactory("gaussian_hmm", features)()
        adapter.reconstruct(fit.winner.artifact)
        test_matrix = self._matrix(test, features)
        filtered = adapter.causal_filter(test_matrix)
        return _hash((model_hashes, filtered.filtered_probabilities.tolist()))

    def outer_test_probabilities(
        self,
        features: tuple[str, ...],
        state_count: int,
        model_hashes: tuple[str, ...],
    ) -> tuple[tuple[object, ...], tuple[tuple[float, ...], ...]]:
        """Return causal TEST probabilities for publication after validation."""

        fit = self._fit(features, state_count)
        if _hash(fit) not in model_hashes:
            raise ValueError("publication model hash differs from the final TRAIN fit")
        adapter = CandidateAdapterFactory("gaussian_hmm", features)()
        adapter.reconstruct(fit.winner.artifact)
        filtered = adapter.causal_filter(self._matrix(self.test, features))
        timestamps = tuple(self.test.loc[:, "timestamp_m1"].tolist())
        return timestamps, tuple(
            tuple(float(value) for value in row) for row in filtered.filtered_probabilities.tolist()
        )

    def as_callbacks(self) -> CanonicalStageCallbacks:
        return CanonicalStageCallbacks(
            evaluate_subset=self.evaluate_subset,
            evaluate_hmm_subset=self.evaluate_hmm_subset,
            hmm_selector_contract_hash=self.hmm_selector_contract_hash,
            fit_final_hmm=self.fit_final_hmm,
            evaluate_gaussian_subset_by_k=_CanonicalGaussianSubsetBatchEvaluator(self),
            evaluate_outer_test=self.evaluate_outer_test,
        )


@dataclass(slots=True)
class _CanonicalGaussianSubsetBatchEvaluator:
    """Flatten candidate/inner-fold HMM starts into the shared CPU frontier."""

    callbacks: CanonicalModelCallbacks
    frontier: SharedTaskFrontier[Any, Any] | None = None

    def __call__(
        self, state_count: int, features: tuple[str, ...]
    ) -> FeatureSubsetScore | None:
        return self.callbacks._inner_score(features, state_count)

    def bind_feature_values(self, values: Mapping[str, Sequence[float]]) -> None:
        self.callbacks.bind_feature_values(values)

    def bind_execution_frontier(self, frontier: object) -> None:
        self.frontier = cast(SharedTaskFrontier[Any, Any] | None, frontier)

    def _inner_plan(self) -> CalendarMonthPlan:
        source_train = self.callbacks._effective_train()
        return plan_calendar_month(
            tuple(source_train.loc[:, "timestamp_m1"].tolist()),
            minimum_train_source_observations=(
                self.callbacks.profile.feature_discovery.inner_train_source_observations
            ),
        )

    def _job_factory(
        self, state_count: int, features: tuple[str, ...]
    ) -> tuple[FrontierFoldJob, ...]:
        source_train = self.callbacks._effective_train()
        jobs: list[FrontierFoldJob] = []
        for fold in self._inner_plan().folds:
            train = source_train.iloc[: fold.train_source_observations]
            matrix = self.callbacks._matrix(train, features)
            job_id = f"{fold.fold_id}:{','.join(features)}"
            jobs.append(
                FrontierFoldJob(
                    features,
                    fold.fold_id,
                    MultistartBatchJob(
                        job_id,
                        matrix,
                        state_count,
                        CandidateAdapterFactory("gaussian_hmm", features),
                    ),
                )
            )
        return tuple(jobs)

    def _evidence_factory(
        self, entry: FrontierFoldJob, result: MultistartResult
    ) -> FeatureSubsetFoldEvidence:
        source_train = self.callbacks._effective_train()
        fold = next(item for item in self._inner_plan().folds if item.fold_id == entry.fold_id)
        test = source_train.iloc[
            fold.train_source_observations : fold.train_source_observations
            + fold.test_source_observations
        ]
        adapter = CandidateAdapterFactory("gaussian_hmm", entry.candidate_subset)()
        adapter.reconstruct(result.winner.artifact)
        filtered = adapter.causal_filter(self.callbacks._matrix(test, entry.candidate_subset))
        target = filtered.log_likelihood / max(1, len(test))
        bounded = (tanh(abs(target)) + 1.0) / 2.0
        plan = self._inner_plan()
        return FeatureSubsetFoldEvidence(
            fold_id=entry.fold_id,
            valid=True,
            latest=entry.fold_id == plan.folds[-1].fold_id,
            target_log_score=target,
            baseline_target_log_score=0.0,
            calibration_error=1.0 - bounded,
            stability_score=result.success_rate,
            support_score=min(1.0, len(test) / 42.0),
        )

    def evaluate_many(
        self,
        feature_sets: Sequence[tuple[str, ...]],
        *,
        state_count: int,
    ) -> tuple[FeatureSubsetScore | None, ...]:
        if self.frontier is None:
            return tuple(self(state_count, features) for features in feature_sets)
        plan = self._inner_plan()
        evaluator = FrontierFeatureSubsetEvaluator(
            self._job_factory,
            self._evidence_factory,
            _hash,
            self.callbacks.source_build_id,
            plan.plan_hash,
            plan.folds[-1].fold_id,
            state_count,
            max_workers=self.callbacks.max_workers,
            frontier=self.frontier,
        )
        return evaluator.evaluate_many(feature_sets, state_count=state_count)


def build_canonical_stage_factory(
    *,
    profile: ModelProfile,
    catalog: FeatureCatalogSnapshot,
    source_build_id: str,
    max_workers: int | None,
) -> Callable[[pd.DataFrame, pd.DataFrame, CalendarMonthFold, int], StageCallbacks]:
    """Return a pickle-safe per-fold callback factory for the public backend."""

    def factory(
        train: pd.DataFrame,
        test: pd.DataFrame,
        _fold: CalendarMonthFold,
        fold_workers: int,
    ) -> StageCallbacks:
        return cast(
            StageCallbacks,
            CanonicalModelCallbacks(
                train=train,
                test=test,
                profile=profile,
                catalog=catalog,
                source_build_id=source_build_id,
                max_workers=min(max_workers or fold_workers, fold_workers),
            ).as_callbacks(),
        )

    return factory


__all__ = ["CanonicalModelCallbacks", "build_canonical_stage_factory"]
