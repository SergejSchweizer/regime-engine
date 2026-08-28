"""Pure model-adapter construction from immutable candidate contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from market_regime_engine.models.gaussian_hmm import (
    HmmlearnGaussianHMMAdapter,
    HmmlearnGMMHMMAdapter,
)
from market_regime_engine.models.student_t_hmm import StudentTHMMAdapter, StudentTHMMSettings
from market_regime_engine.profiles.config import ModelProfile


class CandidateContract(Protocol):
    @property
    def candidate_id(self) -> str: ...

    @property
    def model_family(self) -> str: ...

    @property
    def state_count(self) -> int: ...

    @property
    def mixture_count(self) -> int: ...

    @property
    def feature_order(self) -> tuple[str, ...]: ...


def adapter_factory(profile: ModelProfile, candidate: CandidateContract) -> Callable[[], object]:
    """Return a fresh adapter factory after validating the candidate/profile contract."""

    if candidate.state_count not in (2, 3, 4, 5):
        raise ValueError("candidate state_count must be 2, 3, 4, or 5")
    if candidate.model_family == "gaussian_hmm" and candidate.mixture_count == 1:
        return lambda: HmmlearnGaussianHMMAdapter(candidate.feature_order)
    if candidate.model_family == "gmm_hmm" and candidate.mixture_count == 2:
        return lambda: HmmlearnGMMHMMAdapter(candidate.feature_order)
    if candidate.model_family == "student_t_hmm" and candidate.mixture_count == 1:
        settings = profile.student_t_hmm
        if settings is None:
            raise ValueError("Student-t candidate requires Student-t profile settings")
        return lambda: StudentTHMMAdapter(
            candidate.feature_order,
            StudentTHMMSettings(
                minimum_nu=settings.minimum_nu,
                maximum_nu=settings.maximum_nu,
                initial_nu=settings.initial_nu,
                n_iter=settings.n_iter,
                tol=settings.tol,
                min_covar=settings.min_covar,
            ),
        )
    raise ValueError("candidate family/mixture contract is unsupported")
