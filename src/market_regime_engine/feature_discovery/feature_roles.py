"""Canonical feature roles and the v1 scalable-selection contract.

This module is the single source of truth for the semantic split used by the
new macro feature-selection profile.  It deliberately does not inspect raw
source relations or infer a role from a caller-provided allowlist.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from market_regime_engine.features.ports import FeatureCatalogSnapshot

TEMPORAL_KEY = "timestamp_m1"
FEATURE_SELECTION_PROFILE_VERSION = "macro_regime_feature_selection_v1"


class FeatureRole(StrEnum):
    TEMPORAL_KEY = "temporal_key"
    CORE = "core"
    TRANSFORMATION = "transformation"
    PCA = "pca"


class TransformationFamily(StrEnum):
    VIX = "vix"
    VIX9D = "vix9d"
    VIX3M = "vix3m"
    VIX6M = "vix6m"
    VIX1Y = "vix1y"
    VSTOXX = "vstoxx"
    MOVE = "move"
    CISS = "ciss"
    EURO_HY_OAS = "euro_hy_oas"
    US_2Y = "us_2y"
    US_10Y = "us_10y"
    ESTR = "estr"
    USD_BROAD = "usd_broad"
    FED = "fed"


class FeatureStage(StrEnum):
    QUALITY = "train_quality"
    FAMILY_PCA = "family_pca"
    CORRELATION = "global_correlation"
    SFFS = "sffs"
    HMM = "hmm_observation"


# Keep these tuples in canonical order.  They are intentionally the only
# core identity inventory in the production codebase.
CORE_FEATURES: tuple[str, ...] = (
    "vix_log_level",
    "vix9d_log_level",
    "vix3m_log_level",
    "vix6m_log_level",
    "vix1y_log_level",
    "vstoxx_log_level",
    "move_log_level",
    "ciss_log_level",
    "euro_hy_oas_log_level",
    "us_2y_log_level",
    "us_10y_log_level",
    "estr_log_level",
    "usd_broad_log_level",
    "vix9d_vix_ratio",
    "vix_vix3m_ratio",
    "vix9d_vix3m_log_ratio",
    "vix3m_minus_vix",
    "vix6m_minus_vix",
    "vix1y_minus_vix",
    "us_10y_minus_us_2y",
)

TRANSFORMATION_FAMILIES: tuple[str, ...] = tuple(item.value for item in TransformationFamily)

FAMILY_NEAR_DUPLICATE_ABS_THRESHOLD = 0.995
FAMILY_NEAR_DUPLICATE_SUBWINDOW_ABS_THRESHOLD = 0.99
FAMILY_PCA_MAX_COMPONENTS = 8
CORRELATION_ABS_THRESHOLD = 0.95
CORRELATION_SUBWINDOW_ABS_THRESHOLD = 0.90
CORRELATION_SUBWINDOWS = 3
CORRELATION_MIN_PAIR_ROWS = 30
CORRELATION_MIN_SUBWINDOW_ROWS = 10
SFFS_MAX_FEATURES = 10

_TRANSFORMATION_RE = re.compile(
    r"^(?P<family>[a-z0-9_]+)_(?P<kind>delta|zscore|momentum_autocorr|return_geom)"
    r"(?:_[a-z0-9]+)+$"
)
_LOG_RETURN_RE = re.compile(r"^(?P<family>[a-z0-9_]+)_log_return_[0-9]+obs$")
_EXPECTED_MOVE_RE = re.compile(r"^(?P<family>[a-z0-9_]+)_next_expected_move_bp$")
_FAMILY_BY_LONGEST_PREFIX = tuple(sorted(TRANSFORMATION_FAMILIES, key=len, reverse=True))
_FAMILY_PC_RE = re.compile(r"^family_pc_(?P<family>[a-z0-9_]+)_(?P<component>[1-8])$")
_PCA_RE = re.compile(r"^pca_pc_00[1-8]$")


@dataclass(frozen=True, slots=True)
class FeatureSelectionProfile:
    """Versioned, persisted defaults for the scalable selection stages."""

    version: str = FEATURE_SELECTION_PROFILE_VERSION
    family_near_duplicate_abs_threshold: float = FAMILY_NEAR_DUPLICATE_ABS_THRESHOLD
    family_near_duplicate_subwindow_abs_threshold: float = (
        FAMILY_NEAR_DUPLICATE_SUBWINDOW_ABS_THRESHOLD
    )
    family_pca_max_components: int = FAMILY_PCA_MAX_COMPONENTS
    correlation_abs_threshold: float = CORRELATION_ABS_THRESHOLD
    correlation_subwindow_abs_threshold: float = CORRELATION_SUBWINDOW_ABS_THRESHOLD
    correlation_subwindows: int = CORRELATION_SUBWINDOWS
    correlation_min_pair_rows: int = CORRELATION_MIN_PAIR_ROWS
    correlation_min_subwindow_rows: int = CORRELATION_MIN_SUBWINDOW_ROWS
    sffs_max_features: int = SFFS_MAX_FEATURES
    correlation_measure: str = "absolute_pearson"
    correlation_selection: str = "redundancy_only_stable_leader"
    pc_count_policy: str = "first_nonzero_rank_components"
    sffs_score_policy: str = "dimension_independent_feature_subset_score"
    outer_test_policy: str = "evaluation_only"

    def __post_init__(self) -> None:
        if self.version != FEATURE_SELECTION_PROFILE_VERSION:
            raise ValueError("unsupported feature-selection profile version")
        if not 0.0 < self.family_near_duplicate_subwindow_abs_threshold <= 1.0:
            raise ValueError("family subwindow threshold must be in (0,1]")
        if not 0.0 < self.family_near_duplicate_abs_threshold <= 1.0:
            raise ValueError("family threshold must be in (0,1]")
        if self.family_near_duplicate_abs_threshold != 0.995:
            raise ValueError("family near-duplicate threshold is pinned to 0.995")
        if self.family_near_duplicate_subwindow_abs_threshold != 0.99:
            raise ValueError("family subwindow threshold is pinned to 0.99")
        if self.family_pca_max_components != 8:
            raise ValueError("family PCA component cap is pinned to 8")
        if self.correlation_abs_threshold != 0.95:
            raise ValueError("correlation threshold is pinned to 0.95")
        if self.correlation_subwindow_abs_threshold != 0.90:
            raise ValueError("correlation subwindow threshold is pinned to 0.90")
        if self.correlation_subwindows != 3:
            raise ValueError("correlation stability requires three subwindows")
        if self.correlation_min_pair_rows != 30 or self.correlation_min_subwindow_rows != 10:
            raise ValueError("correlation support minima are pinned to 30 and 10")
        if self.sffs_max_features != 10:
            raise ValueError("SFFS feature cap is pinned to 10")
        if self.correlation_measure != "absolute_pearson":
            raise ValueError("only absolute Pearson correlation is supported")
        if self.correlation_selection != "redundancy_only_stable_leader":
            raise ValueError("correlation selection must be redundancy-only")
        if self.pc_count_policy != "first_nonzero_rank_components":
            raise ValueError("PC count must use the first non-zero-rank components")
        if self.sffs_score_policy != "dimension_independent_feature_subset_score":
            raise ValueError("SFFS must use a dimension-independent score")
        if self.outer_test_policy != "evaluation_only":
            raise ValueError("outer TEST is evaluation-only")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "family_near_duplicate_abs_threshold": self.family_near_duplicate_abs_threshold,
            "family_near_duplicate_subwindow_abs_threshold": (
                self.family_near_duplicate_subwindow_abs_threshold
            ),
            "family_pca_max_components": self.family_pca_max_components,
            "correlation_abs_threshold": self.correlation_abs_threshold,
            "correlation_subwindow_abs_threshold": self.correlation_subwindow_abs_threshold,
            "correlation_subwindows": self.correlation_subwindows,
            "correlation_min_pair_rows": self.correlation_min_pair_rows,
            "correlation_min_subwindow_rows": self.correlation_min_subwindow_rows,
            "sffs_max_features": self.sffs_max_features,
            "correlation_measure": self.correlation_measure,
            "correlation_selection": self.correlation_selection,
            "pc_count_policy": self.pc_count_policy,
            "sffs_score_policy": self.sffs_score_policy,
            "outer_test_policy": self.outer_test_policy,
        }

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureRoleAssignment:
    feature_name: str
    role: FeatureRole
    family: str | None = None

    def __post_init__(self) -> None:
        if self.role is FeatureRole.TRANSFORMATION:
            if self.family not in TRANSFORMATION_FAMILIES:
                raise ValueError("transformations require exactly one canonical family")
        elif self.family is not None:
            raise ValueError("temporal and core roles cannot have a transformation family")

    @property
    def direct_hmm_candidate(self) -> bool:
        return self.role in (FeatureRole.CORE, FeatureRole.PCA)

    @property
    def family_pca_input(self) -> bool:
        return self.role is FeatureRole.TRANSFORMATION


def _family_for_name(feature_name: str) -> str | None:
    match = (
        _TRANSFORMATION_RE.fullmatch(feature_name)
        or _LOG_RETURN_RE.fullmatch(feature_name)
        or _EXPECTED_MOVE_RE.fullmatch(feature_name)
    )
    if match is None:
        return None
    family = match.group("family")
    if family in TRANSFORMATION_FAMILIES:
        return family
    # The regular expression intentionally permits only a grammar; this
    # prefix check makes overlapping names (us_10y/us_2y) unambiguous.
    for candidate in _FAMILY_BY_LONGEST_PREFIX:
        if feature_name.startswith(candidate + "_"):
            return candidate
    return None


def classify_feature_name(feature_name: str) -> FeatureRoleAssignment:
    """Classify one view column, failing closed for unknown future columns."""

    if feature_name == TEMPORAL_KEY:
        return FeatureRoleAssignment(feature_name, FeatureRole.TEMPORAL_KEY)
    if feature_name in CORE_FEATURES:
        return FeatureRoleAssignment(feature_name, FeatureRole.CORE)
    if _PCA_RE.fullmatch(feature_name) is not None:
        return FeatureRoleAssignment(feature_name, FeatureRole.PCA)
    family = _family_for_name(feature_name)
    if family is not None:
        return FeatureRoleAssignment(feature_name, FeatureRole.TRANSFORMATION, family)
    raise ValueError(f"unclassifiable feature column: {feature_name}")


@dataclass(frozen=True, slots=True)
class FeatureRoleContract:
    assignments: tuple[FeatureRoleAssignment, ...]
    profile: FeatureSelectionProfile = FeatureSelectionProfile()

    def __post_init__(self) -> None:
        names = tuple(item.feature_name for item in self.assignments)
        if len(names) != len(set(names)):
            raise ValueError("feature-role assignments must be unique")
        if sum(item.role is FeatureRole.TEMPORAL_KEY for item in self.assignments) > 1:
            raise ValueError("feature-role contract has duplicate temporal keys")

    @property
    def temporal_keys(self) -> tuple[str, ...]:
        return tuple(
            item.feature_name for item in self.assignments if item.role is FeatureRole.TEMPORAL_KEY
        )

    @property
    def core_features(self) -> tuple[str, ...]:
        return tuple(
            item.feature_name for item in self.assignments if item.role is FeatureRole.CORE
        )

    @property
    def pca_features(self) -> tuple[str, ...]:
        return tuple(item.feature_name for item in self.assignments if item.role is FeatureRole.PCA)

    @property
    def transformation_features(self) -> tuple[str, ...]:
        return tuple(
            item.feature_name
            for item in self.assignments
            if item.role is FeatureRole.TRANSFORMATION
        )

    @property
    def direct_hmm_candidates(self) -> tuple[str, ...]:
        return tuple(item.feature_name for item in self.assignments if item.direct_hmm_candidate)

    @property
    def family_pca_inputs(self) -> tuple[str, ...]:
        return self.transformation_features

    def validate_complete_catalog(self) -> tuple[str, ...]:
        """Require the complete canonical source-view identity inventory."""

        if self.temporal_keys != (TEMPORAL_KEY,):
            raise ValueError("catalog must contain exactly timestamp_m1 as its temporal key")
        missing = tuple(name for name in CORE_FEATURES if name not in self.core_features)
        if missing:
            raise ValueError(f"catalog is missing canonical core features: {', '.join(missing)}")
        return tuple(item.feature_name for item in self.assignments)

    @property
    def profile_hash(self) -> str:
        payload = {
            "profile": self.profile.canonical_dict(),
            "assignments": [
                {"feature_name": item.feature_name, "role": item.role.value, "family": item.family}
                for item in self.assignments
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()

    @property
    def contract_hash(self) -> str:
        """Stable identity to persist beside fold and model evidence."""

        return self.profile_hash

    def evidence_metadata(self) -> dict[str, object]:
        """Return the complete role/profile identity for audit artifacts."""

        return {
            "feature_selection_profile_version": self.profile.version,
            "feature_selection_profile_hash": self.profile.profile_hash,
            "feature_role_contract_hash": self.contract_hash,
            "feature_role_assignments": tuple(
                (item.feature_name, item.role.value, item.family) for item in self.assignments
            ),
        }

    def assignment(self, feature_name: str) -> FeatureRoleAssignment:
        for item in self.assignments:
            if item.feature_name == feature_name:
                return item
        raise KeyError(feature_name)

    def validate_hmm_features(self, feature_names: Iterable[str]) -> tuple[str, ...]:
        """Accept core representatives or canonical family PCs only."""

        result = tuple(feature_names)
        if not result or len(result) != len(set(result)):
            raise ValueError("HMM feature order must be non-empty and duplicate-free")
        for name in result:
            if _is_family_pc(name):
                continue
            assignment = self.assignment(name)
            if not assignment.direct_hmm_candidate:
                raise ValueError("only core features may enter HMM candidates directly")
        return result

    def validate_stage_features(
        self,
        stage: FeatureStage,
        feature_names: Iterable[str],
    ) -> tuple[str, ...]:
        """Validate the semantic boundary for one downstream stage.

        This method checks semantic eligibility only.  It never replaces the
        later TRAIN-only finite, coverage, variance, or pairwise-support
        validation performed by statistical stages.
        """

        result = tuple(feature_names)
        if not result or len(result) != len(set(result)):
            raise ValueError("stage feature order must be non-empty and duplicate-free")
        for name in result:
            if name == TEMPORAL_KEY:
                raise ValueError("timestamp_m1 is a temporal key and cannot enter a feature stage")
            generated = _is_family_pc(name)
            if stage is FeatureStage.QUALITY:
                if generated:
                    raise ValueError("generated family PCs cannot enter raw quality ranking")
                self.assignment(name)
            elif stage is FeatureStage.FAMILY_PCA:
                assignment = self.assignment(name)
                if not assignment.family_pca_input:
                    raise ValueError("family PCA accepts transformations only")
            elif stage in (FeatureStage.CORRELATION, FeatureStage.SFFS, FeatureStage.HMM):
                if not generated:
                    assignment = self.assignment(name)
                    if not assignment.direct_hmm_candidate:
                        raise ValueError(
                            "generated transformations may enter downstream stages "
                            "only as family PCs"
                        )
        return result


def build_feature_role_contract(
    feature_names: Iterable[str],
    *,
    profile: FeatureSelectionProfile | None = None,
) -> FeatureRoleContract:
    """Build a canonical contract in source/catalog order."""

    assignments = tuple(classify_feature_name(name) for name in feature_names)
    return FeatureRoleContract(
        assignments, FeatureSelectionProfile() if profile is None else profile
    )


def build_feature_role_contract_from_catalog(
    catalog: FeatureCatalogSnapshot,
    *,
    profile: FeatureSelectionProfile | None = None,
) -> FeatureRoleContract:
    """Build and validate the role contract for one discovered source catalog.

    The catalog is the only runtime input that determines which non-core
    columns exist.  Every supplied column must classify successfully, and the
    complete canonical temporal/core inventory is required before any stage
    can consume the contract.  The type-only import keeps multiprocessing
    ``spawn`` free of a feature-source import cycle.
    """

    if catalog.timestamp_column != TEMPORAL_KEY:
        raise ValueError("catalog timestamp column must be timestamp_m1")
    contract = build_feature_role_contract(
        (catalog.timestamp_column, *catalog.feature_names), profile=profile
    )
    contract.validate_complete_catalog()
    return contract


def family_pc_name(family: str, component: int) -> str:
    """Return the sole canonical identity for a retained family PC."""

    if family not in TRANSFORMATION_FAMILIES:
        raise ValueError("family PC requires one canonical transformation family")
    if component < 1 or component > FAMILY_PCA_MAX_COMPONENTS:
        raise ValueError("family PC component must be between 1 and 8")
    return f"family_pc_{family}_{component}"


def _is_family_pc(feature_name: str) -> bool:
    match = _FAMILY_PC_RE.fullmatch(feature_name)
    return match is not None and match.group("family") in TRANSFORMATION_FAMILIES


__all__ = [
    "CORE_FEATURES",
    "CORRELATION_ABS_THRESHOLD",
    "CORRELATION_MIN_PAIR_ROWS",
    "CORRELATION_MIN_SUBWINDOW_ROWS",
    "CORRELATION_SUBWINDOWS",
    "CORRELATION_SUBWINDOW_ABS_THRESHOLD",
    "FAMILY_NEAR_DUPLICATE_ABS_THRESHOLD",
    "FAMILY_NEAR_DUPLICATE_SUBWINDOW_ABS_THRESHOLD",
    "FAMILY_PCA_MAX_COMPONENTS",
    "FEATURE_SELECTION_PROFILE_VERSION",
    "SFFS_MAX_FEATURES",
    "TEMPORAL_KEY",
    "TRANSFORMATION_FAMILIES",
    "FeatureRole",
    "FeatureRoleAssignment",
    "FeatureRoleContract",
    "FeatureSelectionProfile",
    "FeatureStage",
    "TransformationFamily",
    "build_feature_role_contract",
    "build_feature_role_contract_from_catalog",
    "classify_feature_name",
    "family_pc_name",
]
