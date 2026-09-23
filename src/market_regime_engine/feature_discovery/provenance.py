"""Canonical feature-role and transformation provenance identities."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

from market_regime_engine.feature_discovery.feature_roles import (
    FeatureRole,
    FeatureRoleContract,
)

_LOG_RETURN_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<family>[a-z0-9_]+)_log_return_(?P<window>[0-9]+)obs$"
)
_TRANSFORMATION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<family>[a-z0-9_]+)_(?P<name>delta|zscore|momentum_autocorr|return_geom)"
    r"_(?P<parameters>[a-z0-9_]+)$"
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class FeatureProvenance:
    """Immutable semantic provenance for one discovered candidate feature."""

    feature_name: str
    role: FeatureRole
    family: str | None
    transformation_name: str | None
    transformation_parameters: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if not self.feature_name or self.feature_name.strip() != self.feature_name:
            raise ValueError("feature provenance requires a trimmed feature name")
        if self.role is FeatureRole.TRANSFORMATION:
            if self.family is None or self.transformation_name is None:
                raise ValueError("transformation provenance is incomplete")
        elif self.family is not None or self.transformation_name is not None:
            raise ValueError("core provenance cannot contain transformation metadata")
        keys = tuple(key for key, _ in self.transformation_parameters)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("transformation parameters must have unique sorted keys")

    @property
    def canonical_dict(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "role": self.role.value,
            "family": self.family,
            "transformation_name": self.transformation_name,
            "transformation_parameters": dict(self.transformation_parameters),
        }

    @property
    def feature_identity(self) -> str:
        encoded = _canonical_json(self.canonical_dict).encode("utf-8")
        return sha256(encoded).hexdigest()


def _transformation_details(
    feature_name: str, family: str
) -> tuple[str, tuple[tuple[str, object], ...]]:
    log_return = _LOG_RETURN_RE.fullmatch(feature_name)
    if log_return is not None:
        if log_return.group("family") != family:
            raise ValueError(f"feature provenance family conflict: {feature_name}")
        return "log_return", (("window_observations", int(log_return.group("window"))),)
    transformation = _TRANSFORMATION_RE.fullmatch(feature_name)
    if transformation is None or transformation.group("family") != family:
        raise ValueError(f"missing transformation provenance: {feature_name}")
    tokens = transformation.group("parameters").split("_")
    parameters = (
        (("variant", tokens[0]),)
        if len(tokens) == 1
        else tuple((f"variant_{index:02d}", token) for index, token in enumerate(tokens))
    )
    return transformation.group("name"), parameters


def build_feature_provenance(
    contract: FeatureRoleContract,
    feature_names: Iterable[str] | None = None,
) -> tuple[FeatureProvenance, ...]:
    """Build deterministic provenance rows from the canonical role contract.

    The result is sorted by canonical identity, so discovery order and process
    completion order cannot change registry ordering or identities.
    """

    names = contract.transformation_features + contract.direct_hmm_candidates
    requested = tuple(names if feature_names is None else feature_names)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("feature provenance names must be non-empty and unique")
    rows: list[FeatureProvenance] = []
    for feature_name in requested:
        assignment = contract.assignment(feature_name)
        if assignment.role in (FeatureRole.CORE, FeatureRole.PCA):
            rows.append(FeatureProvenance(feature_name, assignment.role, None, None, ()))
        elif assignment.role is FeatureRole.TRANSFORMATION:
            if assignment.family is None:
                raise ValueError(f"missing transformation family: {feature_name}")
            transformation_name, parameters = _transformation_details(
                feature_name, assignment.family
            )
            rows.append(
                FeatureProvenance(
                    feature_name,
                    assignment.role,
                    assignment.family,
                    transformation_name,
                    parameters,
                )
            )
        else:
            raise ValueError(f"temporal key is not a candidate feature: {feature_name}")
    return tuple(sorted(rows, key=lambda row: row.feature_identity))


__all__ = ["FeatureProvenance", "build_feature_provenance"]
