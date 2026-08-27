"""Strict YAML/mapping loader for versioned model profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import yaml

from market_regime_engine.profiles.config import (
    EvaluationGates,
    FeatureSelectionConfig,
    GaussianHMMConfig,
    GMMHMMConfig,
    ModelProfile,
    StudentTHMMConfig,
    WalkForwardConfig,
)

type ProfileDataclass = (
    ModelProfile
    | FeatureSelectionConfig
    | WalkForwardConfig
    | GaussianHMMConfig
    | GMMHMMConfig
    | StudentTHMMConfig
    | EvaluationGates
)


def _strict_kwargs(cls: type[ProfileDataclass], raw: Mapping[str, Any]) -> dict[str, Any]:
    field_definitions = fields(cls)
    allowed = {field.name for field in field_definitions}
    unknown = set(raw) - allowed
    required = {
        field.name
        for field in field_definitions
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = required - set(raw)
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing keys for {cls.__name__}: {sorted(missing)}")
    return dict(raw)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def load_profile_mapping(raw: Mapping[str, Any]) -> ModelProfile:
    top = _strict_kwargs(ModelProfile, raw)
    feature_selection_raw = _require_mapping(top.pop("feature_selection"), "feature_selection")
    walk_forward_raw = _require_mapping(top.pop("walk_forward"), "walk_forward")
    gaussian_hmm_raw = _require_mapping(top.pop("gaussian_hmm"), "gaussian_hmm")
    gates_raw = _require_mapping(top.pop("gates"), "gates")
    gmm_hmms_raw = top.pop("gmm_hmms", ())
    student_t_raw = top.pop("student_t_hmm", None)

    fs_kwargs = _strict_kwargs(FeatureSelectionConfig, feature_selection_raw)
    fs_kwargs["static_features"] = tuple(fs_kwargs["static_features"])
    hmm_kwargs = _strict_kwargs(GaussianHMMConfig, gaussian_hmm_raw)
    hmm_kwargs["candidate_states"] = tuple(hmm_kwargs["candidate_states"])
    hmm_kwargs["seeds"] = tuple(hmm_kwargs["seeds"])

    if not isinstance(gmm_hmms_raw, Sequence) or isinstance(gmm_hmms_raw, (str, bytes)):
        raise ValueError("gmm_hmms must be a sequence of mappings")
    gmm_hmms = tuple(
        GMMHMMConfig(**_strict_kwargs(GMMHMMConfig, _require_mapping(item, "gmm_hmms item")))
        for item in gmm_hmms_raw
    )
    student_t_hmm = None
    if student_t_raw is not None:
        student_kwargs = _strict_kwargs(
            StudentTHMMConfig,
            _require_mapping(student_t_raw, "student_t_hmm"),
        )
        student_kwargs["candidate_states"] = tuple(student_kwargs["candidate_states"])
        student_t_hmm = StudentTHMMConfig(**student_kwargs)
    return ModelProfile(
        **top,
        feature_selection=FeatureSelectionConfig(**fs_kwargs),
        walk_forward=WalkForwardConfig(**_strict_kwargs(WalkForwardConfig, walk_forward_raw)),
        gaussian_hmm=GaussianHMMConfig(**hmm_kwargs),
        gates=EvaluationGates(**_strict_kwargs(EvaluationGates, gates_raw)),
        gmm_hmms=gmm_hmms,
        student_t_hmm=student_t_hmm,
    )


def load_profile(path: str | Path) -> ModelProfile:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("profile YAML root must be a mapping")
    return load_profile_mapping(raw)
