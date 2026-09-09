from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import log

import numpy as np
import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.feature_discovery.contracts import ProvisionalTeacherReference
from market_regime_engine.feature_discovery.quality import filter_outer_train_quality
from market_regime_engine.feature_discovery.scoring import (
    score_all_raw_features,
    score_raw_feature,
)
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)
HASH = "a" * 64
FEATURES = ("f0", "f1", "f2")
TEACHER_COUNT = 126
SOURCE_COUNT = 504


def lineage() -> SourceLineage:
    return SourceLineage(
        source_dataset="xetra_gold",
        source_build_id="build-1",
        data_sha256=HASH,
        schema_version=2,
        feature_version=4,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=BASE,
        row_count=SOURCE_COUNT,
        min_timestamp=BASE,
        max_timestamp=BASE + timedelta(days=SOURCE_COUNT - 1),
    )


def snapshot(*, monotonic_transform: bool = False) -> FeatureSnapshot:
    values: list[tuple[float, float, float | None]] = []
    for index in range(SOURCE_COUNT):
        state = (index // 13) % 2
        f0 = float(np.exp(index / 100.0) if monotonic_transform else index)
        f1 = ((-1.0) ** index) * (0.5 if state == 0 else 2.0)
        f2 = float(np.sin(index / 11.0))
        if index < 20:
            f2 = None
        values.append((f0, f1, f2))
    return FeatureSnapshot(
        lineage=lineage(),
        feature_names=FEATURES,
        rows=tuple(
            FeatureRow(BASE + timedelta(days=index), row) for index, row in enumerate(values)
        ),
    )


def quality(snapshot_value: FeatureSnapshot):
    catalog = FeatureCatalogSnapshot.from_entries(
        snapshot_value.lineage,
        "timestamp_m1",
        tuple(FeatureCatalogEntry(name, index + 1) for index, name in enumerate(FEATURES)),
    )
    return filter_outer_train_quality(
        catalog,
        snapshot_value,
        BASE,
        BASE + timedelta(days=SOURCE_COUNT - 1),
    )


def teacher(*, swapped: bool = False, degenerate: bool = False) -> ProvisionalTeacherReference:
    probabilities = []
    for index in range(TEACHER_COUNT):
        state = (index // 13) % 2
        row = (0.95, 0.05) if state == 0 else (0.05, 0.95)
        probabilities.append(row[::-1] if swapped else row)
    if degenerate:
        probabilities = [(1.0, 0.0)] * TEACHER_COUNT
    return ProvisionalTeacherReference(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        timestamps=tuple(BASE + timedelta(days=index) for index in range(TEACHER_COUNT)),
        filtered_probabilities=tuple(probabilities),
        dominant_states=tuple(int(np.argmax(row)) for row in probabilities),
        valid_inner_fold_ids=("fold_001",),
        source_build_id="build-1",
        inner_plan_hash=HASH,
        prototype_features=("prototype_0",),
    )


def test_all_quality_eligible_features_are_scored_on_teacher_support() -> None:
    source = snapshot()
    scores = score_all_raw_features(source, quality(source), teacher())

    assert tuple(score.feature_name for score in scores) == FEATURES
    assert all(score.canonical_ordinal == index for index, score in enumerate(scores, start=1))
    assert scores[0].eligible is True
    assert scores[1].eligible is True
    assert scores[2].eligible is False
    assert scores[2].observation_count == 106
    assert scores[2].coverage == pytest.approx(106 / 126)
    assert "coverage_below_0.90" in (scores[2].exclusion_reason or "")


def test_information_ratio_matches_independent_rank_bin_joint_mass_oracle() -> None:
    source = snapshot()
    result = score_raw_feature(source, quality(source), teacher(), "f0")
    state = tuple((0.95, 0.05) if (index // 13) % 2 == 0 else (0.05, 0.95) for index in range(126))
    bin_count = 126
    bins = tuple(min(9, index * 10 // bin_count) for index in range(bin_count))
    joint = tuple(
        tuple(
            sum(row[state_index] for index, row in enumerate(state) if bins[index] == bin_index)
            / bin_count
            for state_index in range(2)
        )
        for bin_index in range(10)
    )
    bin_masses = tuple(sum(row) for row in joint)
    state_masses = tuple(sum(row[state_index] for row in joint) for state_index in range(2))
    mutual_information = sum(
        mass * log(mass / (bin_masses[bin_index] * state_masses[state_index]))
        for bin_index, row in enumerate(joint)
        for state_index, mass in enumerate(row)
        if mass > 0.0
    )

    assert np.asarray(result.joint_bin_state_masses) == pytest.approx(np.asarray(joint))
    assert result.mutual_information == pytest.approx(mutual_information)
    assert result.state_information_ratio == pytest.approx(
        mutual_information / result.state_entropy
    )


def test_primary_score_is_label_and_monotonic_transform_invariant() -> None:
    original = snapshot()
    original_result = score_raw_feature(original, quality(original), teacher(), "f0")
    transformed = snapshot(monotonic_transform=True)
    transformed_result = score_raw_feature(transformed, quality(transformed), teacher(), "f0")
    swapped_result = score_raw_feature(original, quality(original), teacher(swapped=True), "f0")

    assert transformed_result.state_information_ratio == pytest.approx(
        original_result.state_information_ratio
    )
    assert swapped_result.state_information_ratio == pytest.approx(
        original_result.state_information_ratio
    )
    assert swapped_result.state_means == pytest.approx(original_result.state_means[::-1])


def test_variance_only_separation_has_information_but_negligible_eta_and_entropy_gate() -> None:
    source = snapshot()
    score = score_raw_feature(source, quality(source), teacher(), "f1")
    degenerate = score_raw_feature(source, quality(source), teacher(degenerate=True), "f0")

    assert score.mutual_information > 0.01
    assert score.eta_squared is not None
    assert score.eta_squared < 0.01
    assert degenerate.eligible is False
    assert degenerate.state_information_ratio is None
    assert "state_entropy_degenerate" in (degenerate.exclusion_reason or "")
