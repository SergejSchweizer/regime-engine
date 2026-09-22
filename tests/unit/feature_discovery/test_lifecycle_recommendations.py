from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from market_regime_engine.feature_discovery.lifecycle_recommendations import (
    FeatureLifecyclePolicy,
    build_lifecycle_report,
    write_lifecycle_report,
)
from market_regime_engine.feature_discovery.metadata_store import (
    FeatureRegistryRow,
    FeatureSelectionMetadataStore,
    FoldFeatureStat,
)

PROFILE = "a" * 64
CATALOG = "b" * 64


def _registry(name: str, role: str, status: str = "ACTIVE") -> FeatureRegistryRow:
    return FeatureRegistryRow(
        f"identity:{name}",
        "macro_loader.macro_features_daily",
        "build-1",
        CATALOG,
        name,
        role,
        "vix" if role == "transformation" else None,
        "delta" if role == "transformation" else None,
        {"window": 1} if role == "transformation" else {},
        datetime(2026, 1, 1, tzinfo=UTC),
        status,
    )


def _row(
    fold: int,
    feature: str,
    *,
    selected: bool = False,
    representative: bool = False,
    credit: float | None = None,
) -> FoldFeatureStat:
    return FoldFeatureStat(
        f"fold-{fold:03d}",
        PROFILE,
        "build-1",
        feature,
        True,
        None,
        feature == "core_feature",
        feature == "generated_feature",
        credit,
        representative,
        True,
        selected,
        1.0 if selected else None,
    )


def test_generated_feature_reaches_candidate_after_twenty_clean_folds() -> None:
    rows = tuple(_row(index, "generated_feature") for index in range(1, 21))
    report = build_lifecycle_report(
        (_registry("generated_feature", "transformation"),),
        rows,
        feature_selection_profile_hash=PROFILE,
    )

    evidence = report.recommendations[0]
    assert evidence.recommended_status == "DEPRECATED_CANDIDATE"
    assert evidence.automatic_transition is True
    assert evidence.eligible_folds == 20
    assert report.policy_profile_hash != PROFILE


def test_recent_selection_or_credit_reactivates_and_core_never_auto_deprecates() -> None:
    generated = tuple(
        _row(index, "generated_feature", selected=index == 20, credit=0.0) for index in range(1, 21)
    )
    generated = (*generated[:-1], _row(20, "generated_feature", credit=1.0e-3))
    core = tuple(_row(index, "core_feature") for index in range(1, 21))
    report = build_lifecycle_report(
        (
            _registry("generated_feature", "transformation", "DEPRECATED_CANDIDATE"),
            _registry("core_feature", "core"),
        ),
        generated + core,
        feature_selection_profile_hash=PROFILE,
    )

    recommendations = {item.feature_name: item for item in report.recommendations}
    assert recommendations["generated_feature"].recommended_status == "ACTIVE"
    assert recommendations["core_feature"].recommended_status == "ACTIVE"
    assert recommendations["core_feature"].automatic_transition is False


def test_report_is_deterministic_and_store_is_authoritative(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path / "metadata")
    registry = _registry("generated_feature", "transformation")
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO feature_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                registry.feature_identity,
                registry.source_dataset,
                registry.source_build_id,
                registry.source_catalog_hash,
                registry.feature_name,
                registry.role,
                registry.family,
                registry.transformation_name,
                '{"window":1}',
                registry.first_seen_utc,
                registry.lifecycle_status,
            ],
        )
    for index in range(1, 21):
        store.commit_fold_feature_stats((_row(index, registry.feature_name),))
    first = store.lifecycle_report(feature_selection_profile_hash=PROFILE)
    second = store.lifecycle_report(feature_selection_profile_hash=PROFILE)
    assert first.report_hash == second.report_hash
    path = write_lifecycle_report(first, tmp_path / "reports" / "lifecycle.json")
    assert path.read_text() == path.read_text()
    assert "DROP" not in path.read_text()


def test_policy_epsilon_is_part_of_the_policy_profile_identity() -> None:
    low = FeatureLifecyclePolicy(pca_credit_epsilon=1.0e-12)
    high = FeatureLifecyclePolicy(pca_credit_epsilon=1.0e-6)
    assert low.profile_hash(PROFILE) != high.profile_hash(PROFILE)
