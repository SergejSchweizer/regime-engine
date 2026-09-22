from __future__ import annotations

from pathlib import Path

from market_regime_engine.feature_discovery.lifecycle_recommendations import (
    build_lifecycle_report,
)
from tests.unit.feature_discovery.test_lifecycle_recommendations import (
    PROFILE,
    _registry,
    _row,
)


def test_nineteen_clean_folds_do_not_cross_the_deprecation_boundary() -> None:
    rows = tuple(_row(index, "generated_feature") for index in range(1, 20))
    report = build_lifecycle_report(
        (_registry("generated_feature", "transformation"),),
        rows,
        feature_selection_profile_hash=PROFILE,
    )
    assert report.recommendations[0].recommended_status == "ACTIVE"


def test_recent_selection_representation_or_credit_blocks_candidate() -> None:
    registry = _registry("generated_feature", "transformation")
    for selected, representative, credit in (
        (True, False, 0.0),
        (False, True, 0.0),
        (False, False, 1.0e-3),
    ):
        rows = tuple(
            _row(
                index,
                "generated_feature",
                selected=selected if index == 20 else False,
                representative=representative if index == 20 else False,
                credit=credit if index == 20 else 0.0,
            )
            for index in range(1, 21)
        )
        report = build_lifecycle_report((registry,), rows, feature_selection_profile_hash=PROFILE)
        assert report.recommendations[0].recommended_status == "ACTIVE"


def test_reactivated_candidate_returns_to_active_without_persisting_state() -> None:
    rows = tuple(
        _row(
            index,
            "generated_feature",
            selected=index == 20,
        )
        for index in range(1, 21)
    )
    report = build_lifecycle_report(
        (_registry("generated_feature", "transformation", "DEPRECATED_CANDIDATE"),),
        rows,
        feature_selection_profile_hash=PROFILE,
    )
    evidence = report.recommendations[0]
    assert evidence.current_status == "DEPRECATED_CANDIDATE"
    assert evidence.recommended_status == "ACTIVE"
    assert evidence.automatic_transition is True


def test_core_features_never_receive_an_automatic_droppable_recommendation() -> None:
    rows = tuple(_row(index, "core_feature") for index in range(1, 21))
    for current in ("ACTIVE", "DEPRECATED_CANDIDATE", "DEPRECATED", "DROPPABLE"):
        report = build_lifecycle_report(
            (_registry("core_feature", "core", current),),
            rows,
            feature_selection_profile_hash=PROFILE,
        )
        evidence = report.recommendations[0]
        assert evidence.recommended_status not in ("DEPRECATED", "DROPPABLE")
        assert evidence.automatic_transition is False


def test_feature_selection_production_path_contains_no_postgres_drop_or_alter() -> None:
    root = Path(__file__).parents[3] / "src" / "market_regime_engine" / "feature_discovery"
    source = "\n".join(path.read_text() for path in root.rglob("*.py"))
    assert "DROP TABLE" not in source.upper()
    assert "ALTER TABLE" not in source.upper()


def test_lifecycle_report_exposes_current_and_recommended_state_without_approval_side_effects() -> (
    None
):
    rows = tuple(_row(index, "generated_feature") for index in range(1, 21))
    report = build_lifecycle_report(
        (_registry("generated_feature", "transformation", "DEPRECATED"),),
        rows,
        feature_selection_profile_hash=PROFILE,
    )
    evidence = report.recommendations[0]
    assert evidence.current_status == "DEPRECATED"
    assert evidence.recommended_status == "DEPRECATED_CANDIDATE"
    assert evidence.automatic_transition is True
