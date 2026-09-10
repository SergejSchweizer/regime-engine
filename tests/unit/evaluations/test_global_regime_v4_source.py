from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd  # type: ignore[import-untyped]
import pytest

import market_regime_engine.evaluations.global_regime_v4 as global_v4
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRequest,
    FeatureRow,
    FeatureSnapshot,
)
from market_regime_engine.profiles.loader import load_profile

START = datetime(2020, 1, 1, tzinfo=UTC)


def _lineage() -> SourceLineage:
    return SourceLineage(
        source_dataset="regime_loader",
        source_build_id="schema-build-1",
        data_sha256="a" * 64,
        schema_version=2,
        feature_version=1,
        source_table="regime_loader",
        synced_at_utc=START,
        row_count=2,
        min_timestamp=START,
        max_timestamp=START + timedelta(days=1),
    )


def test_v4_source_entrypoint_requests_the_complete_catalog(monkeypatch) -> None:
    lineage = _lineage()
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("feature_a", 1), FeatureCatalogEntry("feature_b", 2)),
    )
    snapshot = FeatureSnapshot(
        lineage,
        ("feature_a", "feature_b"),
        (
            FeatureRow(START, (1.0, None)),
            FeatureRow(START + timedelta(days=1), (2.0, 3.0)),
        ),
    )
    bound_catalog = catalog.with_materialization(snapshot)

    class Source:
        def __init__(self) -> None:
            self.request: FeatureRequest | None = None

        def read_schema_wide_with_catalog(self, request: FeatureRequest):
            self.request = request
            return bound_catalog, snapshot

    source = Source()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    captured: dict[str, object] = {}

    def fake_evaluate(rows: pd.DataFrame, **kwargs):
        captured["rows"] = rows
        captured.update(kwargs)
        return "evaluated"

    monkeypatch.setattr(global_v4, "evaluate_global_regime_v4", fake_evaluate)
    result = global_v4.evaluate_global_regime_v4_from_source(source, profile=profile)

    assert result == "evaluated"
    assert source.request is not None
    assert source.request.feature_names == ()
    assert source.request.mode.value == "feature_selection"
    rows = captured["rows"]
    assert isinstance(rows, pd.DataFrame)
    assert tuple(rows.columns) == ("timestamp_m1", "feature_a", "feature_b")
    assert tuple(rows["feature_a"]) == (1.0, 2.0)
    assert pd.isna(rows["feature_b"].iloc[0])
    assert rows["feature_b"].iloc[1] == 3.0
    assert captured["catalog"] == bound_catalog


def test_v4_source_entrypoint_fails_closed_for_invalid_snapshot_contracts() -> None:
    lineage = _lineage()
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("feature_a", 1), FeatureCatalogEntry("feature_b", 2)),
    )
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    invalid_profile = load_profile("configs/profiles/xetra_v3.yaml")

    class Source:
        def __init__(self, result) -> None:
            self.result = result

        def read_schema_wide_with_catalog(self, request: FeatureRequest):
            del request
            return self.result

    valid_snapshot = FeatureSnapshot(
        lineage,
        catalog.feature_names,
        (FeatureRow(START, (1.0, 2.0)),),
    )
    bound_catalog = catalog.with_materialization(valid_snapshot)

    with pytest.raises(ValueError, match="canonical Xetra v4 profile"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source((bound_catalog, valid_snapshot)), profile=invalid_profile
        )

    mismatched_snapshot = FeatureSnapshot(
        lineage,
        ("feature_a",),
        (FeatureRow(START, (1.0,)),),
    )
    with pytest.raises(ValueError, match="columns do not match"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source((catalog, mismatched_snapshot)), profile=profile
        )

    missing_digest = type(
        "Snapshot",
        (),
        {
            "feature_names": catalog.feature_names,
            "materialized_feature_data_sha256": None,
            "rows": (object(),),
        },
    )()
    with pytest.raises(ValueError, match="missing its materialization digest"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source((catalog, missing_digest)), profile=profile
        )

    mismatched_digest = type(
        "Snapshot",
        (),
        {
            "feature_names": catalog.feature_names,
            "materialized_feature_data_sha256": "b" * 64,
            "rows": (object(),),
        },
    )()
    with pytest.raises(ValueError, match="digests differ"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source((catalog, mismatched_digest)), profile=profile
        )

    empty_snapshot = FeatureSnapshot(lineage, catalog.feature_names, ())
    empty_catalog = catalog.with_materialization(empty_snapshot)
    with pytest.raises(ValueError, match="contains no rows"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source((empty_catalog, empty_snapshot)), profile=profile
        )
