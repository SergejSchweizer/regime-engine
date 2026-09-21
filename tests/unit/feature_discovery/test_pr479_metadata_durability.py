from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from market_regime_engine.feature_discovery.metadata_store import FeatureSelectionMetadataStore
from tests.unit.feature_discovery.test_metadata_store import make_bundle


def read_logical_rows(database: Path, table: str) -> list[tuple[object, ...]]:
    with duckdb.connect(str(database), read_only=True) as connection:
        columns = connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'main' AND table_name = ?
            ORDER BY ordinal_position
            """,
            [table],
        ).fetchall()
        names = ", ".join(f'"{column[0]}"' for column in columns)
        return connection.execute(f'SELECT {names} FROM "{table}" ORDER BY ALL').fetchall()


def test_reopen_preserves_byte_equivalent_logical_rows(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)
    store.commit_fold(make_bundle())
    before = {table: read_logical_rows(store.database, table) for table in store.table_names()}

    reopened = FeatureSelectionMetadataStore(tmp_path)
    after = {table: read_logical_rows(reopened.database, table) for table in reopened.table_names()}
    assert after == before


def test_concurrent_readers_observe_only_committed_fold_state(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)
    store.commit_fold(make_bundle())

    def read_counts(_: int) -> tuple[int, int]:
        with duckdb.connect(str(store.database), read_only=True) as connection:
            feature_row = connection.execute("SELECT count(*) FROM fold_feature_stats").fetchone()
            model_row = connection.execute("SELECT count(*) FROM fold_model_stats").fetchone()
        assert feature_row is not None and model_row is not None
        return int(feature_row[0]), int(model_row[0])

    with ThreadPoolExecutor(max_workers=8) as executor:
        counts = list(executor.map(read_counts, range(32)))
    assert counts == [(1, 1)] * 32


def test_global_view_matches_independent_base_table_aggregation(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)
    store.commit_fold(make_bundle())
    with duckdb.connect(str(store.database), read_only=True) as connection:
        view_rows = connection.execute(
            """
            SELECT feature_name, fold_count, eligible_fold_count,
                   final_selection_count, representative_fold_count,
                   mean_ablation_loss
            FROM feature_global_stats
            ORDER BY feature_name
            """
        ).fetchall()
        independent_rows = connection.execute(
            """
            SELECT feature_name,
                   count(*),
                   count(*) FILTER (WHERE eligible),
                   count(*) FILTER (WHERE final_selection),
                   count(*) FILTER (WHERE representative),
                   avg(ablation_loss) FILTER (WHERE ablation_loss IS NOT NULL)
            FROM fold_feature_stats
            GROUP BY feature_name
            ORDER BY feature_name
            """
        ).fetchall()
    assert view_rows == independent_rows


def test_conflicting_existing_feature_identity_fails_closed(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)
    original = make_bundle()
    store.commit_fold(original)

    conflicting = replace(
        original,
        fold_id="fold-002",
        profile_hash="c" * 64,
        feature_registry=(replace(original.feature_registry[0], source_catalog_hash="e" * 64),),
        fold_feature_stats=tuple(
            replace(row, fold_id="fold-002", profile_hash="c" * 64)
            for row in original.fold_feature_stats
        ),
        pca_loadings=tuple(
            replace(row, fold_id="fold-002", profile_hash="c" * 64) for row in original.pca_loadings
        ),
        correlation_mapping=tuple(
            replace(row, fold_id="fold-002", profile_hash="c" * 64)
            for row in original.correlation_mapping
        ),
        sffs_steps=tuple(
            replace(row, fold_id="fold-002", profile_hash="c" * 64) for row in original.sffs_steps
        ),
        fold_model_stats=tuple(
            replace(row, fold_id="fold-002", profile_hash="c" * 64)
            for row in original.fold_model_stats
        ),
    )

    with pytest.raises(ValueError, match="conflicting immutable feature identity"):
        store.commit_fold(conflicting)

    assert len(read_logical_rows(store.database, "feature_registry")) == 1
