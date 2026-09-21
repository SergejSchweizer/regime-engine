from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from market_regime_engine.feature_discovery.metadata_store import (
    CorrelationMapping,
    FeatureRegistryRow,
    FeatureSelectionMetadataStore,
    FoldFeatureStat,
    FoldMetadataBundle,
    FoldModelStat,
    PcaLoading,
    SFFSStepRecord,
)


def make_bundle(*, changed: bool = False) -> FoldMetadataBundle:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    source_build_id = "source-build"
    profile_hash = "b" * 64
    return FoldMetadataBundle(
        fold_id="fold-001",
        profile_hash=profile_hash,
        source_build_id=source_build_id,
        feature_registry=(
            FeatureRegistryRow(
                "feature:macro_loader:f1",
                "macro_loader.macro_features",
                source_build_id,
                "a" * 64,
                "f1",
                "direct",
                "macro",
                "identity",
                {},
                timestamp,
                "active",
            ),
        ),
        fold_feature_stats=(
            FoldFeatureStat(
                "fold-001",
                profile_hash,
                source_build_id,
                "f1",
                True,
                None,
                True,
                False,
                None,
                False,
                True,
                True,
                0.1,
            ),
        ),
        pca_loadings=(
            PcaLoading("fold-001", profile_hash, source_build_id, "macro", 1, "f1", 0.9, 0.81, 0.7),
        ),
        correlation_mapping=(
            CorrelationMapping(
                "fold-001",
                profile_hash,
                source_build_id,
                "f1",
                "f1",
                1.0,
                1.0,
                None,
                None,
                100,
                30,
                0,
                0,
                True,
                "representative",
            ),
        ),
        sffs_steps=(
            SFFSStepRecord(
                "fold-001",
                profile_hash,
                source_build_id,
                2,
                1,
                "add",
                "f1",
                "c" * 64,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                True,
                None,
            ),
        ),
        fold_model_stats=(
            FoldModelStat(
                "fold-001",
                profile_hash,
                source_build_id,
                "d" * 64,
                2,
                "gaussian_hmm",
                not changed,
                {"log_likelihood": -1.2 if not changed else -1.3},
                "mlflow-run-1",
            ),
        ),
    )


def query_count(store: FeatureSelectionMetadataStore, table: str) -> int:
    with duckdb.connect(str(store.database), read_only=True) as connection:
        row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def test_bootstrap_has_exact_durable_schema_and_global_view(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)

    assert store.database == tmp_path / "feature_selection.duckdb"
    assert store.table_names() == (
        "correlation_mapping",
        "feature_registry",
        "fold_feature_stats",
        "fold_model_stats",
        "pca_loadings",
        "sffs_steps",
    )
    assert store.view_names() == ("feature_global_stats",)


def test_sffs_steps_can_be_committed_before_the_fold_model_bundle(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)
    rows = make_bundle().sffs_steps

    assert store.commit_sffs_steps(rows) is True
    assert store.commit_sffs_steps(rows) is True
    assert query_count(store, "sffs_steps") == 1


def test_fold_commit_is_atomic_idempotent_and_rejects_conflicting_replay(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)
    bundle = make_bundle()

    assert store.commit_fold(bundle) is True
    assert store.commit_fold(bundle) is False
    assert query_count(store, "feature_registry") == 1
    assert query_count(store, "fold_feature_stats") == 1
    assert query_count(store, "fold_model_stats") == 1
    with pytest.raises(ValueError, match="conflicting immutable"):
        store.commit_fold(make_bundle(changed=True))

    with duckdb.connect(str(store.database), read_only=True) as connection:
        assert connection.execute("SELECT fold_count FROM feature_global_stats").fetchone() == (1,)


def test_failed_fold_commit_rolls_back_every_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)

    def fail_insert(
        connection: duckdb.DuckDBPyConnection,
        bundle: FoldMetadataBundle,
        bundle_hash: str,
    ) -> None:
        raise RuntimeError("injected fold failure")

    monkeypatch.setattr(store, "_insert_fold_rows", fail_insert)
    with pytest.raises(RuntimeError, match="injected fold failure"):
        store.commit_fold(make_bundle())

    for table in store.table_names():
        assert query_count(store, table) == 0


def test_reopen_preserves_logical_rows_without_raw_vectors_or_credentials(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)
    store.commit_fold(make_bundle())
    reopened = FeatureSelectionMetadataStore(tmp_path)

    assert query_count(reopened, "feature_registry") == 1
    with duckdb.connect(str(reopened.database), read_only=True) as connection:
        columns = connection.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'main'
            """
        ).fetchall()
    assert not any(
        "password" in column.lower() or "vector" in column.lower() for _, column in columns
    )
