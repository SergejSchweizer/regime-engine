from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import duckdb
import pytest

from market_regime_engine.feature_discovery.metadata_store import (
    FeatureSelectionMetadataStore,
    FoldFeatureStat,
    FoldMetadataBundle,
    PcaLoading,
)
from market_regime_engine.mlflow_support.feature_selection_evidence import (
    render_cumulative_feature_stats,
)
from tests.unit.feature_discovery.test_metadata_store import make_bundle

PROFILE = "b" * 64


def _stats(
    fold_id: str, *, selected: bool, credit: float | None, loss: float | None
) -> tuple[FoldFeatureStat, ...]:
    return (
        FoldFeatureStat(
            fold_id,
            PROFILE,
            "build",
            "source_a",
            True,
            None,
            False,
            True,
            credit,
            False,
            True,
            selected,
            loss,
        ),
        FoldFeatureStat(
            fold_id,
            PROFILE,
            "build",
            "direct_a",
            True,
            None,
            True,
            False,
            None,
            True,
            True,
            selected,
            loss,
        ),
    )


def _view_rows(store: FeatureSelectionMetadataStore) -> list[tuple[object, ...]]:
    with duckdb.connect(str(store.database), read_only=True) as connection:
        return connection.execute(
            "SELECT * FROM feature_global_stats ORDER BY feature_name"
        ).fetchall()


def test_independent_aggregation_reproduces_every_global_view_field(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)
    store.commit_fold_feature_stats(_stats("fold-001", selected=True, credit=1.0, loss=4.0))
    store.commit_fold_feature_stats(_stats("fold-002", selected=False, credit=2.0, loss=None))

    with duckdb.connect(str(store.database), read_only=True) as connection:
        raw = connection.execute(
            """
            SELECT feature_name, eligible, direct_participation, representative,
                   final_selection, ablation_loss, pca_credit
            FROM fold_feature_stats ORDER BY feature_name, fold_id
            """
        ).fetchall()
    expected: list[tuple[object, ...]] = []
    for feature in ("direct_a", "source_a"):
        values = [row for row in raw if row[0] == feature]
        eligible = sum(row[1] for row in values)
        selected = sum(row[4] for row in values)
        losses = [row[5] for row in values if row[5] is not None]
        credits = [row[6] for row in values if row[6] is not None]
        expected.append(
            (
                feature,
                2,
                eligible,
                eligible,
                eligible,
                selected,
                selected,
                sum(row[3] for row in values),
                sum(row[3] for row in values),
                sum(row[2] and row[4] for row in values),
                selected / eligible,
                sum(losses) / len(losses) if losses else None,
                4.0 if losses else None,
                sum(credits) if credits else None,
                sum(credits) / len(credits) if credits else None,
                "fold-001" if selected else None,
                1 if selected else 2,
            )
        )
    view = _view_rows(store)
    assert len(view) == 2
    for actual, wanted in zip(view, expected, strict=True):
        assert actual[0] == wanted[0]
        assert actual[1:9] == wanted[1:9]
        assert actual[9] == pytest.approx(wanted[9])
        assert actual[10:16] == wanted[10:16]


def test_normalized_selected_pc_credit_conserves_absolute_ablation_loss() -> None:
    loadings = (
        PcaLoading("fold-001", PROFILE, "build", "vix", 1, "source_a", 0.6, 0.36, 0.8),
        PcaLoading("fold-001", PROFILE, "build", "vix", 1, "source_b", 0.8, 0.64, 0.8),
    )
    assert sum(item.squared_loading for item in loadings) == pytest.approx(1.0)
    assert sum(5.0 * item.squared_loading for item in loadings) == pytest.approx(5.0)


def test_failed_or_uncommitted_fold_contributes_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)

    def fail_insert(
        connection: duckdb.DuckDBPyConnection, bundle: FoldMetadataBundle, bundle_hash: str
    ) -> None:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(store, "_insert_fold_rows", fail_insert)
    with pytest.raises(RuntimeError, match="injected failure"):
        store.commit_fold(make_bundle())
    with duckdb.connect(str(store.database), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM feature_global_stats").fetchone() == (0,)


def test_reversal_preserves_cumulative_view_and_plot_hashes(tmp_path: Path) -> None:
    stores = []
    for index, order in enumerate((("fold-001", "fold-002"), ("fold-002", "fold-001"))):
        store = FeatureSelectionMetadataStore(tmp_path / f"store-{index}")
        for fold_id in order:
            store.commit_fold_feature_stats(
                _stats(fold_id, selected=fold_id == "fold-001", credit=1.0, loss=4.0)
            )
        stores.append(store)
    view_hashes = []
    plot_hashes = []
    for index, store in enumerate(stores):
        view_payload = json.dumps(_view_rows(store), sort_keys=True, default=str).encode()
        view_hashes.append(sha256(view_payload).hexdigest())
        paths = render_cumulative_feature_stats(
            store, fold_id="fold-002", output_dir=tmp_path / f"plots-{index}"
        )
        table = next(path for path in paths if path.name == "feature_global_stats.json")
        plot_hashes.append(sha256(table.read_bytes()).hexdigest())
    assert view_hashes[0] == view_hashes[1]
    assert plot_hashes[0] == plot_hashes[1]
