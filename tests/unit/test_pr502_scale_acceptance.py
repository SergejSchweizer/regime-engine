from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.feature_discovery import pipeline as selection_pipeline
from market_regime_engine.feature_discovery.ablation import HMMSubsetEvaluation
from market_regime_engine.feature_discovery.feature_roles import (
    CORE_FEATURES,
    TEMPORAL_KEY,
    TRANSFORMATION_FAMILIES,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.metadata_store import FeatureSelectionMetadataStore
from market_regime_engine.feature_discovery.pipeline import run_canonical_feature_selection
from market_regime_engine.feature_discovery.quality import filter_outer_train_quality
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)
from market_regime_engine.runtime.cpu import available_cpu_count
from market_regime_engine.runtime.parallel import ParallelExecutionPlan
from market_regime_engine.runtime.performance import PerformanceRecorder


class _ScaleOracle:
    def bind_feature_values(self, _values: dict[str, tuple[float, ...]]) -> None:
        return None

    def evaluate_subset(self, features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, float(len(features)))

    def evaluate_hmm_subset(self, features: tuple[str, ...]) -> HMMSubsetEvaluation:
        score = self.evaluate_subset(features)
        fit_hash = sha256("|".join(features).encode()).hexdigest()
        return HMMSubsetEvaluation(score, "gaussian_hmm", 2, "b" * 64, fit_hash)


def test_10000_feature_scale_stays_post_pca_and_records_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transformation_names = tuple(
        f"{family}_delta_{index}obs" for index in range(768) for family in TRANSFORMATION_FAMILIES
    )
    names = (*CORE_FEATURES, *transformation_names)
    assert len(names) >= 10_000
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    rng = np.random.default_rng(502_10_000)
    rows = 36
    matrix = np.zeros((rows, len(names)), dtype=np.float64)
    quality_input_count = len(CORE_FEATURES) + 8 * len(TRANSFORMATION_FAMILIES)
    matrix[:, :quality_input_count] = rng.normal(size=(rows, quality_input_count))
    lineage = SourceLineage(
        source_dataset="hermetic.macro_features",
        source_build_id="build-502-scale",
        data_sha256="c" * 64,
        schema_version=1,
        feature_version=1,
        source_table="hermetic.macro_features",
        synced_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
        row_count=rows,
        min_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        max_timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=rows - 1),
    )
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        TEMPORAL_KEY,
        tuple(FeatureCatalogEntry(name, index + 1) for index, name in enumerate(names)),
    )
    snapshot = FeatureSnapshot(
        lineage=lineage,
        feature_names=catalog.feature_names,
        rows=tuple(
            FeatureRow(
                datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=row_index),
                tuple(float(value) for value in matrix[row_index]),
            )
            for row_index in range(rows)
        ),
    )
    quality = filter_outer_train_quality(
        catalog,
        snapshot,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=rows - 1),
        max_workers=None,
    )
    quality_names = quality.eligible_features
    values = {
        name: tuple(float(value) for value in matrix[:, index]) for index, name in enumerate(names)
    }
    observed_global_candidate_counts: list[int] = []
    selection_module = cast(Any, selection_pipeline)
    original_global_reduction = selection_module.prune_global_correlated_features

    def observe_global_reduction(*args: object, **kwargs: object) -> object:
        feature_values = args[0]
        assert isinstance(feature_values, dict)
        observed_global_candidate_counts.append(len(feature_values))
        return original_global_reduction(*args, **kwargs)

    monkeypatch.setattr(
        selection_module,
        "prune_global_correlated_features",
        observe_global_reduction,
    )
    oracle = _ScaleOracle()
    report_path = tmp_path / "scale-performance.json"
    recorder = PerformanceRecorder(report_path)
    family_plan = ParallelExecutionPlan.create(len(TRANSFORMATION_FAMILIES))
    with recorder.stage(
        "10000_feature_selection",
        task_count=len(names),
        worker_count=family_plan.worker_count,
    ):
        result = run_canonical_feature_selection(
            values,
            contract,
            quality_eligible_features=quality_names,
            evaluate_subset=oracle.evaluate_subset,
            evaluate_hmm_subset=oracle.evaluate_hmm_subset,
            hmm_selector_contract_hash="b" * 64,
            max_sffs_features=3,
            max_workers=None,
            metadata_store=FeatureSelectionMetadataStore(tmp_path / "metadata"),
            metadata_fold_id="fold-502-scale",
            metadata_source_build_id="build-502-scale",
            metadata_state_count=2,
        )
    recorder.update_metadata(
        {
            "candidate_counts": {
                "discovered_features": len(names),
                "quality_survivors": len(quality_names),
                "global_correlation_input": observed_global_candidate_counts[-1],
                "sffs_representatives": len(result.global_reduction.representatives),
                "hmm_selected": len(result.selected_features),
            },
            "effective_worker_counts": {
                "family_pca": family_plan.worker_count,
                "global_correlation": ParallelExecutionPlan.create(
                    observed_global_candidate_counts[-1]
                ).worker_count,
            },
        }
    )
    recorder.finish(status="COMPLETE")

    assert observed_global_candidate_counts == [
        len(result.global_reduction.representatives) + len(result.global_reduction.removed_features)
    ]
    assert observed_global_candidate_counts[0] < 1_000
    assert all(
        name.startswith("family_pc_") or name in CORE_FEATURES
        for name in result.sffs.selected_features
    )
    assert not set(result.sffs.selected_features) & set(transformation_names)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["metadata"]["candidate_counts"]["discovered_features"] >= 10_000
    assert report["metadata"]["candidate_counts"]["quality_survivors"] < 1_000
    assert report["metadata"]["candidate_counts"]["global_correlation_input"] < 1_000
    assert report["peak_parent_rss_mib"] < 64 * 1024
    assert report["stages"][0]["worker_count"] == family_plan.worker_count
    assert available_cpu_count() >= 1
