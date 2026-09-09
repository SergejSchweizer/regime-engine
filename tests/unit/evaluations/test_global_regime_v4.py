from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

import market_regime_engine.evaluations.global_regime_v4 as global_v4
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation.selection import (
    CandidateSelectionEvidence,
    StatisticalChampionSelection,
)
from market_regime_engine.evaluation.walk_forward import AdapterFactory, WalkForwardEvaluation
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan, plan_walk_forward
from market_regime_engine.evaluations.teacher_reference import FrozenTeacherRefit
from market_regime_engine.feature_discovery.contracts import (
    PrefixEvaluation,
    PrefixSearchResult,
    PrototypeSet,
    ProvisionalTeacherReference,
)
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile

START = datetime(2020, 1, 1, tzinfo=UTC)
HASH = "a" * 64


@dataclass(frozen=True)
class _WinnerStub:
    ranked_features: tuple[str, ...]


def _catalog() -> FeatureCatalogSnapshot:
    lineage = SourceLineage(
        source_dataset="xetra_gold",
        source_build_id="build-1",
        data_sha256=HASH,
        schema_version=2,
        feature_version=4,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=START,
        row_count=1449,
        min_timestamp=START,
        max_timestamp=START + timedelta(days=1448),
    )
    entries = tuple(
        FeatureCatalogEntry(name, index + 1) for index, name in enumerate(("f0", "f1", "f2"))
    )
    return FeatureCatalogSnapshot.from_entries(lineage, "timestamp_m1", entries)


def _rows(count: int = 1449) -> pd.DataFrame:
    index = np.arange(count, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(START + timedelta(days=int(value)) for value in index),
            "f0": np.sin(index / 17.0),
            "f1": np.cos(index / 23.0),
            "f2": np.sin(index / 7.0) + index / 1000.0,
        }
    )


def _candidate(catalog: FeatureCatalogSnapshot) -> ResolvedCandidateProfile:
    return ResolvedCandidateProfile(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        covariance_type="full",
        feature_order=("f0", "f1"),
        feature_dimension=2,
        source_build_id=catalog.lineage.source_build_id,
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
        original_feature_universe=catalog.feature_names,
        preliminary_medoids=(),
        feature_contract_version=4,
    )


def _teacher_reference() -> ProvisionalTeacherReference:
    return ProvisionalTeacherReference(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        timestamps=(START,),
        filtered_probabilities=((0.5, 0.5),),
        dominant_states=(0,),
        valid_inner_fold_ids=("fold_001",),
        source_build_id="build-1",
        inner_plan_hash=HASH,
        prototype_features=("f0",),
    )


def _teacher_refit(timestamps: tuple[datetime, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        model_artifact=GaussianHMMArtifact(
            state_count=2,
            feature_order=("f0",),
            start_probabilities=(0.5, 0.5),
            transition_matrix=((0.8, 0.2), (0.2, 0.8)),
            means=((-1.0,), (1.0,)),
            full_covariances=(((1.0,),), ((1.0,),)),
        ),
        test_timestamps=timestamps,
        test_filtered_probabilities=((0.5, 0.5),) * len(timestamps),
    )


def _model_evaluation(timestamps: tuple[datetime, ...]) -> SimpleNamespace:
    fold = SimpleNamespace(
        oos_predictive_log_likelihood_per_observation=-1.0,
        oos_timestamps=timestamps,
        oos_filtered_probabilities=((0.5, 0.5),) * len(timestamps),
    )
    return SimpleNamespace(valid_folds=(fold,))


def _selection_stub(catalog: FeatureCatalogSnapshot) -> global_v4.V4ConfigurationSelection:
    candidate = _candidate(catalog)
    prefix = PrefixSearchResult(
        ranked_features=("f0", "f1"),
        evaluations=(
            PrefixEvaluation(
                prefix_length=2,
                feature_order=("f0", "f1"),
                candidate_id="gaussian_hmm_k2_full",
                shared_timestamp_count=10,
                shared_teacher_coverage=1.0,
                soft_regime_nmi=0.5,
            ),
        ),
        selected_prefix_length=2,
        selected_candidate_id="gaussian_hmm_k2_full",
    )
    champion = SimpleNamespace(champion_candidate_id=candidate.candidate_id)
    constructor = cast(Any, global_v4.V4ConfigurationSelection)
    return cast(
        global_v4.V4ConfigurationSelection,
        constructor(
            source_build_id="build-1",
            catalog_hash=catalog.catalog_hash,
            quality=SimpleNamespace(source_build_id="build-1", catalog_hash=catalog.catalog_hash),
            distance=SimpleNamespace(),
            clusters=SimpleNamespace(),
            prototypes=SimpleNamespace(),
            teacher_evaluation=SimpleNamespace(),
            teacher_reference=_teacher_reference(),
            feature_scores=(),
            winner_selection=SimpleNamespace(),
            prefix_search=prefix,
            final_grid=SimpleNamespace(selection=champion),
            final_candidate=candidate,
            feature_discovery_hash=HASH,
        ),
    )


def test_configuration_selection_contract_rejects_inconsistent_evidence() -> None:
    catalog = _catalog()
    valid = _selection_stub(catalog)
    assert valid.final_candidate.candidate_id == "gaussian_hmm_k2_full"

    def values(**changes: object) -> dict[str, object]:
        result = {field.name: getattr(valid, field.name) for field in fields(valid)}
        result.update(changes)
        return result

    wrong_prefix = SimpleNamespace(
        evaluations=(SimpleNamespace(feature_order=("f0", "f2")),),
        selected_prefix_length=2,
    )
    cases = (
        values(quality=SimpleNamespace(source_build_id="other", catalog_hash=catalog.catalog_hash)),
        values(catalog_hash=HASH),
        values(final_grid=SimpleNamespace(selection=None)),
        values(
            final_grid=SimpleNamespace(
                selection=SimpleNamespace(champion_candidate_id="gaussian_hmm_k3_full")
            )
        ),
        values(prefix_search=wrong_prefix),
        values(feature_discovery_hash="Z" * 64),
    )
    messages = (
        "source build",
        "catalog hash",
        "final statistical champion",
        "final-grid champion",
        "selected prefix",
        "lowercase SHA-256",
    )
    for case, message in zip(cases, messages, strict=True):
        with pytest.raises(ValueError, match=message):
            cast(Any, global_v4.V4ConfigurationSelection)(**case)


def test_selection_pipeline_runs_all_train_only_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    rows = _rows(20)
    prefix = PrefixSearchResult(
        ranked_features=("f0", "f1"),
        evaluations=(
            PrefixEvaluation(
                prefix_length=2,
                feature_order=("f0", "f1"),
                candidate_id="gaussian_hmm_k2_full",
                shared_timestamp_count=10,
                shared_teacher_coverage=1.0,
                soft_regime_nmi=0.5,
            ),
        ),
        selected_prefix_length=2,
        selected_candidate_id="gaussian_hmm_k2_full",
    )
    champion = StatisticalChampionSelection(
        champion_candidate_id="gaussian_hmm_k2_full",
        champion_state_count=2,
        ranked_candidate_ids=("gaussian_hmm_k2_full",),
        evidence=(
            CandidateSelectionEvidence(
                candidate_id="gaussian_hmm_k2_full",
                state_count=2,
                accepted=True,
                rejection_reasons=(),
                rank=1,
            ),
        ),
    )
    calls: list[str] = []

    def stage(name: str, value: object) -> Callable[..., object]:
        def invoke(*args: object, **kwargs: object) -> object:
            del args, kwargs
            calls.append(name)
            return value

        return invoke

    monkeypatch.setattr(
        global_v4,
        "filter_outer_train_quality",
        stage(
            "quality",
            SimpleNamespace(
                source_build_id="build-1",
                catalog_hash=catalog.catalog_hash,
                result_hash=HASH,
            ),
        ),
    )
    monkeypatch.setattr(
        global_v4,
        "global_absolute_spearman_distance",
        stage("distance", SimpleNamespace(matrix_hash=HASH)),
    )
    monkeypatch.setattr(
        global_v4,
        "select_global_clusters",
        stage("clusters", SimpleNamespace(solution_hash=HASH)),
    )
    monkeypatch.setattr(
        global_v4,
        "select_temporary_prototypes",
        stage(
            "prototypes",
            PrototypeSet(
                cluster_ids=("cluster_000",),
                prototypes=("f0",),
                mean_distances=(("f0", 0.0),),
            ),
        ),
    )
    teacher_evaluation = SimpleNamespace(inner_plan=SimpleNamespace(plan_hash=HASH))
    teacher_reference = _teacher_reference()
    monkeypatch.setattr(
        global_v4,
        "select_provisional_teacher",
        stage("teacher", teacher_evaluation),
    )
    monkeypatch.setattr(
        global_v4,
        "build_provisional_teacher_reference",
        stage("teacher_reference", teacher_reference),
    )
    monkeypatch.setattr(global_v4, "score_all_raw_features", stage("scores", ("score",)))
    monkeypatch.setattr(
        global_v4,
        "select_cluster_winners",
        stage("winners", _WinnerStub(ranked_features=("f0", "f1"))),
    )
    monkeypatch.setattr(global_v4, "search_ranked_prefixes", stage("prefix", prefix))
    monkeypatch.setattr(
        global_v4,
        "evaluate_final_v4_grid",
        stage(
            "final_grid",
            SimpleNamespace(
                candidate_grid="grid",
                selection=champion,
                no_champion_reason=None,
            ),
        ),
    )

    selection = global_v4.select_v4_configuration(
        rows,
        catalog=catalog,
        profile=profile,
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
    )

    assert calls == [
        "quality",
        "distance",
        "clusters",
        "prototypes",
        "teacher",
        "teacher_reference",
        "scores",
        "winners",
        "prefix",
        "final_grid",
    ]
    assert selection.final_candidate.candidate_id == champion.champion_candidate_id
    assert selection.final_candidate.feature_order == ("f0", "f1")
    assert len(selection.feature_discovery_hash) == 64


def test_selection_input_contracts_and_hash_defaults_fail_closed() -> None:
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    rows = _rows(20)

    with pytest.raises(TypeError, match="pandas DataFrame"):
        global_v4._validate_train_inputs(object(), catalog, profile, "build-1")
    with pytest.raises(TypeError, match="feature catalog"):
        global_v4._validate_train_inputs(
            rows, cast(FeatureCatalogSnapshot, object()), profile, "build-1"
        )
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        global_v4._utc(datetime(2020, 1, 1), "timestamp")
    with pytest.raises(ValueError, match="source build"):
        global_v4._validate_train_inputs(rows, catalog, profile, "other-build")

    definition, execution = global_v4._selection_hashes(
        profile, catalog, START + timedelta(days=19), None, None
    )
    assert len(definition) == 64
    assert len(execution) == 64


def test_snapshot_conversion_preserves_missing_values_and_pd_isna_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    values = _rows(2)
    values.loc[0, "f0"] = pd.NA
    snapshot = global_v4._as_feature_snapshot(values, catalog)
    assert snapshot.rows[0].values[0] is None

    original_isna = pd.isna
    monkeypatch.setattr(pd, "isna", lambda value: (_ for _ in ()).throw(ValueError()))
    try:
        fallback = global_v4._as_feature_snapshot(_rows(1), catalog)
    finally:
        monkeypatch.setattr(pd, "isna", original_isna)
    assert fallback.rows[0].values[0] == pytest.approx(0.0)


def test_global_policy_input_and_outer_continuation_failures_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    rows = _rows()

    with pytest.raises(TypeError, match="pandas DataFrame"):
        global_v4.evaluate_global_regime_v4(
            cast(pd.DataFrame, object()), catalog=catalog, profile=profile
        )
    with pytest.raises(ValueError, match="source build"):
        global_v4.evaluate_global_regime_v4(
            rows, catalog=catalog, profile=profile, source_build_id="wrong"
        )
    with pytest.raises(ValueError, match="timestamp_m1"):
        global_v4.evaluate_global_regime_v4(
            rows.drop(columns=["timestamp_m1"]), catalog=catalog, profile=profile
        )
    duplicate = rows.copy()
    duplicate.loc[1, "timestamp_m1"] = duplicate.loc[0, "timestamp_m1"]
    with pytest.raises(ValueError, match="strictly increasing"):
        global_v4.evaluate_global_regime_v4(duplicate, catalog=catalog, profile=profile)

    monkeypatch.setattr(
        global_v4, "select_v4_configuration", lambda *args, **kwargs: _selection_stub(catalog)
    )
    result = global_v4.evaluate_global_regime_v4(
        rows,
        catalog=catalog,
        profile=profile,
        outer_runner=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("fit failed")),
    )
    assert result.valid_fold_count == 0
    assert all(
        "outer refit/TEST continuation failed" in (fold.failure_reason or "")
        for fold in result.outer_folds
    )


def test_failure_configuration_requires_two_catalog_features() -> None:
    lineage = _catalog().lineage
    one_feature = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("f0", 1),),
    )
    fold = plan_walk_forward(
        tuple(_rows()["timestamp_m1"]), load_profile("configs/profiles/xetra_v4.yaml").walk_forward
    ).folds[0]
    with pytest.raises(ValueError, match="two catalog features"):
        global_v4._fallback_configuration(one_feature, "build-1", fold, "reason")


def test_outer_policy_passes_only_train_rows_to_each_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows()
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    candidate = _candidate(catalog)
    seen_lengths: list[int] = []

    def select(train_rows: pd.DataFrame, **kwargs: object) -> SimpleNamespace:
        del kwargs
        seen_lengths.append(len(train_rows))
        return SimpleNamespace(
            final_candidate=candidate,
            teacher_reference=_teacher_reference(),
            catalog_hash=catalog.catalog_hash,
            feature_discovery_hash=HASH,
        )

    def outer_runner(
        source_rows: pd.DataFrame,
        plan: WalkForwardPlan,
        profile: ModelProfile,
        candidate: ResolvedCandidateProfile,
        candidate_adapter_factory: AdapterFactory,
    ) -> WalkForwardEvaluation:
        del profile, candidate, candidate_adapter_factory
        fold = plan.folds[0]
        timestamps = tuple(source_rows["timestamp_m1"].iloc[-63:])
        assert timestamps[-1] == fold.test_end
        return cast(WalkForwardEvaluation, _model_evaluation(timestamps))

    monkeypatch.setattr(global_v4, "select_v4_configuration", select)
    result = global_v4.evaluate_global_regime_v4(
        rows,
        catalog=catalog,
        profile=profile,
        outer_runner=outer_runner,
        teacher_refitter=cast(
            Callable[..., FrozenTeacherRefit],
            lambda train, test, **kwargs: _teacher_refit(tuple(test["timestamp_m1"])),
        ),
    )

    assert seen_lengths == [1260, 1323, 1386]
    assert len(result.outer_folds) == 3
    assert result.valid_fold_count == 3
    assert result.production_eligible is True


def test_failed_outer_selection_does_not_reuse_a_previous_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows()
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    calls = 0

    def fail_selection(train_rows: pd.DataFrame, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise ValueError(f"synthetic selection failure {calls}")

    monkeypatch.setattr(global_v4, "select_v4_configuration", fail_selection)
    result = global_v4.evaluate_global_regime_v4(rows, catalog=catalog, profile=profile)

    assert calls == 3
    assert result.valid_fold_count == 0
    assert result.production_eligible is False
    assert all(not fold.valid for fold in result.outer_folds)
    assert (
        len({fold.final_configuration.feature_discovery_hash for fold in result.outer_folds}) == 3
    )
    assert all(
        "TRAIN-only v4 selection failed" in (fold.failure_reason or "")
        for fold in result.outer_folds
    )


def test_train_snapshot_rejects_test_only_column_and_keeps_catalog_order() -> None:
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    train = _rows(20).drop(columns=["f2"])

    with pytest.raises(ValueError, match="missing catalog columns"):
        global_v4.select_v4_configuration(train, catalog=catalog, profile=profile)

    snapshot = global_v4._as_feature_snapshot(_rows(20), catalog)
    assert snapshot.feature_names == ("f0", "f1", "f2")
    assert snapshot.rows[0].timestamp == START
