from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import market_regime_engine.evaluations.k_feature_selection as module


def test_k_feature_selection_rejects_invalid_task_contracts() -> None:
    frame = pd.DataFrame({"timestamp_m1": [datetime(2024, 1, 1, tzinfo=UTC)]})
    cutoff = frame.loc[0, "timestamp_m1"]
    with pytest.raises(TypeError, match="pandas DataFrame"):
        module.run_k_feature_selection(
            [], source_snapshot_id="s", validation_cutoff=cutoff, selector=lambda **_: None
        )
    with pytest.raises(ValueError, match="source_snapshot_id"):
        module.run_k_feature_selection(
            frame, source_snapshot_id=" ", validation_cutoff=cutoff, selector=lambda **_: None
        )
    with pytest.raises(ValueError, match="unique ordered subset"):
        module.run_k_feature_selection(
            frame,
            source_snapshot_id="s",
            validation_cutoff=cutoff,
            selector=lambda **_: None,
            requested_state_counts=(3, 2),
        )
    with pytest.raises(ValueError, match="at least one"):
        module.run_k_feature_selection(
            frame,
            source_snapshot_id="s",
            validation_cutoff=cutoff,
            selector=lambda **_: None,
            requested_state_counts=(),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        module.run_k_feature_selection(
            frame,
            source_snapshot_id="s",
            validation_cutoff=datetime(2024, 1, 1),
            selector=lambda **_: None,
        )


def test_k_feature_selection_runs_ordered_tasks_and_captures_failures() -> None:
    cutoff = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame({"timestamp_m1": [cutoff], "f0": [1.0]})

    def selector(rows: pd.DataFrame, *, state_count: int, **_: object) -> None:
        assert rows is not frame
        if state_count == 3:
            raise RuntimeError("synthetic failure")
        return None

    results = module.run_k_feature_selection(
        frame,
        source_snapshot_id="snapshot",
        validation_cutoff=cutoff,
        selector=selector,
        requested_state_counts=(2, 3),
        max_workers=1,
    )
    assert tuple(result.state_count for result in results) == (2, 3)
    assert results[0].eligible is False
    assert "no eligible" in (results[0].rejection_reason or "")
    assert results[1].rejection_reason == "RuntimeError: synthetic failure"


def test_k_feature_selection_real_wrapper_validates_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"timestamp_m1": [datetime(2024, 1, 1, tzinfo=UTC)]})
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> tuple[object, ...]:
        captured.update(kwargs)
        return ("ok",)

    monkeypatch.setattr(module, "run_k_feature_selection", fake_run)
    assert module.run_real_k_feature_selection(
        frame,
        catalog=object(),
        profile=object(),
        source_snapshot_id="snapshot",
        source_build_id="build",
        validation_cutoff=datetime(2024, 1, 1, tzinfo=UTC),
        deployment_cutoff=datetime(2024, 1, 2, tzinfo=UTC),
        requested_state_counts=(2,),
        max_workers=1,
    ) == ("ok",)
    assert captured["requested_state_counts"] == (2,)

    with pytest.raises(ValueError, match="unique ordered subset"):
        module.run_real_k_feature_selection(
            frame,
            catalog=object(),
            profile=object(),
            source_snapshot_id="snapshot",
            source_build_id="build",
            validation_cutoff=datetime(2024, 1, 1, tzinfo=UTC),
            deployment_cutoff=datetime(2024, 1, 2, tzinfo=UTC),
            requested_state_counts=(2, 2),
        )


def test_fixed_k_selection_runs_the_complete_hermetic_discovery_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutoff = datetime(2024, 1, 2, tzinfo=UTC)
    deployment = datetime(2024, 1, 3, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp_m1": [datetime(2024, 1, 1, tzinfo=UTC), cutoff],
            "f0": [1.0, 2.0],
        }
    )
    catalog = SimpleNamespace(
        feature_names=("f0", "pca_pc_001"),
        catalog_hash="catalog-hash",
        lineage=SimpleNamespace(source_build_id="build"),
    )
    snapshot = SimpleNamespace(
        lineage=SimpleNamespace(source_build_id="build"),
        rows=(
            SimpleNamespace(timestamp=frame.iloc[0, 0]),
            SimpleNamespace(timestamp=cutoff),
        ),
    )
    profile = SimpleNamespace(
        profile_hash="profile-hash",
        profile_id="xetra",
        profile_config_version=4,
        pca=SimpleNamespace(component_count=1, variance_threshold=0.9),
    )
    calls: list[str] = []
    monkeypatch.setattr(module, "validate_pca_feature_universe", lambda *args, **kwargs: ("f0",))
    monkeypatch.setattr(module, "_as_feature_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(
        module,
        "filter_outer_train_quality",
        lambda *args, **kwargs: calls.append("quality") or SimpleNamespace(result_hash="quality"),
    )
    monkeypatch.setattr(
        module,
        "global_absolute_spearman_distance",
        lambda *args, **kwargs: calls.append("distance") or SimpleNamespace(matrix_hash="distance"),
    )
    clusters = SimpleNamespace(solution_hash="clusters")
    monkeypatch.setattr(
        module,
        "select_global_clusters",
        lambda *args, **kwargs: calls.append("clusters") or clusters,
    )
    monkeypatch.setattr(
        module,
        "select_temporary_prototypes",
        lambda *args, **kwargs: calls.append("prototypes") or SimpleNamespace(prototypes=("f0",)),
    )
    monkeypatch.setattr(
        module,
        "build_inner_walk_forward_plan",
        lambda *args, **kwargs: calls.append("plan") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        module,
        "build_model_clock_preflight",
        lambda *args, **kwargs: calls.append("preflight") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        module,
        "require_model_clock_eligible",
        lambda *args, **kwargs: calls.append("eligible"),
    )
    monkeypatch.setattr(
        module,
        "ResolvedCandidateProfile",
        lambda **kwargs: calls.append("candidate") or SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(module, "adapter_factory", lambda *args, **kwargs: object())
    fold = SimpleNamespace(
        fold_id="inner-1",
        oos_timestamps=(datetime(2024, 1, 1, 12, tzinfo=UTC),),
        oos_filtered_probabilities=((0.7, 0.3),),
    )
    evaluation = SimpleNamespace(
        valid_folds=(fold,),
        folds=(SimpleNamespace(valid=True, failure_reason=None),),
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        evaluation_plan_hash="b" * 64,
    )
    monkeypatch.setattr(
        module,
        "run_provisional_gaussian_candidate",
        lambda *args, **kwargs: calls.append("teacher") or evaluation,
    )
    monkeypatch.setattr(
        module,
        "score_all_raw_features",
        lambda *args, **kwargs: calls.append("scores") or (SimpleNamespace(),),
    )
    monkeypatch.setattr(
        module,
        "select_cluster_winners",
        lambda *args, **kwargs: calls.append("winners") or SimpleNamespace(ranked_features=("f0",)),
    )
    selected = SimpleNamespace(
        valid=True,
        candidate_id="gaussian_hmm_k2_full",
        feature_order=("f0",),
    )
    monkeypatch.setattr(
        module,
        "search_ranked_prefixes",
        lambda *args, **kwargs: (
            calls.append("prefixes")
            or SimpleNamespace(
                selected_prefix_length=2,
                selected_candidate_id=selected.candidate_id,
                evaluations=(selected,),
            )
        ),
    )
    monkeypatch.setattr(module, "content_hash", lambda value: "a" * 64)
    monkeypatch.setattr(
        module,
        "KChampionSelection",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    payload = module.select_k_specific_feature_configuration(
        frame,
        catalog=catalog,
        profile=profile,
        state_count=2,
        source_snapshot_id="snapshot",
        source_build_id="build",
        validation_cutoff=cutoff,
        deployment_cutoff=deployment,
        max_workers=1,
    )
    assert payload.selection.slot_id == "k2"
    assert payload.selection.state_count == 2
    assert payload.teacher_identity.startswith("gaussian_hmm_k2_full:")
    assert calls == [
        "quality",
        "distance",
        "clusters",
        "prototypes",
        "plan",
        "preflight",
        "eligible",
        "candidate",
        "teacher",
        "scores",
        "winners",
        "prefixes",
    ]
