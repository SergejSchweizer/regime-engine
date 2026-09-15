from __future__ import annotations

from pathlib import Path

import pytest

from market_regime_engine.mlflow_support.k_slot_metrics import (
    KCandidateMetricProjection,
    KSlotMetadata,
    KSlotMetricProjection,
    build_four_k_slot_projection,
    build_k_slot_projection,
    validate_k_metric_comparison,
    validate_k_metric_points,
)
from market_regime_engine.mlflow_support.k_slot_plots import (
    build_cross_k_plot_payload,
    build_k_plot_payload,
    render_k_plot_payload,
)
from market_regime_engine.mlflow_support.ports import MetricPoint


def metadata(k: int, *, feature_order: tuple[str, ...] = ("f0", "f1")) -> KSlotMetadata:
    return KSlotMetadata(
        slot_id=f"k{k}",
        state_count=k,
        feature_order=feature_order,
        policy_version="k_champion_policy.v1",
        comparison_domain_id="k_specific_shared_feature_vector.v1",
        source_build_id="source-1",
        source_data_sha256="a" * 64,
        evaluation_plan_hash="b" * 64,
    )


def slot(k: int, *, eligible: bool = True, cross_k_metric: bool = False) -> KSlotMetricProjection:
    info = metadata(k)
    metric_points = [
        MetricPoint("fit_quality_oos_predictive_loglik_per_obs", 0.5, 0, 100),
    ]
    if cross_k_metric:
        metric_points.append(MetricPoint("valid_fold_rate", 1.0, 0, 100))
    candidates = tuple(
        KCandidateMetricProjection(
            logged_model_id=f"k{k}-{family}",
            model_family=family,
            metadata=info,
            metric_points=tuple(metric_points),
        )
        for family in ("gaussian_hmm", "gmm_hmm", "student_t_hmm")
    )
    return build_k_slot_projection(
        candidates,
        eligible=eligible,
        selected_logged_model_id=candidates[0].logged_model_id if eligible else None,
        selected_metric_points=candidates[0].metric_points if eligible else (),
        unavailable_reason=None if eligible else "slot failed validation",
    )


def test_per_k_plot_uses_three_families_and_cross_k_rejects_likelihood(
    tmp_path: Path,
) -> None:
    per_k = build_k_plot_payload(slot(2), "fit_quality_oos_predictive_loglik_per_obs")
    assert per_k.status == "available"
    assert len(per_k.series) == 3
    mixed = build_cross_k_plot_payload(
        tuple(slot(k) for k in (2, 3, 4, 5)), "fit_quality_oos_predictive_loglik_per_obs"
    )
    assert mixed.status == "not_available"
    cross = build_cross_k_plot_payload(
        tuple(slot(k, cross_k_metric=True) for k in (2, 3, 4, 5)), "valid_fold_rate"
    )
    assert cross.status == "available"
    assert cross.comparison_domain_id == "cross_k_dimension_independent.v1"
    assert tuple(series.model_family for series in cross.series) == (
        "K=2",
        "K=3",
        "K=4",
        "K=5",
    )
    assert cross.no_evaluation_recomputation
    manifest = render_k_plot_payload(cross, tmp_path)
    assert manifest["canonical_payload_hash"] == cross.canonical_payload_hash
    assert manifest["logged_model_ids"] == list(cross.logged_model_ids)
    assert manifest["feature_order_hashes"] == list(cross.feature_order_hashes)
    assert manifest["source_data_hashes"] == list(cross.source_data_hashes)
    assert manifest["no_evaluation_recomputation"] is True


def test_ineligible_cross_k_plot_writes_only_an_unavailable_manifest(tmp_path: Path) -> None:
    payload = build_cross_k_plot_payload(
        tuple(slot(k, eligible=k != 5, cross_k_metric=True) for k in (2, 3, 4, 5)),
        "valid_fold_rate",
    )
    assert payload.status == "not_available"
    manifest = render_k_plot_payload(payload, tmp_path)
    assert manifest["png_path"] is None
    assert not tuple(tmp_path.glob("*.png"))
    assert manifest["unavailable_reason"]


def test_slot_projection_requires_all_families_and_four_slots() -> None:
    with pytest.raises(ValueError, match="exactly K=2,3,4,5"):
        build_four_k_slot_projection((slot(2),))
    with pytest.raises(ValueError, match="three model families"):
        build_k_slot_projection(
            tuple(
                KCandidateMetricProjection(
                    logged_model_id=f"only-{index}",
                    model_family=family,
                    metadata=metadata(2),
                    metric_points=(MetricPoint("valid_fold_rate", 1.0, 0, 100),),
                )
                for index, family in enumerate(("gaussian_hmm", "gmm_hmm"))
            ),
            eligible=False,
            unavailable_reason="missing family",
        )


def test_same_feature_metric_comparison_rejects_feature_hash_mismatch() -> None:
    left = slot(2)
    assert left.metadata.feature_order_sha256
    base_tags = metadata(2).tags(scope="candidate")
    tags = {
        "one": base_tags,
        "two": {
            **base_tags,
            "regime_engine.feature_order_sha256": "b" * 64,
            "regime_engine.feature_dimension": "3",
        },
    }
    with pytest.raises(ValueError, match="feature dimensions"):
        validate_k_metric_comparison(
            "fit_quality_oos_predictive_loglik_per_obs",
            {
                key: (MetricPoint("fit_quality_oos_predictive_loglik_per_obs", 0.1, 0, 1),)
                for key in tags
            },
            tags,
            state_count=2,
        )


def test_k_metric_validation_requires_complete_lineage_and_identity() -> None:
    points = {"model": (MetricPoint("valid_fold_rate", 1.0, 0, 1),)}
    complete = metadata(2).tags(scope="candidate")

    missing = {
        key: value for key, value in complete.items() if key != "regime_engine.source_build_id"
    }
    with pytest.raises(ValueError, match="required lineage tags"):
        validate_k_metric_points(points, {"model": missing})

    mismatched = {
        "one": complete,
        "two": {**complete, "regime_engine.source_build_id": "different-source"},
    }
    with pytest.raises(ValueError, match="source_build_id lineage"):
        validate_k_metric_comparison(
            "valid_fold_rate",
            {
                "one": points["model"],
                "two": points["model"],
            },
            mismatched,
            state_count=2,
        )


def test_k_plot_inputs_are_canonical_across_order_and_process_count() -> None:
    from market_regime_engine.mlflow_support.k_slot_plots import prepare_k_plot_payloads

    slots = tuple(slot(k) for k in (2, 3, 4, 5))
    keys = ("fit_quality_oos_predictive_loglik_per_obs",)
    serial = prepare_k_plot_payloads(slots, keys, max_workers=1)
    parallel = prepare_k_plot_payloads(tuple(reversed(slots)), tuple(reversed(keys)), max_workers=2)

    assert tuple(item.canonical_payload_hash for item in serial) == tuple(
        item.canonical_payload_hash for item in parallel
    )
    assert tuple(item.slot_id for item in serial) == ("k2", "k3", "k4", "k5")
