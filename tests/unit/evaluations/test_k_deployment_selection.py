from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from market_regime_engine.contracts.core import (
    K_CHAMPION_POLICY_ID,
    K_CHAMPION_POLICY_VERSION,
)
from market_regime_engine.evaluations.k_champion_contract import (
    K_SLOT_COMPARISON_DOMAIN,
    K_SLOT_PROMOTION_SCORE_VERSION,
    KChampionSelection,
    feature_order_hash,
)
from market_regime_engine.evaluations.k_deployment_selection import (
    KDeploymentArtifact,
    KDeploymentSelectionResult,
    KDeploymentSlotResult,
    _run_slot,
    select_k_deployment_packages,
)

HASH = "a" * 64
START = datetime(2026, 1, 1, tzinfo=UTC)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_m1": [START + timedelta(days=index) for index in range(3)],
            "f0": [1.0, 2.0, 3.0],
        }
    )


def _selection(slot_id: str, cutoff: datetime) -> KChampionSelection:
    features = ("f0",)
    return KChampionSelection(
        slot_id=slot_id,
        state_count=int(slot_id[1:]),
        model_family="gaussian_hmm",
        candidate_identity=f"gaussian_hmm_{slot_id}_full",
        feature_order=features,
        feature_order_hash=feature_order_hash(features),
        source_snapshot_id="snapshot-1",
        profile_id="xetra",
        profile_config_version=4,
        policy_id=K_CHAMPION_POLICY_ID,
        policy_version=K_CHAMPION_POLICY_VERSION,
        validation_cutoff=START,
        deployment_cutoff=cutoff,
        comparison_domain_id=K_SLOT_COMPARISON_DOMAIN,
        promotion_score_version=K_SLOT_PROMOTION_SCORE_VERSION,
        reference_teacher_id="teacher-1",
        artifact_hash=HASH,
        source_build_id="build-1",
        source_catalog_hash=HASH,
    )


def _artifact(selection: KChampionSelection, cutoff: datetime) -> KDeploymentArtifact:
    return KDeploymentArtifact(
        slot_id=selection.slot_id,
        selection_hash=selection.selection_hash,
        artifact_hash=HASH,
        package_directory=f"/tmp/{selection.slot_id}",
        source_snapshot_id="snapshot-1",
        deployment_cutoff=cutoff,
        state_count=selection.state_count,
        model_family=selection.model_family,
        feature_order=selection.feature_order,
        feature_order_hash=selection.feature_order_hash,
        policy_version=selection.policy_version,
        source_build_id="build-1",
        source_catalog_hash=HASH,
    )


def _validation(*, eligible: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        source_snapshot_id="snapshot-1",
        validation_cutoff=START,
        slots=tuple(
            SimpleNamespace(
                slot_id=slot,
                eligible=eligible,
                rejection_reasons=() if eligible else ("validation failed",),
            )
            for slot in ("k2", "k3", "k4", "k5")
        ),
    )


def test_selects_and_refits_each_eligible_k_slot() -> None:
    rows = _rows()
    cutoff = rows["timestamp_m1"].iloc[-1]
    selected: list[str] = []
    refit: list[str] = []

    def selector(
        source: pd.DataFrame, *, slot_id: str, deployment_cutoff: datetime
    ) -> KChampionSelection:
        assert len(source) == 3
        assert deployment_cutoff == cutoff
        selected.append(slot_id)
        return _selection(slot_id, cutoff)

    def refitter(
        source: pd.DataFrame, *, selection: KChampionSelection, slot_id: str
    ) -> KDeploymentArtifact:
        assert source["timestamp_m1"].iloc[-1] == cutoff
        refit.append(slot_id)
        return _artifact(selection, cutoff)

    result = select_k_deployment_packages(
        rows,
        validation=_validation(),
        deployment_cutoff=cutoff,
        source_build_id="build-1",
        source_catalog_hash=HASH,
        selector=selector,
        refitter=refitter,
        max_workers=1,
    )

    assert tuple(item.slot_id for item in result.slots) == ("k2", "k3", "k4", "k5")
    assert len(result.eligible_artifacts) == 4
    assert selected == ["k2", "k3", "k4", "k5"]
    assert refit == selected


def test_ineligible_slots_are_reported_without_selector_or_refit() -> None:
    calls: list[str] = []
    rows = _rows()
    cutoff = rows["timestamp_m1"].iloc[-1]
    validation = _validation(eligible=False)

    def selector(*args: object, **kwargs: object) -> None:
        calls.append("selector")
        return None

    def refitter(*args: object, **kwargs: object) -> None:
        calls.append("refitter")
        return None

    result = select_k_deployment_packages(
        rows,
        validation=validation,
        deployment_cutoff=cutoff,
        source_build_id="build-1",
        source_catalog_hash=HASH,
        selector=selector,
        refitter=refitter,
        max_workers=1,
    )
    assert not result.eligible_artifacts
    assert all(
        item.reason == "slot failed outer validation: validation failed" for item in result.slots
    )
    assert calls == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows, cutoff: rows.drop(columns=["timestamp_m1"]), "requires timestamp_m1"),
        (lambda rows, cutoff: rows.iloc[[1, 0, 2]], "strictly increasing"),
        (lambda rows, cutoff: rows, "exactly through deployment cutoff"),
    ],
)
def test_deployment_selection_rejects_source_contract_drift(change, message: str) -> None:
    rows = _rows()
    cutoff = rows["timestamp_m1"].iloc[-1]
    if message == "exactly through deployment cutoff":
        cutoff = cutoff + timedelta(days=1)
    with pytest.raises((ValueError, TypeError), match=message):
        select_k_deployment_packages(
            change(rows, cutoff),
            validation=_validation(),
            deployment_cutoff=cutoff,
            source_build_id="build-1",
            source_catalog_hash=HASH,
            selector=lambda *args, **kwargs: None,
            refitter=lambda *args, **kwargs: None,
            max_workers=1,
        )


def test_deployment_artifact_and_slot_result_contracts_fail_closed() -> None:
    cutoff = START + timedelta(days=2)
    selection = _selection("k2", cutoff)
    artifact = _artifact(selection, cutoff)
    assert KDeploymentSlotResult("k2", True, selection, artifact).eligible
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(artifact, source_catalog_hash="A" * 64)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda item: replace(item, slot_id="k1"), "slot must be k2"),
        (lambda item: replace(item, state_count=3), "state count must match"),
        (lambda item: replace(item, model_family="unsupported"), "family is unsupported"),
        (lambda item: replace(item, feature_order=()), "feature order must be non-empty"),
        (lambda item: replace(item, policy_version="wrong"), "policy version is unsupported"),
        (lambda item: replace(item, source_build_id=" build"), "source build must be non-empty"),
    ],
)
def test_deployment_artifact_identity_fields_are_strict(change, message: str) -> None:
    selection = _selection("k2", START + timedelta(days=2))
    with pytest.raises(ValueError, match=message):
        change(_artifact(selection, START + timedelta(days=2)))


def test_deployment_slot_and_result_contracts_reconcile_every_identity() -> None:
    cutoff = START + timedelta(days=2)
    selection = _selection("k2", cutoff)
    artifact = _artifact(selection, cutoff)
    with pytest.raises(ValueError, match="eligibility"):
        KDeploymentSlotResult("k2", True, None, None)
    with pytest.raises(ValueError, match="trimmed reason"):
        KDeploymentSlotResult("k2", False, None, None, " bad")
    with pytest.raises(ValueError, match="bound"):
        KDeploymentSlotResult("k2", True, selection, replace(artifact, selection_hash="b" * 64))
    valid = KDeploymentSlotResult("k2", True, selection, artifact)
    slots = (
        valid,
        *tuple(
            KDeploymentSlotResult(slot, False, None, None, "not eligible")
            for slot in ("k3", "k4", "k5")
        ),
    )
    result = KDeploymentSelectionResult("snapshot-1", START, cutoff, slots)
    assert result.eligible_artifacts == (artifact,)
    with pytest.raises(ValueError, match="exactly k2"):
        KDeploymentSelectionResult("snapshot-1", START, cutoff, tuple(reversed(slots)))


def test_run_slot_converts_selector_and_refitter_contract_errors_to_rejections() -> None:
    rows = _rows()
    cutoff = rows["timestamp_m1"].iloc[-1]
    validation = _validation().slots[0]
    base = (
        rows,
        "k2",
        validation,
        "snapshot-1",
        "build-1",
        HASH,
        START,
        cutoff,
    )
    no_selection = _run_slot((*base, lambda *args, **kwargs: None, lambda *args, **kwargs: None))
    assert "no configuration" in (no_selection.reason or "")
    wrong_slot = _run_slot(
        (
            *base,
            lambda *args, **kwargs: _selection("k3", cutoff),
            lambda *args, **kwargs: _artifact(_selection("k3", cutoff), cutoff),
        )
    )
    assert "different K slot" in (wrong_slot.reason or "")
    bad_refit = _run_slot(
        (
            *base,
            lambda *args, **kwargs: _selection("k2", cutoff),
            lambda *args, **kwargs: object(),
        )
    )
    assert "K deployment refitter" in (bad_refit.reason or "")


def test_deployment_input_identity_validation_is_fail_closed() -> None:
    rows = _rows()
    cutoff = rows["timestamp_m1"].iloc[-1]
    kwargs = dict(
        validation=_validation(),
        deployment_cutoff=cutoff,
        source_build_id="build-1",
        source_catalog_hash=HASH,
        selector=lambda *args, **kwargs: None,
        refitter=lambda *args, **kwargs: None,
        max_workers=1,
    )
    with pytest.raises(TypeError, match="pandas DataFrame"):
        select_k_deployment_packages(object(), **kwargs)
    with pytest.raises(ValueError, match="source_build_id"):
        select_k_deployment_packages(rows, **{**kwargs, "source_build_id": " build-1"})
    with pytest.raises(ValueError, match="source_catalog_hash"):
        select_k_deployment_packages(rows, **{**kwargs, "source_catalog_hash": "bad"})
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        select_k_deployment_packages(
            rows, **{**kwargs, "deployment_cutoff": cutoff.replace(tzinfo=None)}
        )
