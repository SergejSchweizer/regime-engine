from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from market_regime_engine.contracts.core import K_CHAMPION_ALIASES
from market_regime_engine.evaluations.k_champion_outer import KChampionSlotValidation
from market_regime_engine.mlflow_support.k_slot_plots import (
    build_cross_k_plot_payload,
    build_k_plot_payload,
    prepare_k_plot_payloads,
    render_k_plot_payload,
)
from market_regime_engine.mlflow_support.registry import MlflowModelRegistry
from tests.fixtures.k_champion.portfolio import (
    FAMILIES,
    HermeticPortfolio,
    build_portfolio,
    canonical_hash_payload,
    independent_canonical_hash,
    independent_portfolio_outer_hash,
    metric_slots,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def portfolio() -> HermeticPortfolio:
    return build_portfolio(max_workers=None)


@dataclass(frozen=True)
class _Version:
    version: str
    source: str
    tags: dict[str, str]


class _LocalRegistry:
    """No-network registry boundary for the four candidate alias assertions."""

    def __init__(self) -> None:
        self.models: set[str] = set()
        self.versions: dict[tuple[str, str], _Version] = {}
        self.aliases: dict[tuple[str, str], str] = {}
        self.counter = 0

    @staticmethod
    def _missing() -> MlflowException:
        return MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)

    def get_registered_model(self, name: str) -> object:
        if name not in self.models:
            raise self._missing()
        return object()

    def create_registered_model(self, name: str) -> object:
        self.models.add(name)
        return object()

    def create_model_version(
        self,
        *,
        name: str,
        source: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> _Version:
        del description
        self.counter += 1
        version = _Version(str(self.counter), source, dict(tags or {}))
        self.versions[(name, version.version)] = version
        return version

    def search_model_versions(self, filter_string: str) -> tuple[_Version, ...]:
        assert filter_string == "name='regime-xetra'"
        return tuple(self.versions.values())

    def get_model_version(self, name: str, version: str) -> _Version:
        try:
            return self.versions[(name, version)]
        except KeyError as exc:
            raise self._missing() from exc

    def get_model_version_by_alias(self, name: str, alias: str) -> _Version:
        try:
            return self.get_model_version(name, self.aliases[(name, alias)])
        except KeyError as exc:
            raise self._missing() from exc

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[(name, alias)] = version

    def set_registered_model_tag(self, name: str, key: str, value: str) -> None:
        del name, key, value


def _portfolio_payload(
    portfolio: HermeticPortfolio, outer: object, slots: object
) -> dict[str, Any]:
    return {
        "outer_result_hash": outer.result_hash,
        "source_data_sha256": portfolio.source_data_sha256,
        "slot_payload_hashes": [slot.canonical_payload_hash for slot in slots],
    }


def test_real_four_slot_portfolio_is_canonical_and_parallel(
    portfolio: HermeticPortfolio, tmp_path: Path
) -> None:
    manifest = json.loads(
        (Path(__file__).parents[1] / "fixtures/k_champion/expectation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(portfolio.plan.folds) == 3
    assert len(portfolio.plan.folds) == manifest["outer_folds"]
    assert len(portfolio.models.candidates) == manifest["candidate_fits"]
    assert len(portfolio.models.refits) == manifest["selected_refits"]
    assert len(portfolio.models.teachers) == manifest["teacher_refits"]
    for fold in portfolio.plan.folds:
        for state_count in (2, 3, 4, 5):
            records = portfolio.models.candidates_for(fold.fold_id, f"k{state_count}")
            assert tuple(item.model_family for item in records) == FAMILIES
            assert len({item.feature_order for item in records}) == 1

    parallel = portfolio.outer(max_workers=None)
    serial = portfolio.outer(max_workers=1)
    assert len(parallel.outer_folds) == manifest["outer_folds"] * manifest["slots"]
    assert tuple(item.slot_id for item in parallel.outer_folds[:4]) == ("k2", "k3", "k4", "k5")
    assert all(item.valid for item in parallel.outer_folds)
    assert all(item.eligible for item in parallel.slots)
    assert parallel.result_hash == serial.result_hash

    deployment = portfolio.deployment(parallel, tmp_path / "packages", max_workers=None)
    assert len(deployment.eligible_artifacts) == manifest["deployment_packages"]
    assert all(
        (Path(item.package_directory) / "model.json").is_file()
        for item in deployment.eligible_artifacts
    )
    assert tuple(item.slot_id for item in deployment.slots) == ("k2", "k3", "k4", "k5")

    slots = metric_slots(portfolio, parallel, deployment)
    assert tuple(slot.metadata.state_count for slot in slots) == (2, 3, 4, 5)
    assert all(len(slot.candidates) == 3 for slot in slots)
    assert all(slot.selected_logged_model_id is not None for slot in slots)

    plot_payloads = prepare_k_plot_payloads(
        slots,
        ("train_loglik_per_obs", "valid_fold_rate"),
        max_workers=None,
    )
    assert len(plot_payloads) == manifest["plot_payloads"]
    assert all(payload.status == "available" for payload in plot_payloads)
    assert sum(len(payload.series) for payload in plot_payloads) == manifest["plot_series"]
    assert all(len(payload.series) == manifest["families_per_slot"] for payload in plot_payloads)
    assert all(payload.no_evaluation_recomputation for payload in plot_payloads)
    manifests = tuple(
        render_k_plot_payload(payload, tmp_path / "plots") for payload in plot_payloads
    )
    assert len(tuple((tmp_path / "plots").glob("*.png"))) == 8
    assert all(manifest["canonical_payload_hash"] for manifest in manifests)

    for forbidden_metric in ("train_loglik_per_obs", "aic", "bic"):
        cross_metric = build_cross_k_plot_payload(slots, forbidden_metric)
        assert cross_metric.status == "not_available"
        assert "cross-K" in (cross_metric.unavailable_reason or "")

    registry_client = _LocalRegistry()
    registry = MlflowModelRegistry(registry_client)
    registered = tuple(
        registry.register_k_slot_package(
            item.selection,
            package_source_uri=f"runs:/hermetic/{item.slot_id}",
            artifact_hash=item.artifact.artifact_hash,
        )
        for item in deployment.slots
        if item.eligible and item.selection is not None and item.artifact is not None
    )
    legacy_champion_version = registered[0].exact_version
    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="champion",
        expected_current_version=None,
        new_version=legacy_champion_version,
        reason="seed unchanged single-champion route",
    )
    aliases = tuple(
        registry.compare_and_swap_alias(
            model_name="regime-xetra",
            alias=K_CHAMPION_ALIASES[index],
            expected_current_version=None,
            new_version=item.exact_version,
            reason="hermetic four-slot acceptance",
        )
        for index, item in enumerate(registered)
    )
    assert aliases == (True,) * manifest["registry_aliases"]
    assert set(registry_client.aliases) == {
        ("regime-xetra", alias) for alias in (*K_CHAMPION_ALIASES, "champion")
    }
    assert registry_client.aliases[("regime-xetra", "champion")] == legacy_champion_version

    payload = _portfolio_payload(portfolio, parallel, slots)
    expected_hash = canonical_hash_payload(payload)
    with __import__("concurrent.futures").futures.ProcessPoolExecutor(max_workers=1) as executor:
        assert executor.submit(independent_canonical_hash, payload).result() == expected_hash


def test_ineligible_slot_has_no_selected_model_package_or_alias(
    portfolio: HermeticPortfolio, tmp_path: Path
) -> None:
    validation = portfolio.outer(max_workers=None)
    k5 = next(slot for slot in validation.slots if slot.slot_id == "k5")
    ineligible_k5 = KChampionSlotValidation(
        slot_id=k5.slot_id,
        source_snapshot_id=k5.source_snapshot_id,
        outer_plan_hash=k5.outer_plan_hash,
        planned_fold_count=k5.planned_fold_count,
        valid_fold_count=k5.valid_fold_count - 1,
        valid_fold_rate=(k5.valid_fold_count - 1) / k5.planned_fold_count,
        latest_complete_fold_valid=False,
        mean_soft_regime_nmi=k5.mean_soft_regime_nmi,
        worst_fold_soft_regime_nmi=k5.worst_fold_soft_regime_nmi,
        mean_common_support=k5.mean_common_support,
        mean_stability=k5.mean_stability,
        fold_result_hashes=k5.fold_result_hashes,
        eligible=False,
        rejection_reasons=("fixture forced ineligible",),
    )
    ineligible_validation = validation.__class__(
        source_snapshot_id=validation.source_snapshot_id,
        profile_id=validation.profile_id,
        profile_config_version=validation.profile_config_version,
        policy_version=validation.policy_version,
        validation_cutoff=validation.validation_cutoff,
        outer_plan_hash=validation.outer_plan_hash,
        outer_folds=validation.outer_folds,
        slots=tuple(ineligible_k5 if slot.slot_id == "k5" else slot for slot in validation.slots),
    )
    deployment = portfolio.deployment(
        ineligible_validation,
        tmp_path / "packages",
        ineligible_slots=("k5",),
        max_workers=None,
    )
    k5_result = next(item for item in deployment.slots if item.slot_id == "k5")
    assert not k5_result.eligible
    assert k5_result.selection is None
    assert k5_result.artifact is None
    assert len(deployment.eligible_artifacts) == 3

    eligible = metric_slots(portfolio, validation, deployment)
    k5_projection = next(slot for slot in eligible if slot.slot_id == "k5")
    assert not k5_projection.eligible
    assert k5_projection.selected_logged_model_id is None
    assert build_k_plot_payload(k5_projection, "train_loglik_per_obs").status == "not_available"
    assert build_cross_k_plot_payload(eligible, "valid_fold_rate").status == "not_available"

    registry_client = _LocalRegistry()
    registry = MlflowModelRegistry(registry_client)
    for item in deployment.slots:
        if item.eligible and item.selection is not None and item.artifact is not None:
            registered = registry.register_k_slot_package(
                item.selection,
                package_source_uri=f"runs:/hermetic/{item.slot_id}",
                artifact_hash=item.artifact.artifact_hash,
            )
            assert registry.compare_and_swap_alias(
                model_name="regime-xetra",
                alias=f"champion-{item.slot_id}",
                expected_current_version=None,
                new_version=registered.exact_version,
                reason="eligible-only acceptance",
            )
    assert set(registry_client.aliases) == {
        ("regime-xetra", "champion-k2"),
        ("regime-xetra", "champion-k3"),
        ("regime-xetra", "champion-k4"),
    }
    assert ("regime-xetra", "champion-k5") not in registry_client.aliases


def test_future_rows_do_not_change_completed_outer_evidence(portfolio: HermeticPortfolio) -> None:
    baseline = portfolio.outer(max_workers=None)
    mutated = portfolio.rows.copy(deep=True)
    future_mask = mutated["timestamp_m1"] > portfolio.validation_cutoff
    for column in ("f0", "f1", "f2", "f3"):
        mutated.loc[future_mask, column] += 1000.0
    future_portfolio = portfolio.__class__(
        rows=mutated,
        profile=portfolio.profile,
        plan=portfolio.plan,
        models=portfolio.models,
        source_data_sha256=portfolio.source_data_sha256,
        validation_cutoff=portfolio.validation_cutoff,
        deployment_cutoff=portfolio.deployment_cutoff,
    )
    assert future_portfolio.outer(max_workers=None).result_hash == baseline.result_hash


def test_full_portfolio_repeats_in_an_independent_process(
    portfolio: HermeticPortfolio,
) -> None:
    baseline = portfolio.outer(max_workers=None).result_hash
    with __import__("concurrent.futures").futures.ProcessPoolExecutor(max_workers=1) as executor:
        repeated = executor.submit(independent_portfolio_outer_hash).result()
    assert repeated == baseline
