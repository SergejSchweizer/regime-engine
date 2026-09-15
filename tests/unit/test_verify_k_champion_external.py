from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts.verify_k_champion_external import inspect_external_registry


@dataclass(frozen=True)
class _Version:
    version: str
    tags: dict[str, str]


class _Client:
    def __init__(self, versions: dict[str, _Version], aliases: dict[str, str]) -> None:
        self._versions = versions
        self._aliases = aliases

    def get_registered_model(self, name: str) -> object:
        assert name == "regime-xetra"
        return SimpleNamespace(aliases=self._aliases)

    def get_model_version_by_alias(self, name: str, alias: str) -> _Version:
        assert name == "regime-xetra"
        return self._versions[self._aliases[alias]]


def _tags(alias: str, *, snapshot: str = "snapshot-1") -> dict[str, str]:
    state_count = alias.rsplit("-k", 1)[1]
    return {
        "regime_engine.slot_id": f"k{state_count}",
        "regime_engine.alias": alias,
        "regime_engine.state_count": state_count,
        "regime_engine.model_family": "gaussian_hmm",
        "regime_engine.policy_id": "k_champion_portfolio",
        "regime_engine.policy_version": "k_champion_portfolio.v1",
        "regime_engine.source_snapshot_id": snapshot,
        "regime_engine.feature_order_sha256": "a" * 64,
        "regime_engine.comparison_domain_id": "k_slot_promotion.v1",
        "regime_engine.promotion_score_version": "k_slot_promotion.v1",
        "regime_engine.reference_teacher_id": f"teacher-{state_count}",
        "regime_engine.validation_cutoff": "2026-08-20T00:00:00Z",
        "regime_engine.deployment_cutoff": "2026-08-21T00:00:00Z",
        "regime_engine.selection_sha256": "b" * 64,
        "regime_engine.artifact_sha256": "c" * 64,
        "regime_engine.idempotency_key": "d" * 64,
    }


def _client(*, missing: str | None = None, snapshot: str = "snapshot-1") -> _Client:
    aliases = {f"champion-k{k}": str(k) for k in (2, 3, 4, 5)}
    versions = {
        version: _Version(version, _tags(alias, snapshot=snapshot))
        for alias, version in aliases.items()
    }
    if missing is not None:
        del versions[aliases[missing]].tags["regime_engine.artifact_sha256"]
    return _Client(versions, aliases)


def test_external_readback_requires_complete_matching_slot_lineage() -> None:
    report = inspect_external_registry(_client(), expected_default_champion=None)

    assert report["present_slot_count"] == 4
    aliases = report["aliases"]
    assert isinstance(aliases, dict)
    assert aliases["champion-k2"]["version"] == "2"
    assert aliases["champion-k5"]["state_count"] == "5"


def test_external_readback_rejects_missing_provenance_tag() -> None:
    with pytest.raises(RuntimeError, match="missing required tags"):
        inspect_external_registry(_client(missing="champion-k3"))


def test_external_readback_rejects_incompatible_snapshot_and_default_alias_change() -> None:
    client = _client()
    client._versions["3"].tags["regime_engine.source_snapshot_id"] = "other-snapshot"
    with pytest.raises(RuntimeError, match="incompatible source_snapshot_id"):
        inspect_external_registry(client)

    with pytest.raises(RuntimeError, match="changed legacy champion alias"):
        inspect_external_registry(_client(), expected_default_champion="99")
