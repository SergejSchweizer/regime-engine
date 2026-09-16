"""MLflow-facing contract for independent K-champion alias moves.

The module deliberately describes an alias operation instead of performing one.
Registration and CAS execution remain operator-controlled; the legacy
``champion`` alias is not a legal target of this policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from market_regime_engine.contracts.core import K_CHAMPION_ALIASES as CORE_K_CHAMPION_ALIASES
from market_regime_engine.contracts.core import KChampionSlot
from market_regime_engine.evaluations.k_champion_contract import (
    KChampionPromotionDecision,
    KChampionSelection,
)

REGISTERED_MODEL_NAME = "regime-xetra"
K_CHAMPION_ALIASES = frozenset(CORE_K_CHAMPION_ALIASES)
LEGACY_ALIAS = "champion"


def alias_for_slot(slot_id: str) -> str:
    try:
        return KChampionSlot(slot_id).alias
    except ValueError as exc:
        raise ValueError("K-champion alias requires slot k2, k3, k4 or k5") from exc


@dataclass(frozen=True, slots=True)
class KChampionPromotionInstruction:
    """An auditable, idempotent alias move for one already-registered version."""

    model_name: str
    slot_id: str
    alias: str
    exact_model_version: str
    expected_current_version: str | None
    selection_hash: str
    idempotency_key: str
    reason: str

    def __post_init__(self) -> None:
        if self.model_name != REGISTERED_MODEL_NAME:
            raise ValueError("K-champion promotion model must be regime-xetra")
        expected_alias = alias_for_slot(self.slot_id)
        if self.alias != expected_alias or self.alias not in K_CHAMPION_ALIASES:
            raise ValueError("promotion alias does not match the K slot")
        if self.alias == LEGACY_ALIAS:
            raise ValueError("K-champion policy cannot mutate the legacy champion alias")
        for value, field in (
            (self.exact_model_version, "exact_model_version"),
            (self.reason, "reason"),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{field} must be a non-empty trimmed string")
        for value, field in (
            (self.selection_hash, "selection_hash"),
            (self.idempotency_key, "idempotency_key"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if self.expected_current_version is not None and (
            not self.expected_current_version
            or self.expected_current_version.strip() != self.expected_current_version
        ):
            raise ValueError("expected_current_version must be trimmed when supplied")


def build_promotion_instruction(
    decision: KChampionPromotionDecision,
    *,
    exact_model_version: str,
    expected_current_version: str | None = None,
    reason: str = "K-slot champion replacement",
) -> KChampionPromotionInstruction:
    """Build a slot alias move only for an eligible winner.

    No model version is deleted and no alias is changed here.  A caller must
    execute the returned instruction through an audited compare-and-swap.
    """

    winner = decision.winner
    if winner is None:
        raise ValueError("ineligible K slot cannot produce a promotion instruction")
    selection: KChampionSelection = winner.selection
    return KChampionPromotionInstruction(
        model_name=REGISTERED_MODEL_NAME,
        slot_id=selection.slot_id,
        alias=selection.alias,
        exact_model_version=exact_model_version,
        expected_current_version=expected_current_version,
        selection_hash=selection.selection_hash,
        idempotency_key=selection.idempotency_key,
        reason=reason,
    )


__all__ = [
    "K_CHAMPION_ALIASES",
    "LEGACY_ALIAS",
    "REGISTERED_MODEL_NAME",
    "KChampionPromotionInstruction",
    "alias_for_slot",
    "build_promotion_instruction",
]
