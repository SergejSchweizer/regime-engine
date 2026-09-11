"""Durable checkpoints for v4 discovery stages within an outer fold."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import TypeVar, cast

from market_regime_engine.evaluation_runs.contracts import (
    EvaluationRunIdentity,
    WorkUnitIdentity,
)
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StageCheckpoint:
    """Execute one stage once and cache its exact Python payload in the run ledger."""

    identity: EvaluationRunIdentity
    store: SQLiteEvaluationRunStore
    scope: str

    def _unit(
        self,
        stage: str,
        parameters: Iterable[tuple[str, str]],
        parent_payloads: Iterable[bytes],
    ) -> WorkUnitIdentity:
        parents = tuple(sha256(payload).hexdigest() for payload in parent_payloads)
        return WorkUnitIdentity(
            evaluation_run_key=self.identity.key,
            unit_type="v4_stage",
            coordinates=(
                ("scope", self.scope),
                ("stage", stage),
            ),
            parent_payload_hashes=parents,
            unit_parameters=tuple(sorted(parameters)),
        )

    def run(
        self,
        stage: str,
        compute: Callable[[], T],
        *,
        parameters: Iterable[tuple[str, str]] = (),
        parent_payloads: Iterable[bytes] = (),
    ) -> T:
        unit = self._unit(stage, parameters, parent_payloads)
        cached = self.store.load_completed_work_unit(self.identity, unit)
        if cached is not None:
            return cast(T, pickle.loads(cached))
        if not self.store.claim_work_unit(self.identity, unit):
            raise RuntimeError(f"stage work unit is currently claimed: {unit.key}")
        try:
            value = compute()
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            self.store.complete_work_unit(self.identity, unit, payload)
            return value
        except BaseException:
            self.store.fail_work_unit(self.identity, unit, "stage computation failed")
            raise


__all__ = ["StageCheckpoint"]
