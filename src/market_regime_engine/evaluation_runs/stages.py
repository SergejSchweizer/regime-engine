"""Durable checkpoints for v4 discovery stages within an outer fold."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import TypeVar, cast

from market_regime_engine.evaluation_runs.contracts import (
    EvaluationRunIdentity,
    WorkUnitIdentity,
    canonical_json,
)
from market_regime_engine.evaluation_runs.store import (
    SQLiteEvaluationRunStore,
    WorkUnitStatus,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StageCheckpoint:
    """Execute one stage once and cache its exact Python payload in the run ledger."""

    identity: EvaluationRunIdentity
    store: SQLiteEvaluationRunStore
    scope: str

    @staticmethod
    def _domain_invalid_payload(exc: ValueError | TypeError) -> bytes:
        return canonical_json(
            {
                "status": WorkUnitStatus.DOMAIN_INVALID.value,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )

    @staticmethod
    def _raise_cached_domain_invalid(payload: bytes) -> None:
        try:
            record = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("cached domain-invalid stage payload is not valid JSON") from exc
        if (
            not isinstance(record, dict)
            or record.get("status") != WorkUnitStatus.DOMAIN_INVALID.value
            or not isinstance(record.get("reason"), str)
            or not record["reason"]
        ):
            raise ValueError("cached domain-invalid stage payload is incompatible")
        raise ValueError(record["reason"])

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
            state = self.store.work_unit_state(self.identity, unit)
            if state is not None and state.status is WorkUnitStatus.DOMAIN_INVALID:
                self._raise_cached_domain_invalid(cached)
            return cast(T, pickle.loads(cached))
        if not self.store.claim_work_unit(self.identity, unit):
            raise RuntimeError(f"stage work unit is currently claimed: {unit.key}")
        try:
            value = compute()
        except (ValueError, TypeError) as exc:
            self.store.complete_work_unit(
                self.identity,
                unit,
                self._domain_invalid_payload(exc),
                domain_invalid=True,
            )
            raise
        try:
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            self.store.complete_work_unit(self.identity, unit, payload)
            return value
        except BaseException:
            self.store.fail_work_unit(self.identity, unit, "stage computation failed")
            raise


__all__ = ["StageCheckpoint"]
