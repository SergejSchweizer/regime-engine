"""Seed-level checkpoint units for deterministic HMM multistart fitting."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from time import monotonic, sleep

from market_regime_engine.evaluation_runs.contracts import EvaluationRunIdentity, WorkUnitIdentity
from market_regime_engine.evaluation_runs.store import (
    SQLiteEvaluationRunStore,
    WorkUnitStatus,
)
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.training.multistart import StartDiagnostic


@dataclass(frozen=True, slots=True)
class SeedFitOutcome:
    diagnostic: StartDiagnostic
    result: FitResult | None


@dataclass(frozen=True, slots=True)
class HMMSeedCheckpoint:
    """Resolve one candidate/fold/seed to one durable ledger unit."""

    run_identity: EvaluationRunIdentity
    store: SQLiteEvaluationRunStore
    candidate_id: str
    fold_id: str
    state_count: int
    scope: str = "default"
    parent_payload_hashes: tuple[str, ...] = ()

    def unit(self, seed: int) -> WorkUnitIdentity:
        return WorkUnitIdentity(
            evaluation_run_key=self.run_identity.key,
            unit_type="hmm_seed",
            coordinates=(
                ("candidate_id", self.candidate_id),
                ("fold_id", self.fold_id),
                ("scope", self.scope),
                ("seed", str(seed)),
            ),
            parent_payload_hashes=self.parent_payload_hashes,
            unit_parameters=(("state_count", str(self.state_count)),),
        )

    def load(self, seed: int) -> SeedFitOutcome | None:
        payload = self.store.load_completed_work_unit(self.run_identity, self.unit(seed))
        if payload is None:
            return None
        outcome = pickle.loads(payload)
        if not isinstance(outcome, SeedFitOutcome) or outcome.diagnostic.seed != seed:
            raise ValueError("cached HMM seed payload is incompatible")
        return outcome

    def save(self, seed: int, outcome: SeedFitOutcome) -> None:
        unit = self.unit(seed)
        deadline = monotonic() + 30.0
        while True:
            if self.store.claim_work_unit(self.run_identity, unit):
                payload = pickle.dumps(outcome, protocol=pickle.HIGHEST_PROTOCOL)
                self.store.complete_work_unit(self.run_identity, unit, payload)
                return
            cached = self.load(seed)
            if cached is not None:
                if cached != outcome:
                    raise ValueError(
                        "HMM seed was completed by another worker with different bytes"
                    )
                return
            state = self.store.work_unit_state(self.run_identity, unit)
            if state is None:
                continue
            if monotonic() >= deadline:
                raise RuntimeError(f"HMM seed remains claimed: {unit.key}")
            if state.status in {WorkUnitStatus.COMPLETE, WorkUnitStatus.DOMAIN_INVALID}:
                continue
            sleep(0.01)


__all__ = ["HMMSeedCheckpoint", "SeedFitOutcome"]
