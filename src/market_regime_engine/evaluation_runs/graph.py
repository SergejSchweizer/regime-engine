"""Deterministic structural work graphs for v4 evaluation execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from market_regime_engine.evaluation_runs.contracts import (
    EvaluationRunIdentity,
    WorkUnitIdentity,
)


@dataclass(frozen=True, slots=True)
class WorkUnitNode:
    identity: WorkUnitIdentity
    parent_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if tuple(sorted(self.parent_keys)) != self.parent_keys:
            raise ValueError("parent_keys must be canonically sorted")
        if self.identity.evaluation_run_key == "":
            raise ValueError("work unit must belong to an evaluation run")


@dataclass(frozen=True, slots=True)
class EvaluationWorkGraph:
    """A topologically ordered graph whose ordering is independent of execution order."""

    run_key: str
    nodes: tuple[WorkUnitNode, ...]

    def __post_init__(self) -> None:
        keys = tuple(node.identity.key for node in self.nodes)
        if len(set(keys)) != len(keys):
            raise ValueError("work graph contains duplicate structural unit keys")
        known = set(keys)
        for node in self.nodes:
            if node.identity.evaluation_run_key != self.run_key:
                raise ValueError("work graph node belongs to another run")
            if any(parent not in known for parent in node.parent_keys):
                raise ValueError("work graph contains an unknown parent unit")
        if keys != self._topological_keys():
            raise ValueError("work graph nodes must be in deterministic topological order")

    def _topological_keys(self) -> tuple[str, ...]:
        parents = {node.identity.key: set(node.parent_keys) for node in self.nodes}
        resolved: set[str] = set()
        ordered: list[str] = []
        while parents:
            ready = sorted(key for key, values in parents.items() if not values - resolved)
            if not ready:
                raise ValueError("work graph contains a dependency cycle")
            ordered.extend(ready)
            resolved.update(ready)
            for key in ready:
                parents.pop(key)
        return tuple(ordered)

    @classmethod
    def from_nodes(
        cls,
        identity: EvaluationRunIdentity,
        nodes: Iterable[WorkUnitNode],
    ) -> EvaluationWorkGraph:
        materialized = tuple(nodes)
        by_key = {node.identity.key: node for node in materialized}
        if len(by_key) != len(materialized):
            raise ValueError("work graph contains duplicate structural unit keys")
        known = set(by_key)
        if any(parent not in known for node in materialized for parent in node.parent_keys):
            raise ValueError("work graph contains an unknown parent unit")
        remaining = {key: set(node.parent_keys) for key, node in by_key.items()}
        resolved: set[str] = set()
        ordered: list[WorkUnitNode] = []
        while remaining:
            ready = sorted(key for key, parents in remaining.items() if not parents - resolved)
            if not ready:
                raise ValueError("work graph contains a dependency cycle")
            ordered.extend(by_key[key] for key in ready)
            resolved.update(ready)
            for key in ready:
                remaining.pop(key)
        return cls(identity.key, tuple(ordered))


def make_work_unit(
    identity: EvaluationRunIdentity,
    *,
    unit_type: str,
    coordinates: tuple[tuple[str, str], ...] = (),
    parent_payload_hashes: tuple[str, ...] = (),
    unit_parameters: tuple[tuple[str, str], ...] = (),
) -> WorkUnitIdentity:
    """Construct one canonical unit with sorted structural coordinates."""

    return WorkUnitIdentity(
        evaluation_run_key=identity.key,
        unit_type=unit_type,
        coordinates=tuple(sorted(coordinates)),
        parent_payload_hashes=parent_payload_hashes,
        unit_parameters=tuple(sorted(unit_parameters)),
    )


__all__ = ["EvaluationWorkGraph", "WorkUnitNode", "make_work_unit"]
