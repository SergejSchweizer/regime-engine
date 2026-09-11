from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_regime_engine.evaluation_runs.contracts import EvaluationRunIdentity
from market_regime_engine.evaluation_runs.graph import (
    EvaluationWorkGraph,
    WorkUnitNode,
    make_work_unit,
)


def run() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        "global_regime_v4",
        "xetra",
        4,
        "a" * 64,
        1,
        "b" * 64,
        "c" * 64,
        datetime(2026, 1, 1, tzinfo=UTC),
        "d" * 40,
        "e" * 64,
        "3.14.7",
    )


def test_graph_order_is_deterministic_topological_and_coordinate_based() -> None:
    identity = run()
    root = make_work_unit(identity, unit_type="outer", coordinates=(("fold", "001"),))
    child = make_work_unit(
        identity,
        unit_type="seed",
        coordinates=(("fold", "001"), ("seed", "11")),
        parent_payload_hashes=("a" * 64,),
    )
    graph = EvaluationWorkGraph.from_nodes(
        identity,
        (
            WorkUnitNode(child, (root.key,)),
            WorkUnitNode(root),
        ),
    )
    assert tuple(node.identity.key for node in graph.nodes) == (root.key, child.key)
    assert child.key == "seed/fold=001/seed=11"


def test_graph_rejects_unknown_parent_and_cycle() -> None:
    identity = run()
    one = make_work_unit(identity, unit_type="one")
    two = make_work_unit(identity, unit_type="two")
    with pytest.raises(ValueError, match="unknown parent"):
        EvaluationWorkGraph.from_nodes(identity, (WorkUnitNode(one, ("missing",)),))
    with pytest.raises(ValueError, match="cycle"):
        EvaluationWorkGraph.from_nodes(
            identity,
            (WorkUnitNode(one, (two.key,)), WorkUnitNode(two, (one.key,))),
        )
