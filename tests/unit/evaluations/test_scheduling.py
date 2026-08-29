from __future__ import annotations

from market_regime_engine.evaluations.scheduling import randomized_order


def test_randomized_order_is_stable_and_contains_every_cell(monkeypatch) -> None:
    monkeypatch.setenv("REGIME_EVALUATION_SCHEDULING_SEED", "same-code-and-data")
    values = ("k2", "k3", "k4", "k5")
    assert randomized_order(values, scope="matrix") == randomized_order(values, scope="matrix")
    assert set(randomized_order(values, scope="matrix")) == set(values)


def test_missing_seed_keeps_canonical_order(monkeypatch) -> None:
    monkeypatch.delenv("REGIME_EVALUATION_SCHEDULING_SEED", raising=False)
    assert randomized_order(("k2", "k3"), scope="matrix") == ("k2", "k3")
