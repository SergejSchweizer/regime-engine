from __future__ import annotations

import ast
import copy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from market_regime_engine.evaluation.calendar_clock import plan_calendar_month
from market_regime_engine.profiles.loader import load_profile_mapping

ROOT = Path(__file__).parents[2]
SOURCE_ROOT = ROOT / "src" / "market_regime_engine"


def _production_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in SOURCE_ROOT.rglob("*.py"))


def test_production_has_no_retired_selector_or_source_fallback_surface() -> None:
    production = _production_text()
    retired_symbols = (
        "pca_only_prefix_selector",
        "l_star_selector",
        "teacher_hmm_selector",
        "clustering_selector",
        "raw_plus_pca_selector",
        "macro_features_daily",
        "macro_raw",
        "schema_wide_feature_source",
        "caller_selected_feature_relation",
    )
    assert all(symbol not in production for symbol in retired_symbols)


def test_canonical_import_graph_has_no_deleted_optional_or_dynamic_edge() -> None:
    deleted_modules = {
        "feature_selection",
        "medoid" + "_multivariate",
        "medoid" + "_univariate",
        "delta1" + "_univariate",
        "univariate_grid",
        "registry_compat",
        "legacy",
    }
    canonical_prefixes = {
        "commands", "evaluation", "feature_discovery", "features",
        "preprocessing", "profiles", "runtime", "training",
    }
    for path in SOURCE_ROOT.rglob("*.py"):
        if path.relative_to(SOURCE_ROOT).parts[0] not in canonical_prefixes:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[-1] not in deleted_modules
                    for alias in node.names
                ), path
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[-1] not in deleted_modules, path
            elif isinstance(node, ast.Call):
                function = node.func
                dynamic = (
                    isinstance(function, ast.Name)
                    and function.id in {"__import__", "import_module"}
                ) or (
                    isinstance(function, ast.Attribute)
                    and function.attr == "import_module"
                )
                if dynamic:
                    # The CLI composition is intentionally lazy so static
                    # verifier runs never import external services. It may
                    # load only the current lifecycle module.
                    assert path.name == "runtime.py"
                    assert any(
                        isinstance(argument, ast.Constant)
                        and argument.value == "market_regime_engine.commands.lifecycle"
                        for argument in node.args
                    )


@pytest.mark.parametrize(
    ("location", "key"),
    (
        ("top", "selector"),
        ("feature_selection", "source_relation"),
        ("feature_discovery", "feature_source"),
        ("feature_discovery", "pca_only_prefix"),
        ("feature_selection", "raw_plus_pca_selector"),
        ("walk_forward", "l_star"),
    ),
)
def test_historical_profile_switches_fail_closed_as_unknown(
    location: str, key: str
) -> None:
    raw = yaml.safe_load((ROOT / "configs/profiles/xetra_v4.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    mutated = copy.deepcopy(raw)
    target = mutated if location == "top" else mutated[location]
    target[key] = "historical-switch"

    with pytest.raises(ValueError, match="unknown keys"):
        load_profile_mapping(mutated)


def test_canonical_calendar_fixture_keeps_pre_deletion_statistical_identity() -> None:
    timestamps = tuple(
        datetime.combine(
            date(2023, 1, 1) + timedelta(days=index), datetime.min.time(), tzinfo=UTC
        )
        for index in range(120)
    )
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=31)

    # Pinned local PR-500 canonical fixture identities. No database or MLflow
    # client is opened by this deterministic regression fixture.
    assert plan.plan_hash == "356fbf8c5fb1fc9154a2c1667e2d077ffe6a00cab77186b530bd958a9ccac7ac"
    assert tuple(fold.month_clock_hash for fold in plan.folds) == (
        "7b1315aa1753570ff462d11f899bf0636083be1f58564fdddc2fce68aa77366c",
        "b33d5bd101ab8d33b170a0b2c3de85ead97d041bf0976ea6bd258d96201a199f",
        "a5ff7fd6b167575a03555deb8f5c63b257da709b24aa72893b6d94b00915c31d",
    )


def test_production_month_clock_has_no_fixed_block_constructor() -> None:
    production = _production_text()
    forbidden_constructors = (
        "walk_forward_splits", "fixed_block", "trading_day_clock", "fixed_row_clock"
    )
    assert all(name not in production for name in forbidden_constructors)
    clock = (SOURCE_ROOT / "evaluation" / "calendar_clock.py").read_text(encoding="utf-8")
    assert "plan_calendar_month" in clock
    assert "_next_month" in clock
