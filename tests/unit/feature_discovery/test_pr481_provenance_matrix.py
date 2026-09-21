from __future__ import annotations

import pytest

from market_regime_engine.feature_discovery.feature_roles import (
    FeatureRole,
    FeatureRoleAssignment,
    FeatureRoleContract,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.provenance import (
    FeatureProvenance,
    build_feature_provenance,
)


def test_matrix_covers_core_and_multiple_families_and_windows() -> None:
    contract = build_feature_role_contract(
        (
            "vix_log_level",
            "vix_log_return_5obs",
            "vix_log_return_20obs",
            "usd_broad_log_level",
            "usd_broad_log_return_5obs",
        )
    )
    rows = build_feature_provenance(contract)
    assert {row.family for row in rows} == {None, "vix", "usd_broad"}
    assert {row.transformation_parameters for row in rows if row.feature_name.endswith("obs")} == {
        (("window_observations", 5),),
        (("window_observations", 20),),
    }


def test_missing_role_family_and_transformation_provenance_fail_closed() -> None:
    contract = build_feature_role_contract(("vix_log_level",))
    with pytest.raises(KeyError):
        build_feature_provenance(contract, ("vix_log_return_5obs",))

    with pytest.raises(ValueError, match="incomplete"):
        FeatureProvenance("vix_log_return_5obs", FeatureRole.TRANSFORMATION, "vix", None, ())

    conflicting = FeatureRoleContract(
        (FeatureRoleAssignment("vix_log_return_5obs", FeatureRole.TRANSFORMATION, "usd_broad"),)
    )
    with pytest.raises(ValueError, match="family conflict"):
        build_feature_provenance(conflicting)


def test_discovery_order_normalization_and_window_separation_are_stable() -> None:
    names = ("vix_log_return_20obs", "vix_log_return_5obs", "vix_log_level")
    first = build_feature_provenance(build_feature_role_contract(names))
    second = build_feature_provenance(build_feature_role_contract(tuple(reversed(names))))
    assert first == second
    assert tuple(row.feature_identity for row in first) == tuple(
        row.feature_identity for row in sorted(first, key=lambda row: row.feature_identity)
    )

    equivalent = FeatureProvenance(
        "vix_log_return_20obs",
        FeatureRole.TRANSFORMATION,
        "vix",
        "log_return",
        (("window_observations", 20),),
    )
    generated = next(row for row in first if row.feature_name == "vix_log_return_20obs")
    assert equivalent == generated
    assert len({row.feature_identity for row in first}) == len(first)


def test_unknown_pattern_is_not_accepted_as_a_name_only_fallback() -> None:
    with pytest.raises(ValueError, match="unclassifiable"):
        build_feature_role_contract(("vix_not_a_supported_transform_5obs",))
