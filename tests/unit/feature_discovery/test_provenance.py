from __future__ import annotations

import pytest

from market_regime_engine.feature_discovery.feature_roles import build_feature_role_contract
from market_regime_engine.feature_discovery.provenance import build_feature_provenance


def test_core_and_transformations_have_structured_canonical_provenance() -> None:
    contract = build_feature_role_contract(
        (
            "vix_log_return_20obs",
            "vix_log_level",
            "vix_delta_5obs",
        )
    )

    rows = build_feature_provenance(contract)
    by_name = {row.feature_name: row for row in rows}
    assert by_name["vix_log_level"].family is None
    assert by_name["vix_log_level"].transformation_name is None
    assert by_name["vix_log_return_20obs"].family == "vix"
    assert by_name["vix_log_return_20obs"].transformation_name == "log_return"
    assert by_name["vix_log_return_20obs"].canonical_dict["transformation_parameters"] == {
        "window_observations": 20
    }
    assert by_name["vix_delta_5obs"].transformation_name == "delta"
    assert by_name["vix_delta_5obs"].canonical_dict["transformation_parameters"] == {
        "variant": "5obs"
    }


def test_provenance_identity_and_order_are_canonical() -> None:
    left = build_feature_provenance(
        build_feature_role_contract(("vix_log_level", "vix_log_return_20obs"))
    )
    right = build_feature_provenance(
        build_feature_role_contract(("vix_log_return_20obs", "vix_log_level"))
    )
    assert left == right
    assert len({row.feature_identity for row in left}) == len(left)

    different_window = build_feature_provenance(
        build_feature_role_contract(("vix_log_return_21obs",))
    )[0]
    assert different_window.feature_identity != next(
        row.feature_identity for row in left if row.feature_name == "vix_log_return_20obs"
    )


def test_missing_or_conflicting_transformation_provenance_fails_closed() -> None:
    with pytest.raises(ValueError, match="unclassifiable"):
        build_feature_role_contract(("vix_unknown_5obs",))

    contract = build_feature_role_contract(("vix_log_level",))
    with pytest.raises(KeyError):
        build_feature_provenance(contract, ("missing_feature",))

    with pytest.raises(ValueError, match="temporal key"):
        build_feature_provenance(
            build_feature_role_contract(("timestamp_m1",)),
            ("timestamp_m1",),
        )
