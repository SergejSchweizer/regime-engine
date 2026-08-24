from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_regime_engine.contracts import LatestInvocation, ReplayInvocation
from market_regime_engine.mlflow_app.dispatch import (
    ApiInputError,
    map_exception,
    parse_invocation_payload,
    parse_json_body,
    parse_oos_query,
    parse_utc_timestamp,
    to_jsonable,
)
from market_regime_engine.serving.latest_handler import StaleDefaultChampionError
from market_regime_engine.serving.replay_limits import ReplayGuardrailError


def test_parse_latest_and_replay_are_strict_and_normalize_utc() -> None:
    latest = parse_invocation_payload(
        {
            "operation": "latest",
            "as_of": "2026-08-24T12:00:00+00:00",
            "model_version": "7",
        }
    )
    assert isinstance(latest, LatestInvocation)
    assert latest.as_of == datetime(2026, 8, 24, 12, tzinfo=UTC)
    assert latest.model_version == "7"

    replay = parse_invocation_payload(
        {
            "operation": "replay",
            "start": "2026-08-20T00:00:00Z",
            "end": "2026-08-24T00:00:00Z",
        }
    )
    assert isinstance(replay, ReplayInvocation)
    assert replay.start.tzinfo is UTC
    assert replay.end.tzinfo is UTC


def test_parse_invocation_rejects_profile_unknown_fields_and_bad_operations() -> None:
    cases = (
        ({"operation": "latest", "profile_id": "xetra"}, "profile_in_body_forbidden"),
        ({"operation": "latest", "start": "2026-01-01T00:00:00Z"}, "unknown_request_field"),
        ({"operation": "replay", "as_of": "2026-01-01T00:00:00Z"}, "unknown_request_field"),
        ({"operation": "replay", "start": "2026-01-01T00:00:00Z"}, "missing_replay_bound"),
        ({}, "missing_operation"),
        ({"operation": "forecast"}, "unsupported_operation"),
        ({"operation": "latest", "model_version": " 7"}, "invalid_model_version"),
    )
    for payload, expected in cases:
        with pytest.raises(ApiInputError) as exc:
            parse_invocation_payload(payload)
        assert exc.value.error_code == expected

    with pytest.raises(ApiInputError, match="JSON object"):
        parse_invocation_payload([])


def test_parse_timestamps_and_replay_interval_fail_closed() -> None:
    assert parse_utc_timestamp("2026-01-01T00:00:00Z", "at").tzinfo is UTC
    invalid_values = (
        None,
        "",
        "not-a-time",
        "2026-01-01T00:00:00",
        "2026-01-01T01:00:00+01:00",
    )
    for value in invalid_values:
        with pytest.raises(ApiInputError, match=r"UTC|RFC3339"):
            parse_utc_timestamp(value, "at")

    with pytest.raises(ApiInputError) as exc:
        parse_invocation_payload(
            {
                "operation": "replay",
                "start": "2026-01-02T00:00:00Z",
                "end": "2026-01-01T00:00:00Z",
            }
        )
    assert exc.value.error_code == "invalid_replay_interval"


def test_parse_json_body_requires_json_and_well_formed_payload() -> None:
    parsed = parse_json_body(b'{"operation":"latest"}', is_json=True)
    assert isinstance(parsed, LatestInvocation)
    with pytest.raises(ApiInputError) as exc:
        parse_json_body(b"{}", is_json=False)
    assert exc.value.error_code == "invalid_content_type"
    with pytest.raises(ApiInputError) as exc:
        parse_json_body(b"{", is_json=True)
    assert exc.value.error_code == "malformed_json"
    with pytest.raises(ApiInputError) as exc:
        parse_json_body(b"\xff", is_json=True)
    assert exc.value.error_code == "malformed_json"


def test_oos_query_is_bounded_utc_and_rejects_unknown_fields() -> None:
    start, end = parse_oos_query(
        {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00+00:00"}
    )
    assert start == datetime(2026, 1, 1, tzinfo=UTC)
    assert end == datetime(2026, 1, 2, tzinfo=UTC)
    assert parse_oos_query({}) == (None, None)
    with pytest.raises(ApiInputError) as exc:
        parse_oos_query({"latest": "true"})
    assert exc.value.error_code == "unknown_query_field"
    with pytest.raises(ApiInputError) as exc:
        parse_oos_query(
            {"start": "2026-01-03T00:00:00Z", "end": "2026-01-02T00:00:00Z"}
        )
    assert exc.value.error_code == "invalid_oos_interval"


def test_to_jsonable_serializes_dataclasses_enums_times_and_collections() -> None:
    invocation = LatestInvocation(as_of=datetime(2026, 1, 1, tzinfo=UTC), model_version="8")
    payload = to_jsonable({"invocation": invocation, "values": (1, True, None)})
    assert payload["invocation"]["operation"] == "latest"
    assert payload["invocation"]["as_of"] == "2026-01-01T00:00:00Z"
    assert payload["values"] == [1, True, None]
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        to_jsonable(datetime(2026, 1, 1))
    with pytest.raises(TypeError, match="unsupported API response"):
        to_jsonable(object())


def test_safe_http_error_mapping_covers_input_guardrail_stale_and_domain_failures() -> None:
    payload, status = map_exception(ApiInputError("bad", "safe"), "r1")
    assert (status, payload["error_code"], payload["schema_version"]) == (
        400,
        "bad",
        "RegimeError.v1",
    )

    guardrail = ReplayGuardrailError(413, "replay_row_limit", "too many", False)
    payload, status = map_exception(guardrail, "r2")
    assert (status, payload["error_code"], payload["retryable"]) == (
        413,
        "replay_row_limit",
        False,
    )

    stale = StaleDefaultChampionError(7.0, 35.0)
    payload, status = map_exception(stale, "r3")
    assert status == 503
    assert payload["error_code"] == "stale_default_champion"
    assert "7.0" not in str(payload)

    payload, status = map_exception(KeyError("secret-ish-name"), "r4")
    assert status == 404
    assert payload["error_code"] == "resource_not_found"
    assert "secret-ish-name" not in str(payload)

    payload, status = map_exception(FileNotFoundError("/private/path"), "r5")
    assert status == 404
    assert payload["error_code"] == "oos_build_not_found"
    assert "/private/path" not in str(payload)

    payload, status = map_exception(ValueError("raw feature vector"), "r6")
    assert status == 422
    assert payload["error_code"] == "semantic_validation_failed"
    assert "raw feature vector" not in str(payload)

    payload, status = map_exception(RuntimeError("password=hidden"), "r7")
    assert status == 503
    assert payload["error_code"] == "dependency_unavailable"
    assert "password=hidden" not in str(payload)
