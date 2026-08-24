"""MLflow Flask application factory with regime-engine route placeholders."""

from __future__ import annotations

from flask import Flask, Response, jsonify
from mlflow.server import app as mlflow_app

_INVOCATION_PATH = "/regime-engine/v1/profiles/<profile_id>/invocations"
_OOS_PATH = "/regime-engine/v1/profiles/<profile_id>/oos-builds/<build_id>"
_HEALTH_PATH = "/regime-engine/v1/health"


def create_app(app: Flask = mlflow_app) -> Flask:
    """Extend the canonical MLflow Flask app without loading models or source services."""
    if "regime_engine_placeholder_invocation" not in app.view_functions:
        app.add_url_rule(
            _INVOCATION_PATH,
            endpoint="regime_engine_placeholder_invocation",
            view_func=_placeholder_invocation,
            methods=["POST"],
        )
    if "regime_engine_placeholder_oos" not in app.view_functions:
        app.add_url_rule(
            _OOS_PATH,
            endpoint="regime_engine_placeholder_oos",
            view_func=_placeholder_oos,
            methods=["GET"],
        )
    if "regime_engine_placeholder_health" not in app.view_functions:
        app.add_url_rule(
            _HEALTH_PATH,
            endpoint="regime_engine_placeholder_health",
            view_func=_placeholder_health,
            methods=["GET"],
        )
    return app


def _placeholder_invocation(profile_id: str) -> tuple[Response, int]:
    return (
        jsonify(
            {
                "schema_version": "RegimeError.v1",
                "error_code": "service_not_composed",
                "message": "regime-engine invocation dependencies are not composed yet",
                "profile_id": profile_id,
                "retryable": True,
            }
        ),
        503,
    )


def _placeholder_oos(profile_id: str, build_id: str) -> tuple[Response, int]:
    return (
        jsonify(
            {
                "schema_version": "RegimeError.v1",
                "error_code": "service_not_composed",
                "message": "regime-engine OOS dependencies are not composed yet",
                "profile_id": profile_id,
                "build_id": build_id,
                "retryable": True,
            }
        ),
        503,
    )


def _placeholder_health() -> Response:
    return jsonify(
        {
            "schema_version": "RegimeHealth.v1",
            "status": "initializing",
            "ready": False,
        }
    )
