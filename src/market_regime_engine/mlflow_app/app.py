"""Final MLflow Flask composition for regime-engine profile serving."""

from __future__ import annotations

import os

from flask import Flask, Response, current_app, jsonify, request
from mlflow.server import app as mlflow_app

from market_regime_engine.contracts import LatestInvocation, ReplayInvocation
from market_regime_engine.mlflow_app.dependencies import (
    ReadinessSnapshot,
    ServiceDependencies,
    resolve_dependencies,
)
from market_regime_engine.mlflow_app.dispatch import (
    map_exception,
    parse_json_body,
    parse_oos_query,
    to_jsonable,
)
from market_regime_engine.mlflow_app.registry_compat import install_postgres_model_version_cast

install_postgres_model_version_cast()

_INVOCATION_PATH = "/regime-engine/v1/profiles/<profile_id>/invocations"
_OOS_PATH = "/regime-engine/v1/profiles/<profile_id>/oos-builds/<build_id>"
_HEALTH_PATH = "/regime-engine/v1/health"
_INVOCATION_ENDPOINT = "regime_engine_placeholder_invocation"
_OOS_ENDPOINT = "regime_engine_placeholder_oos"
_HEALTH_ENDPOINT = "regime_engine_placeholder_health"
_EXTENSION_KEY = "regime_engine_dependencies"
_GENAI_NAVIGATION_FILTER = """<script>
(() => {
  const hideGenAiNavigation = () => {
    document.querySelectorAll("a, button").forEach((element) => {
      const label = (element.textContent || "").trim().toLowerCase();
      const href = element.getAttribute("href") || "";
      if (label === "genai" || href.startsWith("/genai")) {
        (element.closest("li, [role=listitem]") || element).style.display = "none";
      }
    });
  };
  hideGenAiNavigation();
  new MutationObserver(hideGenAiNavigation).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
})();
</script>"""


def _dependencies() -> ServiceDependencies | None:
    explicit = current_app.extensions.get(_EXTENSION_KEY)
    if explicit is not None and not isinstance(explicit, ServiceDependencies):
        raise TypeError("regime-engine Flask extension contains invalid dependencies")
    return resolve_dependencies(explicit)


def _fallback_request_id() -> str:
    return "unavailable"


def _request_id(dependencies: ServiceDependencies | None) -> str:
    return dependencies.new_request_id() if dependencies is not None else _fallback_request_id()


def _error_response(exc: Exception, request_id: str) -> tuple[Response, int]:
    payload, status_code = map_exception(exc, request_id)
    return jsonify(payload), status_code


def _service_not_composed(request_id: str) -> tuple[Response, int]:
    payload = {
        "schema_version": "RegimeError.v1",
        "request_id": request_id,
        "error_code": "service_not_composed",
        "message": "regime-engine serving dependencies are not composed",
        "retryable": True,
        "details": {},
    }
    return jsonify(payload), 503


def _invocation(profile_id: str) -> tuple[Response, int] | Response:
    dependencies = _dependencies()
    request_id = _request_id(dependencies)
    if dependencies is None:
        return _service_not_composed(request_id)
    try:
        invocation = parse_json_body(request.get_data(cache=False), is_json=request.is_json)
        request_time = dependencies.request_time_utc()
        if isinstance(invocation, LatestInvocation):
            result = dependencies.latest_handler.handle(
                request_id=request_id,
                profile_id=profile_id,
                invocation=invocation,
                request_time_utc=request_time,
            )
        elif isinstance(invocation, ReplayInvocation):
            result = dependencies.replay_handler.handle(
                request_id=request_id,
                profile_id=profile_id,
                invocation=invocation,
                request_time_utc=request_time,
            )
        else:
            raise AssertionError("unreachable invocation type")
        return jsonify(to_jsonable(result))
    except Exception as exc:
        return _error_response(exc, request_id)


def _oos(profile_id: str, build_id: str) -> tuple[Response, int] | Response:
    dependencies = _dependencies()
    request_id = _request_id(dependencies)
    if dependencies is None:
        return _service_not_composed(request_id)
    try:
        if any(len(request.args.getlist(key)) != 1 for key in request.args):
            from market_regime_engine.mlflow_app.dispatch import ApiInputError

            raise ApiInputError(
                "duplicate_query_field",
                "OOS query fields may appear at most once",
            )
        start, end = parse_oos_query(request.args.to_dict(flat=True))
        result = dependencies.oos_handler.handle(
            profile_id=profile_id,
            build_id=build_id,
            start=start,
            end=end,
        )
        return jsonify(to_jsonable(result))
    except Exception as exc:
        return _error_response(exc, request_id)


def _health() -> Response:
    dependencies = _dependencies()
    if dependencies is None:
        snapshot = ReadinessSnapshot("not_ready", False)
        status = "initializing"
    else:
        try:
            snapshot = dependencies.readiness()
            status = snapshot.status
        except Exception:
            snapshot = ReadinessSnapshot("not_ready", False)
            status = "not_ready"
    return jsonify(
        {
            "schema_version": "RegimeHealth.v1",
            "status": status,
            "ready": snapshot.ready,
        }
    )


def _filter_genai_navigation(response: Response) -> Response:
    """Hide MLflow's optional GenAI navigation from the self-hosted UI only."""

    if request.path != "/" or response.mimetype != "text/html":
        return response
    html = response.get_data(as_text=True)
    response.set_data(html.replace("</body>", f"{_GENAI_NAVIGATION_FILTER}</body>"))
    return response


def _install_route(
    app: Flask,
    *,
    rule: str,
    endpoint: str,
    view_func: object,
    methods: list[str],
) -> None:
    if endpoint in app.view_functions:
        app.view_functions[endpoint] = view_func  # type: ignore[assignment]
        return
    app.add_url_rule(
        rule,
        endpoint=endpoint,
        view_func=view_func,  # type: ignore[arg-type]
        methods=methods,
    )


def create_app(
    app: Flask = mlflow_app,
    dependencies: ServiceDependencies | None = None,
) -> Flask:
    """Extend canonical MLflow Flask routes with injected regime-engine handlers."""

    if dependencies is None and os.environ.get("REGIME_AUTO_COMPOSE") == "1":
        from market_regime_engine.deployment import configure_serving_defaults

        configure_serving_defaults()
    app.extensions[_EXTENSION_KEY] = dependencies
    if "regime_engine_genai_navigation_filter" not in app.extensions:
        app.after_request(_filter_genai_navigation)
        app.extensions["regime_engine_genai_navigation_filter"] = True
    _install_route(
        app,
        rule=_INVOCATION_PATH,
        endpoint=_INVOCATION_ENDPOINT,
        view_func=_invocation,
        methods=["POST"],
    )
    _install_route(
        app,
        rule=_OOS_PATH,
        endpoint=_OOS_ENDPOINT,
        view_func=_oos,
        methods=["GET"],
    )
    _install_route(
        app,
        rule=_HEALTH_PATH,
        endpoint=_HEALTH_ENDPOINT,
        view_func=_health,
        methods=["GET"],
    )
    return app
