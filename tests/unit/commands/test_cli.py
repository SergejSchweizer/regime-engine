from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest

from market_regime_engine import cli
from market_regime_engine.commands import actions, runtime
from market_regime_engine.commands.contracts import (
    OperatorAction,
    OperatorRequest,
    OperatorResult,
)
from market_regime_engine.commands.errors import OperatorCommandError


class FakeService:
    def __init__(self) -> None:
        self.requests: list[OperatorRequest] = []

    def execute(self, request: OperatorRequest) -> OperatorResult:
        self.requests.append(request)
        return OperatorResult(
            action=request.action,
            profile_id=request.profile_id,
            fields=(("request", request.canonical_json()),),
        )


@pytest.mark.parametrize(
    ("argv", "action", "parameters"),
    [
        (["evaluate", "--profile", "xetra"], OperatorAction.EVALUATE, ()),
        (
            ["final-refit", "--profile", "xetra", "--evaluation-id", "eval-1"],
            OperatorAction.FINAL_REFIT,
            (("evaluation_id", "eval-1"),),
        ),
        (
            ["publish-oos", "--profile", "xetra", "--evaluation-id", "eval-1"],
            OperatorAction.PUBLISH_OOS,
            (("evaluation_id", "eval-1"),),
        ),
        (
            [
                "register",
                "--profile",
                "xetra",
                "--production-package",
                "/tmp/package",
                "--oos-build-id",
                "walk-forward-oos-abc",
            ],
            OperatorAction.REGISTER,
            (
                ("oos_build_id", "walk-forward-oos-abc"),
                ("production_package", "/tmp/package"),
            ),
        ),
        (["status", "--profile", "xetra"], OperatorAction.STATUS, ()),
    ],
)
def test_cli_dispatches_exact_actions(
    argv: list[str],
    action: OperatorAction,
    parameters: tuple[tuple[str, str], ...],
) -> None:
    service = FakeService()
    stdout = StringIO()
    stderr = StringIO()
    assert cli.main(argv, service=service, stdout=stdout, stderr=stderr) == 0
    assert stderr.getvalue() == ""
    assert service.requests == [OperatorRequest(action, "xetra", parameters)]
    payload = json.loads(stdout.getvalue())
    assert payload["action"] == action.value
    assert payload["profile_id"] == "xetra"


def test_parser_exposes_only_operator_lifecycle_actions() -> None:
    help_text = cli.build_parser().format_help()
    for command in ("evaluate", "final-refit", "register", "publish-oos", "status"):
        assert command in help_text
    assert "serve" not in help_text


def test_register_requires_explicit_oos_publication_evidence() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "register",
                "--profile",
                "xetra",
                "--production-package",
                "/tmp/package",
            ]
        )
    assert exc.value.code == 2


def test_cli_returns_deterministic_operator_error() -> None:
    class FailingService:
        def execute(self, request: OperatorRequest) -> OperatorResult:
            del request
            raise OperatorCommandError("source_unavailable", "feature source is unavailable")

    stdout = StringIO()
    stderr = StringIO()
    assert (
        cli.main(
            ["status", "--profile", "xetra"],
            service=FailingService(),
            stdout=stdout,
            stderr=stderr,
        )
        == 2
    )
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "error: source_unavailable: feature source is unavailable\n"


def test_action_adapter_rejects_mismatched_service_identity() -> None:
    class MismatchService:
        def execute(self, request: OperatorRequest) -> OperatorResult:
            return OperatorResult(
                action=OperatorAction.STATUS,
                profile_id=request.profile_id,
                fields=(("state", "ok"),),
            )

    with pytest.raises(ValueError, match="mismatched command identity"):
        actions.evaluate(MismatchService(), profile_id="xetra")


def test_request_result_and_error_validation() -> None:
    request = OperatorRequest(
        OperatorAction.REGISTER,
        "xetra",
        (("oos_build_id", "oos-1"), ("production_package", "/tmp/p")),
    )
    assert request.parameter("oos_build_id") == "oos-1"
    assert request.parameter("missing") is None
    assert json.loads(request.canonical_json())["parameters"]["production_package"] == "/tmp/p"
    result = OperatorResult(OperatorAction.STATUS, "xetra", (("state", "ready"),))
    assert json.loads(result.canonical_json())["fields"] == {"state": "ready"}

    with pytest.raises(ValueError, match="profile_id"):
        OperatorRequest(OperatorAction.STATUS, "", ())
    with pytest.raises(ValueError, match="sorted"):
        OperatorRequest(OperatorAction.STATUS, "xetra", (("z", "1"), ("a", "2")))
    with pytest.raises(ValueError, match="unique"):
        OperatorRequest(OperatorAction.STATUS, "xetra", (("a", "1"), ("a", "2")))
    with pytest.raises(ValueError, match="keys"):
        OperatorRequest(OperatorAction.STATUS, "xetra", (("", "1"),))
    with pytest.raises(ValueError, match="values"):
        OperatorRequest(OperatorAction.STATUS, "xetra", (("a", ""),))
    with pytest.raises(ValueError, match="result profile_id"):
        OperatorResult(OperatorAction.STATUS, " ", ())
    with pytest.raises(ValueError, match="sorted"):
        OperatorResult(OperatorAction.STATUS, "xetra", (("z", "1"), ("a", "2")))
    with pytest.raises(ValueError, match="operator error code"):
        OperatorCommandError("", "message")
    with pytest.raises(ValueError, match="operator error message"):
        OperatorCommandError("code", " ")


def test_default_runtime_loader_is_lazy_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        exc = ModuleNotFoundError(name)
        exc.name = "market_regime_engine.commands.lifecycle"
        raise exc

    monkeypatch.setattr(runtime, "import_module", missing)
    with pytest.raises(OperatorCommandError, match="runtime_not_configured"):
        runtime.load_default_operator_service()

    def nested_missing(name: str) -> object:
        del name
        exc = ModuleNotFoundError("dependency")
        exc.name = "dependency"
        raise exc

    monkeypatch.setattr(runtime, "import_module", nested_missing)
    with pytest.raises(ModuleNotFoundError, match="dependency"):
        runtime.load_default_operator_service()


def test_default_runtime_requires_callable_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "import_module", lambda name: SimpleNamespace(name=name))
    with pytest.raises(OperatorCommandError, match="build_operator_service"):
        runtime.load_default_operator_service()

    service = FakeService()
    monkeypatch.setattr(
        runtime,
        "import_module",
        lambda name: SimpleNamespace(name=name, build_operator_service=lambda: service),
    )
    assert runtime.load_default_operator_service() is service


def test_dispatch_rejects_unreachable_command() -> None:
    namespace = SimpleNamespace(command="serve", profile="xetra")
    with pytest.raises(AssertionError, match="unreachable command"):
        cli._dispatch(namespace, FakeService())
