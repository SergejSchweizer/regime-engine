"""Operator CLI for explicit regime-engine lifecycle actions without standalone serving."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from market_regime_engine.commands import actions
from market_regime_engine.commands.contracts import OperatorResult, OperatorService
from market_regime_engine.commands.errors import OperatorCommandError
from market_regime_engine.commands.runtime import load_default_operator_service


def _profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        required=True,
        help="public profile ID, for example xetra",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regime-engine",
        description="Explicit statistical-model lifecycle commands for regime-engine.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subcommands.add_parser(
        "evaluate",
        help="run deterministic feature selection and walk-forward candidate evaluation",
    )
    _profile_argument(evaluate_parser)

    refit_parser = subcommands.add_parser(
        "final-refit",
        help="refit the statistically selected candidate using immutable evaluation evidence",
    )
    _profile_argument(refit_parser)
    refit_parser.add_argument("--evaluation-id", required=True)

    publish_parser = subcommands.add_parser(
        "publish-oos",
        help="publish immutable walk-forward OOS predictions for an evaluation",
    )
    _profile_argument(publish_parser)
    publish_parser.add_argument("--evaluation-id", required=True)

    register_parser = subcommands.add_parser(
        "register",
        help="register a final-refit package after explicit OOS publication",
    )
    _profile_argument(register_parser)
    register_parser.add_argument("--production-package", required=True)
    register_parser.add_argument(
        "--oos-build-id",
        required=True,
        help="immutable walk_forward_oos build proving publication completed before registration",
    )

    status_parser = subcommands.add_parser(
        "status",
        help="show deterministic source/evaluation/registry lifecycle status",
    )
    _profile_argument(status_parser)
    return parser


def _dispatch(
    namespace: argparse.Namespace,
    service: OperatorService,
) -> OperatorResult:
    command = str(namespace.command)
    profile_id = str(namespace.profile)
    if command == "evaluate":
        return actions.evaluate(service, profile_id=profile_id)
    if command == "final-refit":
        return actions.final_refit(
            service,
            profile_id=profile_id,
            evaluation_id=str(namespace.evaluation_id),
        )
    if command == "publish-oos":
        return actions.publish_oos(
            service,
            profile_id=profile_id,
            evaluation_id=str(namespace.evaluation_id),
        )
    if command == "register":
        return actions.register(
            service,
            profile_id=profile_id,
            production_package=str(namespace.production_package),
            oos_build_id=str(namespace.oos_build_id),
        )
    if command == "status":
        return actions.status(service, profile_id=profile_id)
    raise AssertionError(f"unreachable command: {command}")


def main(
    argv: Sequence[str] | None = None,
    *,
    service: OperatorService | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    errors = stderr if stderr is not None else sys.stderr
    namespace = build_parser().parse_args(argv)
    try:
        runtime = service if service is not None else load_default_operator_service()
        result = _dispatch(namespace, runtime)
    except OperatorCommandError as exc:
        print(f"error: {exc.code}: {exc.message}", file=errors)
        return 2
    print(result.canonical_json(), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
