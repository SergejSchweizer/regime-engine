"""Deterministic human-readable rendering of evaluation statistics."""

from __future__ import annotations

import json
from collections.abc import Mapping

from market_regime_engine.evaluation_statistics.contracts import (
    GLOBAL_V4_EVALUATION_ID,
    RunStatistics,
    _evaluation_id_value,
)


def _append_evidence(lines: list[str], value: object, indent: str = "") -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            item = value[key]
            if isinstance(item, Mapping):
                lines.append(f"{indent}- {key}:")
                _append_evidence(lines, item, indent + "  ")
            else:
                encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                lines.append(f"{indent}- {key}: {encoded}")
        return
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    lines.append(f"{indent}- value: {encoded}")


def _append_global_v4_formulas(lines: list[str]) -> None:
    lines.extend(
        [
            "",
            "## Recomputable v4 formulas",
            "",
            "- Distance: `d_ij = 1 - |rho_ij|`.",
            "- State-information ratio: `SIR = I(X; Z) / H(Z)`; `eta^2` is diagnostic only.",
            "- Soft regime NMI: `NMI = 2 I(P; Q) / (H(P) + H(Q))` on exact shared timestamps.",
            "- Outer validity: `valid_fold_rate = valid_folds / planned_folds`; production "
            "requires rate `>= 0.80`, at least 3 valid folds, and a valid latest fold.",
        ]
    )


def render_statistics(statistics: RunStatistics, sha256: str | None = None) -> str:
    lines = [
        "# Evaluation Statistics",
        "",
        f"- Schema version: {statistics.schema_version}",
        f"- Evaluation: {_evaluation_id_value(statistics.evaluation_id)}",
        f"- MLflow run ID: {statistics.mlflow_run_id}",
        f"- Run type: {statistics.run_type.value}",
        f"- Status: {statistics.status.value}",
    ]
    if sha256 is not None:
        lines.append(f"- Statistics SHA-256: {sha256}")
    if statistics.evidence:
        lines.extend(["", "## Evidence", ""])
        _append_evidence(lines, statistics.evidence)
    if _evaluation_id_value(statistics.evaluation_id) == GLOBAL_V4_EVALUATION_ID:
        _append_global_v4_formulas(lines)
    return "\n".join(lines) + "\n"
