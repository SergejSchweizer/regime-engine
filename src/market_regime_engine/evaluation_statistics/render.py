"""Deterministic human-readable rendering of evaluation statistics."""

from market_regime_engine.evaluation_statistics.contracts import RunStatistics, _evaluation_id_value


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
    return "\n".join(lines) + "\n"
