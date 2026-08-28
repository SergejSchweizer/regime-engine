"""Fail-closed MLflow tracking with immutable local statistics mirrors."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from market_regime_engine.evaluation.selection import select_statistical_champion
from market_regime_engine.evaluation_statistics.contracts import RunStatistics, RunType, Status
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.contracts import EvaluationId, FeatureSpec
from market_regime_engine.evaluations.delta1_univariate import Delta1UnivariateEvaluation
from market_regime_engine.evaluations.medoid_multivariate import MedoidMultivariateEvaluation
from market_regime_engine.evaluations.medoid_univariate import MedoidUnivariateEvaluation
from market_regime_engine.evaluations.univariate_grid import UnivariateFeatureGrid
from market_regime_engine.mlflow_support.ports import TrackingPort
from market_regime_engine.training.candidate_grid import CandidateGridEvaluation

EvaluationResult = (
    MedoidMultivariateEvaluation | MedoidUnivariateEvaluation | Delta1UnivariateEvaluation
)


def _write_candidate_comparison_table(
    path: Path, grid: CandidateGridEvaluation, *, feature_name: str | None
) -> None:
    """Write the complete canonical statistical-comparison evidence as CSV."""

    try:
        selection = select_statistical_champion(grid)
        evidence = {item.candidate_id: item for item in selection.evidence}
    except ValueError:
        evidence = {}
    fields = (
        "feature_name",
        "statistical_rank",
        "candidate_id",
        "state_count_fewer_is_better",
        "accepted",
        "rejection_reasons",
        "oos_predictive_loglik_mean_higher_is_better",
        "oos_predictive_loglik_std_lower_is_better",
        "oos_predictive_loglik_worst_fold_higher_is_better",
        "bic_mean_lower_is_better",
        "aic_mean_lower_is_better",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for aggregate in grid.aggregates:
            item = evidence.get(aggregate.candidate_id)
            mean = aggregate.oos_predictive_loglik_mean
            std = aggregate.oos_predictive_loglik_std
            worst = aggregate.oos_predictive_loglik_worst_fold
            writer.writerow(
                {
                    "feature_name": feature_name or "all_medoids",
                    "statistical_rank": "" if item is None or item.rank is None else item.rank,
                    "candidate_id": aggregate.candidate_id,
                    "state_count_fewer_is_better": aggregate.state_count,
                    "accepted": "" if item is None else item.accepted,
                    "rejection_reasons": "" if item is None else "; ".join(item.rejection_reasons),
                    "oos_predictive_loglik_mean_higher_is_better": mean,
                    "oos_predictive_loglik_std_lower_is_better": std,
                    "oos_predictive_loglik_worst_fold_higher_is_better": worst,
                    "bic_mean_lower_is_better": aggregate.bic_mean,
                    "aic_mean_lower_is_better": aggregate.aic_mean,
                }
            )


def _track_candidate_comparison_tables(
    port: TrackingPort,
    parent_run_id: str,
    grids: tuple[tuple[str | None, CandidateGridEvaluation], ...],
) -> None:
    with TemporaryDirectory(prefix="regime-engine-candidate-comparison-") as directory:
        root = Path(directory)
        for feature_name, grid in grids:
            name = "candidate_comparison.csv" if feature_name is None else f"{feature_name}.csv"
            path = root / name
            _write_candidate_comparison_table(path, grid, feature_name=feature_name)
            port.log_artifact(parent_run_id, str(path), "candidate_comparison")


@dataclass(frozen=True, slots=True)
class EvaluationTrackingResult:
    parent_run_id: str
    feature_run_ids: tuple[tuple[str, str], ...]
    candidate_run_ids: tuple[tuple[str, str], ...]
    statistics_root: str


def track_statistics_run(
    port: TrackingPort,
    writer: StatisticsWriter,
    *,
    run_name: str,
    statistics: RunStatistics,
    parent_run_id: str | None = None,
) -> tuple[str, str]:
    """Create one MLflow run and its exact immutable finalized statistics mirror."""

    if statistics.status is not Status.RUNNING:
        raise ValueError("statistics tracking requires an initial RUNNING dossier")
    run_id = port.start_run(run_name=run_name, parent_run_id=parent_run_id)
    started = replace(statistics, mlflow_run_id=run_id, parent_run_id=parent_run_id)
    try:
        directory = writer.start(started)
        finalized = replace(started, status=Status.FINISHED, ended_at=datetime.now(UTC))
        digest = writer.finalize(finalized)
        path = directory / "statistics.json"
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("finalized statistics hash mismatch")
        port.log_params(run_id, {"statistics_sha256": digest})
        port.log_artifact(run_id, str(path), "statistics")
        port.end_run(run_id)
        return run_id, digest
    except BaseException:
        port.fail_run(run_id)
        raise


def _running_statistics(
    evaluation_id: EvaluationId,
    run_type: RunType,
    run_name: str,
    evidence: dict[str, object],
) -> RunStatistics:
    return RunStatistics(
        evaluation_id,
        "pending",
        run_type,
        run_name,
        Status.RUNNING,
        datetime.now(UTC),
        evidence=evidence,
    )


def _candidate_evidence(grid: CandidateGridEvaluation, candidate_id: str) -> dict[str, object]:
    aggregate = next(item for item in grid.aggregates if item.candidate_id == candidate_id)
    evaluation = next(item for item in grid.evaluations if item.candidate_id == candidate_id)
    return {
        "identity": {"candidate_id": candidate_id},
        "lineage": {
            "source_build_id": grid.source_build_id,
            "evaluation_plan_hash": grid.evaluation_plan_hash,
            "feature_selection_definition_hash": grid.feature_selection_definition_hash,
            "feature_selection_execution_hash": grid.feature_selection_execution_hash,
        },
        "input": {"feature_order": grid.feature_order},
        "model": {"state_count": evaluation.state_count},
        "folds": {
            "planned_count": len(evaluation.folds),
            "valid_count": len(evaluation.valid_folds),
        },
        "aggregate": asdict(aggregate),
    }


def _track_grid(
    port: TrackingPort,
    writer: StatisticsWriter,
    grid: CandidateGridEvaluation,
    parent_run_id: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            candidate.candidate_id,
            track_statistics_run(
                port,
                writer,
                run_name=candidate.candidate_id,
                parent_run_id=parent_run_id,
                statistics=_running_statistics(
                    EvaluationId.MEDOID_MULTIVARIATE,
                    RunType.CANDIDATE,
                    candidate.candidate_id,
                    _candidate_evidence(grid, candidate.candidate_id),
                ),
            )[0],
        )
        for candidate in grid.evaluations
    )


def track_evaluation_result(
    port: TrackingPort,
    writer: StatisticsWriter,
    *,
    result: EvaluationResult,
) -> EvaluationTrackingResult:
    """Track one v3 evaluation result with a one-to-one immutable local mirror per run."""

    grids: tuple[CandidateGridEvaluation, ...]
    feature_grids: tuple[UnivariateFeatureGrid, ...]
    if isinstance(result, MedoidMultivariateEvaluation):
        evaluation_id = EvaluationId.MEDOID_MULTIVARIATE
        lineage, feature_spec, grids = result.lineage, result.feature_spec, (result.candidate_grid,)
        champion, reason = (
            result.medoid_multivariate_statistical_champion,
            result.no_champion_reason,
        )
        feature_grids = ()
    elif isinstance(result, MedoidUnivariateEvaluation):
        evaluation_id = EvaluationId.MEDOID_UNIVARIATE
        lineage, feature_spec = result.lineage, result.feature_spec
        grids = ()
        champion, reason = result.medoid_univariate_evaluation_champion, result.no_champion_reason
        feature_grids = result.feature_grids
    else:
        evaluation_id = EvaluationId.DELTA1_UNIVARIATE
        lineage = result.lineage
        feature_spec = FeatureSpec(
            evaluation_id, tuple(grid.feature_name for grid in result.feature_grids)
        )
        grids = ()
        champion, reason = result.delta1_univariate_evaluation_champion, result.no_champion_reason
        feature_grids = result.feature_grids
    parent_evidence: dict[str, object] = {
        "identity": {"evaluation_id": evaluation_id.value},
        "lineage": asdict(lineage),
        "input": {"feature_order": feature_spec.feature_order},
        "champion": {"candidate_id": champion, "no_champion_reason": reason},
    }
    parent_run_id, _ = track_statistics_run(
        port,
        writer,
        run_name=evaluation_id.value,
        statistics=_running_statistics(
            evaluation_id, RunType.PARENT, evaluation_id.value, parent_evidence
        ),
    )
    feature_run_ids: list[tuple[str, str]] = []
    candidate_run_ids: list[tuple[str, str]] = []
    comparison_grids = tuple((None, grid) for grid in grids) + tuple(
        (feature_grid.feature_name, feature_grid.candidate_grid) for feature_grid in feature_grids
    )
    _track_candidate_comparison_tables(port, parent_run_id, comparison_grids)
    for grid in grids:
        candidate_run_ids.extend(_track_grid(port, writer, grid, parent_run_id))
    for feature_grid in feature_grids:
        feature_run_id, _ = track_statistics_run(
            port,
            writer,
            run_name=feature_grid.feature_name,
            parent_run_id=parent_run_id,
            statistics=_running_statistics(
                evaluation_id,
                RunType.FEATURE,
                feature_grid.feature_name,
                {"input": {"feature_name": feature_grid.feature_name}},
            ),
        )
        feature_run_ids.append((feature_grid.feature_name, feature_run_id))
        for candidate in feature_grid.candidate_grid.evaluations:
            candidate_run_id, _ = track_statistics_run(
                port,
                writer,
                run_name=candidate.candidate_id,
                parent_run_id=feature_run_id,
                statistics=_running_statistics(
                    evaluation_id,
                    RunType.CANDIDATE,
                    candidate.candidate_id,
                    _candidate_evidence(feature_grid.candidate_grid, candidate.candidate_id),
                ),
            )
            candidate_run_ids.append((candidate.candidate_id, candidate_run_id))
    return EvaluationTrackingResult(
        parent_run_id, tuple(feature_run_ids), tuple(candidate_run_ids), str(writer.preflight())
    )
