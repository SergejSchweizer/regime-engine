"""Idempotency markers for completed evaluation batches."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Protocol

_EXPERIMENT_NAME = "regime-engine-evaluation"
_MARKER_RUN_NAME = "xetra-v3-evaluation-batch"
_FINGERPRINT_PARAM = "regime_engine_evaluation_fingerprint"


class _Experiment(Protocol):
    experiment_id: str


class _RunInfo(Protocol):
    status: str


class _RunData(Protocol):
    params: dict[str, str]


class _Run(Protocol):
    info: _RunInfo
    data: _RunData


class MlflowClientLike(Protocol):
    def get_experiment_by_name(self, name: str) -> _Experiment | None: ...

    def create_experiment(self, name: str) -> str: ...

    def search_runs(
        self, experiment_ids: list[str], filter_string: str, max_results: int
    ) -> list[_Run]: ...

    def create_run(self, experiment_id: str, tags: dict[str, str]) -> _Run: ...

    def log_param(self, run_id: str, key: str, value: str) -> None: ...

    def set_terminated(self, run_id: str, status: str) -> None: ...


def xetra_v3_evaluation_fingerprint(*, git_commit: str, data_sha256: str) -> str:
    """Return a stable key for one code revision and one immutable source snapshot."""

    if not git_commit or not data_sha256:
        raise ValueError("git_commit and data_sha256 must be non-empty")
    payload = json.dumps(
        {"dataset_sha256": data_sha256, "git_commit": git_commit, "scope": "xetra_v3"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def completed_xetra_v3_evaluation_exists(client: MlflowClientLike, fingerprint: str) -> bool:
    """Whether MLflow has a finished batch marker for ``fingerprint``."""

    experiment = client.get_experiment_by_name(_EXPERIMENT_NAME)
    if experiment is None:
        return False
    query = (
        f"attributes.status = 'FINISHED' and params.{_FINGERPRINT_PARAM} = '{fingerprint}' "
        f"and tags.mlflow.runName = '{_MARKER_RUN_NAME}'"
    )
    return any(
        run.info.status == "FINISHED" and run.data.params.get(_FINGERPRINT_PARAM) == fingerprint
        for run in client.search_runs([experiment.experiment_id], query, max_results=1)
    )


def record_completed_xetra_v3_evaluation(
    client: MlflowClientLike,
    *,
    fingerprint: str,
    git_commit: str,
    data_sha256: str,
    parent_run_ids: tuple[str, ...],
) -> str:
    """Write the completion marker only after every V3 hierarchy has finished."""

    experiment = client.get_experiment_by_name(_EXPERIMENT_NAME)
    experiment_id = (
        client.create_experiment(_EXPERIMENT_NAME)
        if experiment is None
        else experiment.experiment_id
    )
    run = client.create_run(experiment_id, {"mlflow.runName": _MARKER_RUN_NAME})
    run_id = getattr(run.info, "run_id", None)
    if not isinstance(run_id, str) or not run_id:
        raise TypeError("MLflow completion marker needs a run_id")
    for key, value in {
        _FINGERPRINT_PARAM: fingerprint,
        "regime_engine_git_commit": git_commit,
        "regime_engine_source_data_sha256": data_sha256,
        "regime_engine_parent_run_ids": ",".join(parent_run_ids),
    }.items():
        client.log_param(run_id, key, value)
    client.set_terminated(run_id, status="FINISHED")
    return run_id
