from __future__ import annotations

from dataclasses import dataclass, field

from market_regime_engine.mlflow_support.evaluation_dedup import (
    completed_xetra_v3_evaluation_exists,
    record_completed_xetra_v3_evaluation,
    xetra_v3_evaluation_fingerprint,
)


@dataclass
class _Experiment:
    experiment_id: str


@dataclass
class _Info:
    status: str
    run_id: str = "marker-run"


@dataclass
class _Data:
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class _Run:
    info: _Info
    data: _Data = field(default_factory=_Data)


class _Client:
    def __init__(self, runs: list[_Run] | None = None) -> None:
        self.experiment = _Experiment("1")
        self.runs = [] if runs is None else runs
        self.logged: dict[str, str] = {}
        self.terminated: tuple[str, str] | None = None

    def get_experiment_by_name(self, name: str) -> _Experiment:
        assert name == "regime-engine-evaluation"
        return self.experiment

    def create_experiment(self, name: str) -> str:
        raise AssertionError("experiment already exists")

    def search_runs(
        self, experiment_ids: list[str], filter_string: str, max_results: int
    ) -> list[_Run]:
        assert experiment_ids == ["1"]
        assert "regime_engine_evaluation_fingerprint" in filter_string
        assert max_results == 1
        return self.runs

    def create_run(self, experiment_id: str, tags: dict[str, str]) -> _Run:
        assert experiment_id == "1"
        assert tags["mlflow.runName"] == "xetra-v3-evaluation-batch"
        return _Run(_Info("RUNNING"))

    def log_param(self, run_id: str, key: str, value: str) -> None:
        assert run_id == "marker-run"
        self.logged[key] = value

    def set_terminated(self, run_id: str, status: str) -> None:
        self.terminated = (run_id, status)


def test_fingerprint_changes_with_code_or_dataset() -> None:
    fingerprint = xetra_v3_evaluation_fingerprint(git_commit="a", data_sha256="b")
    assert fingerprint != xetra_v3_evaluation_fingerprint(git_commit="c", data_sha256="b")
    assert fingerprint != xetra_v3_evaluation_fingerprint(git_commit="a", data_sha256="d")


def test_completed_marker_is_required_to_skip() -> None:
    fingerprint = xetra_v3_evaluation_fingerprint(git_commit="a", data_sha256="b")
    assert not completed_xetra_v3_evaluation_exists(
        _Client(
            [_Run(_Info("FAILED"), _Data({"regime_engine_evaluation_fingerprint": fingerprint}))]
        ),
        fingerprint,
    )
    assert completed_xetra_v3_evaluation_exists(
        _Client(
            [_Run(_Info("FINISHED"), _Data({"regime_engine_evaluation_fingerprint": fingerprint}))]
        ),
        fingerprint,
    )


def test_completion_marker_records_full_identity() -> None:
    client = _Client()
    fingerprint = xetra_v3_evaluation_fingerprint(git_commit="a", data_sha256="b")
    assert (
        record_completed_xetra_v3_evaluation(
            client,
            fingerprint=fingerprint,
            git_commit="a",
            data_sha256="b",
            parent_run_ids=("one", "two", "three"),
        )
        == "marker-run"
    )
    assert client.logged["regime_engine_evaluation_fingerprint"] == fingerprint
    assert client.logged["regime_engine_parent_run_ids"] == "one,two,three"
    assert client.terminated == ("marker-run", "FINISHED")
