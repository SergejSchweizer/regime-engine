"""External-service-backed Xetra v4 lifecycle implementation."""

from __future__ import annotations

import json
import os
import pickle
import platform
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import psycopg
from mlflow.exceptions import MlflowException

from market_regime_engine.commands.lifecycle import (
    EvaluationOutcome,
    FinalRefitOutcome,
    LifecycleStatus,
    OOSPublicationOutcome,
    RegistrationOutcome,
)
from market_regime_engine.contracts import PredictionMode
from market_regime_engine.evaluation.walk_forward import run_walk_forward_candidate
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluation_runs.snapshot import ArrowDatasetSnapshotStore
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.deployment_selection import (
    select_deployment_configuration,
)
from market_regime_engine.evaluations.global_regime_v4 import (
    V4ConfigurationSelection,
    evaluate_global_regime_v4_from_source,
)
from market_regime_engine.feature_discovery.contracts import AdaptiveEvaluationResult
from market_regime_engine.features.ports import FeatureRequest, FeatureSnapshot
from market_regime_engine.features.postgres_settings import FeaturePostgresSettings
from market_regime_engine.features.postgres_source import PostgresFeatureSource
from market_regime_engine.mlflow_support.evaluation_tracking import (
    build_global_v4_evidence,
    track_global_v4_evaluation,
)
from market_regime_engine.mlflow_support.model_package import (
    load_production_package,
    save_production_package,
)
from market_regime_engine.mlflow_support.model_publishing import publish_production_package
from market_regime_engine.mlflow_support.registry import MlflowModelRegistry
from market_regime_engine.mlflow_support.settings import MLflowSettings
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort
from market_regime_engine.predictions.store import PredictionStore
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.training.adapter_factory import adapter_factory
from market_regime_engine.training.final_refit import final_production_refit


class _RecordingSource:
    def __init__(self, source: PostgresFeatureSource) -> None:
        self._source = source
        self.catalog: Any | None = None
        self.snapshot: FeatureSnapshot | None = None

    def read_schema_wide_with_catalog(self, request: FeatureRequest) -> Any:
        if self.catalog is not None and self.snapshot is not None:
            return self.catalog, self.snapshot
        catalog, snapshot = self._source.read_schema_wide_with_catalog(request)
        self.catalog = catalog
        self.snapshot = snapshot
        return catalog, snapshot


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(temporary, path)


def _read_pickle(path: Path, expected: type[Any]) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"lifecycle state is missing: {path.name}")
    value = pickle.loads(path.read_bytes())
    if not isinstance(value, expected):
        raise ValueError(f"lifecycle state {path.name} has an incompatible type")
    return value


def _commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _rows(snapshot: FeatureSnapshot) -> pd.DataFrame:
    frame = pd.DataFrame([row.values for row in snapshot.rows], columns=snapshot.feature_names)
    frame.insert(0, "timestamp_m1", [row.timestamp for row in snapshot.rows])
    return frame


def _candidate(configuration: Any, catalog: Any) -> ResolvedCandidateProfile:
    return ResolvedCandidateProfile(
        candidate_id=configuration.candidate_id,
        state_count=configuration.state_count,
        covariance_type="full",
        feature_order=configuration.feature_order,
        feature_dimension=len(configuration.feature_order),
        source_build_id=configuration.source_build_id or catalog.lineage.source_build_id,
        feature_selection_definition_hash=configuration.selection_definition_hash,
        feature_selection_execution_hash=configuration.selection_execution_hash,
        original_feature_universe=catalog.feature_names,
        model_family=configuration.model_family,
        mixture_count=2 if configuration.model_family == "gmm_hmm" else 1,
    )


def _configured_state_root(root: Path) -> Path:
    configured = os.environ.get("REGIME_ENGINE_STATE_ROOT") or os.environ.get(
        "REGIME_EVALUATION_CHECKPOINT_ROOT"
    )
    if not configured:
        raise RuntimeError(
            "REGIME_ENGINE_STATE_ROOT or REGIME_EVALUATION_CHECKPOINT_ROOT must be configured "
            "to a persistent deployment volume"
        )
    state_root = Path(configured).expanduser()
    if not state_root.is_absolute():
        raise RuntimeError("lifecycle state root must be an absolute path")
    resolved_state_root = state_root.resolve()
    resolved_root = root.resolve()
    if resolved_state_root == resolved_root or resolved_root in resolved_state_root.parents:
        raise RuntimeError("lifecycle state root must be outside the repository")
    return resolved_state_root


class V4LifecycleBackend:
    """Run the complete Xetra v4 lifecycle against external NAS services."""

    def __init__(self) -> None:
        self.root = Path(os.environ.get("REGIME_ENGINE_ROOT", Path(__file__).resolve().parents[3]))
        self.state_root = _configured_state_root(self.root) / "xetra"
        self.profile = load_profile(self.root / "configs/profiles/xetra_v4.yaml")
        self._source: PostgresFeatureSource | None = None
        self.mlflow_settings = MLflowSettings.from_environment()

    @property
    def source(self) -> PostgresFeatureSource:
        if self._source is None:
            settings = FeaturePostgresSettings.from_env(os.environ)
            self._source = PostgresFeatureSource(
                lambda: cast(Any, psycopg.connect(**cast(Any, settings.connection_kwargs())))
            )
        return self._source

    @property
    def _evaluation_path(self) -> Path:
        return self.state_root / "evaluation.pkl"

    @property
    def _source_path(self) -> Path:
        return self.state_root / "source.pkl"

    @property
    def _selection_path(self) -> Path:
        return self.state_root / "deployment-selection.pkl"

    @property
    def _package_path(self) -> Path:
        return self.state_root / "production-package"

    @property
    def _oos_path(self) -> Path:
        return self.state_root / "oos"

    def _capture_source(self) -> tuple[Any, FeatureSnapshot]:
        recording = _RecordingSource(self.source)
        recording.read_schema_wide_with_catalog(FeatureRequest.all_features())
        if recording.catalog is None or recording.snapshot is None:
            raise RuntimeError("source did not return a catalog and snapshot")
        _atomic_pickle(self._source_path, (recording.catalog, recording.snapshot))
        return recording.catalog, recording.snapshot

    def _saved_source(self) -> tuple[Any, FeatureSnapshot]:
        if self._source_path.is_file():
            value = pickle.loads(self._source_path.read_bytes())
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("saved lifecycle source state is malformed")
            return value
        return self._capture_source()

    def _registry_version(self, alias: str) -> str | None:
        try:
            return MlflowModelRegistry().resolve_alias("regime-xetra", alias).exact_version
        except MlflowException as exc:
            if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
                return None
            raise

    def status(self, profile_id: str) -> LifecycleStatus:
        if profile_id != "xetra":
            raise ValueError("only xetra is supported")
        catalog, _snapshot = self._capture_source()
        completed: str | None = None
        metadata_path = self.state_root / "completed.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            completed = metadata.get("source_build_id")
        return LifecycleStatus(
            current_source_build_id=catalog.lineage.source_build_id,
            completed_source_build_id=completed,
            champion_version=self._registry_version("champion"),
            challenger_version=self._registry_version("challenger"),
        )

    def evaluate(self, profile_id: str, source_build_id: str) -> EvaluationOutcome:
        if profile_id != "xetra":
            raise ValueError("only xetra is supported")
        if self._evaluation_path.is_file():
            result = _read_pickle(self._evaluation_path, AdaptiveEvaluationResult)
            if result.source_build_id != source_build_id:
                raise ValueError("saved evaluation source build differs from cycle build")
            return EvaluationOutcome(
                "global_regime_v4",
                source_build_id,
                next(
                    fold.final_configuration.candidate_id
                    for fold in reversed(result.outer_folds)
                    if fold.valid
                ),
            )
        recording = _RecordingSource(self.source)
        recording.read_schema_wide_with_catalog(FeatureRequest.all_features())
        if recording.catalog is None or recording.snapshot is None:
            raise RuntimeError("source did not return a catalog and snapshot")
        catalog, snapshot = recording.catalog, recording.snapshot
        _atomic_pickle(self._source_path, (catalog, snapshot))
        if catalog.lineage.source_build_id != source_build_id:
            raise ValueError("source build changed before evaluation")
        selections: dict[int, V4ConfigurationSelection] = {}
        result = evaluate_global_regime_v4_from_source(
            recording,
            profile=self.profile,
            snapshot_store=ArrowDatasetSnapshotStore(self.state_root / "snapshots"),
            repository_commit_sha=_commit(self.root),
            uv_lock_sha256=_file_sha256(self.root / "uv.lock"),
            python_version=platform.python_version(),
            selection_sink=selections.__setitem__,
        )
        if result.source_build_id != source_build_id:
            raise ValueError("source build changed during evaluation")
        if not result.production_eligible:
            raise ValueError("v4 evaluation failed production-eligibility gates")
        evidence = build_global_v4_evidence(
            result,
            catalog=catalog,
            snapshot=snapshot,
            profile=self.profile,
            selections=selections,
            repository_commit_sha=_commit(self.root),
        )
        track_global_v4_evaluation(
            FileMlflowTrackingPort(
                self.mlflow_settings.tracking_uri,
                experiment_name="regime-engine-evaluation",
            ),
            StatisticsWriter(self.state_root / "statistics"),
            evidence=evidence,
            result=result,
            selections=selections,
        )
        _atomic_pickle(self._evaluation_path, result)
        return EvaluationOutcome(
            "global_regime_v4",
            source_build_id,
            next(
                fold.final_configuration.candidate_id
                for fold in reversed(result.outer_folds)
                if fold.valid
            ),
        )

    def final_refit(self, profile_id: str, evaluation_id: str) -> FinalRefitOutcome:
        if profile_id != "xetra" or evaluation_id != "global_regime_v4":
            raise ValueError("final refit requires the completed global_regime_v4 evaluation")
        if (self._package_path / "MLmodel").is_file():
            load_production_package(self._package_path)
            return FinalRefitOutcome(str(self._package_path))
        validation = _read_pickle(self._evaluation_path, AdaptiveEvaluationResult)
        catalog, snapshot = self._saved_source()
        frame = _rows(snapshot)
        deployment = select_deployment_configuration(
            frame,
            catalog=catalog,
            profile=self.profile,
            validation=validation,
        )
        candidate = _candidate(deployment.configuration, catalog)
        validation_rows = frame.loc[
            frame["timestamp_m1"] <= validation.validation_evaluation_cutoff
        ].copy()
        validation_plan = plan_walk_forward(
            tuple(validation_rows["timestamp_m1"]), self.profile.walk_forward
        )
        winning_evaluation = run_walk_forward_candidate(
            validation_rows,
            plan=validation_plan,
            profile=self.profile,
            candidate=candidate,
            adapter_factory=cast(Any, adapter_factory(self.profile, candidate)),
        )
        artifact = final_production_refit(
            frame,
            lineage=catalog.lineage,
            candidate=candidate,
            winning_evaluation=winning_evaluation,
            deployment_selection=deployment,
            profile=self.profile,
        )
        package = save_production_package(artifact, self._package_path)
        _atomic_pickle(self._selection_path, deployment)
        return FinalRefitOutcome(str(package))

    def publish_oos(self, profile_id: str, evaluation_id: str) -> OOSPublicationOutcome:
        if profile_id != "xetra" or evaluation_id != "global_regime_v4":
            raise ValueError("OOS publication requires the completed global_regime_v4 evaluation")
        result = _read_pickle(self._evaluation_path, AdaptiveEvaluationResult)
        build_id = sha256(f"walk_forward_oos:{result.result_hash}".encode()).hexdigest()
        store = PredictionStore(self._oos_path)
        try:
            store.load_manifest("xetra", build_id)
        except FileNotFoundError:
            rows: list[dict[str, object]] = []
            for fold in result.outer_folds:
                if not fold.valid:
                    continue
                for timestamp, probabilities in zip(
                    fold.oos_timestamps,
                    fold.oos_filtered_probabilities,
                    strict=True,
                ):
                    state_ids = tuple(f"state_{index:03d}" for index in range(len(probabilities)))
                    dominant_index = max(range(len(probabilities)), key=probabilities.__getitem__)
                    rows.append(
                        {
                            "profile_id": "xetra",
                            "schema_version": "RegimePrediction.v1",
                            "prediction_mode": PredictionMode.WALK_FORWARD_OOS.value,
                            "timestamp": timestamp,
                            "state_ids": list(state_ids),
                            "state_probabilities": list(probabilities),
                            "dominant_state": state_ids[dominant_index],
                            "confidence": probabilities[dominant_index],
                            "entropy": 0.0,
                            "candidate_id": fold.final_configuration.candidate_id,
                            "fold_id": f"outer_fold_{fold.fold_index:03d}",
                            "evaluation_plan_hash": result.policy_hash,
                            "feature_selection_definition_hash": (
                                fold.final_configuration.selection_definition_hash
                            ),
                            "feature_selection_execution_hash": (
                                fold.final_configuration.selection_execution_hash
                            ),
                        }
                    )
            rows.sort(key=lambda row: cast(Any, row)["timestamp"])
            catalog, _snapshot = self._saved_source()
            store.publish(
                build_id=build_id,
                profile_id="xetra",
                prediction_mode=PredictionMode.WALK_FORWARD_OOS,
                rows=rows,
                source_build_id=result.source_build_id,
                source_data_sha256=catalog.lineage.data_sha256,
                source_schema_version=catalog.lineage.schema_version,
                source_feature_version=catalog.lineage.feature_version,
                source_synced_at_utc=catalog.lineage.synced_at_utc,
                feature_contract_hash=catalog.catalog_hash,
                feature_selection_definition_hash=None,
                feature_selection_execution_hash=None,
                created_at_utc=catalog.lineage.synced_at_utc,
            )
        return OOSPublicationOutcome(build_id)

    def register_challenger(
        self,
        profile_id: str,
        production_package: str,
        oos_build_id: str,
    ) -> RegistrationOutcome:
        if profile_id != "xetra":
            raise ValueError("only xetra is supported")
        package = Path(production_package).resolve()
        artifact = load_production_package(package)
        manifest = PredictionStore(self._oos_path).load_manifest("xetra", oos_build_id)
        if manifest.source_build_id != artifact.source_build_id:
            raise ValueError("OOS publication source build differs from production artifact")
        published = publish_production_package(artifact, package)
        self.state_root.mkdir(mode=0o750, parents=True, exist_ok=True)
        (self.state_root / "completed.json").write_text(
            json.dumps(
                {
                    "source_build_id": artifact.source_build_id,
                    "registered_model": published.registered.model_name,
                    "exact_version": published.registered.exact_version,
                    "oos_build_id": oos_build_id,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return RegistrationOutcome(published.registered.exact_version)


__all__ = ["V4LifecycleBackend"]
