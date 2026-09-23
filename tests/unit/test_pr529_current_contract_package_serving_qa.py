from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from market_regime_engine.features.postgres_settings import FeaturePostgresSettings
from market_regime_engine.mlflow_support.model_package import (
    production_artifact_from_payload,
    production_artifact_json,
)
from market_regime_engine.mlflow_support.settings import MLflowSettings
from market_regime_engine.profiles.loader import load_profile_mapping
from tests.unit.mlflow_support.test_model_package import artifact

ROOT = Path(__file__).parents[2]
SOURCE_ROOT = ROOT / "src" / "market_regime_engine"


@pytest.mark.parametrize("version", (1, 2, 3))
def test_historical_profile_versions_are_rejected(version: int) -> None:
    raw = yaml.safe_load((ROOT / "configs/profiles/xetra_v4.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    mutated = copy.deepcopy(raw)
    mutated["profile_config_version"] = version

    with pytest.raises(ValueError, match="version 4"):
        load_profile_mapping(mutated)


@pytest.mark.parametrize(
    "field",
    (
        "feature_selection_definition_hash",
        "feature_selection_execution_hash",
        "data_time_semantics",
        "pca_scaler",
        "source_build_id",
        "source_catalog_hash",
    ),
)
def test_package_missing_current_identity_fields_is_rejected(field: str) -> None:
    payload = copy.deepcopy(json.loads(production_artifact_json(artifact())))
    payload.pop(field)

    with pytest.raises(ValueError):
        production_artifact_from_payload(payload)


def test_current_package_round_trip_preserves_canonical_bytes_and_hash() -> None:
    payload = production_artifact_json(artifact())
    restored = production_artifact_from_payload(json.loads(payload))
    assert production_artifact_json(restored) == payload
    restored_hash = hashlib.sha256(production_artifact_json(restored).encode()).hexdigest()
    original_hash = hashlib.sha256(payload.encode()).hexdigest()
    assert restored_hash == original_hash


def test_removed_environment_names_fail_closed() -> None:
    with pytest.raises(ValueError, match="password or password file is required"):
        FeaturePostgresSettings.from_env(
            {
                "REGIME_FEATURE_PGDATABASE": "macro_loader",
                "REGIME_FEATURE_PGPASSWORD_SECRET_FILE": "/not-used",
            }
        )
    with pytest.raises(ValueError, match="production tracking URI"):
        MLflowSettings.from_environment({"MLFLOW_TRACKING_URI": "file:///tmp/mlruns"})


def test_production_import_path_contains_only_current_package_reader_and_names() -> None:
    production = "\n".join(path.read_text(encoding="utf-8") for path in SOURCE_ROOT.rglob("*.py"))
    removed_symbols = (
        "class PCATwoStageScalerArtifact",
        "def fit_pca_hmm_scaler",
        "RegimeEnginePCATwoStageScaler",
        "REGIME_EVALUATION_CHECKPOINT_ROOT",
        "REGIME_FEATURE_PGPASSWORD_SECRET_FILE",
    )
    assert all(symbol not in production for symbol in removed_symbols)

    allowed_historical_prose = {
        "src/market_regime_engine/contracts/core.py": {"legacy/global"},
        "src/market_regime_engine/features/ports.py": {"compatibility label"},
        "src/market_regime_engine/feature_discovery/contracts.py": {
            "compatibility name",
            "compatibility alias",
        },
        "src/market_regime_engine/feature_discovery/lifecycle_recommendations.py": {
            "lifecycle_deprecated",
        },
    }
    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        allowed = allowed_historical_prose.get(relative, set())
        for line in path.read_text(encoding="utf-8").splitlines():
            lowered = line.casefold()
            if "legacy" in lowered or "compatibility" in lowered:
                assert any(token in lowered for token in allowed), (relative, line)
