from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_regime_engine.evaluation_runs.contracts import (
    DatasetSnapshotIdentity,
    EvaluationRunIdentity,
    WorkUnitIdentity,
    canonical_json,
    content_hash,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def dataset_identity() -> DatasetSnapshotIdentity:
    return DatasetSnapshotIdentity(
        source_dataset="gold",
        source_build_id="build-1",
        data_sha256="a" * 64,
        schema_version=1,
        feature_version=1,
        data_time_semantics="current_vintage_observation_day",
        row_count=2,
        min_timestamp=NOW,
        max_timestamp=NOW,
        source_catalog_hash="b" * 64,
        materialized_feature_data_sha256="c" * 64,
        materialized_row_count=2,
        materialized_min_timestamp=NOW,
        materialized_max_timestamp=NOW,
        source_table="regime_loader.regime_features_daily",
    )


def run_identity() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        evaluation_id="global_regime_v4",
        profile_id="xetra",
        profile_config_version=4,
        profile_hash="d" * 64,
        evaluation_contract_version=1,
        evaluation_plan_hash="e" * 64,
        dataset_snapshot_key=dataset_identity().key,
        evaluation_cutoff=NOW,
        repository_commit_sha="f" * 40,
        uv_lock_sha256="1" * 64,
        python_version="3.14.7",
    )


def test_identities_are_canonical_and_independent_of_operational_metadata() -> None:
    identity = run_identity()
    assert identity.key == content_hash(identity.as_dict())
    assert canonical_json(identity.as_dict()).endswith(b"\n")
    assert replace_run(identity, repository_commit_sha="0" * 40).key != identity.key

    unit = WorkUnitIdentity(
        evaluation_run_key=identity.key,
        unit_type="candidate_seed",
        coordinates=(("candidate_id", "gaussian_hmm_k2_full"), ("seed", "11")),
        parent_payload_hashes=("2" * 64,),
        unit_parameters=(("state_count", "2"),),
    )
    changed_parent = WorkUnitIdentity(
        evaluation_run_key=identity.key,
        unit_type=unit.unit_type,
        coordinates=unit.coordinates,
        parent_payload_hashes=("3" * 64,),
        unit_parameters=unit.unit_parameters,
    )
    assert unit.key == "candidate_seed/candidate_id=gaussian_hmm_k2_full/seed=11"
    assert unit.work_unit_input_hash != changed_parent.work_unit_input_hash


def replace_run(identity: EvaluationRunIdentity, **changes: object) -> EvaluationRunIdentity:
    values = {field: getattr(identity, field) for field in identity.__dataclass_fields__}
    values.update(changes)
    return EvaluationRunIdentity(**values)


def test_work_unit_coordinates_and_parameters_must_be_sorted() -> None:
    identity = run_identity()
    with pytest.raises(ValueError, match="canonically sorted"):
        WorkUnitIdentity(
            evaluation_run_key=identity.key,
            unit_type="unit",
            coordinates=(("z", "1"), ("a", "2")),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        WorkUnitIdentity(
            evaluation_run_key=identity.key,
            unit_type="unit",
            parent_payload_hashes=("not-a-hash",),
        )
