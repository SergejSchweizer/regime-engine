from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluations.run_store import (
    DatasetSnapshotKey,
    EvaluationRunKey,
    FileDatasetSnapshotStore,
    FileEvaluationRunStore,
)
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)

NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _lineage() -> SourceLineage:
    return SourceLineage(
        source_dataset="regime_loader",
        source_build_id="build-1",
        data_sha256="a" * 64,
        schema_version=2,
        feature_version=4,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=NOW,
        row_count=2,
        min_timestamp=NOW,
        max_timestamp=NOW + timedelta(days=1),
    )


def _catalog_and_snapshot() -> tuple[FeatureCatalogSnapshot, FeatureSnapshot]:
    lineage = _lineage()
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("feature_a", 1), FeatureCatalogEntry("feature_b", 2)),
    )
    snapshot = FeatureSnapshot(
        lineage,
        ("feature_a", "feature_b"),
        (
            FeatureRow(NOW, (1.0, None)),
            FeatureRow(NOW + timedelta(days=1), (2.0, 3.0)),
        ),
    )
    return catalog.with_materialization(snapshot), snapshot


def _run_key(dataset_key: DatasetSnapshotKey) -> EvaluationRunKey:
    return EvaluationRunKey(
        evaluation_id="global_regime_v4",
        profile_id="xetra",
        profile_config_version=4,
        profile_hash="b" * 64,
        evaluation_contract_version=1,
        evaluation_plan_hash="c" * 64,
        dataset_snapshot_key=dataset_key.key,
        evaluation_cutoff=NOW + timedelta(days=1),
        repository_commit_sha="d" * 40,
        uv_lock_sha256="e" * 64,
        python_version="3.14.7",
    )


def test_dataset_snapshot_key_is_bound_to_materialized_catalog() -> None:
    catalog, snapshot = _catalog_and_snapshot()
    key = DatasetSnapshotKey.from_catalog(catalog)
    assert key.materialized_feature_data_sha256 == snapshot.materialized_feature_data_sha256
    assert len(key.key) == 64

    changed = FeatureSnapshot(
        snapshot.lineage,
        snapshot.feature_names,
        (FeatureRow(NOW, (10.0, None)), FeatureRow(NOW + timedelta(days=1), (2.0, 3.0))),
    )
    changed_key = DatasetSnapshotKey.from_catalog(catalog.with_materialization(changed))
    assert changed_key.key != key.key


def test_file_snapshot_store_round_trip_and_immutable_bytes(tmp_path: Path) -> None:
    catalog, snapshot = _catalog_and_snapshot()
    key = DatasetSnapshotKey.from_catalog(catalog)
    store = FileDatasetSnapshotStore(tmp_path / "snapshots")

    store.finalize(key, snapshot)
    loaded = store.load(key)
    assert loaded == snapshot
    store.finalize(key, snapshot)

    payload_path = tmp_path / "snapshots" / key.key / "snapshot.json"
    payload_path.write_bytes(payload_path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="payload hash"):
        store.load(key)
    with pytest.raises(ValueError, match="immutable"):
        store.finalize(key, snapshot)


@pytest.mark.parametrize(
    "manifest_corruption",
    (
        "missing",
        "invalid_json",
        "object",
        "version",
        "key",
        "hash_missing",
        "hash_invalid",
        "hash_wrong",
    ),
)
def test_file_snapshot_store_rejects_corrupt_manifests(
    tmp_path: Path, manifest_corruption: str
) -> None:
    catalog, snapshot = _catalog_and_snapshot()
    key = DatasetSnapshotKey.from_catalog(catalog)
    root = tmp_path / "snapshots"
    store = FileDatasetSnapshotStore(root)
    store.finalize(key, snapshot)
    manifest_path = root / key.key / "manifest.json"
    if manifest_corruption == "missing":
        manifest_path.unlink()
        message = "missing or incomplete"
    else:
        manifest = json.loads(manifest_path.read_bytes())
        if manifest_corruption == "invalid_json":
            manifest_path.write_bytes(b"{")
            message = "not valid JSON"
        else:
            if manifest_corruption == "object":
                manifest = []
                message = "must be an object"
            elif manifest_corruption == "version":
                manifest["format_version"] = 99
                message = "manifest version"
            elif manifest_corruption == "key":
                manifest["dataset_snapshot_key"] = "wrong"
                message = "key does not match"
            elif manifest_corruption == "hash_missing":
                manifest.pop("payload_sha256")
                message = "hash is missing"
            elif manifest_corruption == "hash_invalid":
                manifest["payload_sha256"] = "invalid"
                message = "payload_sha256"
            else:
                manifest["payload_sha256"] = "0" * 64
                message = "does not match manifest"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
    with pytest.raises(ValueError, match=message):
        store.load(key)


def test_file_snapshot_store_rejects_partial_snapshot(tmp_path: Path) -> None:
    catalog, snapshot = _catalog_and_snapshot()
    key = DatasetSnapshotKey.from_catalog(catalog)
    store = FileDatasetSnapshotStore(tmp_path / "snapshots")
    directory = tmp_path / "snapshots" / key.key
    directory.mkdir(parents=True)
    (directory / "snapshot.json").write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        store.finalize(key, snapshot)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source_dataset", " bad", "source_dataset"),
        ("source_table", "", "source_table"),
        ("source_build_id", " bad", "source_build_id"),
        ("data_sha256", "not-a-hash", "data_sha256"),
        ("schema_version", 0, "source schema"),
        ("feature_version", 0, "source schema"),
        ("data_time_semantics", "historical", "data_time_semantics"),
        ("row_count", -1, "row counts"),
        ("min_timestamp", NOW + timedelta(days=2), "timestamp bounds"),
        ("materialized_row_count", 0, "empty materialization"),
        ("materialized_min_timestamp", None, "non-empty materialization"),
        ("materialized_min_timestamp", NOW + timedelta(days=2), "materialized timestamp"),
    ),
)
def test_dataset_snapshot_key_rejects_invalid_contract_fields(
    field: str, value: object, message: str
) -> None:
    catalog, _ = _catalog_and_snapshot()
    key = DatasetSnapshotKey.from_catalog(catalog)
    with pytest.raises(ValueError, match=message):
        replace(key, **{field: value})


def test_dataset_snapshot_key_rejects_missing_lineage_identity() -> None:
    lineage = replace(_lineage(), row_count=None, min_timestamp=None, max_timestamp=None)
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("feature_a", 1),),
    )
    with pytest.raises(ValueError, match="row count"):
        DatasetSnapshotKey.from_catalog(catalog)

    bound_catalog, _ = _catalog_and_snapshot()
    unbound_catalog = replace(
        bound_catalog,
        materialized_feature_data_sha256=None,
        materialized_row_count=None,
        materialized_min_timestamp=None,
        materialized_max_timestamp=None,
    )
    with pytest.raises(ValueError, match="bound"):
        DatasetSnapshotKey.from_catalog(unbound_catalog)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("profile_config_version", 0, "contract versions"),
        ("evaluation_contract_version", 0, "contract versions"),
        ("profile_hash", "not-a-hash", "profile_hash"),
        ("evaluation_plan_hash", "not-a-hash", "evaluation_plan_hash"),
        ("dataset_snapshot_key", "not-a-hash", "dataset_snapshot_key"),
        ("uv_lock_sha256", "not-a-hash", "uv_lock_sha256"),
        ("repository_commit_sha", "not-a-git-sha", "Git commit SHA"),
        ("python_version", "", "python_version"),
        ("evaluation_cutoff", NOW.replace(tzinfo=None), "evaluation_cutoff"),
    ),
)
def test_evaluation_run_key_rejects_invalid_contract_fields(
    field: str, value: object, message: str
) -> None:
    catalog, _ = _catalog_and_snapshot()
    key = _run_key(DatasetSnapshotKey.from_catalog(catalog))
    with pytest.raises(ValueError, match=message):
        replace(key, **{field: value})
    assert replace(key, evaluation_cutoff=None).as_dict()["evaluation_cutoff"] is None


def _rewrite_snapshot_payload(root: Path, key: DatasetSnapshotKey, payload: bytes) -> None:
    directory = root / key.key
    (directory / "snapshot.json").write_bytes(payload)
    manifest = {
        "format_version": 1,
        "dataset_snapshot_key": key.key,
        "payload_sha256": sha256(payload).hexdigest(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("invalid_json", "not valid JSON"),
        ("not_an_object", "must be an object"),
        ("format_version", "format version"),
        ("dataset_key", "key does not match"),
        ("identity", "identity does not match"),
        ("lineage", "lineage is missing"),
        ("row_count", "row_count is invalid"),
        ("versions", "versions are invalid"),
        ("feature_names", "features or rows are malformed"),
        ("feature_name_type", "feature names are malformed"),
        ("row", "row is malformed"),
        ("float", "invalid float"),
        ("float_type", "not a float encoding"),
        ("timestamp", "not a valid ISO timestamp"),
        ("timestamp_missing", "must be an ISO timestamp"),
        ("skipped_count", "skipped row count is invalid"),
        ("lineage_identity", "lineage does not match"),
    ),
)
def test_file_snapshot_store_rejects_corrupt_payloads(
    tmp_path: Path, corruption: str, message: str
) -> None:
    catalog, snapshot = _catalog_and_snapshot()
    key = DatasetSnapshotKey.from_catalog(catalog)
    root = tmp_path / "snapshots"
    store = FileDatasetSnapshotStore(root)
    store.finalize(key, snapshot)
    if corruption == "invalid_json":
        payload = b"{"  # The manifest is rewritten so decoding is exercised.
    else:
        document = json.loads((root / key.key / "snapshot.json").read_bytes())
        if corruption == "not_an_object":
            payload = b"[]"
        else:
            if corruption == "format_version":
                document["format_version"] = 99
            elif corruption == "dataset_key":
                document["dataset_snapshot_key"] = "wrong"
            elif corruption == "identity":
                document["identity"] = {}
            elif corruption == "lineage":
                document["lineage"] = None
            elif corruption == "row_count":
                document["lineage"]["row_count"] = "2"
            elif corruption == "versions":
                document["lineage"]["schema_version"] = 0
            elif corruption == "feature_names":
                document["feature_names"] = "not-a-list"
            elif corruption == "feature_name_type":
                document["feature_names"][0] = 1
            elif corruption == "row":
                document["rows"][0] = None
            elif corruption == "float":
                document["rows"][0]["values"][0] = "not-a-float"
            elif corruption == "float_type":
                document["rows"][0]["values"][0] = 1
            elif corruption == "timestamp":
                document["rows"][0]["timestamp"] = "not-a-timestamp"
            elif corruption == "timestamp_missing":
                document["rows"][0].pop("timestamp")
            elif corruption == "skipped_count":
                document["skipped_incomplete_row_count"] = -1
            elif corruption == "lineage_identity":
                document["lineage"]["source_dataset"] = "other"
            payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
    _rewrite_snapshot_payload(root, key, payload)
    with pytest.raises(ValueError, match=message):
        store.load(key)


def test_file_snapshot_store_rejects_mismatched_materialization_and_lineage_on_finalize(
    tmp_path: Path,
) -> None:
    catalog, snapshot = _catalog_and_snapshot()
    key = DatasetSnapshotKey.from_catalog(catalog)
    store = FileDatasetSnapshotStore(tmp_path / "snapshots")
    wrong_lineage = replace(snapshot.lineage, source_dataset="other")
    with pytest.raises(ValueError, match="lineage"):
        store.finalize(key, replace(snapshot, lineage=wrong_lineage))

    wrong_rows = (
        FeatureRow(NOW, (10.0, None)),
        FeatureRow(NOW + timedelta(days=1), (2.0, 3.0)),
    )
    wrong_snapshot = FeatureSnapshot(snapshot.lineage, snapshot.feature_names, wrong_rows)
    with pytest.raises(ValueError, match="matrix hash"):
        store.finalize(key, wrong_snapshot)


def test_file_snapshot_store_rejects_payload_identity_mismatches() -> None:
    catalog, original = _catalog_and_snapshot()
    original_key = DatasetSnapshotKey.from_catalog(catalog)
    one_row = FeatureSnapshot(
        original.lineage,
        original.feature_names,
        (FeatureRow(NOW, (1.0, None)),),
    )
    changed_rows = FeatureSnapshot(
        original.lineage,
        original.feature_names,
        (
            FeatureRow(NOW, (10.0, None)),
            FeatureRow(NOW + timedelta(days=1), (2.0, 3.0)),
        ),
    )
    with pytest.raises(ValueError, match="matrix hash"):
        FileDatasetSnapshotStore._decode(
            original_key, FileDatasetSnapshotStore._payload(original_key, changed_rows)
        )

    one_row_key = replace(
        original_key,
        materialized_feature_data_sha256=one_row.materialized_feature_data_sha256,
    )
    with pytest.raises(ValueError, match="row count"):
        FileDatasetSnapshotStore._decode(
            one_row_key, FileDatasetSnapshotStore._payload(one_row_key, one_row)
        )

    early_key = replace(original_key, materialized_min_timestamp=NOW - timedelta(days=1))
    with pytest.raises(ValueError, match="minimum"):
        FileDatasetSnapshotStore._decode(
            early_key, FileDatasetSnapshotStore._payload(early_key, original)
        )

    late_key = replace(original_key, materialized_max_timestamp=NOW + timedelta(days=2))
    with pytest.raises(ValueError, match="maximum"):
        FileDatasetSnapshotStore._decode(
            late_key, FileDatasetSnapshotStore._payload(late_key, original)
        )


def test_file_run_store_claims_reuses_and_finalizes_atomic_units(tmp_path: Path) -> None:
    catalog, _ = _catalog_and_snapshot()
    dataset_key = DatasetSnapshotKey.from_catalog(catalog)
    key = _run_key(dataset_key)
    store = FileEvaluationRunStore(tmp_path / "runs")
    input_hash = sha256(b"outer-fold-1-input").hexdigest()

    assert store.open_run(key).status == "RUNNING"
    assert store.claim_work_unit(key, "outer-fold-1", input_hash)
    assert not store.claim_work_unit(key, "outer-fold-1", input_hash)
    payload = b"outer-fold-result"
    store.complete_work_unit(key, "outer-fold-1", input_hash, payload)
    assert store.load_completed_work_unit(key, "outer-fold-1", input_hash) == payload
    store.complete_work_unit(key, "outer-fold-1", input_hash, payload)
    with pytest.raises(ValueError, match="immutable"):
        store.complete_work_unit(key, "outer-fold-1", input_hash, b"different")
    with pytest.raises(ValueError, match="input identity"):
        store.claim_work_unit(key, "outer-fold-1", "f" * 64)

    final_payload = b"final-evaluation-result"
    root_hash = sha256(b"canonical-result").hexdigest()
    store.complete_run(key, root_hash, final_payload)
    assert store.open_run(key).status == "COMPLETE"
    assert store.load_completed_run(key) == final_payload
    assert not store.claim_work_unit(key, "outer-fold-2", "1" * 64)
    store.complete_run(key, root_hash, final_payload)
    with pytest.raises(ValueError, match="immutable"):
        store.complete_run(key, "b" * 64, final_payload)
    with pytest.raises(ValueError, match="immutable"):
        store.complete_run(key, root_hash, b"different")


def test_file_run_store_reclaims_expired_units_and_fails_closed(tmp_path: Path) -> None:
    catalog, _ = _catalog_and_snapshot()
    key = _run_key(DatasetSnapshotKey.from_catalog(catalog))
    store = FileEvaluationRunStore(tmp_path / "runs", lease_seconds=2)
    with pytest.raises(ValueError, match="opened"):
        store.claim_work_unit(key, "unit", "1" * 64)
    with pytest.raises(ValueError, match="missing"):
        store.load_completed_work_unit(key, "unit", "1" * 64)
    store.open_run(key)
    with pytest.raises(ValueError, match="lease_seconds"):
        store.claim_work_unit(key, "unit", "1" * 64, lease_seconds=0)
    with pytest.raises(ValueError, match="non-empty"):
        store.complete_work_unit(key, "unit", "1" * 64, b"")
    with pytest.raises(ValueError, match="claimed"):
        store.complete_work_unit(key, "unit", "1" * 64, b"payload")

    assert store.claim_work_unit(key, "unit", "1" * 64, lease_seconds=1)
    assert not store.claim_work_unit(key, "unit", "1" * 64)
    ledger_path = tmp_path / "runs" / key.key / "run.json"
    ledger = json.loads(ledger_path.read_bytes())
    ledger["work_units"]["unit"]["lease_expires_at"] = (NOW - timedelta(seconds=1)).isoformat()
    ledger_path.write_text(
        json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    assert store.claim_work_unit(key, "unit", "1" * 64)
    with pytest.raises(ValueError, match="terminal"):
        store.complete_run(key, "a" * 64, b"result")

    store.complete_work_unit(key, "unit", "1" * 64, b"payload")
    unit_path = next((tmp_path / "runs" / key.key / "units").glob("*.payload"))
    unit_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="payload hash"):
        store.load_completed_work_unit(key, "unit", "1" * 64)


def test_file_run_store_handles_domain_invalid_and_final_payload_corruption(
    tmp_path: Path,
) -> None:
    catalog, _ = _catalog_and_snapshot()
    key = _run_key(DatasetSnapshotKey.from_catalog(catalog))
    store = FileEvaluationRunStore(tmp_path / "runs")
    store.open_run(key)
    input_hash = "1" * 64
    assert store.claim_work_unit(key, "invalid", input_hash)
    ledger_path = tmp_path / "runs" / key.key / "run.json"
    ledger = json.loads(ledger_path.read_bytes())
    ledger["work_units"]["invalid"].update({"status": "DOMAIN_INVALID", "lease_expires_at": None})
    ledger_path.write_text(
        json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    assert not store.claim_work_unit(key, "invalid", input_hash)
    assert store.load_completed_work_unit(key, "invalid", input_hash) is None
    store.complete_run(key, "a" * 64, b"result")
    final_path = tmp_path / "runs" / key.key / "final.payload"
    final_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="payload hash"):
        store.load_completed_run(key)


def test_file_run_store_rejects_invalid_ledgers_and_payload_metadata(tmp_path: Path) -> None:
    catalog, _ = _catalog_and_snapshot()
    key = _run_key(DatasetSnapshotKey.from_catalog(catalog))
    store = FileEvaluationRunStore(tmp_path / "runs")
    store.open_run(key)
    ledger_path = tmp_path / "runs" / key.key / "run.json"

    ledger_path.write_bytes(b"{")
    with pytest.raises(ValueError, match="not valid JSON"):
        store.open_run(key)
    ledger_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        store.open_run(key)

    ledger_path.write_text(
        json.dumps(FileEvaluationRunStore._new_ledger(key), sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    ledger = FileEvaluationRunStore._new_ledger(key)
    ledger["status"] = "BROKEN"
    ledger_path.write_text(
        json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="status is invalid"):
        store.open_run(key)


def test_file_run_store_requires_terminal_units_and_rejects_corruption(tmp_path: Path) -> None:
    catalog, _ = _catalog_and_snapshot()
    key = _run_key(DatasetSnapshotKey.from_catalog(catalog))
    store = FileEvaluationRunStore(tmp_path / "runs")
    store.open_run(key)
    with pytest.raises(ValueError, match="terminal"):
        store.complete_run(key, "a" * 64, b"result")

    ledger_path = tmp_path / "runs" / key.key / "run.json"
    ledger_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="ledger version"):
        store.open_run(key)
