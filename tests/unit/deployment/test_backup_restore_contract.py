BACKUP = "scripts/local_mlflow_backup.sh"
VERIFY = "scripts/verify_mlflow_backup.sh"
RESTORE = "scripts/local_mlflow_restore.sh"
UPGRADE = "scripts/verified_mlflow_db_upgrade.sh"
DOC = "docs/ops/backup_restore.md"


def _text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_operation_scripts_are_bash_valid_and_local_compose_only() -> None:
    subprocess = __import__("subprocess")
    for path in (BACKUP, VERIFY, RESTORE, UPGRADE):
        subprocess.run(["bash", "-n", path], check=True)
        text = _text(path)
        assert "docker context show" in text or path == UPGRADE
        assert "docker compose" in text
        assert "ssh://" not in text
        assert "tcp://" not in text
        assert "docker login" not in text
        assert "docker compose push" not in text


def test_backup_is_quiesced_and_manifest_hashes_database_and_artifacts() -> None:
    text = _text(BACKUP)
    assert "scripts/verify_local_compose.sh" in text
    assert "docker compose stop mlflow" in text
    assert "pg_dump --format=custom" in text
    assert "mlflow-artifacts.tar.gz" in text
    assert "application_image_id=$image_id" in text
    assert "repository_git_sha=$git_sha" in text
    assert "mlflow_version=$mlflow_version" in text
    assert "postgres_version=$postgres_version" in text
    assert "database_dump_sha256=$db_sha" in text
    assert "artifact_archive_sha256=$artifact_sha" in text
    assert "scripts/verify_mlflow_backup.sh" in text
    assert "docker compose up -d --no-build mlflow" in text


def test_verifier_rejects_hash_version_provenance_and_secret_failures() -> None:
    text = _text(VERIFY)
    assert '"$(field manifest_version)" == "1"' in text
    assert '"$(field mlflow_version)" == "3.15.1"' in text
    assert "database dump hash mismatch" in text
    assert "artifact archive hash mismatch" in text
    assert "pg_restore --list" in text
    assert "forbidden credential material" in text
    for field in (
        "application_image_id=",
        "repository_git_sha=",
        "mlflow_version=",
        "postgres_version=",
    ):
        assert field in text


def test_restore_requires_explicit_confirmation_exact_image_and_git_then_verifies() -> None:
    text = _text(RESTORE)
    assert "--confirm-destructive-restore" in text
    assert "scripts/verify_mlflow_backup.sh" in text
    assert "exact locally built application image" in text
    assert "exact repository Git SHA" in text
    assert "dropdb --if-exists --force" in text
    assert "pg_restore --exit-on-error" in text
    assert "tar -C /mlflow/artifacts -xzf -" in text
    assert "information_schema.tables" in text
    assert "restore_verified=true" in text
    assert "docker compose up -d --no-build mlflow" in text


def test_database_upgrade_is_impossible_through_wrapper_without_verified_backup() -> None:
    text = _text(UPGRADE)
    verify_position = text.index("scripts/verify_mlflow_backup.sh")
    stop_position = text.index("docker compose stop mlflow")
    upgrade_position = text.index("regime-engine-mlflow-db-upgrade")
    assert verify_position < stop_position < upgrade_position
    assert "docker compose run --rm --no-deps" in text
    assert "docker compose up -d --no-build mlflow" in text
    assert "docker compose build" not in text


def test_operations_doc_has_explicit_local_rebuild_and_verify_before_revoke_rotation() -> None:
    text = _text(DOC)
    assert "scripts/local_mlflow_backup.sh" in text
    assert "scripts/local_mlflow_restore.sh" in text
    assert "scripts/verified_mlflow_db_upgrade.sh" in text
    assert "docker compose build --pull mlflow" in text
    assert "docker compose up -d --no-build" in text
    assert "verify-before-revoke" in text
    assert "REGIME_FEATURE_PGPASSWORD_SECRET_FILE" in text
    assert "MLFLOW_BACKEND_DB_PASSWORD_SECRET_FILE" in text
    assert "old secret" in text.lower()
    assert "new secret" in text.lower()
