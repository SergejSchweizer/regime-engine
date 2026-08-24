# MLflow backend backup, restore, migration and secret rotation

All procedures in this document run from the clean deployment checkout on the host that owns the local Unix-socket Docker daemon. They operate on the `regime-engine` Compose project only. A remote Docker host, registry-hosted application image, Swarm and Kubernetes are not part of the contract.

## Verified backup

Create a new backup directory on local durable storage:

```sh
scripts/local_mlflow_backup.sh /srv/backups/regime-engine/2026-08-24T143000Z
```

The script first runs `scripts/verify_local_compose.sh`, records the running local application image ID, repository Git SHA, MLflow version and PostgreSQL version, then stops only the `mlflow` service. With writers quiesced it creates a custom-format `pg_dump` of the private MLflow backend and a gzip archive of `/mlflow/artifacts`. `manifest.env` contains SHA-256 hashes for both payloads and no credentials. `scripts/verify_mlflow_backup.sh` checks both hashes, archive readability, dump readability, version/provenance fields and absence of credential-like manifest fields before the application service is restarted with `docker compose up -d --no-build mlflow`.

A backup is not considered successful merely because files exist. Verification must return `backup_verified=true`.

## Restore

Restore is destructive and requires the exact local image ID and Git SHA recorded in the backup:

```sh
scripts/local_mlflow_restore.sh /srv/backups/regime-engine/2026-08-24T143000Z \
  --confirm-destructive-restore
```

The script re-verifies the backup before stopping MLflow. It recreates only the private MLflow backend database, restores it with `pg_restore --exit-on-error`, clears and restores the artifact volume from the matching archive, verifies that the restored public metadata schema is non-empty and that the artifact path is readable, then restarts with `--no-build` and re-runs local Compose verification. The external feature PostgreSQL is never mutated or restored by this procedure.

## MLflow database migration

Normal Compose startup never performs `mlflow db upgrade`. Before any explicit database migration, create and verify a backup from the exact currently running image/check-out. Then use only the guarded wrapper:

```sh
scripts/verified_mlflow_db_upgrade.sh /srv/backups/regime-engine/2026-08-24T143000Z
```

The wrapper refuses to reach the migration command until `scripts/verify_mlflow_backup.sh` succeeds. It quiesces MLflow, executes the image-owned `/usr/local/bin/regime-engine-mlflow-db-upgrade` as a one-shot local Compose container, restarts with `--no-build`, and verifies the deployment.

If an application or dependency change requires a new image, rebuild it explicitly on the deployment host first:

```sh
scripts/local_compose_build.sh
# exact underlying build command: docker compose build --pull mlflow
scripts/local_compose_up.sh
# exact normal start command: docker compose up -d --no-build
```

Never combine migration with an implicit image rebuild.

## Backend secret rotation

Backend rotation is **verify-before-revoke**. Keep the old secret file until the new credential is proven end to end.

1. Create a new secret file outside the repository with mode `0600`; do not replace the old file yet.
2. From `mlflow-postgres`, change only the configured MLflow backend role password using a local administrator session. Do not print either password.
3. Temporarily point `MLFLOW_BACKEND_DB_PASSWORD_SECRET_FILE` at the new secret file and recreate only `mlflow` with `docker compose up -d --no-build --force-recreate mlflow`.
4. Run `scripts/verify_local_compose.sh` and the PR-034 unified MLflow smoke. Tracking and registry reads/writes must pass.
5. Only after verification succeeds, make the new secret-file path permanent and securely revoke/delete the old secret. If verification fails, restore the old database password from the retained old secret and recreate MLflow against the old file.

The PostgreSQL backend container's `POSTGRES_PASSWORD_FILE` is an initialization input for its data directory; changing that file alone does not rotate an already initialized database role.

## Feature PostgreSQL secret rotation

The external feature database is owned by `regime-loader` infrastructure, so this repository never changes that SQL role itself. Rotation is again **verify-before-revoke**:

1. Have the external database operator issue a new password for the existing quoted role `"regime-engine"` while retaining a rollback path for the old secret.
2. Store the new value in a new `0600` file and point `REGIME_FEATURE_PGPASSWORD_SECRET_FILE` at that file locally.
3. Before revoking the old secret, run the external read-only feature-PG smoke with the new credential and `sslmode=require`.
4. Recreate only `mlflow` with `docker compose up -d --no-build --force-recreate mlflow`, then run local Compose verification and the optional `xetra` latest smoke.
5. Only after all checks pass may the external operator revoke the old secret. On failure, restore the old secret path before any revocation.

Neither rotation procedure writes credentials into `.env`, logs, model artifacts, backup manifests or API responses.
