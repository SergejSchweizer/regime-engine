# Local two-service deployment

PR-061 defines the production Compose topology for `regime-engine`. The deployment host owns a local checkout of this repository and talks only to its local Docker daemon over a Unix socket. Remote Docker contexts/builders, Docker Swarm, Kubernetes and an application-image registry are outside the MVP contract.

## Topology

`compose.yaml` declares exactly two services:

- `mlflow`: the repository-built `regime-engine-mlflow:local` image, publishing only `5000:5000`;
- `mlflow-postgres`: private MLflow metadata PostgreSQL, with no host port.

The feature PostgreSQL at `10.10.1.3:54321` is external and never becomes a Compose service. MLflow backend metadata and feature-source credentials/settings are separate namespaces.

Persistent MLflow state uses explicit host bind mounts below
`/volume2/docker/mlflow`: PostgreSQL data is in `postgres/` and MLflow
artifacts are in `artifacts/`. These paths are local-host storage, not a remote
Docker volume or an application-image registry. Existing named-volume state
must be moved only with a verified backup and the documented storage migration
procedure before this topology is first started.

## Immutable upstream images

The application Dockerfile pins `python:3.14.7-slim-bookworm` through `docker/python-base.lock`. Compose pins the official `postgres:18.6-alpine` input through `docker/postgres-backend.lock` and the exact multi-platform index digest. PostgreSQL 18 uses `/var/lib/postgresql` as the official volume target; the named backend volume is mounted there.

The repository-owned application image is never pulled from or pushed to Docker Hub, GHCR or another registry. Its canonical tag is exactly `regime-engine-mlflow:local`, with `pull_policy: never`.

## Required local configuration

Copy `.env.example` to `.env` on the deployment host and replace placeholders locally. Do not commit `.env` or password files. Two Docker secrets are mounted:

- MLflow backend password -> `/run/secrets/mlflow_backend_password`;
- feature PostgreSQL password -> `/run/secrets/regime_feature_password`.

Secret files remain mode `0600`. Because Compose mounts file secrets as host bind
mounts, the container entrypoint starts as root solely to read the backend secret
and immediately launches MLflow as its configured non-root runtime UID/GID
(`10001:10001` by default); the application process does not run as root.

Some NAS filesystems enforce host ACLs for bind mounts independently of POSIX
mode bits. On such hosts set `MLFLOW_RUNTIME_UID`, `MLFLOW_RUNTIME_GID`,
`MLFLOW_POSTGRES_RUNTIME_UID`, and `MLFLOW_POSTGRES_RUNTIME_GID` in the ignored
local `.env` to the UID/GID that owns `/volume2/docker/mlflow` (for this host,
`1016` and `100`). This keeps database and artifact writes non-root while
leaving the portable image defaults unchanged.

`REGIME_FEATURE_PGDATABASE` and the MLflow backend database/user are required runtime values with no guessed defaults. The configured trusted-LAN feature server does not offer TLS, so feature transport is fixed to `sslmode=disable`; do not silently change the transport mode.

The trusted-LAN host/origin defaults contain only `10.10.1.3`, localhost and loopback, with their explicit `:5000` Host-header forms. Do not use wildcard Host/CORS values. Network/firewall policy must keep port 5000 private to trusted clients/operators.

## Build

The only supported production application-image build is:

```sh
scripts/local_compose_build.sh
```

The wrapper verifies that Docker is reached through a local Unix socket, rejects remote Buildx endpoints, records the current repository Git SHA and UTC build timestamp as image build arguments, and executes exactly:

```sh
docker compose build --pull mlflow
```

The build may pull the pinned Python base input. It does not push or pull a repository-owned application image.

## Start

Normal production startup is:

```sh
scripts/local_compose_up.sh
```

The wrapper first requires `regime-engine-mlflow:local` to exist locally, then executes exactly:

```sh
docker compose up -d --no-build
```

Startup therefore never rebuilds the application image and cannot silently fetch it from a registry. The pinned official PostgreSQL backend image may be pulled only when it is absent locally.

Normal startup performs no MLflow schema migration. Migration is the explicit
PR-065 guarded operation in [the backup/restore guide](ops/backup_restore.md):
operators must first create and verify a backup, then run only
`scripts/verified_mlflow_db_upgrade.sh <verified-backup-directory>`. It must
never be combined with an implicit image rebuild.

## Verify

After startup run:

```sh
scripts/verify_local_compose.sh
```

Verification fails closed unless the project is `regime-engine`, the running service set is exactly `mlflow` plus `mlflow-postgres`, the application container uses the local canonical tag, its actual Docker image ID has no registry RepoDigest, the embedded Git revision equals the local checkout, MLflow is exactly `3.15.1`, only host port 5000 is published, and the backend database remains private. It prints only non-secret deployment evidence: project, service set, local image ID, Git SHA, MLflow version, build timestamp and port mapping.

## Stop

```sh
scripts/local_compose_down.sh
```

The default down path preserves named backend/artifact volumes. Deleting volumes is an explicit destructive operator action and is not performed by the wrapper.

## Runtime resource contract

The defaults are four Gunicorn workers and four gthread threads per worker. Each worker may own a feature PostgreSQL pool of at most four connections, and the global feature-connection budget is sixteen, so the required inequality is `4 * 4 <= 16`. Replay admission remains one per worker, yielding at most four admitted replay requests under the exact default topology.

All BLAS thread counts are pinned to one. Replay bounds, cache TTL and source/model staleness thresholds are passed explicitly from Compose and match the contract-owner values. No second serving port, proxy, Prometheus exporter or separate Python serving environment is introduced.
