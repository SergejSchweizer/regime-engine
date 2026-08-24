# Local MLflow / regime-engine container image

PR-032 defines the single repository-owned application image used by the MVP. The canonical local tag is `regime-engine-mlflow:local`. The image is built only from this repository's root build context on the deployment host; it is not pulled from or pushed to an application-image registry.

## Immutable upstream base

The Dockerfile pins `python:3.14.7-slim-bookworm` by the official multi-platform index digest recorded in `docker/python-base.lock`. The lock also records the Linux/amd64 manifest digest for operator inspection. A tag without its digest is not an acceptable production input.

## Build provenance

The build must supply the repository Git SHA and UTC build timestamp as Docker build arguments:

```sh
docker build \
  --build-arg REGIME_ENGINE_GIT_SHA="$(git rev-parse HEAD)" \
  --build-arg REGIME_ENGINE_BUILD_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t regime-engine-mlflow:local \
  .
```

Those values are written to OCI labels. The image also records Python `3.14.7` and MLflow `3.15.1` labels. `/opt/regime-engine/build/python-packages.tsv` and `/opt/regime-engine/build/python-sbom.json` provide sorted, local dependency/SBOM evidence for the installed image.

PR-061 owns the production Compose build command (`docker compose build --pull mlflow`) and startup command (`docker compose up -d --no-build`). This PR does not introduce a remote image registry, remote builder, or implicit startup build.

## Runtime process

The image runs as UID/GID `10001` and starts exactly one MLflow server process. The custom application is selected with `--app-name regime-engine`; MLflow is forced through Gunicorn `gthread` with the contract defaults:

- `MLFLOW_WORKERS=4`
- `MLFLOW_THREADS_PER_WORKER=4`
- `MLFLOW_HTTP_TIMEOUT_SECONDS=120`
- `MLFLOW_GRACEFUL_TIMEOUT_SECONDS=30`
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

The process binds `0.0.0.0:5000`. There is no Uvicorn application, second model-serving process, port `5001`, reverse proxy, or Prometheus exporter.

## MLflow backend and artifacts

The MLflow backend database is the private Compose service `mlflow-postgres:5432`. The following runtime values are mandatory and deliberately have no credential defaults:

```text
MLFLOW_BACKEND_DB_NAME
MLFLOW_BACKEND_DB_USER
MLFLOW_BACKEND_DB_PASSWORD_FILE
```

The password is read from the mounted Docker secret file and URI-escaped before the PostgreSQL backend URI is constructed. For Compose file secrets, the entrypoint begins as root only to read the mode-`0600` secret and immediately drops the MLflow process to UID/GID `10001` using `setpriv`. The script does not enable shell tracing or print the credential-bearing URI. `MLFLOW_ARTIFACT_ROOT` defaults to `/mlflow/artifacts`.

Feature PostgreSQL credentials remain a separate runtime concern and must use the `REGIME_FEATURE_*` variables defined by `DATA_SOURCE.md`; they are never folded into the MLflow backend variables.

## Explicit database migration

Normal image startup never executes `mlflow db upgrade`. Backend schema migration is an explicit one-shot operator action:

```sh
docker compose exec -T mlflow /usr/local/bin/regime-engine-mlflow-db-upgrade
```

The service must be quiesced and backed up according to the later operational migration/restore contract before this command is used. The migration script consumes the same backend database secret file without printing its contents.

## Local verification

After a local build, operators can inspect provenance and the dependency evidence without contacting an external service:

```sh
docker image inspect regime-engine-mlflow:local
docker run --rm --entrypoint cat regime-engine-mlflow:local /opt/regime-engine/build/python-packages.tsv
docker run --rm --entrypoint cat regime-engine-mlflow:local /opt/regime-engine/build/python-sbom.json
```

The application image tag is mutable local state; deployment evidence must therefore record the actual local Docker image ID together with the Git SHA rather than treating the tag as immutable provenance.
