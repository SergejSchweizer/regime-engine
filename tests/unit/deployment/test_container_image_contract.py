DOCKERFILE = "Dockerfile"
LOCK = "docker/python-base.lock"
ENTRYPOINT = "scripts/mlflow_entrypoint.sh"
UPGRADE = "scripts/mlflow_db_upgrade.sh"
DOCKERIGNORE = ".dockerignore"
DIGEST = "sha256:23c59390fc717bf09f9336908199a0ae75d9c4264bf296123f94ad772fea3b52"


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_python_base_is_exact_tag_plus_digest_and_matches_lock() -> None:
    dockerfile = _read(DOCKERFILE)
    lock = _read(LOCK)
    assert f"python:3.14.7-slim-bookworm@{DIGEST}" in dockerfile
    assert "image=python:3.14.7-slim-bookworm" in lock
    assert f"index_digest={DIGEST}" in lock
    assert "linux_amd64_manifest_digest=sha256:" in lock
    assert ":latest" not in dockerfile


def test_image_uses_frozen_repo_install_and_exact_runtime_versions() -> None:
    dockerfile = _read(DOCKERFILE)
    assert "COPY uv.lock pyproject.toml README.md ./" in dockerfile
    assert "apt-get install --no-install-recommends --yes build-essential" in dockerfile
    assert "build-essential libstdc++6" in dockerfile
    assert "apt-mark manual libstdc++6" in dockerfile
    assert "CC=g++ CXX=g++ python -m pip install --no-cache-dir -r uv.lock" in dockerfile
    assert "python -m pip install --no-cache-dir --no-deps ." in dockerfile
    assert "apt-get purge --auto-remove --yes build-essential" in dockerfile
    assert 'hmmlearn.__version__ == "0.3.3"' in dockerfile
    assert "from hmmlearn.hmm import GaussianHMM" in dockerfile
    assert 'mlflow.__version__ == "3.15.1"' in dockerfile
    assert 'platform.python_version() == "3.14.7"' in dockerfile
    assert 'io.regime-engine.mlflow-version="3.15.1"' in dockerfile


def test_image_is_non_root_and_contains_inspectable_dependency_sbom() -> None:
    dockerfile = _read(DOCKERFILE)
    assert "USER 10001:10001" in dockerfile
    assert "/opt/regime-engine/build/python-packages.tsv" in dockerfile
    assert "/opt/regime-engine/build/python-sbom.json" in dockerfile
    assert '"bomFormat":"CycloneDX"' in dockerfile
    assert "REGIME_ENGINE_GIT_SHA" in dockerfile
    assert "REGIME_ENGINE_BUILD_TIMESTAMP" in dockerfile


def test_entrypoint_runs_one_mlflow_gunicorn_service_on_5000() -> None:
    script = _read(ENTRYPOINT)
    assert "exec setpriv --reuid 10001 --regid 10001 --init-groups" in script
    assert "mlflow server" in script
    assert "--app-name regime-engine" in script
    assert "--host 0.0.0.0" in script
    assert "--port 5000" in script
    assert '--workers "$MLFLOW_WORKERS"' in script
    assert "--worker-class gthread" in script
    assert "--threads ${MLFLOW_THREADS_PER_WORKER}" in script
    assert "--timeout ${MLFLOW_HTTP_TIMEOUT_SECONDS}" in script
    assert "--graceful-timeout ${MLFLOW_GRACEFUL_TIMEOUT_SECONDS}" in script
    assert "uvicorn" not in script.lower()
    assert "models serve" not in script.lower()
    assert ":5001" not in script
    assert "prometheus" not in script.lower()


def test_entrypoint_defaults_match_exact_runtime_contract() -> None:
    script = _read(ENTRYPOINT)
    for setting in (
        'MLFLOW_WORKERS="${MLFLOW_WORKERS:-4}"',
        'MLFLOW_THREADS_PER_WORKER="${MLFLOW_THREADS_PER_WORKER:-4}"',
        'MLFLOW_HTTP_TIMEOUT_SECONDS="${MLFLOW_HTTP_TIMEOUT_SECONDS:-120}"',
        'MLFLOW_GRACEFUL_TIMEOUT_SECONDS="${MLFLOW_GRACEFUL_TIMEOUT_SECONDS:-30}"',
        'MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts}"',
    ):
        assert setting in script
    dockerfile = _read(DOCKERFILE)
    for setting in ("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1"):
        assert setting in dockerfile


def test_backend_secret_is_required_but_never_echoed() -> None:
    for path in (ENTRYPOINT, UPGRADE):
        script = _read(path)
        assert "MLFLOW_BACKEND_DB_PASSWORD_FILE" in script
        assert 'cat "$MLFLOW_BACKEND_DB_PASSWORD_FILE"' not in script
        assert "set -x" not in script
        assert "printf '%s' \"$value\"" in script
    assert "mlflow db upgrade" not in _read(ENTRYPOINT)
    assert 'exec mlflow db upgrade "$backend_uri"' in _read(UPGRADE)


def test_docker_context_excludes_local_state_and_secrets() -> None:
    ignored = _read(DOCKERIGNORE)
    for value in (".git", ".venv", "tests", "docs", "ops", ".env"):
        assert value in ignored
