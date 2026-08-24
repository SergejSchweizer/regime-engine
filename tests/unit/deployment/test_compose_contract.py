COMPOSE = "compose.yaml"
LOCK = "docker/postgres-backend.lock"
ENV_EXAMPLE = ".env.example"
BUILD = "scripts/local_compose_build.sh"
UP = "scripts/local_compose_up.sh"
DOWN = "scripts/local_compose_down.sh"
VERIFY = "scripts/verify_local_compose.sh"
DEPLOYMENT_DOC = "docs/deployment.md"
POSTGRES_DIGEST = "sha256:432b3b824c0769275ec9b0947736ef8b376d6997bcaa9de29818f613819c2feb"
AMD64_DIGEST = "sha256:63bdc97d67b5133bf0e5ebd500bec6d046fa851dc81340d838f0347e616107e8"


def _text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _compose() -> dict[str, object]:
    value = __import__("yaml").safe_load(_text(COMPOSE))
    assert isinstance(value, dict)
    return value


def test_compose_has_exact_project_services_and_public_port() -> None:
    config = _compose()
    assert config["name"] == "regime-engine"
    services = config["services"]
    assert isinstance(services, dict)
    assert set(services) == {"mlflow", "mlflow-postgres"}
    mlflow = services["mlflow"]
    backend = services["mlflow-postgres"]
    assert mlflow["ports"] == ["5000:5000"]
    assert "ports" not in backend


def test_mlflow_is_local_build_only_and_never_registry_pulled() -> None:
    services = _compose()["services"]
    mlflow = services["mlflow"]
    assert mlflow["image"] == "regime-engine-mlflow:local"
    assert mlflow["pull_policy"] == "never"
    assert mlflow["build"]["context"] == "."
    assert mlflow["build"]["dockerfile"] == "Dockerfile"
    text = _text(COMPOSE).lower()
    banned = (
        "ghcr.io",
        "docker.io/sergejschweizer",
        "5001",
        "prometheus",
        "nginx",
        "traefik",
    )
    for token in banned:
        assert token not in text


def test_postgres_backend_is_official_digest_pinned_and_persistent() -> None:
    services = _compose()["services"]
    backend = services["mlflow-postgres"]
    assert backend["image"] == f"postgres:18.6-alpine@{POSTGRES_DIGEST}"
    assert backend["pull_policy"] == "missing"
    assert "mlflow_backend_data:/var/lib/postgresql" in backend["volumes"]
    lock = _text(LOCK)
    assert "image=postgres:18.6-alpine" in lock
    assert f"index_digest={POSTGRES_DIGEST}" in lock
    assert f"linux_amd64_manifest_digest={AMD64_DIGEST}" in lock


def test_backend_and_feature_database_settings_are_strictly_namespaced() -> None:
    services = _compose()["services"]
    environment = services["mlflow"]["environment"]
    assert environment["MLFLOW_BACKEND_DB_PASSWORD_FILE"] == (
        "/run/secrets/mlflow_backend_password"
    )
    assert environment["REGIME_FEATURE_PGHOST"] == "${REGIME_FEATURE_PGHOST:-10.10.1.3}"
    assert environment["REGIME_FEATURE_PGPORT"] == "${REGIME_FEATURE_PGPORT:-54321}"
    assert environment["REGIME_FEATURE_PGUSER"] == ("${REGIME_FEATURE_PGUSER:-regime-engine}")
    assert environment["REGIME_FEATURE_PGPASSWORD_FILE"] == ("/run/secrets/regime_feature_password")
    assert environment["REGIME_FEATURE_PGSSLMODE"] == "${REGIME_FEATURE_PGSSLMODE:-require}"
    generic_pg = {"PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"}
    assert not (generic_pg & set(environment))


def test_exact_runtime_defaults_and_trusted_lan_policy_are_composed() -> None:
    environment = _compose()["services"]["mlflow"]["environment"]
    expected = {
        "MLFLOW_WORKERS": "${MLFLOW_WORKERS:-4}",
        "MLFLOW_THREADS_PER_WORKER": "${MLFLOW_THREADS_PER_WORKER:-4}",
        "MLFLOW_HTTP_TIMEOUT_SECONDS": "${MLFLOW_HTTP_TIMEOUT_SECONDS:-120}",
        "MLFLOW_GRACEFUL_TIMEOUT_SECONDS": "${MLFLOW_GRACEFUL_TIMEOUT_SECONDS:-30}",
        "REGIME_MODEL_ALIAS_CACHE_TTL_SECONDS": "${REGIME_MODEL_ALIAS_CACHE_TTL_SECONDS:-30}",
        "REGIME_PG_POOL_MIN_SIZE": "${REGIME_PG_POOL_MIN_SIZE:-1}",
        "REGIME_PG_POOL_MAX_SIZE": "${REGIME_PG_POOL_MAX_SIZE:-4}",
        "REGIME_PG_ACQUIRE_TIMEOUT_SECONDS": "${REGIME_PG_ACQUIRE_TIMEOUT_SECONDS:-5}",
        "REGIME_PG_STATEMENT_TIMEOUT_SECONDS": ("${REGIME_PG_STATEMENT_TIMEOUT_SECONDS:-30}"),
        "REGIME_FEATURE_PG_CONNECTION_BUDGET": ("${REGIME_FEATURE_PG_CONNECTION_BUDGET:-16}"),
        "REGIME_REPLAY_MAX_ROWS": "${REGIME_REPLAY_MAX_ROWS:-10000}",
        "REGIME_REPLAY_MAX_INTERNAL_ROWS": "${REGIME_REPLAY_MAX_INTERNAL_ROWS:-15000}",
        "REGIME_REPLAY_MAX_RANGE_DAYS": "${REGIME_REPLAY_MAX_RANGE_DAYS:-14610}",
        "REGIME_REPLAY_TIMEOUT_SECONDS": "${REGIME_REPLAY_TIMEOUT_SECONDS:-60}",
        "REGIME_REPLAY_MAX_RESPONSE_BYTES": ("${REGIME_REPLAY_MAX_RESPONSE_BYTES:-26214400}"),
        "REGIME_REPLAY_MAX_CONCURRENCY_PER_WORKER": (
            "${REGIME_REPLAY_MAX_CONCURRENCY_PER_WORKER:-1}"
        ),
        "REGIME_SOURCE_STALE_WARN_DAYS": "${REGIME_SOURCE_STALE_WARN_DAYS:-4}",
        "REGIME_SOURCE_STALE_FAIL_DAYS": "${REGIME_SOURCE_STALE_FAIL_DAYS:-7}",
        "REGIME_MODEL_STALE_WARN_DAYS": "${REGIME_MODEL_STALE_WARN_DAYS:-14}",
        "REGIME_MODEL_STALE_FAIL_DAYS": "${REGIME_MODEL_STALE_FAIL_DAYS:-35}",
    }
    for name, value in expected.items():
        assert environment[name] == value
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["MLFLOW_SERVER_ALLOWED_HOSTS"] == (
        "${MLFLOW_SERVER_ALLOWED_HOSTS:-10.10.1.3,10.10.1.3:5000,localhost,localhost:5000,"
        "127.0.0.1,127.0.0.1:5000}"
    )
    assert "*" not in environment["MLFLOW_SERVER_ALLOWED_HOSTS"]
    assert "*" not in environment["MLFLOW_SERVER_CORS_ALLOWED_ORIGINS"]
    workers = int(expected["MLFLOW_WORKERS"].removesuffix("}").split(":-")[-1])
    pool_max = int(expected["REGIME_PG_POOL_MAX_SIZE"].removesuffix("}").split(":-")[-1])
    budget = int(expected["REGIME_FEATURE_PG_CONNECTION_BUDGET"].removesuffix("}").split(":-")[-1])
    assert workers * pool_max <= budget


def test_secrets_are_file_backed_and_env_example_contains_placeholders_only() -> None:
    config = _compose()
    secrets = config["secrets"]
    backend_secret = secrets["mlflow_backend_password"]["file"]
    feature_secret = secrets["regime_feature_password"]["file"]
    assert backend_secret.startswith("${MLFLOW_BACKEND_DB_PASSWORD_SECRET_FILE:?")
    assert feature_secret.startswith("${REGIME_FEATURE_PGPASSWORD_SECRET_FILE:?")
    env = _text(ENV_EXAMPLE)
    assert "<required-mlflow-backend-database>" in env
    assert "<required-mlflow-backend-user>" in env
    assert "<required-regime-loader-serving-database>" in env
    assert "password=" not in env.lower()


def test_scripts_enforce_local_build_and_no_build_start_contract() -> None:
    build = _text(BUILD)
    up = _text(UP)
    down = _text(DOWN)
    verify = _text(VERIFY)
    assert "docker compose build --pull mlflow" in build
    assert "docker compose up -d --no-build" in up
    assert "docker image inspect regime-engine-mlflow:local" in up
    for script in (build, up, down, verify):
        assert "docker context show" in script
        assert "docker context inspect" in script
        assert "unix://*" in script
        assert "remote DOCKER_HOST is forbidden" in script
    assert "docker buildx inspect" in build
    assert "Endpoint:" in build
    assert ".Current" not in build
    combined = "\n".join((build, up, down, verify, _text(DEPLOYMENT_DOC)))
    banned = (
        "docker compose push",
        "docker login",
        "ssh://",
        "tcp://",
        "kubectl",
        "docker stack",
    )
    for token in banned:
        assert token not in combined


def test_normal_compose_path_never_runs_database_migration_and_verifies_provenance() -> None:
    compose = _text(COMPOSE)
    up = _text(UP)
    assert "mlflow db upgrade" not in compose
    assert "mlflow db upgrade" not in up
    verify = _text(VERIFY)
    for evidence in (
        "compose_project=",
        "application_image_id=",
        "repository_git_sha=",
        "mlflow_version=",
        "build_timestamp=",
        "port_5000=",
        "custom_image_repo_digests=none",
    ):
        assert evidence in verify


def test_production_contract_uses_compose_yaml_not_example_file() -> None:
    assert __import__("os").path.exists(COMPOSE)
    assert not __import__("os").path.exists("compose.example.yaml")
    assert "--build" not in _text(UP)
