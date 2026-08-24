FROM python:3.14.7-slim-bookworm@sha256:23c59390fc717bf09f9336908199a0ae75d9c4264bf296123f94ad772fea3b52

ARG REGIME_ENGINE_GIT_SHA=unknown
ARG REGIME_ENGINE_BUILD_TIMESTAMP=unknown

LABEL org.opencontainers.image.title="regime-engine-mlflow" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.revision="${REGIME_ENGINE_GIT_SHA}" \
      org.opencontainers.image.created="${REGIME_ENGINE_BUILD_TIMESTAMP}" \
      org.opencontainers.image.description="MLflow 3.15.1 with the regime-engine custom Flask application" \
      io.regime-engine.mlflow-version="3.15.1" \
      io.regime-engine.python-version="3.14.7"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_WORKERS=4 \
    MLFLOW_THREADS_PER_WORKER=4 \
    MLFLOW_HTTP_TIMEOUT_SECONDS=120 \
    MLFLOW_GRACEFUL_TIMEOUT_SECONDS=30 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /opt/regime-engine

COPY uv.lock pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts/mlflow_entrypoint.sh /usr/local/bin/regime-engine-mlflow-entrypoint
COPY scripts/mlflow_db_upgrade.sh /usr/local/bin/regime-engine-mlflow-db-upgrade

RUN apt-get update \
    && apt-get install --no-install-recommends --yes build-essential libstdc++6 \
    && apt-mark manual libstdc++6 \
    && CC=g++ CXX=g++ python -m pip install --no-cache-dir -r uv.lock \
    && python -m pip install --no-cache-dir --no-deps . \
    && apt-get purge --auto-remove --yes build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -c 'from hmmlearn.hmm import GaussianHMM; import hmmlearn, mlflow, platform; assert GaussianHMM; assert platform.python_version() == "3.14.7"; assert hmmlearn.__version__ == "0.3.3"; assert mlflow.__version__ == "3.15.1"' \
    && groupadd --system --gid 10001 regime-engine \
    && useradd --system --uid 10001 --gid regime-engine --home-dir /opt/regime-engine --shell /usr/sbin/nologin regime-engine \
    && install -d -o regime-engine -g regime-engine /mlflow/artifacts /opt/regime-engine/build \
    && python -c 'import importlib.metadata as m, json, pathlib; items=sorted((d.metadata["Name"],d.version) for d in m.distributions() if d.metadata["Name"]); pathlib.Path("/opt/regime-engine/build/python-packages.tsv").write_text("".join(f"{n}\\t{v}\\n" for n,v in items), encoding="utf-8"); pathlib.Path("/opt/regime-engine/build/python-sbom.json").write_text(json.dumps({"bomFormat":"CycloneDX","specVersion":"1.6","version":1,"components":[{"type":"library","name":n,"version":v,"purl":f"pkg:pypi/{n.lower().replace(chr(95),chr(45))}@{v}"} for n,v in items]}, sort_keys=True, separators=(",",":")), encoding="utf-8")' \
    && chmod 0555 /usr/local/bin/regime-engine-mlflow-entrypoint /usr/local/bin/regime-engine-mlflow-db-upgrade \
    && chown -R regime-engine:regime-engine /opt/regime-engine/build

USER 10001:10001

EXPOSE 5000

ENTRYPOINT ["/usr/local/bin/regime-engine-mlflow-entrypoint"]
