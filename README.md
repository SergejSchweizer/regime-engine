# Regime Engine

`regime-engine` is the implementation repository for the Python distribution `market-regime-engine` and import package `market_regime_engine`.

The MVP is a statistical regime service built around full-covariance Gaussian HMMs. It reads the external `regime-loader` feature PostgreSQL serving replica and exposes predictions through the same MLflow 3.15.1 service that owns tracking, registry, and artifacts.

## Canonical identity

- Python: **3.14.7**
- MLflow: **3.15.1**
- Gaussian HMM backend: **hmmlearn 0.3.3**
- public profile: `xetra`
- profile config version: `1`
- feature-selection policy: `xetra_semantic_medoid_v1`
- registered model: `regime-xetra`
- production alias: `champion`
- prediction contract: `RegimePrediction.v1`

## Bootstrap

Bootstrap is fail-closed:

```bash
./scripts/bootstrap.sh
.venv/bin/python -m pytest tests/test_hmm_bootstrap_smoke.py
```

The script rejects any interpreter other than Python 3.14.7 and installs the exact pinned dependency roots from `uv.lock`. The HMM smoke must fit a K=2 `covariance_type="full"` model; there is no reduced-covariance/backend fallback.

## Data and scientific claim boundary

The production feature source is the external PostgreSQL service at `10.10.1.3:54321`, read through the dedicated TLS-required `"regime-engine"` role. Source data has `data_time_semantics=current_vintage_observation_day`: evaluation is causal/split-leak-free relative to the current-vintage observation sequence, but it does not claim provider-release-time historical-vintage safety.

After features are frozen, only rows where every selected feature is finite and non-null are HMM observations. Missing rows stay as gap evidence and do not create extra transition powers.

## One-port MLflow service

Production has one public service at `http://10.10.1.3:5000`. The same MLflow process serves standard MLflow routes and the profile API. There is no standalone FastAPI/Uvicorn server, no `mlflow models serve`, no `:5001`, no reverse proxy, and no Prometheus endpoint.

The production Compose deployment contains exactly `mlflow` and private `mlflow-postgres`. The repository application image is local-only and is built and started explicitly:

```bash
docker compose build --pull mlflow
docker compose up -d --no-build
```

The custom image is `regime-engine-mlflow:local` with `pull_policy: never`; it is not published to an application registry. Operator/model-cycle commands run inside the local service with `docker compose exec -T mlflow ...`.

## Statistical lifecycle

Feature selection uses first-fold TRAIN only. K=2/K=3/K=4 full-covariance Gaussian HMM candidates are compared by the deterministic walk-forward contract in `EVALUATION.md`. No ETF/portfolio/trading metric participates in feature selection or model ranking.

A walk-forward fold model is never registered for production. After a statistical champion is selected, the winning K is fit again from scratch on all eligible current-vintage observations through the exact evaluation cutoff; only this final-refit artifact may become a `regime-xetra` version and later receive the `challenger`/`champion` alias.

## Contract ownership

- `BACKLOG.md`: implementation PR scope/dependencies/API/deployment/operations plan
- `CONTRIBUTING.md`: Git/weak-agent rules
- `DATA_SOURCE.md`: feature PostgreSQL, lineage, time/missing-value semantics
- `EVALUATION.md`: feature selection, HMM fitting, walk-forward, alignment, ranking, final refit
- `PLOT_STYLE.md`: diagnostic plot presentation
- `ARCHITECTURE.md`: durable architecture overview
- `docs/model_lifecycle.md`: lifecycle and serving continuation

Consumer portfolio/economic evaluation belongs downstream and is deliberately outside this repository.
