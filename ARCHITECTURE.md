# Regime Engine Architecture

## Canonical identity

Repository `SergejSchweizer/regime-engine` ships Python distribution `market-regime-engine`, import package `market_regime_engine`, MLflow app entry point `regime-engine`, public profile `xetra` config version `1`, registered model `regime-xetra`, and production alias `champion`.

## Ownership boundary

```text
regime-loader
  -> immutable Gold
  -> external feature PostgreSQL 10.10.1.3:54321
  -> regime-engine
  -> MLflow tracking/registry/artifacts + regime profile API
  -> portfell / future consumers
```

The engine is statistical only. ETF/portfolio returns, weights, Sharpe, Sortino, Calmar, drawdown, Expected Shortfall, transaction costs, trading labels, and profitability never influence feature selection or statistical champion ranking.

Input lineage has `data_time_semantics=current_vintage_observation_day`. Walk-forward evaluation is causal and split-leak-free with respect to that current-vintage observation sequence; it does not claim historical provider-release-time/vintage safety.

After feature selection is frozen, an HMM observation exists only when every selected feature is finite and non-null. Missing timestamps remain gap evidence. One HMM transition is taken per retained observation, never per elapsed calendar day. This same complete-case observation clock is used by evaluation, final refit, latest, and replay.

## Production serving topology

Production exposes exactly one MLflow 3.15.1 HTTP service on `10.10.1.3:5000`:

- standard MLflow UI/tracking/registry/artifact routes;
- `POST /regime-engine/v1/profiles/{profile_id}/invocations`;
- `GET /regime-engine/v1/profiles/{profile_id}/oos-builds/{build_id}`;
- `GET /regime-engine/v1/health`.

There is no standalone FastAPI/Uvicorn service, `mlflow models serve`, port 5001, reverse proxy, or Prometheus exporter. MLflow custom apps are Flask/WSGI and the deployment forces Gunicorn `gthread` workers.

Compose contains exactly `mlflow` and private `mlflow-postgres`. Only `mlflow` publishes `5000:5000`. Feature PostgreSQL is external and uses a dedicated read-only trusted-LAN `"regime-engine"` role with explicit plaintext `sslmode=disable`; feature credentials and MLflow-backend credentials use separate environment namespaces.

## Local-only application image

The deployment host has a local checkout and uses its local Unix-socket Docker daemon. The repository-owned image is built locally as `regime-engine-mlflow:local` and is never pushed to or pulled from an application registry.

Production build and startup are deliberately separate:

```bash
docker compose build --pull mlflow
docker compose up -d --no-build
```

`compose.yaml` owns the production contract. It declares `build`, `image: regime-engine-mlflow:local`, and `pull_policy: never` for the app service. Startup fails if the local image is absent rather than silently pulling it. Operator/model-cycle commands run through `docker compose exec -T mlflow ...`.

## Statistical lifecycle

The Xetra profile selects features only from first-fold TRAIN data, evaluates exactly K=2/K=3/K=4 full-covariance Gaussian HMM candidates in expanding walk-forward folds, and chooses a statistical champion using the deterministic ranking in `EVALUATION.md`.

No walk-forward fold model is registered. After selection, a mandatory fresh final production refit uses the frozen features and all eligible source observations through the exact evaluation cutoff, aligns state IDs to the last valid winning-K fold, and persists the inference origin, trained-through timestamp, and terminal filtered probabilities. Only this final-refit artifact can become a `regime-xetra` model version and be assigned `challenger`/`champion`.

Latest and fixed-model replay are causal forward-filter operations. Replay start is never a new HMM initial condition. Walk-forward OOS prediction builds remain immutable and are retrieved by explicit build ID; fixed-model replay is never substituted for OOS evidence.

## Security and capacity

The MVP is trusted-private-LAN only. Port 5000 must not be Internet exposed. Host/CORS configuration is explicit and non-wildcard. The canonical feature PostgreSQL endpoint does not offer TLS, so feature transport uses explicit `sslmode=disable`; secrets/credential-bearing DSNs/raw feature vectors/model binaries are excluded from logs and API errors.

Each Gunicorn worker owns a process-local model cache and psycopg pool. With production defaults: 4 workers x 4 threads, pool max 4, feature-PG connection budget 16, and one admitted replay per worker. Replay uses bounded synchronous request-thread work with cooperative deadlines; there is no hidden unbounded executor.

## Contract ownership

- `BACKLOG.md`: PR scope/dependencies/API/deployment/operations plan
- `CONTRIBUTING.md`: Git and weak-agent rules
- `DATA_SOURCE.md`: source PostgreSQL/lineage/time/missing-value semantics
- `EVALUATION.md`: feature selection/HMM/walk-forward/alignment/ranking/final refit
- `PLOT_STYLE.md`: visualization rendering

Implementation must fail closed rather than inventing a fallback when these contracts cannot be met.
