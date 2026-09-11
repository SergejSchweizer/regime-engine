# Regime Engine

`regime-engine` is the implementation repository for the Python distribution `market-regime-engine` and import package `market_regime_engine`.

The MVP is a statistical regime service built around full-covariance Gaussian HMMs. It reads the external `regime-loader` feature PostgreSQL serving replica and exposes predictions through the same MLflow 3.15.1 service that owns tracking, registry, and artifacts.

## Canonical identity

- Python: **3.14.7**
- MLflow: **3.15.1**
- Gaussian HMM backend: **hmmlearn 0.3.3**
- public profile: `xetra`
- profile config version: `4`
- feature-discovery policy: `xetra_global_regime_v4`
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

The production feature source is the external PostgreSQL service at `10.10.1.3:54321`, read through the dedicated trusted-LAN plaintext `"regime-engine"` role (`sslmode=disable`). Source data has `data_time_semantics=current_vintage_observation_day`: evaluation is causal/split-leak-free relative to the current-vintage observation sequence, but it does not claim provider-release-time historical-vintage safety.

For the opt-in external feature-PG verifier, copy `config.example.yaml` to the
Git-ignored `config.yaml` and set the connection metadata plus a local
password-file path. The file must never contain a password or DSN.

After features are frozen, only rows where every selected feature is finite and non-null are HMM observations. Missing rows stay as gap evidence and do not create extra transition powers.

## External MLflow service

Production uses the existing MLflow service at `http://10.10.1.3:5000`. The same
service provides standard MLflow tracking, registry, artifacts, and the
`regime-engine` profile API. There is no repository-owned MLflow container,
Compose file, local MLflow PostgreSQL backend, standalone FastAPI/Uvicorn
server, reverse proxy, or second serving port.

The lifecycle/evaluation commands default to this endpoint and reject local or
alternate MLflow URIs. The feature PostgreSQL remains external at
`10.10.1.3:54321` and is accessed through the read-only `regime-engine` role.

Run the complete Xetra v4 evaluation as one cron-safe command:

```bash
./scripts/run_xetra_v4_cron.sh
```

## Statistical lifecycle

Feature discovery and candidate selection use outer-fold TRAIN data only. The v4 contract evaluates the dynamic source catalog, Gaussian K2-K5, two-mixture GMM-HMM K2-K5, and Student-t K2-K5 candidates using the deterministic walk-forward contract in `EVALUATION.md`. No ETF/portfolio/trading metric participates in discovery or model ranking.

A walk-forward fold model is never registered for production. After a
production-eligible statistical evaluation, deployment selection is rerun once
on the complete source through its maximum, then the selected HMM is fit again
from scratch on all eligible current-vintage observations through that
deployment cutoff. The package is uploaded to NAS MLflow and registered using a
remote `runs:/...` URI; only this final-refit artifact may become a
`regime-xetra` version. Registration can update `challenger` only; `champion`
promotion remains an explicit operator action.

The complete v4 workflow is documented in [docs/regime_evaluations.md](docs/regime_evaluations.md).

## Contract ownership

- `BACKLOG.md`: implementation PR scope/dependencies/API/deployment/operations plan
- `CONTRIBUTING.md`: Git/weak-agent rules
- `DATA_SOURCE.md`: feature PostgreSQL, lineage, time/missing-value semantics
- `EVALUATION.md`: feature selection, HMM fitting, walk-forward, alignment, ranking, final refit
- `PLOT_STYLE.md`: diagnostic plot presentation
- `ARCHITECTURE.md`: durable architecture overview
- `docs/model_lifecycle.md`: lifecycle and serving continuation
- `docs/model_lifecycle_operations.md`: external MLflow lifecycle operations

Consumer portfolio/economic evaluation belongs downstream and is deliberately outside this repository.
