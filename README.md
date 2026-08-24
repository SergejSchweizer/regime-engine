# Regime Engine

`regime-engine` is the implementation repository for the Python distribution `market-regime-engine` and import package `market_regime_engine`.

The MVP is a statistical regime service built around full-covariance Gaussian HMMs. It reads immutable/current-vintage Gold features from the external `regime-loader` PostgreSQL serving replica and exposes predictions through the single MLflow 3.15.1 service described by the repository contracts.

## Pinned bootstrap

- Python: **3.14.7**
- MLflow: **3.15.1**
- Gaussian HMM backend: **hmmlearn 0.3.3**
- public profile: `xetra`
- registered model: `regime-xetra`
- production alias: `champion`

Bootstrap is fail-closed:

```bash
./scripts/bootstrap.sh
.venv/bin/python -m pytest tests/test_hmm_bootstrap_smoke.py
```

The bootstrap script rejects any interpreter other than Python 3.14.7 and installs the exact pinned dependency roots from `uv.lock`. The HMM smoke test must fit a K=2 `covariance_type="full"` model; there is no fallback covariance/backend.

## Architecture contracts

Implementation details are governed by `BACKLOG.md`, `CONTRIBUTING.md`, `DATA_SOURCE.md`, `EVALUATION.md`, and `PLOT_STYLE.md`. Consumer portfolio/trading economics are deliberately outside this repository.
