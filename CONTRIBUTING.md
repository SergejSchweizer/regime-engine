# Contributing and Git Policy

Status date: 2026-08-23

`BACKLOG.md` is the single authoritative implementation plan for `SergejSchweizer/regime-engine`. The consolidated backlog has no Wave-7/Wave-8 override/addendum mechanism: each PR has one effective dependency set, scope, allowed-file list, and acceptance list.

## Canonical identities

- GitHub repository: `SergejSchweizer/regime-engine`
- repository short name: `regime-engine`
- Python distribution: `market-regime-engine`
- Python import package: `market_regime_engine`
- initial public profile ID: `xetra`
- Xetra registered model: `regime-xetra`
- production MLflow alias: `champion`

Do not substitute legacy `xetra_cross_asset_v1` as the public profile ID or `engine-champion` as a serving alias.

## PR naming

Canonical PR name:

```text
PR-<three-digit-number>-<kebab-case-slug>
```

PR title:

```text
PR-014-gaussian-hmm-adapter: Implement full-covariance Gaussian HMM adapter
```

Branch:

```text
pr/PR-014-gaussian-hmm-adapter
```

Commit subject:

```text
<type>(PR-014-gaussian-hmm-adapter): <imperative description>
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Generic/WIP commit subjects are forbidden. The squash commit follows the same canonical naming rule.

## Required Git state

Before branch creation:

```text
git switch main
git pull --ff-only
git status --short
git branch --show-current
```

Required state is a clean tree on `main`.

Immediately before final push:

```text
git status --short
git branch --show-current
```

Required state is an empty status and the exact PR branch declared by `BACKLOG.md`.

## Weak-agent scope discipline

An implementation agent receives one PR section plus only the already-merged contracts/interfaces needed by that PR.

The agent must:

1. edit only allowed files;
2. implement every acceptance checkbox;
3. ship tests in the same PR;
4. stop when a dependency is unmerged or required work is outside scope;
5. never invent thresholds, aliases, profile IDs, database names, fallback libraries, covariance modes, API fields, ports, or deployment services;
6. never edit `BACKLOG.md`;
7. never broaden into portfolio/economic logic;
8. never contact NAS services from required CI tests.

If a pinned dependency such as Python 3.14.7, MLflow 3.15.1, or `hmmlearn==0.3.3` fails its required compatibility test, the agent stops. It does not change the pinned architecture independently.

## Contract-owner files

These are normative contract-owner files:

- `BACKLOG.md`: implementation scope, dependencies, constants, API/deployment contracts, execution plan;
- `DATA_SOURCE.md`: upstream PostgreSQL source, lineage, time/missing-value semantics and credentials;
- `EVALUATION.md`: exact statistical/HMM/evaluation/final-refit semantics;
- `PLOT_STYLE.md`: presentation/rendering only;
- `CONTRIBUTING.md`: Git/weak-agent execution rules.

Weak implementation agents do not rewrite contract-owner files unless their PR explicitly lists that file and exact purpose.

`PLOT_STYLE.md` can never alter statistical semantics. `DATA_SOURCE.md` can never alter feature-selection/model-selection semantics. `EVALUATION.md` can never introduce consumer portfolio metrics.

## Production source and serving boundaries

Production features come from the external `regime-loader` PostgreSQL serving replica at `10.10.1.3:54321` using the dedicated read-only user `regime-engine`. Direct upstream Parquet is not the production source.

Production serving is one MLflow service at `http://10.10.1.3:5000`, extended by the `regime-engine` MLflow Flask app and explicitly run through Gunicorn. There is no separate FastAPI/Uvicorn application, model-serving port 5001, reverse proxy, or Prometheus exposure.

## Required tests

Push/merge required tests are hermetic. Only explicitly marked `external_service` smoke tests may contact:

- feature PostgreSQL `10.10.1.3:54321`;
- MLflow `http://10.10.1.3:5000`.

Those external tests are opt-in and excluded from required gates.
