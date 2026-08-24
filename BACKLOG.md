# Regime Engine — Consolidated Implementation Backlog

Status date: 2026-08-24

This is the single authoritative implementation backlog for `SergejSchweizer/regime-engine`.

There are no legacy Wave overrides or addenda. Each PR below has exactly one effective branch, dependency set, scope, allowed-file set and acceptance list. Historical/superseded wording must not be consulted by implementation agents.

---

# 1. Canonical identities

| Concept | Canonical value |
|---|---|
| GitHub repository | `SergejSchweizer/regime-engine` |
| repository short name | `regime-engine` |
| Python distribution | `market-regime-engine` |
| import package | `market_regime_engine` |
| Python | `3.14.7` |
| MLflow | `3.15.1` |
| Gaussian HMM backend | `hmmlearn==0.3.3` |
| MLflow app entry point | `regime-engine` |
| MLflow service | `http://10.10.1.3:5000` |
| feature PostgreSQL | `10.10.1.3:54321` |
| feature PostgreSQL user | `regime-engine` |
| initial public profile ID | `xetra` |
| profile configuration version | `1` |
| feature-selection policy | `xetra_semantic_medoid_v1` |
| registered Xetra model | `regime-xetra` |
| production alias | `champion` |
| non-production alias | `challenger` |
| prediction contract | `RegimePrediction.v1` |
| invocation response | `RegimeInvocationResponse.v1` |
| error response | `RegimeError.v1` |
| local Compose project | `regime-engine` |
| local custom MLflow image tag | `regime-engine-mlflow:local` |

`xetra_cross_asset_v1` is not a public profile ID. Use `profile_id=xetra` plus `profile_config_version=1`.

`engine-champion` is not an MLflow alias. *Statistical champion* is an evaluation status; `champion` is the production MLflow alias assigned only after final production refit.

PR IDs `039`–`044` and `051`–`055` are retired historical planning/documentation IDs and must never be reused.

---

# 2. Normative contract ownership

To eliminate duplicated definitions, each topic has one owner:

| Topic | Authoritative file |
|---|---|
| PR scope, dependencies, API/deployment/operations plan | `BACKLOG.md` |
| Git/weak-agent rules | `CONTRIBUTING.md` |
| feature PostgreSQL, lineage, time/missing-value semantics | `DATA_SOURCE.md` |
| feature selection, HMM, walk-forward, alignment, ranking, final refit | `EVALUATION.md` |
| plot rendering/presentation | `PLOT_STYLE.md` |

Implementation PRs reference those contracts; they must not duplicate or reinterpret them.

If an implementation cannot satisfy a pinned contract, the agent stops. It does not invent a fallback.

---

# 3. Non-negotiable architecture

Ownership direction:

```text
regime-loader
  -> immutable Gold
  -> external PostgreSQL serving replica 10.10.1.3:54321
  -> regime-engine
  -> MLflow / predictions / profile API
  -> portfell and future consumers
```

`regime-engine` does not own provider acquisition or portfolio/economic evaluation.

No ETF return, portfolio weight/return, Sharpe, Sortino, Calmar, drawdown, Expected Shortfall, transaction cost, trading label or regime profitability may influence engine feature selection or statistical champion selection.

Production serves one public HTTP endpoint family from one MLflow service:

```text
http://10.10.1.3:5000
  MLflow UI / Tracking / Registry / artifacts
  POST /regime-engine/v1/profiles/{profile_id}/invocations
  GET  /regime-engine/v1/profiles/{profile_id}/oos-builds/{build_id}
  GET  /regime-engine/v1/health
```

No second `mlflow models serve`, standalone FastAPI/Uvicorn service, public `:5001`, nginx, Traefik, reverse proxy or Prometheus exporter is permitted.

---

# 4. Exact serving/runtime topology

Compose has exactly:

```text
docker-compose
├── mlflow
└── mlflow-postgres
```

Only `mlflow` publishes `5000:5000`. `mlflow-postgres` has no host port and stores MLflow relational metadata only. The feature PostgreSQL remains external.

## Local-only build and execution contract

The Compose project is built and executed locally on the deployment host. For production the deployment host is the NAS `10.10.1.3` with a local checkout of this repository.

`local` has the following exact meaning:

- Compose commands target the deployment host's local Docker daemon through a local Unix socket; TCP/SSH remote Docker contexts, remote BuildKit builders, Docker Swarm and Kubernetes are outside the MVP contract.
- The repository-owned `mlflow` image is built locally from this repository's `Dockerfile` and local build context.
- The canonical local custom-image tag is exactly `regime-engine-mlflow:local`.
- The custom `regime-engine-mlflow:local` image is never pulled from or pushed to Docker Hub, GHCR or another image registry in the MVP.
- Third-party/base images may be downloaded only as pinned upstream inputs: the pinned Python base image during local `mlflow` build and the pinned official PostgreSQL image for `mlflow-postgres`.
- The production build step is explicit and separate from startup: `docker compose build --pull mlflow`.
- Production startup is exactly `docker compose up -d --no-build`; startup must not rebuild the application image implicitly.
- The `mlflow` service declares the local build context plus `image: regime-engine-mlflow:local` and `pull_policy: never`, so startup cannot silently replace the local custom image with a registry image.
- If `regime-engine-mlflow:local` is absent, production startup fails rather than pulling an application image.
- The Compose file itself is `compose.yaml`, committed in this repository; `compose.example.yaml` is not the production deployment contract.
- Deployment verification records the local Docker image ID, repository Git SHA, MLflow version and build timestamp so the running local image is auditable.
- Cron/operator model-cycle commands execute inside the already-running local `mlflow` service via `docker compose exec -T mlflow ...`; no separate remote execution environment is introduced.

MLflow custom apps are Flask/WSGI. MLflow 3.15.1 must therefore be forced to Gunicorn; its default Uvicorn server is forbidden for this deployment.

Canonical startup inside the locally built image:

```text
mlflow server
  --app-name regime-engine
  --host 0.0.0.0
  --port 5000
  --workers ${MLFLOW_WORKERS}
  --gunicorn-opts "--worker-class gthread --threads ${MLFLOW_THREADS_PER_WORKER} --timeout ${MLFLOW_HTTP_TIMEOUT_SECONDS} --graceful-timeout ${MLFLOW_GRACEFUL_TIMEOUT_SECONDS}"
```

Pinned production defaults:

```text
MLFLOW_WORKERS=4
MLFLOW_THREADS_PER_WORKER=4
MLFLOW_HTTP_TIMEOUT_SECONDS=120
MLFLOW_GRACEFUL_TIMEOUT_SECONDS=30
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1

REGIME_MODEL_ALIAS_CACHE_TTL_SECONDS=30
REGIME_PG_POOL_MIN_SIZE=1
REGIME_PG_POOL_MAX_SIZE=4
REGIME_PG_ACQUIRE_TIMEOUT_SECONDS=5
REGIME_PG_STATEMENT_TIMEOUT_SECONDS=30
REGIME_FEATURE_PG_CONNECTION_BUDGET=16

REGIME_REPLAY_MAX_ROWS=10000
REGIME_REPLAY_MAX_INTERNAL_ROWS=15000
REGIME_REPLAY_MAX_RANGE_DAYS=14610
REGIME_REPLAY_TIMEOUT_SECONDS=60
REGIME_REPLAY_MAX_RESPONSE_BYTES=26214400
REGIME_REPLAY_MAX_CONCURRENCY_PER_WORKER=1

REGIME_SOURCE_STALE_WARN_DAYS=4
REGIME_SOURCE_STALE_FAIL_DAYS=7
REGIME_MODEL_STALE_WARN_DAYS=14
REGIME_MODEL_STALE_FAIL_DAYS=35
```

Startup validation requires:

```text
MLFLOW_WORKERS * REGIME_PG_POOL_MAX_SIZE <= REGIME_FEATURE_PG_CONNECTION_BUDGET
```

With defaults this is exactly `4 * 4 <= 16`.

---

# 5. Database/environment separation

Feature source variables are exactly those defined in `DATA_SOURCE.md`, including production `REGIME_FEATURE_PGSSLMODE=require`.

The external feature database name has no default and is never guessed.

MLflow backend database variables are separately named and also have no invented credential values:

```text
MLFLOW_BACKEND_DB_NAME=<required runtime value>
MLFLOW_BACKEND_DB_USER=<required runtime value>
MLFLOW_BACKEND_DB_PASSWORD_FILE=/run/secrets/mlflow_backend_password
MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts
```

Compose maps these runtime values to its private `mlflow-postgres` service. The feature DB and MLflow backend DB must never share generic `PG*` environment variables.

Feature reader role is exact SQL identifier `"regime-engine"`; role/bootstrap details are owned by PR-057 and `DATA_SOURCE.md`.

---

# 6. Security/threat model

MVP is trusted-private-LAN only:

- port 5000 must not be Internet-exposed;
- host/NAS firewall restricts it to trusted private clients/operators;
- allowed hosts default to `10.10.1.3`, `localhost`, `127.0.0.1` plus explicitly configured trusted hostnames;
- no wildcard Host/CORS configuration;
- browser CORS is same-origin by default;
- current MVP accepts trusted-LAN registry/operator access rather than claiming multi-tenant application-layer isolation;
- future untrusted/public access requires a separate versioned authentication/authorization design composed into the same MLflow app.

Feature PostgreSQL transport is TLS-required (`sslmode=require`). An implementation may not silently downgrade it.

No secret, credential-bearing DSN, raw feature vector or model binary payload may appear in logs/errors/API output.

---

# 7. Public API contract

## Invocation

```text
POST /regime-engine/v1/profiles/{profile_id}/invocations
```

`profile_id` is path-only. A body `profile_id` or unknown body field is rejected.

Latest body:

```json
{
  "operation": "latest",
  "as_of": "optional RFC3339 UTC timestamp",
  "model_version": "optional exact immutable model version"
}
```

Replay body:

```json
{
  "operation": "replay",
  "start": "required RFC3339 UTC timestamp",
  "end": "required RFC3339 UTC timestamp",
  "model_version": "optional exact immutable model version"
}
```

Rules:

- UTC `Z`/zero offset required; normalized to UTC;
- replay interval is inclusive `[start,end]`;
- latest forbids `start/end`; replay forbids `as_of`;
- absent version resolves configured `champion`, pins its exact immutable version before source access, and records alias resolution time;
- explicit version bypasses alias lookup;
- model-version pinning does not pin an unavailable historical source vintage;
- consumers never provide features, source build, DB/table, scaler, HMM or state-mapping details.

If latest omits `as_of`, effective upper source bound is the current validated sync-state `max_timestamp`; the response prediction timestamp is the latest complete model observation at or before that bound.

## Success envelope

`RegimeInvocationResponse.v1` contains at least:

```text
schema_version
request_id
profile_id
operation
prediction_mode
requested time fields
model name/version/alias/alias_resolved_at_utc/trained_through_timestamp
source dataset/build/hash/schema/feature/synced_at_utc/data_time_semantics
feature_contract_hash
feature_selection_definition_hash nullable
feature_selection_execution_hash nullable
warmup_observation_count
skipped_incomplete_row_count
predictions[]
```

`latest` returns exactly one prediction. Replay must return at least one complete model observation or fail `422 no_complete_observations`.

Modes are exactly `fixed_model_latest` and `fixed_model_replay`.

## OOS retrieval

```text
GET /regime-engine/v1/profiles/{profile_id}/oos-builds/{build_id}?start=<optional>&end=<optional>
```

Explicit immutable build ID is mandatory; no silent latest. Mode is exactly `walk_forward_oos`. Fixed replay is never substituted.

## Error envelope

`RegimeError.v1`:

```text
schema_version
request_id
error_code
safe message
retryable boolean
safe details
```

HTTP mapping:

- `400` malformed/forbidden input;
- `404` unknown profile/model version/OOS build;
- `413` replay range/row/internal-row/response-size limit;
- `422` model/source/semantic/no-complete-observation failure;
- `503` dependency/capacity/stale default source/champion failure;
- `504` cooperative replay deadline after underlying replay work has actually stopped.

---

# 8. Serving correctness/high-load rules

Each Gunicorn worker owns its own process-local model cache and psycopg pool.

Model cache:

- key: `(profile_id, exact_model_version)`;
- alias resolution TTL 30 s;
- single-flight load per key under concurrent requests;
- max two loaded versions per profile/worker: current + at most one previous;
- LRU eviction only when no active request references the model;
- new alias target completely loads/validates before atomic replacement;
- failed new target causes explicit failure; previous object is never falsely labelled current.

Replay:

- admission is one process-local semaphore per worker by default;
- no separate background/unbounded executor; work runs synchronously in current Gunicorn `gthread` request thread;
- filtering/inference is chunked for cooperative deadline checks;
- capacity slot is released only after underlying work stops;
- PG statement timeout bounds DB work;
- preflight checks requested interval and conservative row/state response estimate;
- final serialization uses a bounded buffer and verifies byte limit before response commit;
- no truncation, silent pagination or prediction-build substitution.

Exact configured upper bounds:

```text
feature PG connections = workers * pool_max = 16
admitted concurrent replay = workers * replay_per_worker = 4
request-thread capacity = workers * threads = 16
```

At admitted replay capacity, PR-062 must show standard MLflow health/tracking/registry read requests and latest remain serviceable under this exact topology. No QoS claim is made if callers bypass admission and externally exhaust all process/thread resources.

Graceful shutdown:

- SIGTERM stops new admitted replay work;
- Gunicorn gets 30 s graceful timeout;
- process-local PG pools close during worker shutdown;
- no request may release a replay slot while hidden work continues after timeout.

---

# 9. Freshness/lifecycle semantics

Staleness uses elapsed UTC seconds divided by exactly `86400`, not date-boundary counting.

For default-champion latest:

```text
source_staleness_days = (request_time_utc - prediction_timestamp)/86400s
model_staleness_days  = max(0,(prediction_timestamp - trained_through_timestamp)/86400s)
```

- warn threshold => custom health `degraded`, serving continues;
- fail threshold => default-champion latest returns `503`;
- explicit-version historical replay remains allowed when its requested source/model contracts are valid.

The MVP concept/model-drift decision mechanism is periodic full reevaluation on a new source build. No uncalibrated online detector may promote/demote a model.

Recommended model-cycle cadence is exactly every 7 days after upstream synchronization. Scheduling itself is NAS/operator-owned.

Promotion/rollback uses compare-and-swap:

```text
expected_current_version
new_version/rollback_version
non-empty reason
```

Mismatch performs no mutation. Every alias move is audited.

---

# 10. Dependency/image reproducibility

- Python exactly `3.14.7`.
- committed frozen `uv.lock`.
- MLflow exactly `3.15.1`.
- `hmmlearn==0.3.3` exactly.
- PR-001 must prove full-covariance HMM smoke under Python 3.14.7; failure blocks architecture rather than triggering a fallback.
- Docker Python tag target: `python:3.14.7-slim-bookworm`.
- MLflow backend PostgreSQL tag target: `postgres:18.6-alpine`.
- tag-only upstream image references are not sufficient: PR-032/061 resolve and commit exact official registry SHA-256 digests for the Python base and PostgreSQL inputs.
- the repository-owned MLflow/regime-engine application image is always built locally as `regime-engine-mlflow:local`; no remote application-image registry is part of the MVP.
- deployment evidence records the local application image ID plus Git SHA; the local tag alone is not treated as immutable provenance.
- `docker compose up` must not build or pull the custom application image; only explicit `docker compose build --pull mlflow` creates/replaces it.
- floating `latest` tags are forbidden for all upstream inputs.
- dependency/image changes require a dedicated compatibility change plus lock/test updates.

Required unit+integration code coverage threshold is 90%. External-service tests are excluded.

---

# 11. Backup/migration/secret rotation

Normal container startup never automatically runs MLflow backend schema migration.

Before a pinned MLflow version change:

1. quiesce/stop `mlflow`;
2. dump MLflow backend PostgreSQL;
3. archive artifact volume;
4. write one manifest with UTC, MLflow/PostgreSQL versions, local application image ID/Git SHA and SHA-256 hashes;
5. run explicit one-shot `mlflow db upgrade`;
6. rebuild the local application image explicitly when required;
7. start service locally and run metadata/registry/artifact smoke.

Restore requires the matching DB dump + artifact archive manifest and MLflow stopped.

Operational docs must define rotation for MLflow backend and feature-PG secrets: provision new secret, restart/reload affected service, verify connectivity, then revoke old secret; secrets are never printed during rotation.

---

# 12. Git/weak-agent rules

Canonical PR: `PR-<three-digit-number>-<kebab-case-slug>`.

Branch: `pr/<canonical-pr-name>`.

Commit: `<type>(<canonical-pr-name>): <imperative description>`.

Every agent:

1. starts from clean/up-to-date `main`;
2. reports `git status --short` and `git branch --show-current` before work and before final push;
3. stops for dirty tree/unmerged dependency/out-of-scope need;
4. edits only allowed files;
5. never edits `BACKLOG.md`;
6. never invents constants/names/fallbacks;
7. ships tests with behavior;
8. never requires NAS endpoints in required CI.

Contract-owner files are modified only by PRs explicitly listing them.

---

# 13. CI/governance target

Push and merge workflows each have parallel `lint`, `type`, `unit`, `integration` jobs and final `push-gate`/`merge-gate` respectively.

- Python 3.14.7 and frozen lock;
- Ruff check/format;
- strict mypy;
- required tests hermetic;
- combined unit/integration coverage >=90%;
- dependency audit runs in the lint lane against the frozen dependency set and fails for known high/critical vulnerabilities unless a repository-tracked, expiry-dated exception exists;
- no external NAS smoke in required gate.

Protected main target is exactly `SergejSchweizer/regime-engine/main`: PR required, strict `merge-gate`, conversations resolved, admins included, force push/deletion disabled, squash merge, auto-merge and merged-branch deletion enabled.

---

# 14. Atomic PRs

## Wave 0 — bootstrap/governance

### PR-001 — Bootstrap exact Python/dependencies

- **Branch:** `pr/PR-001-bootstrap-python314`
- **Depends on:** none
- **Allowed:** `.python-version`, `.gitignore`, `pyproject.toml`, `uv.lock`, `README.md`, `src/market_regime_engine/__init__.py`, bootstrap scripts, package/HMM smoke tests, `tests/conftest.py`

Acceptance:

- [ ] Canonical package identities and Python 3.14.7.
- [ ] Frozen exact MLflow/hmmlearn/runtime/dev dependencies; dependency-audit tooling included.
- [ ] No direct standalone FastAPI/Uvicorn server dependency.
- [ ] Bootstrap uses frozen lock and rejects wrong Python.
- [ ] `hmmlearn==0.3.3` full-covariance K=2 smoke fit succeeds on Python 3.14.7; otherwise PR fails without fallback.
- [ ] `.venv`, secrets, local MLflow/artifacts/caches ignored.

### PR-002 — Push quality gate

- **Branch:** `pr/PR-002-push-quality-gate`
- **Depends on:** PR-001
- **Allowed:** `.github/workflows/push-gate.yml`

Acceptance: parallel lint/type/unit/integration; frozen Python/deps; lint includes dependency audit; combined coverage >=90%; hermetic; final job exactly `push-gate`; superseded runs cancelled.

### PR-003 — Merge quality gate

- **Branch:** `pr/PR-003-merge-quality-gate`
- **Depends on:** PR-001
- **Allowed:** `.github/workflows/merge-gate.yml`

Acceptance mirrors PR-002 for PRs to main; final exactly `merge-gate`; any required-lane failure prevents success.

### PR-004 — Repository governance

- **Branch:** `pr/PR-004-repository-governance`
- **Depends on:** PR-003
- **Allowed:** `scripts/configure_github_governance.sh`, `docs/repository_governance.md`

Acceptance: targets exactly `SergejSchweizer/regime-engine/main`; authenticated admin required; applies/verifies section-13 protection/merge settings idempotently.

## Wave 1 — durable contracts/ports

### PR-005 — Synchronize durable architecture docs

- **Branch:** `pr/PR-005-architecture-contract`
- **Depends on:** PR-001
- **Allowed:** `ARCHITECTURE.md`, `CONTRIBUTING.md`, `DATA_SOURCE.md`, `EVALUATION.md`, `README.md`, `docs/model_lifecycle.md`

Acceptance: all canonical identities, current-vintage claim boundary, complete-case observation clock, final-refit requirement, one-port Gunicorn MLflow architecture, local-only Compose build/run contract, trusted-LAN/TLS source boundary and `champion` terminology match current contract-owner files; no legacy addendum/alias/profile wording.

### PR-006 — Core versioned contracts

- **Branch:** `pr/PR-006-core-domain-contracts`
- **Depends on:** PR-001
- **Allowed:** `src/market_regime_engine/contracts/*`, `tests/unit/contracts/*`

Acceptance:

- [ ] Immutable source/model/prediction/invocation/error contracts.
- [ ] Profile ID separate from config version.
- [ ] `data_time_semantics` and exact source lineage fields.
- [ ] selection definition/execution hashes distinct.
- [ ] final-refit temporal/filter-state fields representable.
- [ ] Gaussian covariance restricted to `full`.
- [ ] contract layer independent of MLflow/HTTP/filesystem/model backend.

### PR-007 — Profile schema/loader

- **Branch:** `pr/PR-007-model-profile-config`
- **Depends on:** PR-006
- **Allowed:** `src/market_regime_engine/profiles/*`, `tests/unit/profiles/*`

Acceptance: implements every pinned Xetra setting/threshold/tolerance in `EVALUATION.md`; unknown keys/duplicate features/reduced covariance fail; static-feature vs selection-policy source mutually exclusive; deterministic profile hash; no hidden defaults.

### PR-008 — FeatureSource + PostgreSQL adapter

- **Branch:** `pr/PR-008-postgres-feature-source`
- **Depends on:** PR-006
- **Allowed:** `src/market_regime_engine/features/__init__.py`, `ports.py`, `postgres_source.py`, unit/integration tests for the adapter

Acceptance:

- [ ] Loader-independent port.
- [ ] One `REPEATABLE READ READ ONLY` snapshot binds sync-state+rows then closes before model work.
- [ ] Selection mode vs resolved-model NULL semantics exactly `DATA_SOURCE.md`.
- [ ] exact ordered columns/timestamp/source bounds; no fill/carry.
- [ ] identifiers safely composed from validated contract names; no raw operator string interpolation.
- [ ] hermetic tests only.

### PR-009 — Train-only preprocessing

- **Branch:** `pr/PR-009-preprocessing-pipeline`
- **Depends on:** PR-006
- **Allowed:** `src/market_regime_engine/preprocessing/*`, unit tests

Acceptance: scaler fit only supplied retained TRAIN rows; exact order; finite parameters; variance <=`1e-12` fails; deterministic serialization; future rows cannot alter parameters.

### PR-010 — Model adapter/artifact protocols

- **Branch:** `pr/PR-010-model-adapter-protocol`
- **Depends on:** PR-006
- **Allowed:** `src/market_regime_engine/models/protocols.py`, `artifacts.py`, unit tests

Acceptance: fit/extract/reconstruct/causal-filter contracts; predictive-scoring accepts terminal TRAIN alpha; full covariance/off-diagonals preserved; shape/finite validation; no MLflow/HTTP coupling.

### PR-011 — Immutable prediction store

- **Branch:** `pr/PR-011-prediction-store`
- **Depends on:** PR-006
- **Allowed:** `src/market_regime_engine/predictions/*`, unit/integration tests

Acceptance: immutable atomic Parquet+manifest builds, explicit build ID, checksum/lineage/time semantics, explicit `walk_forward_oos` vs fixed-model modes, no silent latest research read.

### PR-012 — MLflow client/registry ports

- **Branch:** `pr/PR-012-mlflow-client-boundary`
- **Depends on:** PR-006
- **Allowed:** `src/market_regime_engine/mlflow_support/settings.py`, `ports.py`, unit tests

Acceptance: production URI exactly `http://10.10.1.3:5000`; no network at import; injectable tracking/registry; alias resolution returns exact version; fold metric history supports explicit step/timestamp; no second serving URI.

### PR-013 — MLflow custom Flask app skeleton

- **Branch:** `pr/PR-013-mlflow-app-skeleton`
- **Depends on:** PR-006
- **Allowed:** `pyproject.toml`, `src/market_regime_engine/mlflow_app/*` skeleton/contracts/errors, unit tests

Acceptance:

- [ ] entry point exactly `regime-engine = market_regime_engine.mlflow_app.app:create_app` in `mlflow.app` group.
- [ ] factory extends `mlflow.server.app` and preserves standard routes.
- [ ] no model/PG/network work at import/factory creation.
- [ ] Flask test client proves standard MLflow route + custom placeholders.
- [ ] no standalone ASGI application.

## Wave 2 — feature selection/profile resolution

### PR-045 — Feature-selection contracts

- **Branch:** `pr/PR-045-feature-selection-contracts`
- **Depends on:** PR-007
- **Allowed:** `src/market_regime_engine/feature_selection/contracts.py`, `__init__.py`, unit tests

Acceptance: immutable block/policy/evidence/result contracts exactly `EVALUATION.md`; separate definition/execution hashes; order-preserving final subset; deterministic SHA-256 canonical JSON; no source/model/MLflow work.

### PR-046 — Exact Xetra 48-feature/eight-block policy

- **Branch:** `pr/PR-046-xetra-feature-blocks`
- **Depends on:** PR-045
- **Allowed:** `configs/feature_selection/xetra_semantic_medoid_v1.yaml`, profile-selection docs/test

Acceptance: exact 48 features, exact eight block membership/order and every feature-selection constant from `EVALUATION.md`; each feature exactly once; no target/economic fields.

### PR-047 — Pure Stage-1 selector

- **Branch:** `pr/PR-047-spearman-medoid-selector`
- **Depends on:** PR-045
- **Allowed:** `src/market_regime_engine/feature_selection/selector.py`, deterministic fixtures/unit tests

Acceptance: Stage 1 exactly `EVALUATION.md` including average-rank Spearman, coverage, `ddof=0`, variance threshold, 504 complete rows and numeric tie tolerance; no Stage 2/source/HMM.

### PR-020 — Expanding walk-forward planner

- **Branch:** `pr/PR-020-walk-forward-splits`
- **Depends on:** PR-007, PR-008
- **Allowed:** `src/market_regime_engine/evaluation/walk_forward_splits.py`, unit tests

Acceptance: exact 1260/63/63/no-partial source-row plan, deterministic fold IDs/UTC bounds/hash, no synthetic rows, final evaluation cutoff exactly last complete fold `test_end`.

### PR-048 — Stage-2 prune/freeze

- **Branch:** `pr/PR-048-prune-freeze-first-train-features`
- **Depends on:** PR-020, PR-046, PR-047
- **Allowed:** `src/market_regime_engine/feature_selection/freeze.py`, unit/integration tests

Acceptance: exact first-TRAIN Stage 2; fixed matrix; strict >0.85; deterministic removals; no recompute/replacement; future rows cannot alter definition hash/evidence while execution lineage may change; failure stops evaluation.

### PR-066 — Feature-selection stability diagnostics only

- **Branch:** `pr/PR-066-feature-selection-stability-diagnostics`
- **Depends on:** PR-020, PR-048
- **Allowed:** `src/market_regime_engine/feature_selection/stability.py`, unit/integration tests

Acceptance:

- [ ] No change to frozen features/hashes/evaluation inputs.
- [ ] First-fold threshold sensitivity runs Stage 2 with thresholds exactly `0.80`, `0.85`, `0.90`; only 0.85 is canonical.
- [ ] Later-fold shadow reruns use each later fold TRAIN sample and canonical 0.85 solely for diagnostics.
- [ ] Per shadow fold output records selected ordered tuple and Jaccard overlap `|intersection|/|union|` versus frozen final tuple.
- [ ] Diagnostic failures are recorded but cannot invalidate/alter canonical selection or champion ranking.

### PR-021 — Xetra profile

- **Branch:** `pr/PR-021-xetra-profile`
- **Depends on:** PR-007, PR-046
- **Allowed:** `configs/profiles/xetra_v1.yaml`, profile doc/test

Acceptance: public `xetra`, config version 1, exact EVALUATION walk-forward/HMM/gate/ranking values including `ranking_abs_tolerance=1e-12`, exactly K2/K3/K4 full candidates, no agent-selected constants.

### PR-049 — Resolve frozen selected-feature profile

- **Branch:** `pr/PR-049-resolve-selected-feature-profile`
- **Depends on:** PR-021, PR-048
- **Allowed:** `src/market_regime_engine/profiles/resolution.py`, unit/integration tests

Acceptance: K2/K3/K4 share exact final order/dimension, both selection hashes and source build; original 48 universe/preliminary medoids retained separately; any mismatch fails before candidate comparison.

## Wave 3 — HMM/evaluation/final refit

### PR-014 — Full-covariance Gaussian HMM adapter

- **Branch:** `pr/PR-014-gaussian-hmm-adapter`
- **Depends on:** PR-009, PR-010
- **Allowed:** `src/market_regime_engine/models/gaussian_hmm.py`, fixtures/unit tests

Acceptance: exact backend/init/tolerance settings from EVALUATION; K2/K3/K4; full covariance only; exact covariance validation; off-diagonal round-trip; exposes stabilized forward primitives; backend reset TEST score cannot masquerade as OOS PLL.

### PR-015 — Deterministic multistart

- **Branch:** `pr/PR-015-hmm-multistart`
- **Depends on:** PR-014
- **Allowed:** `src/market_regime_engine/training/multistart.py`, unit tests

Acceptance: exact eight seeds, 6/8 +0.75 gates, TRAIN-loglik winner with EVALUATION numeric tie tolerance then lower seed, all start failures retained diagnostically.

### PR-016 — Causal filter + continued OOS likelihood

- **Branch:** `pr/PR-016-causal-forward-filter`
- **Depends on:** PR-014
- **Allowed:** `src/market_regime_engine/inference/filtering.py`, `predictive_likelihood.py`, unit tests

Acceptance: exact stabilized alpha recursion; explicit continuation from terminal TRAIN alpha; TEST PLL continuation hand-test; future invariance; one transition per retained observation.

### PR-017 — Transition-horizon forecast

- **Branch:** `pr/PR-017-transition-forecasts`
- **Depends on:** PR-016
- **Allowed:** `src/market_regime_engine/inference/forecasting.py`, unit tests

Acceptance: horizon is retained-observation steps; h0=current; matrix-power equivalence; finite/normalized; no calendar-day claim.

### PR-018 — Persistent state alignment

- **Branch:** `pr/PR-018-state-alignment`
- **Depends on:** PR-014
- **Allowed:** `src/market_regime_engine/states/signatures.py`, `alignment.py`, unit tests

Acceptance: exact EVALUATION signature, rounded-10-decimal first-fold sort key/tie failure, RMS cost, exhaustive K! mapping, `1e-10` ambiguity rule, drift diagnostic only, final-refit alignment reusable.

### PR-019 — Model diagnostics

- **Branch:** `pr/PR-019-model-diagnostics`
- **Depends on:** PR-014, PR-016
- **Allowed:** `src/market_regime_engine/evaluation/diagnostics.py`, unit tests

Acceptance: exact AIC/BIC, train-vs-OOS occupancy distinction/gates, natural-log entropy, confidence, retained-observation duration, actual-time switches/year, covariance validity/tolerances.

### PR-022 — Leak-free walk-forward runner

- **Branch:** `pr/PR-022-walk-forward-runner`
- **Depends on:** PR-015, PR-016, PR-018, PR-019, PR-020, PR-049
- **Allowed:** `src/market_regime_engine/evaluation/walk_forward.py`, unit/integration tests

Acceptance: frozen features never rerun; TRAIN-only scaler/HMM; TEST PLL continues from TRAIN alpha; complete-case observation clock; exact usable-row gates; state alignment only TRAIN/prior reference; filtered OOS only; invalid folds explicit; future mutation cannot change earlier folds.

### PR-023 — MLflow evaluation tracking/plots

- **Branch:** `pr/PR-023-mlflow-experiment-tracking`
- **Depends on:** PR-012, PR-022
- **Allowed:** `src/market_regime_engine/mlflow_support/tracking.py`, `plots.py`, unit/file-MLflow integration tests

Acceptance: parent/candidate/fold hierarchy, deterministic `fold_*` histories, TEST-end plot axis, invalid gaps/no interpolation, complete transition/full-covariance artifacts/heatmaps, candidate comparison plots and deterministic manifest; all `PLOT_STYLE.md` requirements.

### PR-024 — Candidate-grid orchestration

- **Branch:** `pr/PR-024-candidate-grid-orchestrator`
- **Depends on:** PR-007, PR-015, PR-022, PR-023
- **Allowed:** `src/market_regime_engine/training/candidate_grid.py`, unit/integration tests

Acceptance: exactly three candidates; identical source/folds/features/hashes; EVALUATION aggregate definitions/tolerances; invalid folds counted/excluded correctly; aligned comparison evidence.

### PR-025 — Statistical champion selection

- **Branch:** `pr/PR-025-statistical-champion-selection`
- **Depends on:** PR-019, PR-024
- **Allowed:** `src/market_regime_engine/evaluation/selection.py`, unit tests

Acceptance: exact EVALUATION hard gates + seven-stage ranking with `1e-12` numeric tie tolerance; zero valid candidate fails; full rejection/rank chain; no alias mutation/economic metric.

### PR-050 — MLflow feature-selection evidence/visual audit

- **Branch:** `pr/PR-050-mlflow-feature-selection-evidence`
- **Depends on:** PR-023, PR-048, PR-049, PR-066
- **Allowed:** `src/market_regime_engine/mlflow_support/feature_selection_tracking.py`, unit/file-MLflow integration tests

Acceptance:

- [ ] canonical selection JSON/scores/within-block/fixed 8x8/pruning evidence + both hashes;
- [ ] summary markdown with eight winners/all removals;
- [ ] Stage-1 scores plot, eight block heatmaps, fixed pre-pruning Stage-2 heatmap, removed features remain visible;
- [ ] logs PR-066 threshold-sensitivity and later-fold shadow diagnostics under `feature_selection/diagnostics/`, visibly labelled non-decision evidence;
- [ ] PLOT_STYLE/manifest source lineage complete;
- [ ] no economic/HMM-subset-selection influence.

### PR-027 — Immutable walk-forward OOS publication

- **Branch:** `pr/PR-027-oos-prediction-publication`
- **Depends on:** PR-011, PR-022
- **Allowed:** `src/market_regime_engine/predictions/oos_publication.py`, unit/integration tests

Acceptance: immutable `walk_forward_oos`, current-vintage source lineage/time semantics, folds/candidate/selection hashes, RegimePrediction.v1 rows, deterministic/idempotent publication.

### PR-063 — Mandatory final production refit

- **Branch:** `pr/PR-063-final-production-refit`
- **Depends on:** PR-015, PR-016, PR-018, PR-019, PR-025, PR-049
- **Allowed:** `src/market_regime_engine/training/final_refit.py`, `src/market_regime_engine/models/production_artifact.py`, unit/integration tests

Acceptance: exact EVALUATION final-refit contract through final planned evaluation cutoff; no selection/ranking rerun; full-sample scaler/multistart; gates reapplied; aligns to last valid winning-K fold; stores origin/trained-through/terminal alpha; OOS evaluation remains immutable.

### PR-026 — MLflow package/registry aliases

- **Branch:** `pr/PR-026-mlflow-model-registry`
- **Depends on:** PR-012, PR-063
- **Allowed:** `src/market_regime_engine/mlflow_support/model_package.py`, `registry.py`, unit/local-registry tests

Acceptance: only PR-063 production artifact registers; exact name `regime-xetra`; complete full model/filter/time/source/selection contract round-trip; aliases only `challenger`/`champion`; CAS primitives/audit evidence; no secret/source-table copy.

## Wave 4 — production PG/runtime/profile service

### PR-057 — Dedicated read-only feature-PG role

- **Branch:** `pr/PR-057-regime-engine-postgres-reader`
- **Depends on:** PR-005
- **Allowed:** `ops/postgres/regime_engine_reader.sql`, bootstrap/verify scripts, SQL unit test, `docs/ops/feature_postgres_reader.md`

Acceptance: exact quoted role; idempotent least privileges from DATA_SOURCE; read-only defaults; mandatory runtime DB/password/admin secret; DB identifier safely passed/quoted by psql rather than raw shell/SQL interpolation; no row/server/writer mutation; catalog privilege verification only.

### PR-058 — Pooled production PG runtime

- **Branch:** `pr/PR-058-postgres-serving-runtime`
- **Depends on:** PR-008
- **Allowed:** `src/market_regime_engine/features/postgres_settings.py`, `postgres_pool.py`, unit/runtime integration tests

Acceptance: exact names/defaults/TLS from DATA_SOURCE/section4; password-file support; lazy process-local pool; acquire/statement timeout; pool closes on worker shutdown; startup connection-budget validation; no credential logging; hermetic tests.

### PR-056 — Profile/model resolver/cache

- **Branch:** `pr/PR-056-profile-model-resolver-cache`
- **Depends on:** PR-007, PR-012, PR-026
- **Allowed:** `src/market_regime_engine/serving/profile_registry.py`, `model_resolver.py`, `model_cache.py`, unit tests

Acceptance: data-driven `xetra -> regime-xetra@champion`; explicit version bypass; TTL; single-flight; max-two-version LRU/ref-count safety; full package validation before swap; invalid new champion fails without stale mislabelling; future crypto mapping requires no route change.

### PR-059 — Replay guardrails

- **Branch:** `pr/PR-059-replay-guardrails`
- **Depends on:** PR-013
- **Allowed:** `src/market_regime_engine/serving/replay_limits.py`, `replay_admission.py`, unit tests

Acceptance: all section4 limits; inclusive-range syntax validation; process semaphore; no extra executor; monotonic cooperative deadline; exact 413/503/504 behavior; slot released only after work stopped.

### PR-029 — Latest handler

- **Branch:** `pr/PR-029-latest-handler`
- **Depends on:** PR-016, PR-018, PR-056, PR-058
- **Allowed:** `src/market_regime_engine/inference/latest.py`, `src/market_regime_engine/serving/latest_handler.py`, unit/integration tests

Acceptance: exact API input; version pinned before source; omitted-as-of uses validated source max; complete-case latest; continuation from stored terminal alpha; exact staleness formulas/warn/fail; RegimePrediction/lineage; no fallback.

### PR-028 — Fixed-model replay handler

- **Branch:** `pr/PR-028-replay-handler`
- **Depends on:** PR-016, PR-018, PR-056, PR-058, PR-059
- **Allowed:** `src/market_regime_engine/inference/replay.py`, `src/market_regime_engine/serving/replay_handler.py`, unit/integration tests

Acceptance: exact API input; model pinned first; EVALUATION warmup/start-invariance; current-vintage source metadata; guardrails/no truncation; same-version+same-source overlapping replay invariant for different requested starts.

### PR-030 — OOS retrieval handler

- **Branch:** `pr/PR-030-oos-prediction-handler`
- **Depends on:** PR-011, PR-027
- **Allowed:** `src/market_regime_engine/predictions/query.py`, `src/market_regime_engine/serving/oos_handler.py`, unit/integration tests

Acceptance: explicit profile/build; bounded UTC slice; no silent latest; `walk_forward_oos` only; no Flask route here.

### PR-060 — Final MLflow app composition

- **Branch:** `pr/PR-060-compose-profile-service`
- **Depends on:** PR-013, PR-028, PR-029, PR-030, PR-056, PR-058, PR-059
- **Allowed:** `src/market_regime_engine/mlflow_app/app.py`, `dependencies.py`, `dispatch.py`, unit/integration app tests

Acceptance: exact routes/schemas/status/error envelope; body profile/unknown fields rejected; injected resolver/source/handlers/guardrails; no HMM math in route; standard MLflow routes still work; custom health is non-secret/readiness-only; no Prometheus.

### PR-031 — Operator CLI

- **Branch:** `pr/PR-031-application-cli`
- **Depends on:** PR-024, PR-026, PR-027, PR-050, PR-063
- **Allowed:** `src/market_regime_engine/cli.py`, `src/market_regime_engine/commands/*`, CLI tests, `pyproject.toml`

Acceptance: evaluate/final-refit/register/publish-oos/status; thin adapters; no standalone `serve`; deterministic errors/help; OOS publication dependency explicit.

### PR-064 — Lifecycle/promotion/rollback/freshness

- **Branch:** `pr/PR-064-model-lifecycle-operations`
- **Depends on:** PR-031, PR-056
- **Allowed:** `src/market_regime_engine/commands/lifecycle.py`, `scripts/model_cycle.sh`, lifecycle tests/doc

Acceptance:

- [ ] CAS promote/rollback.
- [ ] New-source-build detection; unchanged build is a deterministic no-op.
- [ ] Changed build executes evaluate -> statistical select -> final refit -> OOS publication -> register challenger.
- [ ] Promotion remains explicit and uses expected-current-version plus non-empty reason.
- [ ] Exact 7-day recommended cadence and staleness thresholds; no economic/uncalibrated drift alias decision.
- [ ] `scripts/model_cycle.sh` is cron-safe and invokes the installed CLI inside the local Compose `mlflow` container with `docker compose exec -T mlflow ...` from the local repository checkout.
- [ ] The lifecycle script never invokes a remote Docker context, remote container host, registry-hosted application image or a second Python environment on the NAS.
- [ ] A process/profile single-run lock prevents overlapping scheduled `xetra` model cycles.

## Wave 5 — image/compose/external proof/operations

### PR-032 — Unified MLflow/regime-engine locally built image

- **Branch:** `pr/PR-032-container-image`
- **Depends on:** PR-031, PR-060
- **Allowed:** `Dockerfile`, `.dockerignore`, `docker/python-base.lock`, `scripts/mlflow_entrypoint.sh`, `scripts/mlflow_db_upgrade.sh`, image-contract test, `docs/container_image.md`

Acceptance:

- [ ] Resolve/commit exact official SHA-256 for `python:3.14.7-slim-bookworm`; Dockerfile uses tag+digest as the local build input.
- [ ] MLflow exactly 3.15.1/frozen package install.
- [ ] Repository-root Docker build context produces the application image locally; no remote build service is required or allowed.
- [ ] Canonical locally built application tag is exactly `regime-engine-mlflow:local`.
- [ ] No Docker Hub/GHCR/private-registry application image is referenced, pushed or required by build, deployment or tests.
- [ ] Local image labels/evidence include repository Git SHA and MLflow/package version; tests verify those labels can be read from the locally built image.
- [ ] One Gunicorn MLflow process with exact app/defaults; never Uvicorn/model-serve/proxy/Prometheus.
- [ ] Normal startup performs no DB migration; one-shot explicit upgrade script.
- [ ] Non-root; secrets not logged.
- [ ] Produces deterministic dependency/SBOM artifact for local image inspection.

### PR-033 — Real feature-PostgreSQL compatibility smoke

- **Branch:** `pr/PR-033-feature-postgres-smoke`
- **Depends on:** PR-021, PR-057, PR-058
- **Allowed:** external PG test/verify script plus deterministic loader-shaped fixture integration

Acceptance: required integration hermetic; external opt-in target exact host/port, runtime DB/password, user `regime-engine`, `sslmode=require`; proves SELECT/source transaction/privileges/read-only; no destructive/write test.

### PR-061 — Exact local two-service Compose

- **Branch:** `pr/PR-061-two-service-mlflow-compose`
- **Depends on:** PR-032, PR-057, PR-058, PR-059, PR-060
- **Allowed:** `compose.yaml`, `.env.example`, `docker/postgres-backend.lock`, `scripts/local_compose_build.sh`, `scripts/local_compose_up.sh`, `scripts/local_compose_down.sh`, `scripts/verify_local_compose.sh`, compose tests, `docs/deployment.md`

Acceptance:

- [ ] `compose.yaml` declares exactly `mlflow` + `mlflow-postgres`; only host port 5000 is published.
- [ ] Compose project runs on the same local Docker daemon/host where the repository checkout exists; scripts reject TCP/SSH remote Docker endpoints and remote builders.
- [ ] `mlflow` declares local `build.context: .`, the repository Dockerfile, `image: regime-engine-mlflow:local`, and `pull_policy: never`.
- [ ] The only supported production application-image build is explicit `docker compose build --pull mlflow` from the local repository checkout.
- [ ] The only supported normal production start is `docker compose up -d --no-build`; it never builds or pulls the custom application image implicitly.
- [ ] If `regime-engine-mlflow:local` does not exist locally, the startup wrapper fails clearly before `up` instead of pulling from a registry.
- [ ] No `docker compose push`, application-image `pull`, registry login, remote buildx builder, Swarm or Kubernetes path exists in deployment scripts/docs.
- [ ] Resolve/commit exact official SHA-256 for `postgres:18.6-alpine`; `mlflow-postgres` uses the pinned official image locally and may pull that exact digest only when absent.
- [ ] Persistent backend/artifact volumes; backend DB private.
- [ ] Docker secrets for backend and feature passwords; env examples placeholders only.
- [ ] Separate namespaced backend vs feature DB settings.
- [ ] Exact workers/threads/pool/budget/replay/staleness/BLAS settings.
- [ ] Allowed-host/CORS trusted-LAN policy; no wildcard/proxy/Prometheus/5001.
- [ ] No automatic DB migration.
- [ ] `scripts/verify_local_compose.sh` records/verifies running project name, service set, local application image ID, repository Git SHA, MLflow version, port mapping and absence of remote custom-image provenance.
- [ ] Compose contract tests fail if production file is renamed to/example-only `compose.example.yaml`, if `mlflow` loses `build`, if `pull_policy` permits registry pulls, or if normal startup includes `--build`.

### PR-034 — Real locally running unified MLflow smoke

- **Branch:** `pr/PR-034-external-mlflow-smoke`
- **Depends on:** PR-023, PR-026, PR-061
- **Allowed:** external MLflow/regime-service tests and verify script

Acceptance:

- [ ] External smoke is opt-in only.
- [ ] Precondition/verification confirms the service under test is the locally running `compose.yaml` deployment and its `mlflow` container uses the locally built `regime-engine-mlflow:local` image ID recorded by PR-061 verification.
- [ ] The same local `:5000` supports standard MLflow + custom health.
- [ ] Disposable tracking/registry/artifact/fold-history round-trip succeeds.
- [ ] Optional read-only `xetra` latest succeeds if champion exists.
- [ ] No 5001/proxy/Prometheus/remote-registry assumption exists.

### PR-062 — Hermetic capacity/failure isolation proof

- **Branch:** `pr/PR-062-serving-capacity-proof`
- **Depends on:** PR-060, PR-061
- **Allowed:** serving capacity/failure integration tests + fixtures

Acceptance: cache single-flight/LRU/atomic swap; pool timeout; all replay 413/503/504 paths; no hidden work after 504; resources released; exact default 4-worker x4-thread topology keeps health/tracking/registry read/latest serviceable at admitted 4-replay capacity; tests require no published application image and can exercise the locally built Compose image where Docker is available; no secret/raw-feature logs.

### PR-065 — Local Compose backup/restore/migration/secret rotation

- **Branch:** `pr/PR-065-mlflow-backup-restore`
- **Depends on:** PR-061
- **Allowed:** backup/restore/verify scripts, manifest-contract test, operations doc

Acceptance:

- [ ] Operations target the local `regime-engine` Compose project through local `docker compose` commands; no remote Docker host or registry artifact is required.
- [ ] Quiesced DB dump + artifact archive + version/hash manifest.
- [ ] Manifest records local application image ID, Git SHA, MLflow/PostgreSQL versions and source archive hashes.
- [ ] Matching restore and artifact/metadata verification.
- [ ] Mandatory successful backup before explicit MLflow DB upgrade; no automatic migration.
- [ ] After an application/dependency change requiring a new image, rebuild is explicit/local before restart.
- [ ] Exact backend/feature secret-rotation procedure with verify-before-revoke.

### PR-035 — Complete hermetic E2E

- **Branch:** `pr/PR-035-engine-e2e-proof`
- **Depends on:** PR-024, PR-026, PR-027, PR-028, PR-029, PR-030, PR-050, PR-060, PR-062, PR-063, PR-066
- **Allowed:** E2E test + fixtures only

Acceptance:

- [ ] 48 source features -> canonical feature selection -> K2/K3/K4 walk-forward -> statistical champion -> mandatory final refit -> local MLflow version.
- [ ] selection definition/execution hash behavior and PR-066 diagnostics proven non-decision.
- [ ] continued TEST PLL and deterministic state alignment exact.
- [ ] full MLflow fold/plot/feature-selection evidence.
- [ ] registered model is final refit, never fold model.
- [ ] profile invocation latest/replay and different-start overlap invariance.
- [ ] OOS remains distinct/current-vintage limitation explicit.
- [ ] guardrail/error paths hermetic; no NAS.

### PR-036 — Final documentation consistency

- **Branch:** `pr/PR-036-final-documentation`
- **Depends on:** PR-033, PR-034, PR-035, PR-061, PR-064, PR-065
- **Allowed:** `README.md`, `API.md`, `OPERATIONS.md`, `ARCHITECTURE.md`, `DATA_SOURCE.md`, `EVALUATION.md`, `CONTRIBUTING.md`, consumer/integration docs

Acceptance: all contract-owner docs consistent; no duplicate/override/addendum text; canonical names only; current-vintage limitation; complete-case clock; selection hashes/diagnostics; continued OOS PLL; final refit; exact API; one-port Gunicorn Compose; exact local `docker compose build --pull mlflow` then `docker compose up -d --no-build` workflow; local application image only/no application registry; cron uses local `docker compose exec -T mlflow`; TLS feature PG; trusted-LAN; cache/load/replay/staleness; lifecycle/CAS; backup/migration/secret rotation; image/version pinning; no Prometheus/5001/proxy.

## Optional post-MVP challengers

### PR-037 — Student-t HMM challenger

- **Branch:** `pr/PR-037-student-t-hmm-challenger`
- **Depends on:** PR-010, PR-022, PR-036
- Requires a dedicated explicit dependency/backend/profile contract change; full covariance, causal continuation and common evaluation interfaces remain mandatory.

### PR-038 — HSMM challenger

- **Branch:** `pr/PR-038-hsmm-challenger`
- **Depends on:** PR-010, PR-022, PR-036
- Requires an explicit duration/protocol contract; no implicit change to MVP candidate universe.

---

# 15. Parallel execution graph

Only merged dependencies unlock work.

```text
A1: PR-001
A2 parallel: PR-002 PR-003 PR-005 PR-006
A3: PR-004 after PR-003

B1 parallel after PR-006: PR-007 PR-008 PR-009 PR-010 PR-011 PR-012 PR-013
B2: PR-045 after 007; PR-014 after 009+010; PR-020 after 007+008; PR-057 after 005; PR-058 after 008; PR-059 after 013
B3 parallel: PR-046+047 after 045; PR-015+016+018 after 014
B4: PR-019 after 014+016; PR-048 after 020+046+047; PR-021 after 007+046
B5 parallel: PR-066 after 020+048; PR-049 after 021+048

C1: PR-022 after 015+016+018+019+020+049
C2 parallel: PR-023 after 012+022; PR-027 after 011+022
C3 parallel: PR-024 after 007+015+022+023; PR-050 after 023+048+049+066
C4: PR-025 after 019+024; PR-063 after 015+016+018+019+025+049; PR-026 after 012+063

D1: PR-056 after 007+012+026; PR-033 when 021+057+058 ready
D2 parallel: PR-029 after 016+018+056+058; PR-028 after 016+018+056+058+059; PR-030 after 011+027
D3: PR-060 after 013+028+029+030+056+058+059; PR-031 after 024+026+027+050+063
D4 parallel: PR-032 after 031+060; PR-064 after 031+056
D5: PR-061 after 032+057+058+059+060
D6 parallel: PR-034 after 023+026+061; PR-062 after 060+061; PR-065 after 061
D7: PR-035 after declared dependencies
D8: PR-036 final
```

Parallel lanes are deliberately separated by file ownership. If a contract-owner file conflict appears, the later PR rebases after dependency merge rather than broad-rewriting the file.

---

# 16. Completion states

## Code-complete MVP

Code-complete means all required PRs through PR-036 plus PR-056–066 are merged, all required hermetic gates pass, and the E2E proof passes. It does **not** claim the NAS runtime has been provisioned successfully.

## Deployment-ready MVP

Deployment-ready additionally requires operator evidence that:

1. the external `"regime-engine"` feature reader was created with the runtime database/secret;
2. feature PostgreSQL accepts TLS `sslmode=require`;
3. PR-033 external feature-PG smoke passes;
4. a clean/up-to-date repository checkout exists on the NAS `10.10.1.3` and Compose targets that host's local Unix-socket Docker daemon;
5. the custom application image was built locally with `docker compose build --pull mlflow`, is present as `regime-engine-mlflow:local`, and its recorded local image ID matches the deployment evidence;
6. the exact two-service `compose.yaml` deployment was started locally with `docker compose up -d --no-build` and is running at `10.10.1.3:5000`;
7. PR-034 unified MLflow/custom-service smoke passes against that local deployment;
8. the configured feature-PG connection budget is adequate for the deployed workers/pools;
9. an initial MLflow backend+artifact backup has been created and verified with local image/Git provenance;
10. port 5000 is restricted to the trusted private LAN and is not public;
11. there is no remote/published `regime-engine-mlflow` application image dependency, no remote Docker context/build service and no implicit build during normal startup;
12. no `:5001`, reverse proxy, Prometheus exposure or second serving process exists.

Only after both states are satisfied is the unified-serving MVP operationally complete.