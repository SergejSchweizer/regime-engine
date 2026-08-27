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

$$
\text{MLFLOW_WORKERS} \cdot \text{REGIME_PG_POOL_MAX_SIZE}
\le \text{REGIME_FEATURE_PG_CONNECTION_BUDGET}.
$$

With defaults, $4 \cdot 4 \le 16$ holds exactly.

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

Feature PostgreSQL transport is explicitly plaintext on the trusted LAN (`sslmode=disable`) because the canonical server does not offer TLS. An implementation may not silently change it.

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

Acceptance: exact names/defaults/plain-transport contract from DATA_SOURCE/section4; password-file support; lazy process-local pool; acquire/statement timeout; pool closes on worker shutdown; startup connection-budget validation; no credential logging; hermetic tests.

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

Acceptance: required integration hermetic; external opt-in target exact host/port, runtime DB/password, user `regime-engine`, `sslmode=disable`; proves non-TLS SELECT/source transaction/privileges/read-only; no destructive/write test.

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

Acceptance: all contract-owner docs consistent; no duplicate/override/addendum text; canonical names only; current-vintage limitation; complete-case clock; selection hashes/diagnostics; continued OOS PLL; final refit; exact API; one-port Gunicorn Compose; exact local `docker compose build --pull mlflow` then `docker compose up -d --no-build` workflow; local application image only/no application registry; cron uses local `docker compose exec -T mlflow`; explicit plaintext feature PG on the trusted LAN; cache/load/replay/staleness; lifecycle/CAS; backup/migration/secret rotation; image/version pinning; no Prometheus/5001/proxy.

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
2. feature PostgreSQL accepts explicit plaintext `sslmode=disable` on the trusted LAN;
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

---

# 17. Evaluation correctness audit — 2026-08-27

This corrective backlog was created after a source-level audit of the current evaluation, model-family, alignment, ranking, MLflow-plot and Xetra-v2 lifecycle paths. These PRs are corrective work on top of the already-merged implementation. They do not authorize economic metrics to enter model selection.

Execution rule: PRs below follow the same clean-main, allowed-file and full-test rules from section 12. Dependencies are explicit; agents must not combine PRs.

### PR-117 — Restore immutable Xetra v1 Gaussian candidate set

- **Branch:** `pr/PR-117-restore-xetra-v1-candidates`
- **Depends on:** none
- **Allowed:** `configs/profiles/xetra_v1.yaml`, `src/market_regime_engine/profiles/config.py`, `tests/unit/profiles/test_loader.py`, `tests/unit/profiles/test_xetra_profile.py`

Acceptance:

- [ ] `xetra_v1.yaml` contains exactly Gaussian `candidate_states: [2, 3, 4]`; K5 is not silently added to profile version 1.
- [ ] Generic Gaussian config validation no longer globally forces `(2,3,4,5)` before the enclosing profile version is known.
- [ ] `ModelProfile`/pinned-profile validation enforces v1 Gaussian candidate states exactly `(2,3,4)` and rejects any v1 K5.
- [ ] Existing v1 backend, covariance, seeds, fitting parameters, walk-forward values, gates and ranking tolerance remain unchanged.
- [ ] v2 remains loadable with Gaussian K2/K3/K4/K5.
- [ ] Tests prove v1 and v2 candidate-state contracts are distinct and immutable.

### PR-118 — Make candidate identities explicitly profile-version aware

- **Branch:** `pr/PR-118-versioned-candidate-universe`
- **Depends on:** PR-117
- **Allowed:** `EVALUATION.md`, `src/market_regime_engine/profiles/resolution.py`, `src/market_regime_engine/training/candidate_grid.py`, `tests/unit/profiles/test_resolution.py`, `tests/unit/training/test_candidate_grid.py`

Acceptance:

- [ ] `EVALUATION.md` defines candidate universes separately: v1 = Gaussian K2/K3/K4; v2 = Gaussian K2–K5 + two-mixture full-covariance GMM K2–K5 + Student-t K2–K5.
- [ ] Resolution selects the exact expected candidate-ID tuple from `profile_config_version`; it never infers a universe from list length.
- [ ] Candidate-grid validation compares against the exact version-specific tuple; `EXPECTED_CANDIDATE_IDS[:len(...)]` prefix semantics are removed.
- [ ] A candidate set with correct length but wrong identity/order fails closed.
- [ ] All candidates in one grid still share source build, fold plan, frozen feature order and both feature-selection hashes.
- [ ] Tests cover exact v1 and v2 identities plus wrong-order, missing-middle and unexpected-extra cases.

### PR-119 — Define transitive anchored numeric-tolerance semantics

- **Branch:** `pr/PR-119-anchored-tolerance-contract`
- **Depends on:** PR-118
- **Allowed:** `EVALUATION.md`

Acceptance:

- [ ] The `1e-12` tolerance is specified as an anchored equivalence rule, not a pairwise comparator relation.
- [ ] For a maximize stage, the anchor is the exact maximum in the current candidate set and the tied set is every value `>= anchor - 1e-12`; for a minimize stage it is the exact minimum and every value `<= anchor + 1e-12`.
- [ ] Feature-selection secondary tie stages are evaluated only inside the anchored tie set from the preceding stage.
- [ ] Multistart winner semantics are explicit: compute global maximum TRAIN log likelihood first, retain starts within `1e-12` of that maximum, then choose the lowest seed.
- [ ] Champion ranking is defined by recursively partitioning each current group from its exact best anchor at each numeric stage, so results are deterministic and transitive.
- [ ] The contract includes the adversarial chain example `a=0`, `b=0.75e-12`, `c=1.5e-12` and states that pairwise tolerance chaining is forbidden.
- [ ] No ranking stage, tolerance value or economic input is otherwise changed.

### PR-120 — Fix Stage-1 medoid tie resolution

- **Branch:** `pr/PR-120-stage1-anchored-ties`
- **Depends on:** PR-119
- **Allowed:** `src/market_regime_engine/feature_selection/selector.py`, `tests/unit/feature_selection/test_selector.py`

Acceptance:

- [ ] Stage-1 minimum medoid score is computed globally within the block before any tie-break.
- [ ] Only features within `1e-12` of that global minimum advance to coverage comparison.
- [ ] Coverage uses the global maximum among that tied subset and the same anchored `1e-12` rule.
- [ ] Earliest configured block position is used only after both numeric anchored tie stages.
- [ ] Result is invariant to internal iteration/order changes that do not change configured feature order.
- [ ] Regression test covers a three-value non-transitive tolerance chain and proves the globally anchored winner.
- [ ] Spearman definition, coverage formula, variance gate and Stage-2 behavior are untouched.

### PR-121 — Fix multistart global-best tie resolution

- **Branch:** `pr/PR-121-multistart-anchored-winner`
- **Depends on:** PR-119
- **Allowed:** `src/market_regime_engine/training/multistart.py`, `tests/unit/training/test_multistart.py`

Acceptance:

- [ ] All successful/converged starts are collected before winner selection.
- [ ] Winner selection first computes the exact global maximum TRAIN log likelihood.
- [ ] Eligible tied starts satisfy `global_max - start_loglik <= 1e-12`.
- [ ] Lowest seed among the eligible tied starts wins.
- [ ] The sequential pairwise `_better` behavior is removed from winner determination.
- [ ] Regression test covers log likelihoods separated by chained sub-tolerance gaps where pairwise iteration would choose a different seed.
- [ ] Eight seeds, 6/8 minimum, 0.75 success-rate gate and diagnostic retention remain unchanged.

### PR-122 — Make statistical champion ranking transitive

- **Branch:** `pr/PR-122-transitive-champion-ranking`
- **Depends on:** PR-119
- **Allowed:** `src/market_regime_engine/evaluation/selection.py`, `tests/unit/evaluation/test_selection.py`

Acceptance:

- [ ] `cmp_to_key` pairwise tolerance ranking is removed.
- [ ] Accepted candidates are recursively ranked using the anchored-group semantics in `EVALUATION.md` for all five numeric stages.
- [ ] Stage order remains exactly: OOS mean desc, OOS std asc, OOS worst desc, BIC mean asc, AIC mean asc, K asc, candidate ID asc.
- [ ] K and candidate ID remain exact tie-breakers and do not use numeric tolerance.
- [ ] Ranking is invariant to the input order of `grid.aggregates`.
- [ ] Tests cover the `0 / 0.75e-12 / 1.5e-12` non-transitive chain at every numeric direction and randomized aggregate permutations.
- [ ] Hard gates and rejection reasons remain unchanged.

### PR-123 — Build state signatures in a fixed alignment coordinate system

- **Branch:** `pr/PR-123-state-signature-coordinate-transform`
- **Depends on:** PR-118
- **Allowed:** `EVALUATION.md`, `src/market_regime_engine/states/signatures.py`, `tests/unit/states/test_alignment.py`, `tests/unit/preprocessing/test_scaling.py`

Acceptance:

- [ ] `EVALUATION.md` states that persistent-state signatures from different folds must be expressed in one fixed feature coordinate system; direct RMS comparison of parameters standardized by different fold scalers is forbidden.
- [ ] The fixed alignment coordinate system is the scaler fitted on retained TRAIN observations of the first planned fold for the frozen final feature set; it is evaluation evidence only and does not replace fold-local TRAIN scaling for model fitting.
- [ ] A pure helper converts a fitted emission mean/covariance from a fold-local standardized system into the fixed alignment system.
- [ ] For fold scaler mean `m_f`, scale `s_f`, reference mean `m_r`, scale `s_r`, transformed mean is `(m_f + s_f ⊙ μ_f - m_r) / s_r`.
- [ ] With `D = diag(s_f / s_r)`, transformed covariance is `D Σ_f D`; no diagonal-only approximation is allowed.
- [ ] Identity-scaler and same-scaler cases reproduce the original signature exactly within numerical tolerance.
- [ ] Tests prove that two mathematically identical raw-space emissions fitted under different scalers yield identical transformed signatures.
- [ ] No OOS observation is used to fit the fixed alignment scaler.

### PR-124 — Use the fixed signature coordinate for fold and final-refit alignment

- **Branch:** `pr/PR-124-wire-fixed-alignment-coordinate`
- **Depends on:** PR-123
- **Allowed:** `src/market_regime_engine/states/alignment.py`, `src/market_regime_engine/evaluation/walk_forward.py`, `src/market_regime_engine/training/final_refit.py`, `tests/unit/evaluation/test_walk_forward.py`, `tests/unit/evaluation/test_walk_forward_validation.py`, `tests/unit/training/test_final_refit.py`, `tests/unit/states/test_alignment.py`

Acceptance:

- [ ] Walk-forward constructs the fixed alignment scaler exactly once from first-planned-fold retained TRAIN rows and never refits it on later folds.
- [ ] First valid model fold, every later valid fold and final production refit are transformed into that same coordinate system before first-sort/RMS alignment.
- [ ] `StateAlignment.aligned_signatures`, `matched_rms`, `total_cost` and `max_drift` refer only to fixed-coordinate signatures.
- [ ] Fold-local scaler continues to be fit on that fold TRAIN only and remains the scaler used for HMM training/OOS filtering.
- [ ] Final-refit model inference continues to use its final-refit scaler; only its alignment signature is transformed.
- [ ] Future TEST mutations cannot change the fixed alignment scaler or any earlier state mapping.
- [ ] Regression fixture with materially different fold scalers preserves the same persistent mapping for unchanged raw-space regimes.
- [ ] Ambiguity tolerance and exhaustive permutation logic remain unchanged.

### PR-125 — Compare accepted candidates on common valid folds

- **Branch:** `pr/PR-125-common-valid-fold-ranking`
- **Depends on:** PR-122
- **Allowed:** `EVALUATION.md`, `src/market_regime_engine/training/candidate_grid.py`, `src/market_regime_engine/evaluation/selection.py`, `tests/unit/training/test_candidate_grid.py`, `tests/unit/evaluation/test_selection.py`

Acceptance:

- [ ] Existing per-candidate valid-fold-rate gate `>=0.80` is applied first and unchanged.
- [ ] After hard-gate rejection, the ranking support is the intersection of valid fold IDs across all remaining accepted candidates.
- [ ] `common_valid_fold_rate = len(common_valid_fold_ids) / planned_fold_count` is recorded and must also be `>=0.80`; otherwise champion selection fails closed because candidates are not sufficiently comparable.
- [ ] OOS mean, population std (`ddof=0`), OOS worst, BIC mean and AIC mean used by champion ranking are recomputed from exactly the common fold IDs.
- [ ] Candidate-specific valid-fold aggregates remain available only as diagnostics and are named distinctly from common-support ranking metrics.
- [ ] No invalid fold is imputed, interpolated or assigned a fabricated penalty.
- [ ] Evidence records the exact common fold IDs and count.
- [ ] Adversarial test proves that a candidate cannot improve its rank merely by being invalid on the hardest otherwise-comparable folds.

### PR-126 — Validate every GMM component covariance fail-closed

- **Branch:** `pr/PR-126-gmm-component-covariance-validation`
- **Depends on:** PR-118
- **Allowed:** `src/market_regime_engine/models/artifacts.py`, `src/market_regime_engine/models/gaussian_hmm.py`, `src/market_regime_engine/evaluation/diagnostics.py`, `tests/unit/models/test_artifacts.py`, `tests/unit/models/test_gaussian_hmm.py`, `tests/unit/evaluation/test_diagnostics.py`

Acceptance:

- [ ] Every `mixture_full_covariances[state][mixture]` receives the same finite shape, maximum asymmetry `<=1e-10`, minimum diagonal variance `>=1e-12` and Cholesky-without-jitter validation as a full state covariance.
- [ ] Validation occurs before any emission-density calculation or backend reconstruction.
- [ ] GMM emission evaluation may symmetrize only after the explicit asymmetry gate has passed.
- [ ] A malformed component covariance cannot pass merely because the moment-matched aggregate state covariance is positive definite.
- [ ] Regression test constructs an invalid component covariance with an otherwise valid aggregate state covariance and requires failure.
- [ ] Gaussian and Student-t validation behavior is not weakened.
- [ ] GMM parameter-count/AIC/BIC formulas are unchanged.

### PR-127 — Separate Student-t scale matrices from distribution covariance

- **Branch:** `pr/PR-127-student-t-scale-covariance-semantics`
- **Depends on:** PR-126
- **Allowed:** `EVALUATION.md`, `src/market_regime_engine/models/artifacts.py`, `src/market_regime_engine/models/gaussian_hmm.py`, `src/market_regime_engine/models/student_t_hmm.py`, `src/market_regime_engine/states/signatures.py`, `src/market_regime_engine/mlflow_support/plots.py`, `tests/unit/models/test_artifacts.py`, `tests/unit/models/test_student_t_hmm.py`, `tests/unit/states/test_alignment.py`, plot unit tests

Acceptance:

- [ ] `EVALUATION.md` explicitly distinguishes multivariate Student-t scale matrix `S` from covariance `C`.
- [ ] For state degrees of freedom `ν>2`, the canonical distribution covariance is exactly `C = ν/(ν-2) * S`.
- [ ] Persisted Student-t emission matrices remain scale matrices for density evaluation; no silent artifact-format reinterpretation is introduced.
- [ ] One backend-neutral helper exposes distribution covariance to code that semantically needs covariance/standard deviation.
- [ ] State-signature standard deviations and any plot/diagnostic labelled covariance or standard deviation use `C`, not `S`.
- [ ] Student-t log-density/quadratic-form calculations continue to use `S`.
- [ ] Exact unit test verifies the covariance factor and verifies signature log-standard-deviation shift `0.5*ln(ν/(ν-2))`.
- [ ] Gaussian and GMM covariance semantics remain unchanged.

### PR-128 — Wire Student-t profile settings into evaluation and final refit

- **Branch:** `pr/PR-128-wire-student-t-profile-settings`
- **Depends on:** PR-118
- **Allowed:** `src/market_regime_engine/training/candidate_grid.py`, `src/market_regime_engine/training/final_refit.py`, `src/market_regime_engine/models/student_t_hmm.py`, `tests/unit/training/test_candidate_grid.py`, `tests/unit/training/test_final_refit.py`, `tests/unit/models/test_student_t_hmm.py`

Acceptance:

- [ ] Student-t adapter settings are built from the active `ModelProfile.student_t_hmm` contract rather than silently relying on `StudentTHMMSettings()` defaults.
- [ ] `minimum_nu`, `maximum_nu`, `initial_nu`, `n_iter`, `tol` and `min_covar` are passed explicitly in walk-forward evaluation.
- [ ] Final production refit uses exactly the same Student-t settings as the winning evaluation profile.
- [ ] No mutable/global settings object is shared between parallel candidate workers.
- [ ] A test changes one legal profile setting and proves the constructed evaluation and final-refit adapters receive that exact value.
- [ ] Missing Student-t config for a Student-t candidate fails before fitting.
- [ ] Gaussian/GMM adapter construction is unchanged.

### PR-129 — Enforce TRAIN likelihood parity before AIC/BIC

- **Branch:** `pr/PR-129-train-likelihood-parity`
- **Depends on:** PR-126, PR-127, PR-128
- **Allowed:** `EVALUATION.md`, `src/market_regime_engine/evaluation/walk_forward.py`, `tests/unit/evaluation/test_walk_forward.py`, `tests/unit/evaluation/test_walk_forward_validation.py`

Acceptance:

- [ ] After fitting the winning start, TRAIN likelihood is recomputed by the same causal forward/emission implementation used for OOS filtering.
- [ ] Fit-returned and causal-filter TRAIN log likelihoods must satisfy $|\Delta \mathcal{L}| \le 10^{-10} \max(1, |\mathcal{L}_{\mathrm{fit}}|, |\mathcal{L}_{\mathrm{filter}}|)$; the exact rule is documented.
- [ ] A parity failure invalidates the fold with an explicit reason; it is never silently averaged.
- [ ] After parity passes, the causal-filter TRAIN log likelihood is the canonical value supplied to AIC/BIC and stored as fold TRAIN likelihood.
- [ ] TEST continuation logic and TEST-only likelihood sum are unchanged.
- [ ] Tests inject a deliberately inconsistent adapter and require fold failure.
- [ ] Gaussian, GMM and Student-t happy-path fixtures all pass the parity check.

### PR-130 — Make weighted OOS plots explicitly diagnostic-only

- **Branch:** `pr/PR-130-weighted-oos-plot-semantics`
- **Depends on:** PR-125
- **Allowed:** `src/market_regime_engine/mlflow_support/plots.py`, `src/market_regime_engine/mlflow_support/tracking.py`, `tests/unit/mlflow_support/test_tracking.py`, plot unit tests, `PLOT_STYLE.md`

Acceptance:

- [ ] Observation-weighted pooled OOS likelihood may remain as a diagnostic value but is never called the statistical rank, score, winner or best candidate.
- [ ] Candidate plot order/highlight uses canonical statistical champion rank when selection evidence is available; deterministic candidate ID order is the fallback for diagnostics without selection evidence.
- [ ] Plot titles/legends explicitly label weighted pooled OOS likelihood as `diagnostic only`.
- [ ] No plot is ordered by weighted OOS likelihood in a way that can visually imply champion selection.
- [ ] A regression fixture where the weighted diagnostic winner differs from the canonical statistical champion still highlights/orders the canonical champion correctly.
- [ ] The canonical unweighted valid/common-fold ranking metrics remain unchanged.
- [ ] No weighted value is fed back into candidate selection or registry logic.

### PR-131 — Remove automatic champion promotion from the Xetra v2 cycle

- **Branch:** `pr/PR-131-explicit-champion-promotion`
- **Depends on:** PR-118
- **Allowed:** `scripts/run_xetra_v2_cycle.py`, `src/market_regime_engine/commands/lifecycle.py`, `tests/unit/commands/test_lifecycle.py`, script contract tests

Acceptance:

- [ ] A successful Xetra v2 evaluation/refit registers the new model and may CAS-update only the `challenger` alias.
- [ ] The cycle never mutates the `champion` alias automatically.
- [ ] Promotion to `champion` remains a separate explicit operator action with `expected_current_version` and a non-empty reason.
- [ ] If challenger registration/alias update fails, champion remains untouched.
- [ ] Script output clearly distinguishes `statistical_champion_candidate_id`, registered challenger version and current production champion version.
- [ ] Test starts with an existing champion, runs a successful cycle with a different challenger and proves champion is unchanged.
- [ ] OOS publication and final-refit behavior are unchanged.

## Corrective execution graph

```text
E1: PR-117
E2 parallel after 117: PR-118
E3 parallel after 118: PR-119 PR-123 PR-126 PR-128 PR-131
E4 parallel after 119: PR-120 PR-121 PR-122
E5: PR-124 after 123
E6: PR-125 after 122
E7: PR-127 after 126
E8: PR-129 after 126+127+128
E9: PR-130 after 125
```

PR-120, PR-121, PR-124, PR-126, PR-128 and PR-131 have disjoint primary implementation files and can be assigned to weak agents in parallel once their dependencies are merged. PR-129 is intentionally late because it validates a single canonical TRAIN likelihood across all supported emission families.

---

# 18. Canonical full evaluation JSON report archive — 2026-08-27

Every completed full statistical evaluation must leave one self-contained, machine-readable report that explains the complete feature-selection decision, preserves every planned fold and its available evidence, summarizes every candidate across folds, and compares all candidates from that exact run. The report is audit/research evidence only: it must never feed back into feature selection, candidate fitting, champion ranking, alias mutation or downstream economic selection.

A **full evaluation** means one parent evaluation for a pinned profile/version/source snapshot/evaluation plan in which the exact configured candidate universe was attempted and every planned fold identity is represented for every candidate. Individual folds may be invalid and candidates may fail hard gates; those failures are part of the report. A report with missing configured candidates or missing planned fold identities must never be labelled complete.

Canonical report identity and storage:

```text
schema = EvaluationReport.v1
MLflow parent artifact = evaluation-report/evaluation_report.v1.json
repository archive = evaluations/<profile_id>/<source_build_id>/<parent_run_id>.json
container report directory = /regime-engine/evaluations
```

The MLflow artifact and repository-archive JSON must be byte-identical canonical JSON for the same parent run. Generated report files are intentionally visible to Git and are not auto-committed or auto-pushed by the model cycle.

Global report rules:

- JSON is UTF-8, deterministic key ordering, no NaN/Infinity, RFC3339 UTC timestamps only.
- Raw source feature rows, database credentials/DSNs, secret-file contents and model binary payloads are forbidden.
- All finite numeric evidence already used by the statistical evaluation is preserved at full precision; presentation rounding is forbidden in canonical JSON.
- Invalid or unavailable metrics are represented explicitly as `null` plus the exact failure/availability reason; they are never replaced by zero, imputed or interpolated.
- Candidate-specific summaries use that candidate's valid folds and must identify the exact support fold IDs.
- Canonical champion-ranking summaries separately use the PR-125 common-valid-fold support and must identify the exact common fold IDs.
- Any extra pairwise/all-model comparison is diagnostic only and cannot alter the selection result.
- Report generation must validate that the reconstructed feature-selection and aggregate/comparison evidence reproduces the already-frozen selection result and already-computed statistical champion; a mismatch fails report generation rather than changing the run result.

### PR-134 — Define `EvaluationReport.v1` contracts and canonical JSON

- **Branch:** `pr/PR-134-evaluation-report-contracts`
- **Depends on:** PR-118, PR-125
- **Allowed:** `src/market_regime_engine/evaluation/report/__init__.py`, `src/market_regime_engine/evaluation/report/contracts.py`, `tests/unit/evaluation/report/test_contracts.py`, `docs/evaluation_report.md`

Acceptance:

- [ ] Define immutable typed contracts for report metadata, feature-selection explanation, fold evidence, candidate summaries, cross-candidate comparisons and report integrity; this PR contains no model fitting, filesystem, MLflow or lifecycle code.
- [ ] Top-level schema identifier is exactly `EvaluationReport.v1` and contains at least `schema_version`, `parent_run_id`, profile ID/config version, source lineage, repository Git SHA, evaluation-plan hash/cutoff, feature-selection definition/execution hashes, exact configured candidate IDs, feature-selection section, candidate sections, comparison section and integrity section.
- [ ] `parent_run_id`, `source_build_id`, hashes, candidate IDs, fold IDs and feature order are required and non-empty; duplicate candidate/fold IDs fail validation.
- [ ] Full-evaluation completeness validation requires the exact versioned candidate-ID tuple and the exact same ordered planned-fold-ID tuple inside every candidate report.
- [ ] Canonical serialization uses sorted keys, compact separators, UTF-8, `ensure_ascii=True`, `allow_nan=False`; every supported value round-trips without precision-reducing formatting.
- [ ] All timestamps are timezone-aware UTC and serialize to one normalized RFC3339 representation.
- [ ] Optional unavailable metric values use JSON `null`; schema contains explicit validity/failure fields so `null` is never ambiguous with zero.
- [ ] Define `report_payload_sha256` as SHA-256 over the canonical report payload with the integrity field excluded; verify/recompute helper rejects any mismatch.
- [ ] Contract explicitly forbids raw source rows, secrets/DSNs and opaque model binary bytes.
- [ ] Unit tests cover canonical byte equality across mapping insertion orders, nonfinite rejection, duplicate/missing identities, hash verification and JSON round-trip.

### PR-135 — Build exact feature-selection decision explanation

- **Branch:** `pr/PR-135-feature-selection-report`
- **Depends on:** PR-134, PR-120
- **Allowed:** `src/market_regime_engine/evaluation/report/feature_selection.py`, `tests/unit/evaluation/report/test_feature_selection.py`

Acceptance:

- [ ] Pure builder accepts only first-planned-fold TRAIN source rows, the active `FeatureSelectionPolicy`, the immutable `FeatureSelectionResult` and optional PR-066 diagnostic evidence; it performs no mutation and returns report data only.
- [ ] Report records the first-TRAIN row count, policy ID and every pinned policy constant: coverage threshold, population-variance threshold, complete-case minimum, Stage-2 absolute-Spearman threshold and numeric tie tolerance.
- [ ] For every semantic block, preserve canonical block order and every configured feature in configured order with non-null count, source-row denominator, exact coverage, population variance (`ddof=0`), eligible flag, exclusion reason, medoid score and configured position.
- [ ] For each block, record exact eligible-feature order, complete-case observation count, absolute Spearman matrix and distance matrix `1-|rho|`; singleton blocks are represented deterministically as `[[1.0]]` correlation and `[[0.0]]` distance with an explicit singleton flag.
- [ ] Stage-1 explanation records the exact global minimum medoid-score anchor, the anchored medoid-tied set, the maximum-coverage anchor inside that set, the anchored coverage-tied set, configured-position tie break and final block winner.
- [ ] Builder recomputes each medoid score from the recorded distance matrix and requires equality with immutable selection evidence within the existing `1e-12` contract; mismatch fails report construction.
- [ ] Stage 2 records preliminary medoids in canonical block order, complete-case observation count, the fixed pre-pruning 8x8 absolute-Spearman matrix and strict conflict condition `|rho| > 0.85`.
- [ ] Stage-2 explanation enumerates every above-threshold pair in exact processing order `(-abs_spearman, left_block_position, right_block_position)` and records survivor set before decision, whether the pair was processed or skipped because a member had already been removed, removed feature when processed, exact removal reason and survivor set after decision.
- [ ] For each preliminary medoid, report a final disposition of `selected` or `removed`, and for removed medoids reference the exact conflict decision that removed it.
- [ ] Re-simulated Stage-2 decisions must reproduce the immutable `conflicts` and ordered `final_features` exactly; mismatch fails report generation and never changes feature selection.
- [ ] Definition/execution hashes and ordered final features are copied from the frozen result and verified; report-only explanation fields do not change either selection hash.
- [ ] When PR-066 diagnostics are supplied, include threshold-sensitivity and later-fold shadow selections under an explicitly `diagnostic_only` subsection; they must not be presented as selection inputs.
- [ ] Raw feature values are never emitted into the report.

### PR-136 — Serialize complete per-fold evaluation evidence

- **Branch:** `pr/PR-136-fold-evaluation-report`
- **Depends on:** PR-134, PR-124, PR-126, PR-127, PR-129
- **Allowed:** `src/market_regime_engine/evaluation/report/folds.py`, `tests/unit/evaluation/report/test_folds.py`

Acceptance:

- [ ] Pure builder takes a `WalkForwardPlan` plus one `WalkForwardEvaluation`; it does not query MLflow, PostgreSQL or the filesystem and does not refit/re-filter any model.
- [ ] Every planned fold is emitted exactly once in plan order, including invalid folds; fold identity includes fold ID/index plus exact UTC TRAIN/TEST start/end bounds.
- [ ] Preserve source/model observation counts, skipped incomplete TRAIN/TEST counts, validity flag and exact failure reason.
- [ ] For valid folds preserve scaler means/scales, model family, K, feature order, start probabilities, transition matrix, state means and full emission matrices without presentation rounding.
- [ ] Gaussian folds preserve full covariance matrices; GMM folds additionally preserve mixture count, weights, component means and every component full covariance; Student-t folds preserve state-specific degrees of freedom and emission scale matrices plus the derived distribution covariance semantics from PR-127.
- [ ] Preserve the complete multistart evidence for every attempted seed: seed, convergence flag, iteration count, TRAIN log likelihood, numerical-validity status/failure reason and winning-start identity; the exact valid-start count and success rate must reconcile.
- [ ] Preserve persistent-state alignment mapping, fixed-coordinate aligned signatures, matched RMS per state, total assignment cost and maximum drift.
- [ ] Preserve all scalar fold statistics already available to evaluation: canonical TRAIN log likelihood, OOS predictive log likelihood total and per observation, AIC, BIC, multistart success rate, mean dominant-state duration, switches/year, OOS entropy mean and OOS confidence mean.
- [ ] Preserve full state-indexed TRAIN hard/soft occupancy and OOS hard/soft occupancy in persistent-state order.
- [ ] Preserve complete OOS timestamp sequence and aligned filtered probability vector for every retained TEST observation; probability vector dimension must equal K, values must be finite/nonnegative/normalized within existing tolerance, and timestamp count must equal retained TEST count.
- [ ] Invalid folds keep exact counts/failure reason and use `null` for unavailable nested model/statistic evidence; no fabricated placeholder model or metric is allowed.
- [ ] Builder validates all report counts against `WalkForwardFoldResult`, the plan and model artifact; any reconciliation mismatch fails.
- [ ] No raw source feature rows or model binary serialization are included.

### PR-137 — Compute complete per-model fold summary statistics

- **Branch:** `pr/PR-137-candidate-summary-report`
- **Depends on:** PR-136
- **Allowed:** `src/market_regime_engine/evaluation/report/candidate_summary.py`, `tests/unit/evaluation/report/test_candidate_summary.py`

Acceptance:

- [ ] One pure summary is built for every configured candidate from its exact ordered fold reports; no cross-candidate ranking occurs in this PR.
- [ ] Record planned/valid/invalid fold counts, valid-fold rate, exact valid-fold IDs, exact invalid-fold IDs and a frequency table of invalid-fold failure reasons.
- [ ] Candidate-valid-fold scalar summaries use valid folds only and expose `count`, arithmetic mean, population standard deviation (`ddof=0`), minimum and maximum for every available scalar fold diagnostic: TRAIN log likelihood, OOS predictive log likelihood total, OOS predictive log likelihood per observation, AIC, BIC, multistart success rate, max state-signature drift, mean state duration, switches/year, OOS entropy mean and OOS confidence mean.
- [ ] For additive likelihood metrics also expose the exact valid-fold sum; no observation weighting is silently substituted for the unweighted fold mean.
- [ ] State-indexed summaries preserve persistent state IDs and compute the same count/mean/population-std/min/max statistics for TRAIN hard occupancy, TRAIN soft occupancy, OOS hard occupancy and OOS soft occupancy.
- [ ] Transition matrices are first placed in persistent-state order and then summarized elementwise across valid folds with count/mean/population-std/min/max; each mean transition row must remain finite and reconcile to row sum 1 within numerical tolerance.
- [ ] Fixed-coordinate state-signature components may be summarized state/component-wise; fold-local covariance, GMM component parameters and any other tensor whose coordinates/mixture labels are not canonically comparable across folds are not averaged and receive an explicit `not_aggregated_reason` instead.
- [ ] The existing `CandidateAggregate` values for OOS mean/std/worst/best, BIC mean and AIC mean are copied and independently recomputed from the exact support; any mismatch above existing tolerance fails report generation.
- [ ] A candidate with zero valid folds has counts/reasons but no fabricated numeric summary; all unavailable aggregates are `null`.
- [ ] Tests hand-check means, population standard deviations, extrema, sums, per-state statistics, transition-matrix summaries and invalid-fold exclusion.

### PR-138 — Compute all-model and pairwise comparison statistics

- **Branch:** `pr/PR-138-cross-candidate-report`
- **Depends on:** PR-125, PR-137
- **Allowed:** `src/market_regime_engine/evaluation/report/comparison.py`, `tests/unit/evaluation/report/test_comparison.py`

Acceptance:

- [ ] Cross-candidate report includes every configured candidate exactly once, including candidates rejected by hard gates.
- [ ] Preserve the canonical `StatisticalChampionSelection` verbatim: champion ID/K, accepted/rejected status, rejection reasons, one-based rank for accepted candidates, ranking tolerance, common-valid-fold IDs/count/rate and ordered ranked candidate IDs.
- [ ] For each accepted candidate, independently recompute canonical common-support OOS mean, population std, worst fold, BIC mean and AIC mean from exactly PR-125 common fold IDs and require equality with the ranking inputs that produced the frozen selection result.
- [ ] Produce one deterministic all-candidate overview table containing candidate-specific valid-fold summaries and canonical ranking metrics side by side with clear support labels; candidate-specific and common-support values must never share an ambiguous field name.
- [ ] Produce one diagnostic pairwise record for every unordered configured-candidate pair in canonical candidate order.
- [ ] Each pairwise record states the exact intersection of valid fold IDs for that pair, pairwise shared-fold count/rate, and uses only those folds for paired calculations; no invalid fold is imputed.
- [ ] For OOS predictive log likelihood per observation, compute foldwise delta `candidate_a - candidate_b` and report count, mean, population std, minimum, maximum, plus wins/ties/losses using exact `1e-12` anchored equality around zero.
- [ ] For BIC and AIC, report paired foldwise deltas with count/mean/population-std/min/max and direction semantics (`lower_is_better`).
- [ ] Also report direct differences between candidate-level valid-fold summary means, but label them `different_support_diagnostic` whenever the two candidate valid-fold-ID sets differ.
- [ ] No p-value, independence claim or significance test is invented for overlapping walk-forward folds.
- [ ] Pairwise diagnostics are explicitly `diagnostic_only` and cannot alter the already-frozen statistical champion.
- [ ] Tests include rejected candidates, unequal valid-fold sets, zero shared folds, exact ties and a case where pairwise diagnostic leadership differs from canonical common-support rank.

### PR-139 — Assemble, hash and persist the canonical MLflow report artifact

- **Branch:** `pr/PR-139-mlflow-evaluation-report`
- **Depends on:** PR-135, PR-136, PR-137, PR-138
- **Allowed:** `src/market_regime_engine/evaluation/report/builder.py`, `src/market_regime_engine/mlflow_support/evaluation_report.py`, `tests/unit/evaluation/report/test_builder.py`, `tests/unit/mlflow_support/test_evaluation_report.py`

Acceptance:

- [ ] One builder assembles `EvaluationReport.v1` only from already-computed source lineage, active profile/policy, frozen feature-selection result, walk-forward plan/grid, statistical selection and MLflow parent-run identity; it does not rerun feature selection or model fitting.
- [ ] Metadata records parent run ID, parent-run status, profile ID/config version, source dataset/build/data hash/schema/feature version/synced timestamp/time semantics, repository Git SHA/build provenance, plan hash, evaluation cutoff, frozen feature order and both selection hashes.
- [ ] Report exact configured candidate IDs must equal the profile-version universe; all candidate reports must carry the same source build, plan hash, feature order and selection hashes.
- [ ] Builder calls PR-135/136/137/138 components and then performs full cross-section reconciliation before serialization; any missing candidate/fold or metric/hash mismatch fails.
- [ ] `report_payload_sha256` is computed exactly according to PR-134 and verified immediately before persistence.
- [ ] Canonical JSON bytes are generated once and are the only bytes accepted for downstream persistence; no second serializer with different rounding/order is allowed.
- [ ] The exact bytes are logged under parent artifact path `evaluation-report/evaluation_report.v1.json`.
- [ ] MLflow artifact metadata/tagging records schema version and payload SHA-256 without using those tags as a substitute for the JSON artifact.
- [ ] Artifact read-back in integration tests is byte-identical to builder output and its SHA-256 verifies.
- [ ] Report persistence failure is explicit; it must not silently mark report evidence complete.
- [ ] No report field changes candidate ranking or aliases.

### PR-140 — Add atomic repository evaluation archive storage

- **Branch:** `pr/PR-140-repository-evaluation-archive`
- **Depends on:** PR-134
- **Allowed:** `src/market_regime_engine/evaluation/report/archive.py`, `tests/unit/evaluation/report/test_archive.py`, `compose.yaml`, `.env.example`, `evaluations/README.md`, `evaluations/.gitkeep`, compose contract tests

Acceptance:

- [ ] Repository archive path is exactly `evaluations/<profile_id>/<source_build_id>/<parent_run_id>.json`; path components are validated identifiers and path traversal is rejected.
- [ ] `evaluations/` remains Git-visible: generated JSON is not added to `.gitignore`; this PR does not perform `git add`, commit, push or branch mutation.
- [ ] Archive writer accepts canonical JSON bytes plus validated report identity only; it never reconstructs/reformats the report.
- [ ] Writer creates parent directories safely and writes a temporary file in the destination directory, flushes and `fsync`s it, atomically replaces/renames to the final path, then `fsync`s the containing directory on supported POSIX systems.
- [ ] If the final path already exists with byte-identical content, the operation is an idempotent success; if existing bytes differ for the same run ID, fail closed and never overwrite silently.
- [ ] Written bytes are re-read and checked against `report_payload_sha256`/canonical report verification before success is returned.
- [ ] Compose mounts only the report directory read-write into the `mlflow` container, never the whole Git repository; target is exactly `/regime-engine/evaluations`.
- [ ] Host source is configurable as `REGIME_EVALUATIONS_HOST_PATH`; default is repository-relative `./evaluations`, allowing MLflow artifacts/PostgreSQL and the Git checkout to reside on different host partitions.
- [ ] Container environment exposes `REGIME_EVALUATION_REPORT_DIR=/regime-engine/evaluations`; no MLflow artifact-root path is reused as the repository archive.
- [ ] Compose still has exactly the existing two services and unchanged public port topology.
- [ ] Tests cover atomic success, idempotent rewrite, conflicting existing file, interrupted-temp cleanup behavior, traversal rejection and arbitrary absolute host path configuration.

### PR-141 — Wire report creation into every canonical full Xetra evaluation cycle

- **Branch:** `pr/PR-141-wire-evaluation-report-cycle`
- **Depends on:** PR-139, PR-140
- **Allowed:** `scripts/run_xetra_v2_cycle.py`, `scripts/model_cycle.sh`, `src/market_regime_engine/commands/lifecycle.py`, `src/market_regime_engine/commands/actions.py`, `src/market_regime_engine/cli.py`, lifecycle/CLI/script contract tests

Acceptance:

- [ ] Every canonical full Xetra evaluation entry point uses one shared report-completion path; no separate report schema/serializer exists in shell versus Python paths.
- [ ] Report generation occurs only after the exact candidate grid, all planned fold results and frozen statistical champion are available and the MLflow parent-run ID is known.
- [ ] The MLflow parent artifact from PR-139 is persisted first; the same canonical bytes are then written through PR-140 to the repository archive path.
- [ ] Successful completion verifies MLflow artifact bytes and repository bytes are byte-identical and both verify against the same payload SHA-256.
- [ ] Cycle output includes `evaluation_report_schema`, `evaluation_report_path`, `evaluation_report_payload_sha256` and parent run ID.
- [ ] A run is not reported as a successful full evaluation if the exact configured candidate universe or any planned fold identity is missing.
- [ ] Invalid folds/candidate rejections do not prevent a report when the full planned evidence set exists; they are represented explicitly inside it.
- [ ] If report assembly, MLflow artifact persistence or repository archive persistence fails, the model cycle fails closed before final production refit/registration/challenger mutation; no newly registered challenger may exist without its completed report.
- [ ] Re-running the archive step for the same parent run is idempotent when bytes are identical and does not create duplicate files.
- [ ] Existing explicit-champion semantics from PR-131 remain unchanged; report generation never promotes `champion`.
- [ ] No automatic Git commit/push is added.

### PR-142 — Prove full report completeness and reproducibility end to end

- **Branch:** `pr/PR-142-evaluation-report-e2e`
- **Depends on:** PR-141
- **Allowed:** `tests/integration/evaluation_report/*`, deterministic report fixtures only

Acceptance:

- [ ] Hermetic fixture represents an exact profile-v2 full candidate universe and multiple planned folds, including at least one invalid fold and at least one rejected candidate; expensive real HMM fitting is not required for this report-only proof.
- [ ] Generated report contains all configured candidates and every planned fold exactly once per candidate in deterministic order.
- [ ] Feature-selection section contains every semantic block/configured feature, exact Stage-1 eligibility/score/tie trace, complete matrices, Stage-2 pair processing trace, final dispositions and both frozen hashes.
- [ ] Per-fold section contains every field required by PR-136, including complete multistart evidence, model parameters/matrices, persistent-state mapping, all scalar diagnostics, occupancies and full OOS probability timeline for valid folds.
- [ ] Per-model summary values are independently hand-computed in the fixture and match counts, means, `ddof=0` standard deviations, extrema, sums, state summaries and transition summaries exactly.
- [ ] Cross-model section exactly reproduces frozen champion ranking/common support and independently hand-checks every unordered pair's shared support, OOS deltas, wins/ties/losses and AIC/BIC deltas.
- [ ] Invalid folds are never averaged or imputed; rejected candidates remain fully visible.
- [ ] Reordering input mappings/diagnostic containers without changing semantic evidence produces byte-identical canonical JSON.
- [ ] MLflow artifact bytes and repository archive bytes are exactly identical and verify against the same payload SHA-256.
- [ ] Report contains no NaN/Infinity, raw source feature rows, credential/DSN text or model binary payload.
- [ ] A deliberate missing-candidate, missing-fold, aggregate mismatch, feature-selection mismatch and checksum mismatch each fail closed.
- [ ] Required CI remains hermetic and contributes to the repository-wide >=90% unit+integration coverage gate.

## Evaluation-report execution graph

```text
R1: PR-134
R2 parallel after 134: PR-135 PR-136 PR-140
R3: PR-137 after 136
R4: PR-138 after 125+137
R5: PR-139 after 135+136+137+138
R6: PR-141 after 139+140
R7: PR-142 after 141
```

PR-135, PR-136 and PR-140 have disjoint primary implementation files and can be assigned to weak agents in parallel. PR-137 is intentionally isolated from cross-model logic; PR-138 is intentionally isolated from persistence; PR-139 is the only assembly/MLflow-artifact PR; PR-141 is the only lifecycle-wiring PR.