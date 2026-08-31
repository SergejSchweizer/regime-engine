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
	ext{MLFLOW\_WORKERS} \cdot \text{REGIME\_PG\_POOL\_MAX\_SIZE}
\le \text{REGIME\_FEATURE\_PG\_CONNECTION\_BUDGET}.
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
# 18. Xetra v3 three-regime-evaluation architecture — 2026-08-28

This section supersedes the previous combined `xetra_univariate_shadow_v1` design. Xetra v3 keeps one canonical 61-feature selection policy but exposes three first-class, independently auditable regime evaluations. The three evaluations may share low-level HMM/walk-forward code, but they have separate input contracts, observation clocks, champion namespaces, MLflow hierarchies and local statistics evidence.

Canonical Xetra v3 identity:

```text
profile_id=xetra
profile_config_version=3
feature_selection_policy=xetra_semantic_medoid_v3
canonical_feature_universe_size=61
semantic_block_count=8
```

Canonical evaluation IDs are exactly:

```text
medoid_multivariate
medoid_univariate
delta1_univariate
```

No `shadow` evaluation ID is part of the final v3 runtime contract.

## Canonical 61-feature policy

The historical ordered v2 48-feature universe remains unchanged in v1/v2. Xetra v3 adds these exact 13 existing PostgreSQL columns once, inside their economic semantic blocks:

| Semantic block | Added v3 feature(s) |
|---|---|
| US equity volatility spot | `vix_delta_1obs` |
| US equity volatility term structure | `vix9d_delta_1obs`, `vix3m_delta_1obs`, `vix6m_delta_1obs`, `vix1y_delta_1obs` |
| Europe equity volatility | `vstoxx_delta_1obs` |
| Rates volatility | `move_delta_1obs` |
| Systemic stress | `ciss_delta_1obs` |
| Credit stress | `euro_hy_oas_delta_1obs` |
| Rates / yield curve | `us_2y_delta_1obs`, `us_10y_delta_1obs`, `estr_delta_1obs` |
| USD FX | `usd_broad_delta_1obs` |

Exact ordered delta tuple:

```text
vix_delta_1obs
vix9d_delta_1obs
vix3m_delta_1obs
vix6m_delta_1obs
vix1y_delta_1obs
vstoxx_delta_1obs
move_delta_1obs
ciss_delta_1obs
euro_hy_oas_delta_1obs
us_2y_delta_1obs
us_10y_delta_1obs
estr_delta_1obs
usd_broad_delta_1obs
```

Stage 1 runs the existing semantic-medoid selector over all 61 canonical v3 features. A `*_delta_1obs` feature can therefore become its block's `preliminary_medoid`. Stage 2 applies the existing cross-block pruning to the eight Stage-1 representatives and freezes the final ordered multivariate feature set. First-fold TRAIN-only semantics, no economic input, no HMM feedback into feature selection, tie rules and missing-value rules remain unchanged.

Xetra v3 uses exactly the same 12 candidate model identities as v2:

```text
gaussian_hmm_k2_full
gaussian_hmm_k3_full
gaussian_hmm_k4_full
gaussian_hmm_k5_full
gmm_hmm_k2_m2_full
gmm_hmm_k3_m2_full
gmm_hmm_k4_m2_full
gmm_hmm_k5_m2_full
student_t_hmm_k2_full
student_t_hmm_k3_full
student_t_hmm_k4_full
student_t_hmm_k5_full
```

## Evaluation A — `medoid_multivariate`

Purpose: choose the canonical production-eligible statistical champion from the frozen Stage-2 multivariate feature set.

Input contract:

```text
features = frozen final ordered Stage-2 feature tuple
models   = exact 12-candidate v3 universe
clock    = canonical complete-case walk-forward clock for that frozen tuple
```

The evaluation runs exactly 12 candidate evaluations. It reuses the canonical candidate hard gates, common-valid-fold support and seven-stage ranking. Its winner is named exactly `medoid_multivariate_statistical_champion`. This is the only evaluation champion that may proceed to final production refit, challenger registration, immutable OOS publication and a later explicit production promotion.

## Evaluation B — `medoid_univariate`

Purpose: determine how well each of the eight Stage-1 semantic-block medoids alone can identify stable regimes, and identify the single medoid/model pair whose OOS segmentation best agrees with the multivariate champion.

Input contract:

```text
features = exactly eight Stage-1 preliminary_medoids in canonical block order
models   = exact 12-candidate v3 universe per feature
clock    = one common complete-case clock built only across these eight medoid features
```

The evaluation runs exactly 8 x 12 = 96 candidate evaluations. Each feature first selects its own `diagnostic_feature_model_winner` using the canonical within-feature statistical ranking; OOS PLL/BIC/AIC are never compared across different feature names. The evaluation-level diagnostic champion is named exactly `medoid_univariate_evaluation_champion`.

## Evaluation C — `delta1_univariate`

Purpose: determine how well each immediate one-observation origin shock alone can identify stable regimes, and identify the delta/model pair whose OOS segmentation best agrees with the multivariate champion.

Input contract:

```text
features = exact ordered 13-delta tuple above
models   = exact 12-candidate v3 universe per feature
clock    = one common complete-case clock built only across these 13 delta features
```

The evaluation runs exactly 13 x 12 = 156 candidate evaluations. Each feature selects its own `diagnostic_feature_model_winner` using the canonical within-feature statistical ranking. The evaluation-level diagnostic champion is named exactly `delta1_univariate_evaluation_champion`.

A feature can appear in both `medoid_univariate` and `delta1_univariate`. The two evaluations remain independent: each uses its own evaluation clock, hashes, MLflow runs and local statistics. Cross-evaluation fitted-model reuse is forbidden unless a future versioned contract explicitly introduces it; identical feature names do not imply identical retained observations.

A complete three-evaluation execution therefore contains exactly 12 + 96 + 156 = 264 candidate evaluations before invalid-fold rejection.

## Univariate evaluation-champion rule

For each valid univariate feature winner, compare its OOS dominant-state sequence with `medoid_multivariate_statistical_champion` only on the ordered intersection of valid OOS timestamps.

- shared valid-fold support below `0.80` makes that feature ineligible for the evaluation-level champion but does not invalidate its within-feature winner;
- primary cross-feature criterion: maximize label-invariant `dominant_state_nmi`;
- NMI ties use the repository anchored absolute tolerance `1e-12`;
- first exact tie-break: maximize shared OOS timestamp count;
- final exact tie-break: lexicographically smallest feature name;
- raw OOS PLL, BIC, AIC, confidence, entropy and economic metrics are forbidden as cross-feature ranking criteria;
- if no feature has sufficient agreement support, the evaluation has no champion and records an explicit reason;
- equal K additionally reports maximum one-to-one permutation hard agreement and the lexicographically smallest maximizing mapping; unequal K reports that metric as unavailable.

Both univariate evaluation champions are diagnostic-only. They never trigger final refit, OOS publication, model registration, challenger/champion alias mutation or an economic decision.

## Mandatory local statistics for every MLflow run

Every MLflow run created by the three evaluations, including parent, feature, candidate and failed runs, must own exactly one immutable local evidence directory:

```text
./evaluations/<evaluation>/<mlflow_run_id>/
```

`<evaluation>` is exactly `medoid_multivariate`, `medoid_univariate` or `delta1_univariate`. `./evaluations/` is runtime evidence and is git-ignored.

Every run directory contains at least:

```text
statistics.json
statistics.md
```

`statistics.json` is the canonical machine-readable record. `statistics.md` is a deterministic human-readable rendering of the same evidence. The JSON must use UTF-8, deterministic key ordering, finite JSON numbers only and explicit nulls for unavailable values.

When applicable to the run type, statistics include all of the following:

- evaluation/run identity: evaluation ID, schema version, MLflow run ID, parent run ID, run type, run name, lifecycle status, start/end UTC, elapsed seconds, repository Git SHA, profile/config version;
- lineage: source dataset/build/data hash, source time semantics, walk-forward plan hash, evaluation definition/execution hashes, feature-selection definition/execution hashes and clock hash;
- input evidence: exact ordered feature tuple, semantic blocks/roles, feature dimension, source-row count, retained/skipped TRAIN/TEST counts and explicit missing/nonfinite/failure reasons;
- model contract: candidate ID, family, K, covariance semantics, GMM mixture count where applicable, Student-t settings where applicable, parameter count, complete multistart seed set and winning seed;
- every planned fold including invalid folds: source/TRAIN/TEST bounds, retained counts, validity/reason, every start's convergence/success/TRAIN likelihood, TRAIN likelihood-parity evidence, TEST OOS PLL sum/per-observation, AIC, BIC, hard/soft occupancy, confidence, entropy, dominant-state durations, switches/year, alignment/state mapping and drift evidence;
- fitted state statistics: start probabilities, transition matrix, persistence, expected/observed duration diagnostics, emission means, backend-neutral distribution covariances, Student-t scale matrices/nu and GMM mixture weights/component means/component covariances when applicable;
- candidate aggregate evidence: valid-fold rate, common-valid-fold IDs/rate, common-support OOS mean/population-std/worst, common-support BIC/AIC means, hard-gate results, rejection reasons and complete anchored ranking/tie evidence;
- multivariate feature-selection evidence when applicable: 61-feature policy identity, all eight preliminary medoids, Stage-2 correlation/pruning evidence and frozen final ordered features;
- univariate agreement evidence when applicable: shared valid-fold IDs/rate, shared timestamp count, NMI, equal-K permutation hard agreement/mapping and unavailable reason;
- evaluation champion evidence when applicable: eligible feature/candidate set, exact ranking stages/ties, selected namespaced champion or explicit no-champion reason.

The local statistics must never contain credentials, DSNs, raw source feature rows or model-binary payloads. Fitted parameter matrices such as transition/covariance matrices are permitted.

The exact finalized `statistics.json` bytes are also logged to the same MLflow run at `statistics/statistics.json`. SHA-256 of those exact bytes is recorded locally and in MLflow metadata; byte/hash mismatch is a run failure. Before any run is created, the evaluation root must be writable. Immediately after MLflow returns a run ID, an atomic `RUNNING` statistics record is written; finalization atomically replaces it with `FINISHED` or `FAILED`. A successful MLflow run without its local statistics mirror is forbidden.

Execution rule: every PR below inherits section-12 clean-main/status/branch/allowed-file/full-test rules. Agents must not combine PRs, edit `BACKLOG.md`, add economic metrics, mutate v1/v2 identities or change another evaluation's files unless explicitly allowed.

### PR-156 — Pin canonical Xetra v3 61-feature policy contract

- **Branch:** `pr/PR-156-xetra-v3-61-feature-contract`
- **Depends on:** PR-125, PR-129
- **Allowed:** `EVALUATION.md`

Acceptance:

- [ ] Define `profile_config_version=3` and `feature_selection_policy=xetra_semantic_medoid_v3`; v1/v2 definitions remain immutable.
- [ ] Define exactly 61 unique canonical features: ordered v2 48 plus the exact ordered 13-delta tuple in this section.
- [ ] Pin every added delta to the exact semantic block listed above; all 61 features belong to exactly one of eight blocks.
- [ ] Preserve Stage-1 and Stage-2 algorithms, first-fold TRAIN-only semantics, thresholds, tie rules, missing-value rules and no-economic/no-HMM-feedback constraints.
- [ ] State that delta1 features are normal Stage-1 candidates and may enter the frozen multivariate final tuple.
- [ ] Define the v3 candidate universe as exactly the same 12 family/K identities as v2.
- [ ] Preserve historical v1/v2 configuration and selection hashes.
- [ ] Do not implement any evaluation orchestration in this PR.

### PR-157 — Add exact Xetra v3 61-feature/eight-block policy

- **Branch:** `pr/PR-157-xetra-v3-feature-policy`
- **Depends on:** PR-156
- **Allowed:** `configs/feature_selection/xetra_semantic_medoid_v3.yaml`, `tests/unit/feature_selection/test_xetra_v3_policy.py`, `docs/profiles/xetra_feature_selection_v3.md`

Acceptance:

- [ ] Create new v3 policy; do not edit v1/v2 policy files.
- [ ] Preserve exact v2 48 feature order/block membership and add the 13 deltas to the exact blocks pinned by PR-156.
- [ ] Policy contains exactly 61 unique features and eight non-overlapping exhaustive blocks.
- [ ] Exact PostgreSQL delta column names match the ordered tuple in this section.
- [ ] Stage-1/Stage-2 constants equal v2 unless PR-156 explicitly says otherwise.
- [ ] Tests assert the full ordered 61-feature tuple, exact block membership/cardinality and absence of target/economic fields.
- [ ] Tests prove v1/v2 policy files remain unchanged.
- [ ] Documentation states v3 is a versioned extension, not an in-place mutation.

### PR-158 — Add Xetra v3 profile with unchanged 12-model universe

- **Branch:** `pr/PR-158-xetra-v3-profile`
- **Depends on:** PR-157, PR-118, PR-128
- **Allowed:** `configs/profiles/xetra_v3.yaml`, `src/market_regime_engine/profiles/config.py`, `src/market_regime_engine/profiles/resolution.py`, `src/market_regime_engine/training/candidate_grid.py`, `tests/unit/profiles/test_xetra_v3_profile.py`, `tests/unit/profiles/test_resolution.py`, `tests/unit/training/test_candidate_grid.py`

Acceptance:

- [ ] Add public `xetra` profile config version 3 referencing exactly `xetra_semantic_medoid_v3`.
- [ ] v3 candidate IDs/order equal v2: Gaussian K2-K5, GMM M2 K2-K5, Student-t K2-K5.
- [ ] v3 walk-forward, multistart, fit, gate and ranking settings equal v2 unless PR-156 explicitly changes them.
- [ ] Version-aware resolution recognizes v3 explicitly and never infers it from list length.
- [ ] Candidate-grid validation accepts only the exact v3 tuple and rejects wrong order, missing-middle and extra candidates.
- [ ] Source feature universe comes from the v3 61-feature policy; final model features come only from frozen first-fold selection.
- [ ] v1/v2 remain loadable and behavior-identical.
- [ ] No evaluation tracking, final refit or registry mutation occurs here.

### PR-144 — Pin the three first-class evaluation contracts

- **Branch:** `pr/PR-144-three-evaluation-contract`
- **Depends on:** PR-158
- **Allowed:** `EVALUATION.md`

Acceptance:

- [ ] Define exactly `medoid_multivariate`, `medoid_univariate`, `delta1_univariate`; remove `xetra_univariate_shadow_v1` from the v3 target contract.
- [ ] Pin exact inputs and candidate cardinalities: multivariate 12, medoid-univariate 8x12=96, delta1-univariate 13x12=156.
- [ ] Pin independent evaluation clocks: multivariate canonical final-feature clock, medoid common clock over exactly eight Stage-1 medoids, delta1 common clock over exactly 13 deltas.
- [ ] Forbid a combined 21-feature univariate clock.
- [ ] Forbid cross-evaluation fit reuse because evaluation clock/hash identity may differ even for the same feature name.
- [ ] Pin production eligibility to `medoid_multivariate_statistical_champion` only.
- [ ] Pin both univariate evaluations as diagnostic-only.
- [ ] Preserve canonical v3 HMM gates/ranking inside a single feature/candidate grid.
- [ ] No implementation code change in this PR.

### PR-145 — Add shared evaluation identities, specs and lineage contracts

- **Branch:** `pr/PR-145-evaluation-contracts`
- **Depends on:** PR-144
- **Allowed:** `src/market_regime_engine/evaluations/__init__.py`, `src/market_regime_engine/evaluations/contracts.py`, `tests/unit/evaluations/test_contracts.py`

Acceptance:

- [ ] Define immutable enum/value objects for exactly the three evaluation IDs, evaluation lineage, feature spec, candidate spec and result identity.
- [ ] Contract module imports no MLflow, PostgreSQL or model adapter.
- [ ] Define the exact ordered 13-delta tuple and reject missing/reordered/duplicate/unexpected delta names.
- [ ] Build the medoid feature set only from `FeatureSelectionResult.evidence.preliminary_medoids`; require exactly eight unique features in canonical block order.
- [ ] Build multivariate input only from the frozen final ordered Stage-2 feature tuple; never substitute preliminary medoids.
- [ ] Build exact 12 structural candidate specs per model input using v3 family/K/mixture settings.
- [ ] Carry source build, plan hash, selection definition/execution hashes, evaluation definition/execution hashes and evaluation-specific clock hash.
- [ ] Deterministic hashes use canonical JSON/SHA-256 and distinguish all three evaluations even on the same source build.
- [ ] Tests cover exact IDs, feature-set cardinalities, ordering, hash separation and fail-closed invalid contracts.

### PR-146 — Extract reusable v3-capable model-adapter factory

- **Branch:** `pr/PR-146-reusable-model-adapter-factory`
- **Depends on:** PR-144, PR-128
- **Allowed:** `src/market_regime_engine/training/adapter_factory.py`, `src/market_regime_engine/training/candidate_grid.py`, `tests/unit/training/test_adapter_factory.py`, `tests/unit/training/test_candidate_grid.py`

Acceptance:

- [ ] Move candidate-to-adapter construction into one pure factory accepting active profile plus structural candidate spec.
- [ ] Preserve Gaussian full-covariance construction.
- [ ] Preserve GMM `mixture_count=2`, full covariance and K2-K5.
- [ ] Pass every active Student-t profile setting explicitly.
- [ ] Production candidate grid uses the factory; no duplicate family dispatch remains.
- [ ] All three evaluation types can create adapters from the same structural candidate contract.
- [ ] v1/v2/v3 production identities/order/settings remain exact.
- [ ] Invalid family/K/mixture/profile combinations fail before fit; no fallback.

### PR-147 — Generalize walk-forward runner to structural candidate specs

- **Branch:** `pr/PR-147-walk-forward-candidate-protocol`
- **Depends on:** PR-144, PR-124, PR-129
- **Allowed:** `src/market_regime_engine/evaluation/walk_forward.py`, `tests/unit/evaluation/test_walk_forward.py`, `tests/unit/evaluation/test_walk_forward_validation.py`

Acceptance:

- [ ] Introduce the minimal structural candidate protocol used by `run_walk_forward_candidate`.
- [ ] Existing production `ResolvedCandidateProfile` satisfies the protocol unchanged.
- [ ] Runner accepts multivariate and one-feature structural specs without weakening v1/v2/v3 invariants.
- [ ] Both paths use the identical scaler, multistart, causal-filter, likelihood-parity, occupancy, alignment and diagnostics implementation.
- [ ] Membership in a specific evaluation is validated upstream; runner does not infer evaluation identity.
- [ ] Existing production fixtures remain behavior-identical.
- [ ] Unsupported version, family/K mismatch, empty feature order and inconsistent dimension fail closed.

### PR-148 — Implement evaluation-scoped observation clocks

- **Branch:** `pr/PR-148-evaluation-observation-clocks`
- **Depends on:** PR-145
- **Allowed:** `src/market_regime_engine/evaluations/clocks.py`, `tests/unit/evaluations/test_clocks.py`

Acceptance:

- [ ] Implement a pure immutable clock result carrying evaluation ID, ordered features, retained-mask hash and per-fold retained/skipped evidence.
- [ ] `medoid_univariate` eligibility is row-wise finite/non-null across exactly the eight preliminary medoids and no other features.
- [ ] `delta1_univariate` eligibility is row-wise finite/non-null across exactly the 13 canonical deltas and no other features.
- [ ] A combined medoid+delta common clock is impossible through the public API.
- [ ] Multivariate evaluation continues to use the canonical complete-case clock for the frozen final tuple; this module does not replace production walk-forward planning.
- [ ] Original source-row boundaries/timestamps and `WalkForwardPlan` are never changed.
- [ ] No fill, interpolation, carry, synthesis or write-back.
- [ ] Tests prove asymmetric missingness can produce different medoid and delta clock hashes while each evaluation gives every one of its features identical retained timestamps.

### PR-149 — Implement reusable single-feature 12-candidate evaluator

- **Branch:** `pr/PR-149-univariate-feature-grid`
- **Depends on:** PR-145, PR-146, PR-147, PR-148
- **Allowed:** `src/market_regime_engine/evaluations/univariate_grid.py`, `tests/unit/evaluations/test_univariate_grid.py`

Acceptance:

- [ ] Input requires exactly one validated feature spec, one evaluation-scoped clock, canonical v3 profile and complete lineage.
- [ ] Evaluation ID must be either `medoid_univariate` or `delta1_univariate`; multivariate use is rejected.
- [ ] Construct exactly 12 ordered candidate specs with feature order `(feature_name,)` and dimension 1.
- [ ] Evaluate through PR-147 runner and PR-146 factory; no duplicate HMM/EM/filter implementation.
- [ ] Preserve all planned folds, invalid folds/reasons, per-start evidence and aggregate diagnostics.
- [ ] Within-feature statistical selection uses canonical hard gates/common-valid-fold/seven-stage ranking and produces `diagnostic_feature_model_winner` or an explicit no-winner reason.
- [ ] No cross-feature ranking, MLflow tracking, local statistics writing, final refit or registry mutation here.

### PR-150 — Implement `medoid_multivariate` evaluation orchestrator

- **Branch:** `pr/PR-150-medoid-multivariate-evaluation`
- **Depends on:** PR-145, PR-146, PR-147, PR-125, PR-158
- **Allowed:** `src/market_regime_engine/evaluations/medoid_multivariate.py`, `tests/unit/evaluations/test_medoid_multivariate.py`

Acceptance:

- [ ] Require canonical v3 source/plan/selection lineage and the frozen Stage-2 final ordered feature tuple.
- [ ] Evaluate exactly the 12 canonical v3 candidates once on the canonical multivariate complete-case clock.
- [ ] Reuse the existing candidate-grid/statistical-selection implementation; no second ranking algorithm is introduced.
- [ ] Return exactly one immutable `medoid_multivariate` result containing all 12 candidate evaluations and `medoid_multivariate_statistical_champion` or explicit failure.
- [ ] Champion identity equals the canonical statistical champion returned by existing selection for the same grid.
- [ ] Preserve complete common-valid-fold/ranking/rejection evidence.
- [ ] Do not final-refit, publish OOS, register models, mutate aliases, track MLflow or write local statistics in this PR.

### PR-151 — Implement label-invariant univariate-to-multivariate agreement

- **Branch:** `pr/PR-151-univariate-multivariate-agreement`
- **Depends on:** PR-149, PR-150
- **Allowed:** `src/market_regime_engine/evaluations/agreement.py`, `tests/unit/evaluations/test_agreement.py`

Acceptance:

- [ ] Require one valid univariate feature winner and the `medoid_multivariate_statistical_champion` from the same source build and walk-forward plan.
- [ ] Shared support is the ordered intersection of valid fold IDs and exact OOS timestamps; no fabricated rows.
- [ ] Record shared fold IDs/count/rate and shared timestamp count.
- [ ] Shared valid-fold rate below `0.80` or zero timestamps returns unavailable agreement with an explicit reason without invalidating the feature winner.
- [ ] Implement exact label-invariant `dominant_state_nmi = 2*I/(H_x+H_y)` with natural logs and exact 1.0 when both entropies are zero.
- [ ] Equal K enumerates all K! permutations and reports maximum hard agreement plus lexicographically smallest maximizing mapping.
- [ ] Unequal K reports permutation metric/mapping null; no many-to-one state match or cross-space signature RMS.
- [ ] Tests cover perfect relabeling, independence, unequal K, constant sequences and insufficient support.

### PR-159 — Pin namespaced champion selection rules

- **Branch:** `pr/PR-159-three-evaluation-champion-contract`
- **Depends on:** PR-144
- **Allowed:** `EVALUATION.md`

Acceptance:

- [ ] Define exact champion names: `medoid_multivariate_statistical_champion`, `medoid_univariate_evaluation_champion`, `delta1_univariate_evaluation_champion`.
- [ ] Only the multivariate champion is production-eligible.
- [ ] Within each univariate feature, model selection remains the canonical 12-candidate statistical ranking.
- [ ] Across feature winners, forbid OOS PLL/BIC/AIC/economic ranking.
- [ ] Define cross-feature rule exactly: NMI descending using anchored `1e-12` ties, shared OOS timestamp count descending, feature name ascending.
- [ ] Support below `0.80` makes a feature ineligible only for evaluation-level champion selection.
- [ ] No eligible feature produces explicit no-champion evidence; no fallback metric is allowed.
- [ ] No code change in this PR.

### PR-161 — Implement `medoid_univariate` evaluation orchestrator

- **Branch:** `pr/PR-161-medoid-univariate-evaluation`
- **Depends on:** PR-149, PR-150, PR-151, PR-159
- **Allowed:** `src/market_regime_engine/evaluations/medoid_univariate.py`, `tests/unit/evaluations/test_medoid_univariate.py`

Acceptance:

- [ ] Require exactly eight Stage-1 preliminary medoids in canonical block order and the PR-148 `medoid_univariate` clock.
- [ ] Execute exactly eight one-feature grids with 12 candidates each = 96 candidate evaluations.
- [ ] Bounded parallelism across the eight feature grids is allowed; output order is canonical block order and completion-order independent.
- [ ] Every feature records its `diagnostic_feature_model_winner` or explicit no-winner reason without aborting other features.
- [ ] Compute agreement for every valid feature winner using PR-151 and select `medoid_univariate_evaluation_champion` using exactly PR-159 rules.
- [ ] Raw OOS PLL/BIC/AIC are never used to rank different medoid features.
- [ ] Return complete feature-grid, agreement, eligibility and champion/tie evidence.
- [ ] No MLflow tracking, statistics writer, final refit, OOS publication, registration or alias mutation.

### PR-162 — Implement `delta1_univariate` evaluation orchestrator

- **Branch:** `pr/PR-162-delta1-univariate-evaluation`
- **Depends on:** PR-149, PR-150, PR-151, PR-159
- **Allowed:** `src/market_regime_engine/evaluations/delta1_univariate.py`, `tests/unit/evaluations/test_delta1_univariate.py`

Acceptance:

- [ ] Require exactly the ordered 13-delta tuple and the PR-148 `delta1_univariate` clock.
- [ ] Execute exactly 13 one-feature grids with 12 candidates each = 156 candidate evaluations.
- [ ] Bounded parallelism across 13 feature grids is allowed; output order is exact delta tuple order and completion-order independent.
- [ ] Every delta records its `diagnostic_feature_model_winner` or explicit no-winner reason without aborting other deltas.
- [ ] Compute agreement for every valid feature winner using PR-151 and select `delta1_univariate_evaluation_champion` using exactly PR-159 rules.
- [ ] Raw OOS PLL/BIC/AIC are never used to rank different delta features.
- [ ] Return complete feature-grid, agreement, eligibility and champion/tie evidence.
- [ ] Even if a delta is also a Stage-1 medoid, this evaluation performs its own fit on the delta evaluation clock; no cross-evaluation fit reuse.
- [ ] No MLflow tracking, statistics writer, final refit, OOS publication, registration or alias mutation.

### PR-160 — Add mandatory per-MLflow-run local statistics writer

- **Branch:** `pr/PR-160-evaluation-run-statistics-writer`
- **Depends on:** PR-145, PR-159
- **Allowed:** `.gitignore`, `src/market_regime_engine/evaluation_statistics/__init__.py`, `src/market_regime_engine/evaluation_statistics/contracts.py`, `src/market_regime_engine/evaluation_statistics/writer.py`, `src/market_regime_engine/evaluation_statistics/render.py`, `tests/unit/evaluation_statistics/*`

Acceptance:

- [ ] Define one versioned statistics schema capable of representing every mandatory field group in this section.
- [ ] Accept only the exact evaluation IDs `medoid_multivariate`, `medoid_univariate`, `delta1_univariate`.
- [ ] Resolve paths exactly as repository-checkout-relative `./evaluations/<evaluation>/<mlflow_run_id>/`.
- [ ] Add `evaluations/` to `.gitignore`; generated evidence is never committed.
- [ ] Preflight creates/verifies the evaluation root and fails before MLflow-run creation if not writable.
- [ ] Create each run directory once; atomically write initial `RUNNING` JSON/Markdown then atomically finalize to `FINISHED` or `FAILED`.
- [ ] Canonical JSON is deterministic UTF-8, finite-only and rejects NaN/Inf, unknown schema fields, raw feature payloads, DSNs and secret fields.
- [ ] SHA-256 is computed from exact finalized `statistics.json` bytes and exposed for MLflow parity verification.
- [ ] Finalized run directories are immutable; overwrite/reuse attempts fail closed.
- [ ] Failed runs retain a safe FAILED dossier with error code/reason and all evidence available before failure.
- [ ] Tests cover all three evaluation IDs, parent/feature/candidate run types, atomicity, crash-temp cleanup, unwritable roots, immutability, deterministic hashing and forbidden-data rejection.

### PR-152 — Track all three evaluations in MLflow with statistics parity

- **Branch:** `pr/PR-152-three-evaluation-mlflow-tracking`
- **Depends on:** PR-150, PR-161, PR-162, PR-160, PR-130
- **Allowed:** `src/market_regime_engine/mlflow_support/evaluation_tracking.py`, `src/market_regime_engine/evaluations/plots.py`, `tests/unit/mlflow_support/test_evaluation_tracking.py`, `tests/unit/evaluations/test_plots.py`, `PLOT_STYLE.md`

Acceptance:

- [ ] Create one independent top-level parent run per evaluation ID for one source snapshot/execution.
- [ ] `medoid_multivariate` parent logs exactly 12 candidate child runs and its production-eligible namespaced champion.
- [ ] `medoid_univariate` parent logs exactly eight feature runs and 96 candidate runs under those features.
- [ ] `delta1_univariate` parent logs exactly 13 feature runs and 156 candidate runs under those features.
- [ ] The same feature name occurring in two evaluations creates independent MLflow/local run identities; no result is silently shared across evaluation namespaces.
- [ ] Every MLflow parent/feature/candidate run has exactly one PR-160 local mirror under the matching evaluation directory.
- [ ] Every local `statistics.json` contains the complete applicable evidence from this section, not only MLflow scalar metrics.
- [ ] Exact finalized local JSON bytes are logged to that same run as `statistics/statistics.json`; SHA-256 matches local and MLflow metadata and is verified.
- [ ] Candidate runs log family/K/mixture, exact feature order, validity, aggregates and complete TEST-end fold histories.
- [ ] Evaluation parents log exact namespaced champion/no-champion evidence and all tie stages.
- [ ] Multivariate parent logs the complete v3 feature-selection lineage/evidence; univariate parents log evaluation-specific common-clock evidence and agreement tables.
- [ ] Plots are evaluation-specific and never visually rank different univariate features by raw/weighted OOS likelihood.
- [ ] Any local statistics creation/finalization/hash failure marks the MLflow run FAILED and fails the evaluation; no best-effort mode.
- [ ] No final refit, OOS publication, registration or alias mutation occurs in tracking.

### PR-153 — Add one standalone Xetra v3 three-evaluation runner

- **Branch:** `pr/PR-153-run-xetra-v3-evaluations`
- **Depends on:** PR-152
- **Allowed:** `scripts/run_xetra_v3_evaluations.py`, `tests/unit/commands/test_xetra_v3_evaluations_script.py`

Acceptance:

- [ ] Load exactly `xetra_v3.yaml` and `xetra_semantic_medoid_v3.yaml`.
- [ ] Open one read-only PostgreSQL source snapshot containing the canonical 61-feature universe and build one canonical source-row walk-forward plan.
- [ ] Run first-fold feature selection once on all 61 features and freeze eight preliminary medoids plus final Stage-2 tuple.
- [ ] Preflight all three local evaluation roots before creating any evaluation parent run.
- [ ] Execute `medoid_multivariate` first because both univariate evaluation champions require its statistical champion as agreement reference.
- [ ] Then execute `medoid_univariate` and `delta1_univariate` independently with their own PR-148 clocks; they may run in parallel after the multivariate champion exists.
- [ ] Full successful execution attempts exactly 264 candidate evaluations: 12 multivariate + 96 medoid-univariate + 156 delta1-univariate.
- [ ] Emit three independent MLflow parent hierarchies and local statistics mirrors.
- [ ] Never call final refit, prediction/OOS publication, registration or alias/CAS mutation.
- [ ] Candidate/feature invalidity is retained in its evaluation; source/contract/hash/tracking/statistics failures fail explicitly.
- [ ] Final stdout JSON reports source build, three parent run IDs, three namespaced champion statuses/identities, each evaluation clock/count/hash, candidate counts, failed-feature counts and local evidence roots.

### PR-154 — Prove the three-evaluation workflow hermetically

- **Branch:** `pr/PR-154-three-evaluation-e2e-proof`
- **Depends on:** PR-153
- **Allowed:** `tests/e2e/test_xetra_v3_three_evaluations.py`, three-evaluation E2E fixtures only

Acceptance:

- [ ] Hermetic fixture exposes exactly the canonical 61 features with controlled asymmetric missingness and no NAS dependency.
- [ ] Prove Stage-1 consumes all 61, yields exactly eight medoids and can select a delta as a medoid; Stage-2 freezes the multivariate tuple.
- [ ] Prove candidate counts exactly 12, 96 and 156 for the three evaluations and total 264.
- [ ] Prove medoid/delta common clocks are independent and can differ under asymmetric missingness; no combined 21-feature clock is constructed.
- [ ] Prove a delta that is also a medoid is independently evaluated in both univariate evaluations and receives distinct evaluation/clock lineage.
- [ ] Inject deterministic runners where necessary so CI need not execute all real fits; injection cannot bypass count/hash/clock/orchestration assertions.
- [ ] Include at least one small real Gaussian and one small real Student-t/GMM walk-forward smoke through the shared runner path.
- [ ] Prove one feature can have zero accepted candidates without removing other feature results.
- [ ] Prove NMI/permutation agreement is label-invariant and uses only shared valid OOS support.
- [ ] Prove multivariate champion equals canonical statistical selection; univariate champions obey NMI/support/tie rules and never use cross-feature PLL.
- [ ] Prove MLflow hierarchy exactly: three parents; multivariate 12 candidates; medoid 8 feature + 96 candidate runs; delta 13 feature + 156 candidate runs.
- [ ] Prove every MLflow run has exactly one matching local `statistics.json`/`statistics.md`, including FAILED runs, and MLflow JSON artifact bytes/hash match local.
- [ ] Prove statistics completeness for fold/start/state/ranking/lineage evidence and rejection of NaN/Inf, raw source rows and secrets.
- [ ] Prove diagnostic evaluations never call final refit, OOS publication, registration or alias mutation via fail-on-call doubles.
- [ ] Required tests remain hermetic and satisfy repository coverage policy.

### PR-155 — Document Xetra v3 three-evaluation architecture

- **Branch:** `pr/PR-155-three-evaluation-documentation`
- **Depends on:** PR-154
- **Allowed:** `docs/regime_evaluations.md`, `README.md`

Acceptance:

- [ ] Document why v3 extends the canonical universe to 61 features without mutating v1/v2.
- [ ] List exact 13 delta features and semantic-block assignments.
- [ ] Explain the exact roles and inputs of `medoid_multivariate`, `medoid_univariate`, `delta1_univariate`.
- [ ] Explain exact candidate counts 12/96/156 and total 264.
- [ ] Explain separate univariate clocks and why a feature appearing in both univariate evaluations is fitted independently.
- [ ] Explain all three champion names and why only the multivariate champion is production-eligible.
- [ ] Explain within-feature model ranking versus cross-feature NMI agreement ranking and prohibition on cross-feature PLL/BIC/AIC comparison.
- [ ] Document exact `./evaluations/<evaluation>/<mlflow_run_id>/` layout, immutable run directories, `statistics.json`/`statistics.md`, FAILED evidence and MLflow byte/hash parity.
- [ ] Document every mandatory statistics field group from this section and the no-secrets/no-raw-source-row rule.
- [ ] Document standalone runner prerequisites without credentials.
- [ ] README links to the document without duplicating normative contract text.

## Three-evaluation execution graph

Only merged dependencies unlock work. PRs with disjoint allowed files are intentionally parallelizable for weak agents.

```text
G0 prerequisites: PR-125 PR-129
G1: PR-156 after PR-125+PR-129
G2: PR-157 after PR-156
G3: PR-158 after PR-157+PR-118+PR-128
G4: PR-144 after PR-158
G5 parallel after PR-144: PR-145 PR-146 PR-147 PR-159
G6 parallel: PR-148 after PR-145; PR-160 after PR-145+PR-159; PR-150 after PR-145+PR-146+PR-147+PR-125+PR-158
G7: PR-149 after PR-145+PR-146+PR-147+PR-148
G8: PR-151 after PR-149+PR-150
G9 parallel: PR-161 after PR-149+PR-150+PR-151+PR-159; PR-162 after PR-149+PR-150+PR-151+PR-159
G10: PR-152 after PR-150+PR-161+PR-162+PR-160+PR-130
G11: PR-153 after PR-152
G12: PR-154 after PR-153
G13: PR-155 after PR-154
```

Parallel ownership is explicit: after PR-144, PR-145/146/147/159 are disjoint; later PR-161 and PR-162 are deliberately independent sibling orchestrators. The architecture has no combined shadow-suite orchestrator and no combined 21-feature univariate clock.

---
# 19. Delta1-univariate MLflow Model Metrics — 2026-08-31

## Scope contract

The first Model Metrics rollout is **only** for `delta1_univariate`. In this scope, a **dataset** means exactly one feature from the canonical ordered 13-delta tuple, and a **model** means one of the exact 12 canonical Xetra v3 candidate IDs. The existing MLflow hierarchy remains `delta1_univariate parent -> 13 feature/dataset runs -> 12 candidate/model runs per feature`; this rollout does not change candidate selection, agreement ranking, final refit, registry aliases, `medoid_multivariate`, or `medoid_univariate`.

For each delta feature run, Model Metrics must present the same deterministic model-centric structure:

```text
model_metrics/
  models/
    <candidate_id>/
      performance/
        train_loglik_per_obs
        oos_predictive_loglik_per_obs
        aic_per_train_obs
        bic_per_train_obs
        multistart_success_rate
      optimization/
        em_convergence
  comparisons/
    oos_predictive_loglik_per_obs_all_models
    em_convergence_all_models
  manifest.json
```

The first rollout uses the existing MLflow nested-run/artifact model and metric-history APIs; it must not fork or patch MLflow's React frontend. The `model_metrics/` namespace is the canonical dataset section inside each delta feature run.

The per-model EM convergence plot is an **optimization diagnostic only**. For each valid walk-forward fold it uses the winning multistart fit only, plots `TRAIN log likelihood / TRAIN model observation count` against EM iteration, never interpolates beyond a fold's recorded iterations, and overlays a deterministic across-fold median plus 25th/75th percentile band. A candidate with no usable convergence history gets explicit unavailable evidence instead of a fabricated curve. The all-model convergence comparison uses one median curve per canonical candidate and is explicitly labelled `optimization diagnostic only — not model selection`.

The canonical cross-model performance comparison remains OOS predictive log likelihood per observation on the common delta evaluation clock. Training likelihood, EM convergence, AIC/BIC, or any other optimization diagnostic must never become a cross-feature ranking criterion and must never change `diagnostic_feature_model_winner` or `delta1_univariate_evaluation_champion`.

### PR-163 — Pin delta1 Model Metrics diagnostic semantics

- **Branch:** `pr/PR-163-delta1-model-metrics-contract`
- **Depends on:** PR-152, PR-162
- **Allowed:** `EVALUATION.md`, `PLOT_STYLE.md`

Acceptance:

- [ ] Scope is explicitly limited to `delta1_univariate`; `medoid_multivariate` and `medoid_univariate` behavior and artifact layout are unchanged.
- [ ] Define `dataset` for this feature exactly as one of the canonical ordered 13 delta features and `model` exactly as one of the canonical 12 v3 candidate IDs.
- [ ] Pin the feature-run artifact namespace exactly to `model_metrics/models/<candidate_id>/...`, `model_metrics/comparisons/...`, and `model_metrics/manifest.json`.
- [ ] Pin the five core per-model performance histories exactly: TRAIN log likelihood per observation, OOS predictive log likelihood per observation, AIC per TRAIN observation, BIC per TRAIN observation, and multistart success rate.
- [ ] Pin one per-model `em_convergence` optimization plot and one `em_convergence_all_models` comparison plot.
- [ ] Define EM x-axis exactly as one-based completed EM iteration and y-axis exactly as `TRAIN log likelihood per observation`.
- [ ] Define one candidate plot as valid-fold winning-start trajectories plus deterministic across-fold median and 25th/75th percentile envelope; no failed/non-winning seed is mixed into the main candidate curve.
- [ ] Define varying iteration lengths with no extrapolation/interpolation beyond a fold's recorded history; aggregation at iteration `i` uses only valid winner histories that actually contain iteration `i`.
- [ ] Define the all-model EM comparison as one median curve per canonical candidate in canonical candidate order; no curve value is interpreted as statistical rank.
- [ ] Every EM plot title/legend contains `optimization diagnostic only — not model selection` or an equivalent unambiguous phrase.
- [ ] Canonical model comparison remains OOS predictive log likelihood per observation; no TRAIN/EM/AIC/BIC value is fed into cross-feature agreement ranking.
- [ ] A candidate with no valid convergence trace is represented as explicitly unavailable and must not be silently omitted from the dataset section.
- [ ] First rollout explicitly reuses MLflow nested runs/artifacts/metric histories and introduces no MLflow frontend fork, injected React bundle, or second web UI.
- [ ] No numerical gate, candidate universe, state alignment rule, evaluation clock, statistical-selection rule, or production eligibility rule changes in this PR.

### PR-164 — Preserve exact EM likelihood histories in HMM fit results

- **Branch:** `pr/PR-164-hmm-em-likelihood-history`
- **Depends on:** PR-163, PR-128, PR-129
- **Allowed:** `src/market_regime_engine/models/protocols.py`, `src/market_regime_engine/models/gaussian_hmm.py`, `src/market_regime_engine/models/student_t_hmm.py`, `src/market_regime_engine/training/multistart.py`, `tests/unit/models/test_gaussian_hmm.py`, `tests/unit/models/test_student_t_hmm.py`, `tests/unit/training/test_multistart.py`, direct `FitResult` fixture tests only

Acceptance:

- [ ] Extend `FitResult` with one immutable ordered `em_log_likelihood_history: tuple[float, ...]` containing the optimizer objective for each completed EM iteration.
- [ ] A production successful fit requires a non-empty finite history with `len(history) == iterations`; NaN/Inf or length mismatch fails closed.
- [ ] Gaussian HMM copies the exact `hmmlearn` convergence-monitor history after fit; no second fit or synthetic reconstruction is used.
- [ ] GMM-HMM copies the exact `hmmlearn` GMM convergence-monitor history under the same semantics.
- [ ] Student-t HMM records the exact likelihood value evaluated by its existing EM loop once per completed iteration; recording does not alter parameter updates or stopping tolerance.
- [ ] Existing material-likelihood-regression validation uses the same captured history rather than a separately reconstructed sequence.
- [ ] `train_log_likelihood` remains the canonical post-fit TRAIN likelihood used by PR-129 parity/AIC/BIC; the EM history is diagnostic and does not replace that scalar.
- [ ] `MultistartResult.winner` preserves the complete winning `FitResult` history without rerunning the winner.
- [ ] Multistart winner selection remains exact global TRAIN-likelihood maximum with anchored `1e-12` tie semantics then lowest seed; history shape never influences the winner.
- [ ] Reconstruction/inference paths do not invent an EM history for already persisted model artifacts.
- [ ] Tests prove exact history length/order for Gaussian, GMM and Student-t fits and prove multistart returns the winner's original history byte-for-byte/value-for-value.
- [ ] Tests prove adding the history does not change final fitted parameters, canonical TRAIN likelihood, OOS continuation likelihood, AIC/BIC, convergence gate, or winning seed for fixed fixtures.

### PR-165 — Add fail-closed pre-finalization payload logging for evaluation runs

- **Branch:** `pr/PR-165-evaluation-tracking-payload-hook`
- **Depends on:** PR-152
- **Allowed:** `src/market_regime_engine/mlflow_support/evaluation_tracking.py`, `tests/unit/mlflow_support/test_evaluation_tracking.py`

Acceptance:

- [ ] Refactor the existing statistics-run helper to support an optional injected payload emitter executed after MLflow run creation/local `RUNNING` evidence creation and before local/MLflow `FINISHED` finalization.
- [ ] The emitter receives only the current run ID and deterministic writable run-evidence directory/context needed to log metrics/artifacts; no global active-run state is required.
- [ ] With no emitter, behavior and exact existing parent/feature/candidate hierarchy remain byte-for-byte/semantically unchanged.
- [ ] Emitter success permits normal statistics finalization, SHA-256 parity logging, artifact logging, and `FINISHED` termination exactly once.
- [ ] Emitter failure marks the MLflow run `FAILED`, finalizes the local dossier as `FAILED` when safely possible, preserves already available safe evidence, and re-raises; a false `FINISHED` state is impossible.
- [ ] Statistics hash parity is computed only after the final lifecycle state is known.
- [ ] The helper does not know delta feature names, model families, plot semantics, or candidate ordering; it is generic tracking infrastructure only.
- [ ] No registry operation, final refit, OOS publication, alias mutation, HTTP/UI customization, or model computation is introduced.
- [ ] Unit tests cover no-emitter compatibility, successful metric/artifact emission, emitter exception, MLflow logging exception, local finalization exception, and exactly-once run termination.

### PR-166 — Render delta1 per-model and all-model EM convergence diagnostics

- **Branch:** `pr/PR-166-delta1-em-convergence-plots`
- **Depends on:** PR-163, PR-164
- **Allowed:** `src/market_regime_engine/mlflow_support/plots.py`, plot-focused unit tests only

Acceptance:

- [ ] Add a pure renderer for one `WalkForwardEvaluation` candidate that consumes only stored winning `FitResult.em_log_likelihood_history`; it never refits a model.
- [ ] Normalize every fold history point exactly as `em_log_likelihood_history[i] / train_model_observation_count` using that same fold's retained TRAIN model count.
- [ ] Candidate plot x-axis is one-based `EM iteration`; y-axis is exactly `TRAIN log likelihood per observation`.
- [ ] Valid fold winner trajectories are visible individually with deterministic fold identity while the candidate median is visually dominant and the 25th/75th percentile envelope is present.
- [ ] Median/quantiles at iteration `i` use only valid folds with an observed `i`; no padding value, forward fill, interpolation, extrapolation, or invented plateau is allowed.
- [ ] Invalid folds and valid folds lacking a trace are counted in explicit plot/manifest metadata; they do not contribute numeric points.
- [ ] A candidate with zero usable traces yields deterministic unavailable plot/evidence rather than raising an unrelated feature-wide failure or drawing a fake zero line.
- [ ] Add an all-candidate renderer for exactly the candidate evaluations supplied by one `UnivariateFeatureGrid`; each candidate contributes only its across-fold median history.
- [ ] All-model legend order follows the exact canonical 12-candidate order and does not reorder by final TRAIN likelihood, convergence speed, AIC/BIC, or OOS score.
- [ ] Both plot titles clearly state the delta feature and `optimization diagnostic only — not model selection`.
- [ ] Source hashes/manifest metadata include feature name, candidate ID(s), fold IDs, winning seeds, TRAIN counts, raw history values, aggregation rule and plot type so the rendering is reproducible.
- [ ] Generated artifacts satisfy the existing `PLOT_STYLE.md` accessibility/label/determinism requirements and include publication-quality vector output where that contract requires it.
- [ ] Renderer code performs no MLflow calls and changes no evaluation metric or selection result.

### PR-167 — Wire delta1 dataset Model Metrics sections into MLflow

- **Branch:** `pr/PR-167-delta1-model-metrics-tracking`
- **Depends on:** PR-164, PR-165, PR-166, PR-152, PR-162
- **Allowed:** `src/market_regime_engine/mlflow_support/evaluation_tracking.py`, `src/market_regime_engine/evaluation_statistics/contracts.py`, `src/market_regime_engine/evaluation_statistics/render.py`, `tests/unit/mlflow_support/test_evaluation_tracking.py`, `tests/unit/evaluation_statistics/test_writer.py`

Acceptance:

- [ ] The code path is activated only when `evaluation_id == delta1_univariate`; both other evaluation IDs retain their existing tracking behavior and artifact structure.
- [ ] Exactly 13 delta feature runs remain the dataset sections, in the canonical ordered delta tuple; no synthetic aggregate dataset run is introduced.
- [ ] Inside every delta feature run create exactly one deterministic `model_metrics/` artifact namespace matching PR-163.
- [ ] Every dataset section lists all 12 canonical candidate IDs even when a candidate has no accepted/valid folds; unavailable candidates retain explicit status/reason evidence.
- [ ] For every candidate, publish the five exact core performance plots/histories pinned by PR-163 under `model_metrics/models/<candidate_id>/performance/` by reusing existing metric definitions/rendering rather than recomputing alternate statistics.
- [ ] For every candidate, publish the PR-166 EM convergence artifact under `model_metrics/models/<candidate_id>/optimization/em_convergence`.
- [ ] Log candidate-level native MLflow metric histories for the across-fold EM median using one metric key with `step = EM iteration`; values exactly equal the plotted median series.
- [ ] Publish exactly one all-model OOS predictive-log-likelihood-per-observation comparison under `model_metrics/comparisons/oos_predictive_loglik_per_obs_all_models` using the existing canonical candidate comparison semantics.
- [ ] Publish exactly one PR-166 all-model EM comparison under `model_metrics/comparisons/em_convergence_all_models`.
- [ ] Write `model_metrics/manifest.json` containing the exact 12 candidate IDs/order, every expected artifact path, availability status, source hash, source metric keys, and comparison plot identities.
- [ ] Candidate local `statistics.json` gains the winning-fold optimization evidence needed to reproduce its EM plot: fold ID, winning seed, TRAIN model observation count, completed iteration count and raw EM history; no raw source row is stored.
- [ ] Local statistics and MLflow evidence remain finite-only and secret-free; raw feature values, DSNs, credentials and model binary payloads remain forbidden.
- [ ] Plot/metric emission occurs through PR-165 before the relevant run is finalized; any emission failure fails that run/evaluation instead of leaving a false `FINISHED` run with an incomplete Model Metrics section.
- [ ] No metric or plot produced here is passed into candidate selection, cross-feature NMI agreement, evaluation champion selection, final refit, registration or alias logic.
- [ ] No cross-feature PLL/AIC/BIC/EM comparison plot is created; comparisons are strictly within one delta dataset across its 12 models.
- [ ] No custom MLflow frontend code, injected JavaScript, second dashboard service or non-MLflow port is introduced.

### PR-168 — Prove delta1 Model Metrics hierarchy hermetically

- **Branch:** `pr/PR-168-delta1-model-metrics-proof`
- **Depends on:** PR-167
- **Allowed:** `tests/integration/mlflow_support/test_delta1_model_metrics.py`, delta1 Model Metrics fixtures only

Acceptance:

- [ ] Hermetic file-store MLflow fixture requires no NAS, feature PostgreSQL, network service, browser automation, or external MLflow instance.
- [ ] Fixture represents exactly the 13 canonical delta datasets and exact 12 canonical candidate identities per dataset.
- [ ] Prove the tracking hierarchy remains one `delta1_univariate` parent, 13 feature/dataset child runs and 156 candidate/model child runs.
- [ ] For each of all 13 feature runs, prove `model_metrics/manifest.json` exists and enumerates exactly 12 candidate sections in canonical order.
- [ ] For each candidate section, prove the five core performance entries and one EM-convergence entry exist or carry explicit unavailable status/reason; silent omission is forbidden.
- [ ] Prove each usable candidate EM plot/metric series equals the stored winning-fold histories normalized by exact per-fold TRAIN model counts and aggregated under the PR-163 missing-iteration rule.
- [ ] Include histories with different convergence lengths and prove no extrapolation, interpolation, padding, or last-value carry occurs.
- [ ] Include one candidate with no usable history and prove it remains listed with unavailable evidence while the other 11 models/dataset and other datasets are unaffected.
- [ ] Prove each dataset has exactly one OOS all-model comparison and one EM all-model comparison, both containing the exact 12 candidate identities/order.
- [ ] Prove the EM comparison is labelled optimization-only and that changing EM histories alone cannot change `diagnostic_feature_model_winner` or `delta1_univariate_evaluation_champion`.
- [ ] Prove no artifact or metric is emitted into `medoid_multivariate` or `medoid_univariate` Model Metrics namespaces by this rollout.
- [ ] Prove local statistics contain the exact optimization trace evidence and remain byte/hash-consistent with the MLflow `statistics/statistics.json` artifact.
- [ ] Inject one plot/logging failure and prove the affected MLflow/local run is `FAILED` with no false finished incomplete section.
- [ ] Include at least one small real Gaussian-HMM, GMM-HMM and Student-t HMM fit asserting non-empty EM histories reach the renderer without refitting.
- [ ] Required tests are deterministic, hermetic and satisfy the repository coverage gate.

## Delta1 Model Metrics execution graph

```text
M0 prerequisites: PR-152 PR-162
M1: PR-163
M2 parallel after PR-163: PR-164; PR-165 may start after PR-152
M3: PR-166 after PR-163+PR-164
M4: PR-167 after PR-164+PR-165+PR-166+PR-152+PR-162
M5: PR-168 after PR-167
```

PR-165 and PR-164 intentionally own disjoint tracking/model files and may run in parallel after their dependencies are merged. PR-166 is rendering-only. PR-167 is the only PR that composes the new delta1-only evidence into the MLflow hierarchy; PR-168 is proof-only.
