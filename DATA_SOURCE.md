# Regime Engine Data-Source Contract

Status date: 2026-08-23

This document defines the production feature-source boundary for `regime-engine`.

## Canonical naming

The upstream data product is named **`regime-loader`** and is maintained in:

```text
https://github.com/SergejSchweizer/regime-loader.git
```

`market-regime-loader` is a retired name and must not be used in repository documentation, source identifiers, integration paths, configuration names, tests, or operator instructions.

## Ownership boundary

`regime-loader` owns acquisition, Bronze/Silver/Gold processing, causal feature construction, Gold publication, and synchronization of the canonical Gold serving replica to PostgreSQL.

`regime-engine` owns model profiles, preprocessing, HMM fitting, leak-free evaluation, causal inference, MLflow registration, prediction artifacts, and inference APIs. It must not duplicate provider acquisition or the upstream feature-building pipeline.

The production dependency direction is:

```text
regime-loader
    -> canonical immutable Gold
    -> PostgreSQL serving replica at 10.10.1.3:54321
    -> regime-engine
    -> MLflow / RegimePrediction.v1 / OOS prediction artifacts / API
    -> downstream consumers
```

## PostgreSQL production source

The production feature source is the PostgreSQL serving replica written by `regime-loader`.

```text
host:              10.10.1.3
port:              54321
dataset_id:        regime_features_daily
feature table:     regime_loader.regime_features_daily
sync-state table:  regime_loader_sync.gold_sync_state
row-digest table:  regime_loader_sync.gold_row_hashes
```

The feature table contains the upstream Gold temporal key and feature columns. The temporal key is:

```text
timestamp_m1 TIMESTAMPTZ(6)
```

Database sessions are interpreted in UTC. Feature columns are nullable `DOUBLE PRECISION` in the upstream serving plane.

The engine must treat PostgreSQL as a serving replica. The authoritative source remains the immutable Gold data product owned by `regime-loader`; the engine must not mutate the feature table or the loader synchronization metadata.

## Source lineage

The engine must preserve upstream build lineage for every training, evaluation, registration, replay, and realtime prediction operation.

For `dataset_id = 'regime_features_daily'`, lineage is read from:

```text
regime_loader_sync.gold_sync_state
```

At minimum the engine must retain:

```text
source_build_id
data_sha256
schema_version
feature_version
row_count
min_timestamp
max_timestamp
synced_at_utc
```

The consumer feature rows are read from:

```text
regime_loader.regime_features_daily
```

The engine must never invent a build identifier from timestamps, row counts, filesystem ordering, or query time.

## Consistent snapshot requirement

Feature rows and their `gold_sync_state` lineage must represent the same committed upstream synchronization state.

A production feature-source adapter must therefore obtain both within one transactionally consistent, read-only PostgreSQL snapshot. The expected semantic is equivalent to:

```text
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
  read regime_loader_sync.gold_sync_state
  read bounded rows from regime_loader.regime_features_daily
  validate rows against sync-state bounds and requested profile
COMMIT;
```

This prevents a loader synchronization commit from being observed between lineage and feature queries.

The adapter must fail closed if the sync-state row is missing, the dataset ID is wrong, semantic versions are incompatible, timestamps are duplicated/non-monotonic, required features are absent, or selected numerical values are non-finite.

## Query semantics

The engine reads only the requested interval and exact ordered feature set required by the validated model profile.

Required behavior:

- order rows by `timestamp_m1` ascending;
- preserve exact profile feature order;
- never forward-fill, backward-fill, interpolate, or silently impute missing upstream values;
- never use observations after the requested `as_of` timestamp;
- preserve `source_build_id`, `data_sha256`, `schema_version`, and `feature_version` in engine lineage;
- reject duplicate timestamps and incompatible feature contracts;
- keep walk-forward train/test slicing inside the engine after the upstream snapshot has been bound to explicit lineage.

## Credentials and least privilege

PostgreSQL credentials are runtime-only. No password or connection string containing a password may be committed, logged, embedded in MLflow artifacts, or returned by an API.

Production configuration uses standard PostgreSQL environment variables:

```text
PGHOST=10.10.1.3
PGPORT=54321
PGDATABASE=<serving database>
PGUSER=<engine read-only role>
PGPASSWORD=<runtime-only secret>
```

The engine should use its own least-privilege read-only database identity. It must not require or reuse the `regime-loader` writer secret. The read-only identity needs only the privileges required to select the consumer feature table and the synchronization state needed for lineage validation.

## CI and test boundary

Required push/merge tests remain hermetic and must not depend on `10.10.1.3:54321`.

Unit and required integration tests use injected PostgreSQL ports/fakes or a local deterministic test database/fixture representing:

- `regime_loader.regime_features_daily`;
- `regime_loader_sync.gold_sync_state`.

A real-NAS PostgreSQL smoke test, if present, must be explicitly marked `external_service`, be opt-in, perform read-only verification, and be excluded from required CI gates.

## Backlog implementation impact

The implementation backlog must interpret the production feature-source work as follows:

- **PR-001:** include a Python-3.14-compatible PostgreSQL client (`psycopg`) in runtime dependencies.
- **PR-005:** document `regime-loader -> PostgreSQL -> engine -> consumers` and the exact production serving-plane boundary.
- **PR-006:** source/build lineage contracts must carry the upstream PostgreSQL sync-state identity and semantic versions.
- **PR-008:** implement the generic `FeatureSource` port plus a PostgreSQL adapter for `regime_loader.regime_features_daily`; production input is PostgreSQL rather than direct loader Parquet.
- **PR-020/021/022/024:** bind every walk-forward evaluation/candidate comparison to one explicit upstream synchronized source state.
- **PR-023/026/027:** persist upstream `source_build_id`, semantic versions, and source hash in MLflow/model/prediction lineage.
- **PR-028/029/031:** batch, latest, train, and evaluate paths resolve features through the PostgreSQL feature-source boundary.
- **PR-032:** deployment exposes PostgreSQL runtime settings/secrets in addition to `MLFLOW_TRACKING_URI`; it does not manage the PostgreSQL lifecycle.
- **PR-033:** verify compatibility with the `regime-loader` PostgreSQL serving contract, with optional read-only external smoke verification.
- **PR-035:** hermetic E2E uses an injected PostgreSQL-shaped feature source, not the real NAS endpoint.
- **PR-036:** final operator documentation states the exact endpoint, tables, lineage contract, and read-only credential model.

Parquet remains appropriate for engine-owned immutable prediction artifacts and local deterministic fixtures. Direct access to the upstream loader's filesystem is not the production feature-source contract.

## Non-goals

This repository does not:

- provision or administer the shared PostgreSQL server;
- write to `regime_loader.regime_features_daily`;
- write to `regime_loader_sync.*`;
- invoke `regime-loader` as a Python package;
- rebuild upstream Gold features;
- perform provider HTTP acquisition;
- store PostgreSQL credentials in Git or MLflow.
