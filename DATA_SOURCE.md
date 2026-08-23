# Regime Engine Data-Source Contract

Status date: 2026-08-23

This document is authoritative for production feature-source transport, lineage, time semantics, missing-value semantics and credential boundaries.

## Canonical upstream

Upstream data product: `regime-loader`

```text
https://github.com/SergejSchweizer/regime-loader.git
```

Production dependency:

```text
regime-loader
  -> immutable Gold
  -> PostgreSQL serving replica 10.10.1.3:54321
  -> regime-engine
  -> MLflow / predictions / API
  -> consumers
```

`regime-engine` never performs provider acquisition or rebuilds upstream Gold.

## Production PostgreSQL source

```text
host:              10.10.1.3
port:              54321
database:          mandatory runtime value; no default
dataset_id:        regime_features_daily
feature table:     regime_loader.regime_features_daily
sync-state table:  regime_loader_sync.gold_sync_state
temporal key:      timestamp_m1 TIMESTAMPTZ(6)
```

The current row-digest table `regime_loader_sync.gold_row_hashes` exists upstream but is not required by the MVP engine read contract.

Feature columns are nullable `DOUBLE PRECISION`. SQL NULL is permitted by the upstream source; NaN/infinity is invalid.

## Dedicated least-privilege identity

Production runtime username is exactly:

```text
regime-engine
```

SQL role identifier is quoted as:

```sql
"regime-engine"
```

Required grants only:

- database `CONNECT` on the explicitly supplied serving database;
- schema `USAGE` on `regime_loader` and `regime_loader_sync`;
- `SELECT` on `regime_loader.regime_features_daily`;
- `SELECT` on `regime_loader_sync.gold_sync_state`.

No writer/admin/ownership/CREATE privileges are required. The engine must never reuse the `regime-loader` writer credential.

## Runtime environment contract

Feature PostgreSQL settings are deliberately namespaced because the same MLflow container has a separate backend PostgreSQL:

```text
REGIME_FEATURE_PGHOST=10.10.1.3
REGIME_FEATURE_PGPORT=54321
REGIME_FEATURE_PGDATABASE=<required runtime value>
REGIME_FEATURE_PGUSER=regime-engine
REGIME_FEATURE_PGPASSWORD_FILE=<preferred production secret file>
REGIME_FEATURE_PGPASSWORD=<optional local/test direct secret>
REGIME_FEATURE_PGSSLMODE=require
```

Production requires PostgreSQL TLS transport through `sslmode=require`. PR-033 external verification must prove the existing feature PostgreSQL accepts this mode. If the server does not support it, deployment is blocked until a deliberate versioned infrastructure/security contract change is approved; an implementation agent must not silently downgrade to `prefer` or `disable`.

Generic `PGHOST`, `PGDATABASE`, `PGUSER`, and `PGPASSWORD` are not the production regime-feature configuration contract.

No password or credential-bearing DSN may be committed, logged, embedded in MLflow artifacts/models, or returned by an API.

## Source lineage

For `dataset_id=regime_features_daily`, read from `regime_loader_sync.gold_sync_state` and preserve at least:

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

The engine never derives/invents `source_build_id` from timestamps, row counts, mtime, directory order, or query time.

## Consistent source snapshot

Lineage and rows for one engine source acquisition must represent one committed loader synchronization state.

Required semantics:

```text
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
  read regime_loader_sync.gold_sync_state
  read bounded feature rows from regime_loader.regime_features_daily
  validate lineage/source bounds
COMMIT;
```

The PostgreSQL transaction ends after source materialization. HMM fitting/evaluation must not hold a long-lived database transaction open.

Fail closed for missing/wrong sync-state, incompatible schema/feature version, duplicate/non-monotonic timestamps, missing requested columns, nonfinite non-null values, or rows outside validated source bounds.

## Time semantics

`timestamp_m1` is upstream observation-day identity. It is not provider release time and does not identify a historical data vintage.

Canonical engine metadata:

```text
data_time_semantics=current_vintage_observation_day
```

Therefore:

- walk-forward is split-leak-free with respect to the current-vintage observation sequence;
- the MVP is not claimed to be provider-release-time/vintage-safe or fully point-in-time tradable;
- a historical `as_of` or fixed-model replay cuts the current serving replica by observation timestamp;
- later upstream historical corrections may change later replays even when `model_version` is pinned;
- immutable `walk_forward_oos` builds preserve the source-build lineage used when they were produced;
- a stronger point-in-time claim requires a future versioned upstream contract carrying availability/vintage information.

## Missing-value and model-observation semantics

There are two source modes.

### Feature-selection mode

- NULL values are allowed.
- Every non-null numeric value must be finite.
- Coverage/complete-case rules in `EVALUATION.md` determine eligibility.
- No forward fill, backward fill, interpolation, implicit carry, or synthetic row.

### Resolved-model mode

For the frozen final model features, a model observation exists only where all final features are non-null and finite.

- incomplete timestamps are excluded from the HMM observation sequence;
- excluded timestamps remain explicit gap/count evidence;
- transitions occur once between consecutive retained observations, not once per calendar day;
- no extra `A^gap_days` transition is applied;
- the same rule is used for evaluation, final refit, latest and replay;
- latest chooses the latest complete model observation at or before `as_of`;
- replay returns only complete model observations inside its requested interval and reports skipped incomplete rows.

## Query semantics

- UTC everywhere;
- SQL rows ordered by `timestamp_m1` ascending;
- only requested bounded interval and exact ordered columns;
- identifiers validated against the registered/profile contract before SQL construction;
- values/bounds are parameterized;
- no observation later than explicit `as_of` may influence that prediction;
- no source mutation is permitted.

## CI/external-service boundary

Required tests remain hermetic and never depend on `10.10.1.3:54321`.

Required tests use injected/fake/local PostgreSQL-shaped sources representing the feature table and sync-state table.

A real NAS smoke test is allowed only when explicitly marked `external_service`, opted in by the operator, authenticated as `regime-engine`, uses `sslmode=require`, and is read-only. It verifies privilege metadata rather than attempting destructive writes.

## Non-goals

This repository does not provision/restart the external PostgreSQL server/database, mutate loader tables/sync metadata, manage the loader writer identity, build upstream features, or claim unavailable historical release-time/vintage semantics.
