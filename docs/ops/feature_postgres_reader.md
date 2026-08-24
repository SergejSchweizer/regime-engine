# Feature PostgreSQL reader

Production access uses the exact quoted PostgreSQL role `"regime-engine"` against the external `regime-loader` serving database. The database name, reader password, and administrator DSN are runtime-only values; none has a repository default.

Required bootstrap environment:

- `REGIME_FEATURE_PGDATABASE`
- `REGIME_FEATURE_PGPASSWORD_FILE`
- `REGIME_FEATURE_PGADMIN_DSN_FILE`

Run `ops/postgres/bootstrap_reader.sh`. It passes the database name and password to `psql` variables and lets PostgreSQL quote them with `format('%I', ...)` / `format('%L', ...)`; shell text is never spliced into SQL identifiers or literals.

The role receives only database `CONNECT`, schema `USAGE` on `regime_loader` and `regime_loader_sync`, and `SELECT` on `regime_loader.regime_features_daily` plus `regime_loader_sync.gold_sync_state`. Its default transaction mode is read-only. The verification script uses only PostgreSQL privilege catalogs and never attempts a destructive write.

The runtime adapter separately requires `sslmode=require`; this provisioning step does not weaken TLS policy or mutate the external feature/sync rows.
