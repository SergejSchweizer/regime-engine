# Feature PostgreSQL reader

Production access uses the exact quoted PostgreSQL role `"macro-loader"` against
the external `macro-loader` serving database. The feature owner is
`"macro-loader-owner"`; the consumer relation is
`macro_loader.macro_features_daily`. The database name, reader password, and
administrator DSN are runtime-only values; none has a repository default.

Required bootstrap environment:

- `REGIME_FEATURE_PGDATABASE`
- `REGIME_FEATURE_PGPASSWORD_FILE`
- `REGIME_FEATURE_PGADMIN_DSN_FILE`

Run `ops/postgres/bootstrap_reader.sh`. It passes the database name and password to `psql` variables and lets PostgreSQL quote them with `format('%I', ...)` / `format('%L', ...)`; shell text is never spliced into SQL identifiers or literals.

The role receives only database `CONNECT`, schema `USAGE` on `macro_loader` and
`macro_loader_sync`, and `SELECT` on `macro_loader.macro_features_daily` plus
`macro_loader_sync.gold_sync_state`. Its default transaction mode is
read-only. The verification script also checks that the feature relation is
owned by `macro-loader-owner`; it uses only PostgreSQL privilege catalogs and
never attempts a destructive write.

The runtime adapter separately requires `sslmode=disable` because this trusted-LAN PostgreSQL server does not offer TLS; this provisioning step does not mutate the external feature/sync rows.
