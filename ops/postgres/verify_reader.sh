#!/usr/bin/env bash
set -euo pipefail

: "${REGIME_FEATURE_PGDATABASE:?REGIME_FEATURE_PGDATABASE is required}"
: "${REGIME_FEATURE_PGADMIN_DSN_FILE:?REGIME_FEATURE_PGADMIN_DSN_FILE is required}"
[[ -r "${REGIME_FEATURE_PGADMIN_DSN_FILE}" ]] || { echo "admin DSN secret file is not readable" >&2; exit 2; }
admin_dsn="$(cat "${REGIME_FEATURE_PGADMIN_DSN_FILE}")"

psql "${admin_dsn}" --no-psqlrc --set=ON_ERROR_STOP=1 --set=target_db="${REGIME_FEATURE_PGDATABASE}" <<'SQL'
WITH role_check AS (
  SELECT rolname,
         rolcanlogin,
         rolcreatedb,
         rolcreaterole,
         rolsuper,
         COALESCE(rolconfig, ARRAY[]::text[]) AS rolconfig
  FROM pg_roles
  WHERE rolname = 'regime-engine'
), assertions AS (
  SELECT
    rolname = 'regime-engine' AS exact_role,
    rolcanlogin AS can_login,
    NOT rolcreatedb AND NOT rolcreaterole AND NOT rolsuper AS no_admin,
    EXISTS (
      SELECT 1 FROM unnest(rolconfig) config
      WHERE config = 'default_transaction_read_only=on'
    ) AS read_only_default,
    has_database_privilege('regime-engine', :'target_db', 'CONNECT') AS db_connect,
    has_schema_privilege('regime-engine', 'regime_loader', 'USAGE') AS feature_schema_usage,
    has_schema_privilege('regime-engine', 'regime_loader_sync', 'USAGE') AS sync_schema_usage,
    has_table_privilege('regime-engine', 'regime_loader.regime_features_daily', 'SELECT') AS feature_select,
    has_table_privilege('regime-engine', 'regime_loader_sync.gold_sync_state', 'SELECT') AS sync_select,
    NOT has_table_privilege('regime-engine', 'regime_loader.regime_features_daily', 'INSERT,UPDATE,DELETE,TRUNCATE,TRIGGER') AS no_feature_write,
    NOT has_table_privilege('regime-engine', 'regime_loader_sync.gold_sync_state', 'INSERT,UPDATE,DELETE,TRUNCATE,TRIGGER') AS no_sync_write,
    NOT has_schema_privilege('regime-engine', 'regime_loader', 'CREATE') AS no_feature_create,
    NOT has_schema_privilege('regime-engine', 'regime_loader_sync', 'CREATE') AS no_sync_create
  FROM role_check
)
SELECT CASE WHEN bool_and(ok) THEN 'reader privileges verified' ELSE pg_catalog.set_config('regime_engine.verify_failed', '1', false) END
FROM assertions,
LATERAL unnest(ARRAY[
  exact_role, can_login, no_admin, read_only_default, db_connect,
  feature_schema_usage, sync_schema_usage, feature_select, sync_select,
  no_feature_write, no_sync_write, no_feature_create, no_sync_create
]) AS checks(ok);

DO $$
BEGIN
  IF current_setting('regime_engine.verify_failed', true) = '1' THEN
    RAISE EXCEPTION 'regime-engine reader privilege verification failed';
  END IF;
END
$$;
SQL
