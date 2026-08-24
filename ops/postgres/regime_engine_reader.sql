\set ON_ERROR_STOP on

-- Required psql variables:
--   target_db      exact serving database name
--   role_password  runtime secret; never echo this script with expanded variables

SELECT format('CREATE ROLE "regime-engine" LOGIN PASSWORD %L', :'role_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'regime-engine')
\gexec

SELECT format('ALTER ROLE "regime-engine" LOGIN PASSWORD %L', :'role_password')
\gexec

ALTER ROLE "regime-engine" SET default_transaction_read_only = on;
ALTER ROLE "regime-engine" SET statement_timeout = '30s';

SELECT format('REVOKE ALL ON DATABASE %I FROM "regime-engine"', :'target_db')
\gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO "regime-engine"', :'target_db')
\gexec

REVOKE ALL ON SCHEMA public FROM "regime-engine";
REVOKE ALL ON SCHEMA regime_loader FROM "regime-engine";
REVOKE ALL ON SCHEMA regime_loader_sync FROM "regime-engine";
GRANT USAGE ON SCHEMA regime_loader TO "regime-engine";
GRANT USAGE ON SCHEMA regime_loader_sync TO "regime-engine";

REVOKE ALL ON ALL TABLES IN SCHEMA regime_loader FROM "regime-engine";
REVOKE ALL ON ALL TABLES IN SCHEMA regime_loader_sync FROM "regime-engine";
GRANT SELECT ON TABLE regime_loader.regime_features_daily TO "regime-engine";
GRANT SELECT ON TABLE regime_loader_sync.gold_sync_state TO "regime-engine";

-- Deliberately no CREATE, TEMP, ownership, sequence, function, writer, or mutation grants.
