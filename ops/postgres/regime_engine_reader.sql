\set ON_ERROR_STOP on

-- The macro-loader serving identities are:
--   reader: "macro-loader"
--   owner:  "macro-loader-owner"
--   schema/table: macro_loader.macro_features_daily
--
-- Required psql variables:
--   target_db      exact serving database name
--   role_password  runtime secret; never echo this script with expanded variables

SELECT format('CREATE ROLE "macro-loader" LOGIN PASSWORD %L', :'role_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'macro-loader')
\gexec

SELECT format('ALTER ROLE "macro-loader" LOGIN PASSWORD %L', :'role_password')
\gexec

ALTER ROLE "macro-loader" SET default_transaction_read_only = on;
ALTER ROLE "macro-loader" SET statement_timeout = '30s';

SELECT format('REVOKE ALL ON DATABASE %I FROM "macro-loader"', :'target_db')
\gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO "macro-loader"', :'target_db')
\gexec

REVOKE ALL ON SCHEMA public FROM "macro-loader";
REVOKE ALL ON SCHEMA macro_loader FROM "macro-loader";
REVOKE ALL ON SCHEMA macro_loader_sync FROM "macro-loader";
GRANT USAGE ON SCHEMA macro_loader TO "macro-loader";
GRANT USAGE ON SCHEMA macro_loader_sync TO "macro-loader";

REVOKE ALL ON ALL TABLES IN SCHEMA macro_loader FROM "macro-loader";
REVOKE ALL ON ALL TABLES IN SCHEMA macro_loader_sync FROM "macro-loader";
GRANT SELECT ON TABLE macro_loader.macro_features_daily TO "macro-loader";
GRANT SELECT ON TABLE macro_loader_sync.gold_sync_state TO "macro-loader";

-- Deliberately no CREATE, TEMP, ownership, sequence, function, writer, or mutation grants.
