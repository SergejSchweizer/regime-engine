from __future__ import annotations

import os

import psycopg
import pytest

from market_regime_engine.features.postgres_settings import FeaturePostgresSettings

pytestmark = pytest.mark.external


def _external_settings() -> FeaturePostgresSettings:
    if os.environ.get("REGIME_RUN_EXTERNAL_FEATURE_PG") != "1":
        pytest.skip("set REGIME_RUN_EXTERNAL_FEATURE_PG=1 to run the external feature-PG smoke")
    return FeaturePostgresSettings.from_env(os.environ)


def test_external_feature_postgres_is_tls_read_only_and_least_privilege() -> None:
    settings = _external_settings()
    assert settings.host == "10.10.1.3"
    assert settings.port == 54321
    assert settings.user == "regime-engine"
    assert settings.sslmode == "require"

    try:
        connection = psycopg.connect(**settings.connection_kwargs())
    except psycopg.OperationalError:
        raise AssertionError(
            "external feature PostgreSQL connection failed; verify the endpoint "
            "accepts TLS with sslmode=require"
        ) from None

    with connection:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute(
                "SELECT current_user, current_setting('transaction_read_only'), "
                "current_setting('transaction_isolation')"
            )
            current_user, read_only, isolation = cursor.fetchone() or (None, None, None)
            assert current_user == "regime-engine"
            assert read_only == "on"
            assert isolation == "repeatable read"

            cursor.execute("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()")
            ssl_row = cursor.fetchone()
            assert ssl_row is not None and ssl_row[0] is True

            cursor.execute(
                "SELECT "
                "has_database_privilege(current_user, current_database(), 'CONNECT'), "
                "has_schema_privilege(current_user, 'regime_loader', 'USAGE'), "
                "has_schema_privilege(current_user, 'regime_loader_sync', 'USAGE'), "
                "has_table_privilege(current_user, "
                "'regime_loader.regime_features_daily', 'SELECT'), "
                "has_table_privilege(current_user, "
                "'regime_loader_sync.gold_sync_state', 'SELECT'), "
                "has_schema_privilege(current_user, 'regime_loader', 'CREATE'), "
                "has_schema_privilege(current_user, 'regime_loader_sync', 'CREATE'), "
                "has_table_privilege(current_user, "
                "'regime_loader.regime_features_daily', "
                "'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')"
            )
            privileges = cursor.fetchone()
            assert privileges is not None
            assert tuple(privileges[:5]) == (True, True, True, True, True)
            assert tuple(privileges[5:]) == (False, False, False)

            cursor.execute(
                "SELECT source_build_id, data_sha256, schema_version, feature_version, "
                "row_count, min_timestamp, max_timestamp, synced_at_utc "
                "FROM regime_loader_sync.gold_sync_state "
                "WHERE dataset_id = %s",
                ("regime_features_daily",),
            )
            lineage = cursor.fetchone()
            assert lineage is not None and len(lineage) == 8
            assert lineage[0]
            assert isinstance(lineage[1], str) and len(lineage[1]) == 64
            assert int(lineage[2]) == 1
            assert int(lineage[3]) == 1
            assert int(lineage[4]) >= 0
            assert lineage[5] <= lineage[6]

            cursor.execute(
                "SELECT timestamp_m1 FROM regime_loader.regime_features_daily "
                "WHERE timestamp_m1 >= %s AND timestamp_m1 <= %s "
                "ORDER BY timestamp_m1 ASC LIMIT 1",
                (lineage[5], lineage[6]),
            )
            feature_row = cursor.fetchone()
            if int(lineage[4]) > 0:
                assert feature_row is not None
                assert lineage[5] <= feature_row[0] <= lineage[6]
        connection.commit()
