from __future__ import annotations

from pathlib import Path


def test_reader_sql_is_exact_least_privilege_and_safe() -> None:
    sql_text = Path("ops/postgres/regime_engine_reader.sql").read_text(encoding="utf-8")
    assert '"regime-engine"' in sql_text
    assert "regime_loader.regime_features_daily" in sql_text
    assert "regime_loader_sync.gold_sync_state" in sql_text
    assert "GRANT SELECT" in sql_text
    assert "GRANT USAGE" in sql_text
    assert "GRANT CONNECT" in sql_text
    assert "default_transaction_read_only = on" in sql_text
    assert "format('GRANT CONNECT ON DATABASE %I" in sql_text
    assert ":'role_password'" in sql_text
    forbidden = ("GRANT INSERT", "GRANT UPDATE", "GRANT DELETE", "GRANT CREATE", "GRANT ALL")
    assert all(token not in sql_text for token in forbidden)


def test_verification_is_catalog_only_not_destructive() -> None:
    verification = Path("ops/postgres/verify_reader.sh").read_text(encoding="utf-8")
    assert "has_database_privilege" in verification
    assert "has_schema_privilege" in verification
    assert "has_table_privilege" in verification
    assert "INSERT INTO" not in verification
    assert "UPDATE regime_loader" not in verification
    assert "DELETE FROM" not in verification
