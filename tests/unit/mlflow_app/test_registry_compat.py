from __future__ import annotations

from market_regime_engine.mlflow_app.registry_compat import install_postgres_model_version_cast


def test_postgres_registry_model_version_lookup_binds_numeric_strings_as_integers(
    monkeypatch,
) -> None:
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore

    observed: list[object] = []

    def fake_original(self, session, name, version, eager=False):
        observed.append(version)
        return object()

    monkeypatch.setattr(SqlAlchemyStore, "_get_sql_model_version", fake_original)
    monkeypatch.setattr(
        SqlAlchemyStore,
        "_regime_engine_postgres_version_cast",
        False,
        raising=False,
    )
    install_postgres_model_version_cast()

    SqlAlchemyStore._get_sql_model_version(object(), object(), "regime-xetra", "17")
    assert observed == [17]
