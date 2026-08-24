"""Narrow compatibility fixes for the pinned MLflow runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def install_postgres_model_version_cast() -> None:
    """Bind numeric registry versions as integers for PostgreSQL.

    MLflow 3.15.1 validates a model version by converting it to ``int`` but
    retains the original REST string when constructing a SQLAlchemy predicate.
    PostgreSQL correctly rejects the resulting ``integer = varchar`` query.
    Keep the public API unchanged while passing the validated integer to the
    common model-version lookup path.
    """
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore

    if getattr(SqlAlchemyStore, "_regime_engine_postgres_version_cast", False):
        return

    original: Callable[..., Any] = SqlAlchemyStore._get_sql_model_version

    def get_sql_model_version(
        self: Any,
        session: Any,
        name: str,
        version: Any,
        eager: bool = False,
    ) -> Any:
        normalized = int(version) if isinstance(version, str) else version
        return original(self, session, name, normalized, eager=eager)

    setattr(SqlAlchemyStore, "_get_sql_model_version", get_sql_model_version)  # noqa: B010
    setattr(SqlAlchemyStore, "_regime_engine_postgres_version_cast", True)  # noqa: B010
