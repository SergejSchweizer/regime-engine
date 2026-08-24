from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep external-service tests opt-in; required CI stays hermetic."""
    if os.getenv("REGIME_RUN_EXTERNAL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="external test: set REGIME_RUN_EXTERNAL_TESTS=1")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip)
