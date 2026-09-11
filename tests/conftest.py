from __future__ import annotations

import os

import pytest

# Each xdist worker is already a CPU-level unit.  Leaving OpenBLAS at its
# default thread count creates one native thread pool per worker and can
# multiply the machine's logical CPU count by an order of magnitude.  Keep
# native kernels single-threaded so pytest-xdist and the engine's explicit
# task pools provide the parallelism.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep external-service tests opt-in; required CI stays hermetic."""
    if os.getenv("REGIME_RUN_EXTERNAL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="external test: set REGIME_RUN_EXTERNAL_TESTS=1")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip)
