"""Deterministic randomized work ordering for resumable evaluation cells."""

from __future__ import annotations

import os
from hashlib import sha256


def randomized_order(values: tuple[str, ...], *, scope: str) -> tuple[str, ...]:
    """Shuffle deterministically from the active code-and-data fingerprint."""

    seed = os.environ.get("REGIME_EVALUATION_SCHEDULING_SEED", "")
    if not seed:
        return values
    return tuple(
        sorted(
            values,
            key=lambda value: sha256(f"{seed}:{scope}:{value}".encode()).hexdigest(),
        )
    )
