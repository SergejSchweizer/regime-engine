"""Atomic checkpoints for expensive evaluation candidates."""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import TypeVar, cast
from uuid import uuid4

T = TypeVar("T")


class EvaluationCheckpointStore:
    def __init__(self, root: Path, *, fingerprint: str) -> None:
        if not fingerprint:
            raise ValueError("checkpoint fingerprint must be non-empty")
        self._root = root / fingerprint

    def load_or_compute(
        self,
        *,
        evaluation_id: str,
        feature_order: tuple[str, ...],
        candidate_id: str,
        compute: Callable[[], T],
    ) -> T:
        scope = sha256(
            json.dumps(
                {"evaluation_id": evaluation_id, "feature_order": feature_order},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        path = self._root / evaluation_id / scope / f"{candidate_id}.pickle"
        if path.is_file():
            with path.open("rb") as handle:
                result = cast(T, pickle.load(handle))
            self._log("checkpoint_hit", evaluation_id, candidate_id)
            return result
        result = compute()
        path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self._log("checkpoint_written", evaluation_id, candidate_id)
        return result

    @staticmethod
    def _log(event: str, evaluation_id: str, candidate_id: str) -> None:
        print(
            json.dumps(
                {"candidate_id": candidate_id, "checkpoint": event, "evaluation_id": evaluation_id},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
