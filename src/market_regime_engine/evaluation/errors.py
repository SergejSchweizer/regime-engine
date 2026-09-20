"""Typed evaluation outcomes shared by serial and process execution paths."""

from __future__ import annotations


class RecoverableEvaluationInvalidity(ValueError):
    """A deterministic data/statistical gate that may become invalid evidence."""

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip() or reason != reason.strip():
            raise ValueError("recoverable invalidity reason must be a non-empty trimmed string")
        if "\n" in reason or "\r" in reason:
            raise ValueError("recoverable invalidity reason must be single-line")
        super().__init__(reason)


__all__ = ["RecoverableEvaluationInvalidity"]
