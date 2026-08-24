"""Deterministic operator-facing command errors."""

from __future__ import annotations


class OperatorCommandError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        if not code or code.strip() != code:
            raise ValueError("operator error code must be a non-empty trimmed string")
        if not message or message.strip() != message:
            raise ValueError("operator error message must be a non-empty trimmed string")
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")
