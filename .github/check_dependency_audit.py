"""Fail closed unless every pip-audit finding has a live tracked exception."""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_EXCEPTION_KEYS = {"id", "package", "version", "expires", "reason"}


def _load_findings(path: Path) -> set[tuple[str, str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    dependencies: Any
    if isinstance(raw, dict):
        dependencies = raw.get("dependencies", [])
    else:
        dependencies = raw
    if not isinstance(dependencies, list):
        raise ValueError("pip-audit JSON must contain a dependency list")
    findings: set[tuple[str, str, str]] = set()
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise ValueError("pip-audit dependency entry must be an object")
        package = dependency.get("name")
        version = dependency.get("version")
        vulns = dependency.get("vulns", [])
        if (
            not isinstance(package, str)
            or not isinstance(version, str)
            or not isinstance(vulns, list)
        ):
            raise ValueError("pip-audit dependency entry has invalid fields")
        for vuln in vulns:
            if not isinstance(vuln, dict) or not isinstance(vuln.get("id"), str):
                raise ValueError("pip-audit vulnerability entry is invalid")
            findings.add((package.lower(), version, vuln["id"]))
    return findings


def _load_exceptions(path: Path, today: date) -> set[tuple[str, str, str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"exceptions"}:
        raise ValueError("exception file must contain only the exceptions key")
    entries = raw["exceptions"]
    if not isinstance(entries, list):
        raise ValueError("exceptions must be a list")
    allowed: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _REQUIRED_EXCEPTION_KEYS:
            raise ValueError("dependency exception fields must match the required schema exactly")
        advisory = entry["id"]
        package = entry["package"]
        version = entry["version"]
        expires = entry["expires"]
        reason = entry["reason"]
        text_values = (advisory, package, version, reason)
        if not all(isinstance(value, str) and value.strip() for value in text_values):
            raise ValueError("dependency exception text fields must be non-empty strings")
        if not isinstance(expires, str):
            raise ValueError("dependency exception expires must be an ISO date string")
        expiry = date.fromisoformat(expires)
        if expiry < today:
            raise ValueError(f"dependency exception expired: {advisory} on {expiry.isoformat()}")
        key = (package.lower(), version, advisory)
        if key in allowed:
            raise ValueError(f"duplicate dependency exception: {key}")
        allowed.add(key)
    return allowed


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_dependency_audit.py AUDIT_JSON EXCEPTIONS_YAML")
    today = datetime.now(UTC).date()
    findings = _load_findings(Path(sys.argv[1]))
    allowed = _load_exceptions(Path(sys.argv[2]), today)
    unapproved = sorted(findings - allowed)
    if unapproved:
        for package, version, advisory in unapproved:
            print(f"unapproved vulnerability: {package}=={version} {advisory}", file=sys.stderr)
        return 1
    unused = sorted(allowed - findings)
    if unused:
        for package, version, advisory in unused:
            print(
                f"stale/unused audit exception: {package}=={version} {advisory}",
                file=sys.stderr,
            )
        return 1
    print(f"dependency audit findings covered by live exceptions: {len(findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
