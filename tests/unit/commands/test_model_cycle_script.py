from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path("scripts/model_cycle.sh")


def test_model_cycle_script_is_bash_valid_and_uses_local_compose_cli_only() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert 'PROFILE="${REGIME_ENGINE_PROFILE:-xetra}"' in text
    assert "docker context show" in text
    assert "docker context inspect" in text
    assert "unix://*" in text
    assert "flock -n 9" in text
    assert 'docker compose -f "$COMPOSE_FILE"' in text
    assert "compose exec -T mlflow regime-engine" in text
    assert "compose exec -T mlflow python" in text


def test_model_cycle_script_runs_exact_changed_source_sequence_without_promotion() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    status = text.index('run_cli status --profile "$PROFILE"')
    evaluate = text.index('run_cli evaluate --profile "$PROFILE"')
    refit = text.index('run_cli final-refit --profile "$PROFILE"')
    publish = text.index('run_cli publish-oos --profile "$PROFILE"')
    register = text.index("run_cli register")
    assert status < evaluate < refit < publish < register
    assert "CURRENT_SOURCE_BUILD" in text
    assert "COMPLETED_SOURCE_BUILD" in text
    assert 'EVALUATION_SOURCE_BUILD" != "$CURRENT_SOURCE_BUILD' in text
    assert "statistical_champion_candidate_id" in text
    assert "production_package" in text
    assert "oos_build_id" in text
    assert "exact_version" in text
    assert "promote" not in text.lower()
    assert "champion alias" not in text.lower()


def test_model_cycle_script_contains_no_remote_or_build_path() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    banned = (
        "docker compose build",
        "docker compose push",
        "docker buildx",
        "docker login",
        "ssh://",
        "tcp://",
        ":5001",
        "uvicorn",
        "prometheus",
    )
    for token in banned:
        assert token not in text
