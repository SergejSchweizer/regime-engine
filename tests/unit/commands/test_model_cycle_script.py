SCRIPT = "scripts/model_cycle.sh"


def _script_text() -> str:
    with open(SCRIPT, encoding="utf-8") as handle:
        return handle.read()


def test_model_cycle_script_is_bash_valid_and_uses_external_mlflow() -> None:
    subprocess = __import__("subprocess")
    subprocess.run(["bash", "-n", SCRIPT], check=True)
    text = _script_text()
    assert "set -euo pipefail" in text
    assert 'PROFILE="${REGIME_ENGINE_PROFILE:-xetra}"' in text
    assert "flock -n 9" in text
    assert ".venv/bin/regime-engine" in text
    assert "MLFLOW_TRACKING_URI=" in text
    assert "http://10.10.1.3:5000" in text


def test_model_cycle_script_runs_exact_changed_source_sequence_without_promotion() -> None:
    text = _script_text()
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


def test_model_cycle_script_contains_no_docker_or_second_service_path() -> None:
    text = _script_text()
    banned = (
        "docker",
        "uvicorn",
        "prometheus",
    )
    for token in banned:
        assert token not in text
