#!/usr/bin/env bash
set -euo pipefail

EXPECTED_PYTHON="3.14.7"
PYTHON_BIN="${PYTHON_BIN:-python3.14}"

actual="$(${PYTHON_BIN} -c 'import platform; print(platform.python_version())')"
if [[ "${actual}" != "${EXPECTED_PYTHON}" ]]; then
  echo "regime-engine requires Python ${EXPECTED_PYTHON}; got ${actual}" >&2
  exit 2
fi

command -v uv >/dev/null 2>&1 || {
  echo "uv is required to bootstrap the frozen environment" >&2
  exit 3
}

uv venv --python "${PYTHON_BIN}" .venv
uv pip install --python .venv/bin/python -r uv.lock
uv pip install --python .venv/bin/python --no-deps -e .

.venv/bin/python - <<'PY'
import platform
import hmmlearn
import mlflow

assert platform.python_version() == "3.14.7"
assert hmmlearn.__version__ == "0.3.3"
assert mlflow.__version__ == "3.15.1"
print("bootstrap validated")
PY
