# Repository parallelization and acceptance audit

Audit date: 2026-09-14  
Repository: `main` at `214df55`  
Scope: read-only audit of acceptance status, CPU sizing, and CI/local test
selectors. No production evaluation or external service mutation was run.

## Verified current state

- `git status --short --branch` reports a clean `main` aligned with
  `origin/main`.
- The only local and remote branches are `main` and `origin/main`.
- GitHub reports no open pull requests and `git ls-remote --heads origin`
  contains only `main`.
- The merge and push workflows run `pytest -n auto tests -m "not integration
  and not external"`; neither workflow contains the full-computation proof.
- The local pre-commit hook runs only
  `integration and not slow and not external`, also with `-n auto`.
- `pyproject.toml` defaults pytest to `-n auto`.

## CPU findings

The live runtime reports 88 logical CPUs, an 88-CPU affinity mask, no active
integer cgroup reduction, and therefore 88 available CPUs. The deployment
`config.yaml` deliberately exports `REGIME_CPU_WORKERS=86`; that is an
operator budget leaving two logical CPUs available to the host, not an
automatic runtime shortfall. With no override, `cpu_worker_count()` uses all
available CPUs and then caps by task count.

Default CPU-bound candidate, prefix, teacher, multistart, evidence, plot, and
audit work uses process pools. Native numerical pools are capped at one lane
per process. Nested outer/inner work partitions the configured CPU budget, so
the default path avoids multiplying worker pools. Thread pools remain only for
I/O-bound MLflow calls or explicitly non-pickleable extension/test callbacks;
those fallbacks are visible in the source and are not the production default
path.

One oversubscription gap was found and fixed in this change: the deployment
export, test bootstrap, and hermetic proof wrapper now also set
`NUMEXPR_NUM_THREADS` to the configured native-thread budget. The process-pool
initializer already enforced this setting for child workers.

## Acceptance ledger findings

The authoritative ledger near the top of `BACKLOG.md` correctly leaves only
the external PR-232 current-Xetra audit and the external PR-250 MLflow
completeness/resume proof open. The current source-run evidence also records
zero valid folds and `production_eligible=false`, so it cannot be treated as
production acceptance.

The following stale or contradictory claims remain in `BACKLOG.md` and were
intentionally not edited because this audit was explicitly forbidden to edit
that file:

- The status table says its Git/PR reconciliation was checked on 2026-09-13,
  while the file status date and repository head are 2026-09-14.
- The current-state prose describes the merged follow-up range only through
  #350 or #352, while merged follow-ups through #363 are recorded later in the
  same file.
- The historical PR-231 section still has unchecked golden/rerun/label/
  likelihood items and says a remote integration gate remains, although the
  authoritative table and merged #361 record PR-231 acceptance complete.
- The historical PCA planning section still labels all eight PCA items “open”
  and “not implemented” even though the authoritative table says #303–#313
  are implemented; the section itself says its planning text is superseded.
- Historical PR-249 and metric-family sections retain unchecked acceptance
  prose even though the authoritative ledger says the code is merged and
  routes remaining external completeness evidence through PR-250.

These are ledger/documentation hygiene issues, not evidence that the
implementations or CI selectors are currently missing. A future backlog-only
reconciliation should either archive those historical checklists explicitly or
replace them with links to the authoritative status table.

## Focused verification performed

The following checks are appropriate for this audit and do not run a full
evaluation or contact external services:

```text
.venv/bin/pytest -n auto \
  tests/unit/runtime/test_cpu.py \
  tests/unit/evaluations/test_process_parallel.py \
  tests/unit/test_export_config_env.py \
  tests/unit/test_ci_contract.py
.venv/bin/ruff check scripts/export_config_env.py tests/conftest.py \
  tests/unit/test_export_config_env.py tests/unit/test_ci_contract.py
.venv/bin/ruff format --check scripts/export_config_env.py tests/conftest.py \
  tests/unit/test_export_config_env.py tests/unit/test_ci_contract.py
git diff --check
```
