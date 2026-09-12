# MLflow evaluation-namespace reset

This procedure applies only to the historical evaluation namespace at the
pinned NAS MLflow service. It never deletes the registered `regime-xetra`
production model or its aliases.

The cleanup command is a dry run unless `--execute` is supplied. Both modes
write a deterministic JSON manifest outside MLflow, by default at
`docs/qa/mlflow_cleanup_report.json`.

## Inspect the deletion manifest

Run this on the deployment host from the repository root:

```bash
.venv/bin/python scripts/mlflow_cleanup.py \
  --scope evaluation-runs \
  --tracking-uri http://10.10.1.3:5000 \
  --report docs/qa/mlflow_cleanup_report.json
```

Review the listed experiment, run IDs and LoggedModel IDs. No objects are
deleted by this invocation.

## Explicit destructive execution

Only execute after reviewing the dry-run manifest and obtaining the required
operator approval:

```bash
.venv/bin/python scripts/mlflow_cleanup.py \
  --scope evaluation-runs \
  --tracking-uri http://10.10.1.3:5000 \
  --confirm-tracking-uri http://10.10.1.3:5000 \
  --execute \
  --report docs/qa/mlflow_cleanup_report.json
```

The command deletes LoggedModels before runs, treats already-absent objects as
no-ops, re-queries the target namespace, and exits unsuccessfully if any
targeted run or LoggedModel survives. Repeating the exact command is safe and
produces a zero-change verified report.

The command rejects every tracking URI other than
`http://10.10.1.3:5000`; it cannot be used to target an unrelated experiment,
tracking service, registered model, or production alias.

## Backend garbage collection

MLflow run deletion is a soft delete. After the report has status `verified`,
an operator with access to the NAS MLflow backend store must run the
backend-supported permanent cleanup/garbage-collection procedure for that
deployment. Record its command, exit code, timestamp and resulting zero
survivor proof alongside the JSON report. Do not point a local command at a
different backend store and do not run this procedure in Push Gate or Merge
Gate; evaluation cleanup is an explicit operational action.
