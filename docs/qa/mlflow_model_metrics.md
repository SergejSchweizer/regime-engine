# MLflow Model Metrics completeness proof

`scripts/verify_mlflow_model_metrics.py` is a read-only verifier. It compares
the complete Model Metrics history of every LoggedModel with an independently
generated expectation and rejects missing, duplicate, conflicting, unknown or
out-of-domain evidence. FileStore histories are read from their persisted
metric files rather than from MLflow's summary view; remote stores use the
LoggedModel history returned by the tracking server. Model search is paged so
the audit is not silently limited to the first 1,000 models.

For resume acceptance, run the verifier after both an uninterrupted reference
evaluation and a killed-and-resumed evaluation have finished:

```bash
.venv/bin/python scripts/verify_mlflow_model_metrics.py \
  --tracking-uri "$RESUMED_MLFLOW_URI" \
  --experiment "$RESUMED_EXPERIMENT" \
  --expectation expectation.json \
  --baseline-tracking-uri "$REFERENCE_MLFLOW_URI" \
  --baseline-experiment "$REFERENCE_EXPERIMENT" \
  --json-out mlflow-model-metrics-proof.json
```

The resumed and reference namespaces are matched by logical LoggedModel name;
operational MLflow model/run IDs are intentionally ignored. The parity proof
requires identical metric key, step, exact floating-point value and timestamp,
as well as identical `regime_engine.*` lineage tags. A successful report has
`status: "verified"`, zero counts in `counts`, and a `resume_parity` object
whose missing, unexpected, duplicate, conflicting and lineage-mismatch counts
are all zero. The `models` and `model_count_by_scope` fields are the durable
per-LoggedModel evidence record.

The focused repository tests exercise this verifier with MLflow's local
FileStore and do not run an HMM evaluation. Production acceptance still
requires the operator to supply a clean historical namespace, independently
capture the uninterrupted and interrupted/resumed external MLflow runs, and
run the command against the authorized NAS MLflow service. Those external
execution records cannot be produced by the fast local test suite.
