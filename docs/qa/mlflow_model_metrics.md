# MLflow Model Metrics completeness proof

`scripts/verify_mlflow_model_metrics.py` is a read-only verifier. It compares
the complete Model Metrics history of every LoggedModel with an independently
generated expectation and rejects missing, duplicate, conflicting, unknown or
out-of-domain evidence. FileStore histories are read from their persisted
metric files rather than from MLflow's summary view; remote stores use the
LoggedModel history returned by the tracking server. Model search is paged so
the audit is not silently limited to the first 1,000 models.

Run the namespace preflight before starting an evaluation. It includes active
and deleted MLflow runs, so a green result is the required zero-survivor record
for the historical evaluation namespace:

```bash
.venv/bin/python scripts/verify_mlflow_model_metrics.py \
  --tracking-uri "http://10.10.1.3:5000" \
  --experiment "regime-engine-evaluation" \
  --require-clean-namespace \
  --json-out mlflow-historical-namespace-proof.json
```

The preflight is read-only. Its report records the experiment ID, complete run
status/lifecycle inventory (including deleted runs), LoggedModel count, and
`historical_objects_zero` flag.
Do not treat an empty post-run namespace as a successful completeness proof:
the completed-evaluation invocation below uses `--require-nonempty` and an
independently generated expectation.

For resume acceptance, run the verifier after both an uninterrupted reference
evaluation and a killed-and-resumed evaluation have finished:

```bash
.venv/bin/python scripts/verify_mlflow_model_metrics.py \
  --tracking-uri "$RESUMED_MLFLOW_URI" \
  --experiment "$RESUMED_EXPERIMENT" \
  --expectation expectation.json \
  --require-nonempty \
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

The expectation must be generated independently from the completed evaluation
plan and pinned snapshot. It must enumerate every logical LoggedModel and every
finite catalogued point; it must not be derived by rereading the same MLflow
history being audited. The resumed run must be interrupted after tracking has
started, restarted from its durable evaluation state, and compared only after
both the resumed and uninterrupted namespaces are terminal.

The focused repository tests exercise this verifier with MLflow's local
FileStore and do not run an HMM evaluation. Production acceptance still
requires the operator to supply a clean historical namespace, independently
capture the uninterrupted and interrupted/resumed external MLflow runs, and
run the command against the authorized NAS MLflow service. Those external
execution records cannot be produced by the fast local test suite.
