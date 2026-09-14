# MLflow Model Metrics completeness proof

`scripts/verify_mlflow_model_metrics.py` is a read-only verifier. It compares
the complete Model Metrics history of every LoggedModel with an independently
generated expectation and rejects missing, duplicate, conflicting, unknown or
out-of-domain evidence. FileStore histories are read from their persisted
metric files rather than from MLflow's summary view; remote stores use the
LoggedModel history returned by the tracking server. Model search is paged so
the audit is not silently limited to the first 1,000 models.

Run the namespace preflight before starting an evaluation. It includes active,
deleted and registry-side inventory, so a green result is the required zero-survivor record
for the historical evaluation namespace:

```bash
.venv/bin/python scripts/verify_mlflow_model_metrics.py \
  --tracking-uri "http://10.10.1.3:5000" \
  --experiment "regime-engine-evaluation" \
  --require-clean-namespace \
  --json-out mlflow-historical-namespace-proof.json
```

The preflight is read-only. Its report records the experiment ID, complete run
status/lifecycle inventory (including deleted runs), LoggedModel count, deleted
LoggedModel IDs, and the complete registered-model/version inventory. The
`historical_objects_zero` flag covers evaluation runs and LoggedModels; registry
objects are reported but are intentionally not treated as evaluation survivors.
The verifier fails closed when the tracking backend cannot enumerate deleted
LoggedModels. In particular, MLflow's HTTP `RestStore` does not expose the
backend's deleted-LoggedModel inventory helper; it reports
`deleted_logged_model_inventory_available=false` and cannot produce a clean
namespace proof until the backend provides that inventory or the audit runs
against a backend with equivalent visibility.
Do not treat an empty post-run namespace as a successful completeness proof:
the completed-evaluation invocation below uses `--require-nonempty` and an
independently generated expectation.

For metric-export resume acceptance, run the verifier after an uninterrupted
reference export and a killed-and-retried export have finished. This parity
contract applies to the dedicated durable metric-export harness, not to the
full Xetra v4 evaluator:

```bash
.venv/bin/python scripts/verify_mlflow_model_metrics.py \
  --tracking-uri "$RETRIED_MLFLOW_URI" \
  --experiment "$RETRIED_EXPERIMENT" \
  --expectation expectation.json \
  --require-nonempty \
  --baseline-tracking-uri "$REFERENCE_MLFLOW_URI" \
  --baseline-experiment "$REFERENCE_EXPERIMENT" \
  --json-out mlflow-model-metrics-proof.json
```

The retried and reference namespaces are matched by logical LoggedModel name;
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
history being audited. Build the strict, content-addressed bundle from an
independent evidence manifest before running the verifier:

```bash
.venv/bin/python scripts/build_mlflow_model_metrics_expectation.py \
  --source independent-model-metrics-evidence.json \
  --output expectation.json
```

The strict verifier checks schema version, provenance, source-artifact hash,
lineage tags and the canonical expectation hash. It also requires every
audited LoggedModel to be `READY` and sourced by a `FINISHED` run:

```bash
.venv/bin/python scripts/verify_mlflow_model_metrics.py \
  --tracking-uri "$RETRIED_MLFLOW_URI" \
  --experiment "$RETRIED_EXPERIMENT" \
  --expectation expectation.json \
  --require-expectation-contract \
  --require-terminal-model-runs \
  --require-nonempty
```

The metric-export harness may be interrupted after tracking has started and
retries from its durable export state. The full Xetra v4 evaluation is
intentionally non-resumable: if it is interrupted, rerun the complete cron
command from the beginning. It must not claim computation-position resume
parity.

The focused repository tests exercise this verifier with MLflow's local
FileStore and do not run an HMM evaluation. Production acceptance still
requires the operator to supply a clean historical namespace, execute one
complete uninterrupted full evaluation, independently capture its MLflow
evidence, and run the command against the authorized NAS MLflow service.
Those external execution records cannot be produced by the fast local test
suite.
