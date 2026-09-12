# Model lifecycle operations

The public lifecycle profile is exactly `xetra`; the registered MLflow model is exactly `regime-xetra`. Statistical evaluation selects the winning Gaussian-HMM candidate, but candidate selection never moves the production alias automatically.

## Scheduled model cycle

The recommended cadence is exactly every 7 days. `scripts/model_cycle.sh` is the cron-safe entry point and must run from the repository checkout with the project `.venv`. It invokes the installed `regime-engine` CLI directly and uses the external MLflow service configured by `MLFLOW_TRACKING_URI` (default `http://10.10.1.3:5000`). It does not build containers, use a Docker context, or contact a second serving process.

A non-blocking `flock` keyed by profile prevents overlapping scheduled cycles. If another `xetra` cycle already owns the lock, the later invocation exits successfully as a deterministic no-op.

The cycle first reads `status`. If `current_source_build_id == completed_source_build_id`, no statistical work is run. A changed source build executes exactly:

1. `evaluate` against the status-pinned source build; the evaluation result includes the statistical champion candidate.
2. `final-refit` for that immutable evaluation ID.
3. `publish-oos` for the same evaluation ID.
4. `register` using both the final-refit production package and explicit immutable OOS build ID. The backend uploads that package to NAS MLflow and registers only its remote `runs:/...` URI.

The evaluation step is one non-resumable computation. It persists the immutable
input snapshot and writes the completed evaluation only after all folds finish;
it does not persist a partial computation-position ledger. If evaluation is
interrupted, the next cycle recomputes it from the beginning.

Registration creates/updates the `challenger` lifecycle state through the backend; it does **not** move `champion`. If the source build observed by `evaluate` differs from the build observed by `status`, the cycle fails rather than silently evaluating a different vintage.

## Promotion and rollback

Production promotion and rollback are explicit operator decisions. `ModelLifecycleOperations.promote()` and `.rollback()` mutate only the `champion` alias and use registry compare-and-swap with both an expected current version and a non-empty reason. A failed CAS returns false and must be treated as a concurrent-state conflict; callers must re-read registry state before retrying. No economic metric, uncalibrated drift score, feature-selection stability diagnostic, or scheduled cycle automatically promotes a challenger.

## Freshness contract

The lifecycle constants are shared with serving semantics:

- source warning: 4 days;
- source failure: 7 days;
- model warning: 14 days;
- model failure: 35 days;
- recommended model-cycle cadence: 7 days.

These thresholds are operational freshness evidence only. They do not replace statistical candidate selection and do not authorize an alias move.

## Cron example

Run from the repository checkout:

```cron
17 2 * * 0 cd /srv/regime-engine && ./scripts/model_cycle.sh >>/var/log/regime-engine-model-cycle.log 2>&1
```

The exact minute is an operator choice; the required cadence is weekly. Production secrets remain environment or password-file inputs and must never be placed in the cron line or script arguments.
