# Xetra v4 Regime Evaluation

Xetra v4 is the only supported evaluation path. It discovers the complete
feature catalog from the external feature PostgreSQL source, filters invalid
features on each outer TRAIN window, clusters the surviving features, builds a
causal teacher, selects a ranked prefix, and evaluates the exact twelve-model
candidate universe.

The candidate universe is:

- Gaussian HMM K2-K5;
- two-mixture full-covariance GMM-HMM K2-K5;
- Student-t HMM K2-K5.

All decisions are made from the current outer TRAIN window. The outer TEST
window is used once for the selected configuration. Failed folds are recorded
explicitly; no previous configuration is reused.

## Evidence and tracking

The run emits one `global_regime_v4` MLflow parent run and one child dossier per
outer fold. Immutable local statistics are written below the configured
evaluation statistics root, and the canonical evidence hash is logged to the
external MLflow service at `MLFLOW_TRACKING_URI`.

Each valid outer fold also projects its complete final candidate grid into
MLflow LoggedModels. The LoggedModels carry stable candidate/dataset/feature
lineage tags, the feature-discovery selection context, and the finite numeric
fold, seed, EM, state, transition, covariance, emission, posterior, and
information-criterion histories. The metric definitions and indexed-key rules
are centralized in `market_regime_engine.mlflow_support.metric_catalog`, so
new MLflow Model Metrics plots can query metric history instead of reparsing
PNG files or rerunning the evaluation.

The current HMM pipeline does not produce return forecasts, forecast errors, or
trading backtests. Those plots require a separate, explicitly defined target
and backtest data contract and are therefore not emitted by this evaluation.

## Cron-safe execution

From the repository checkout:

```bash
./scripts/run_xetra_v4_cron.sh
```

The wrapper loads `.env`, reads the ignored `config.yaml` feature-source
metadata, verifies external MLflow health, enforces a non-blocking lock, and
runs the complete v4 evaluation against one read-only source snapshot.

`REGIME_EVALUATION_CHECKPOINT_ROOT` (or its alias
`REGIME_ENGINE_STATE_ROOT`) must point to an absolute persistent volume outside
the checkout. The evaluation prints an immutable run key. Pass that key to
`scripts/run_xetra_v4_evaluation.py --run-key` after an interruption to replay
the durable snapshot and completed units without recapturing live PostgreSQL.
Missing or corrupt snapshot/ledger state fails closed and requires a new run
key; it is never silently rebuilt under the old identity.

## CPU parallelism

Every production evaluation and tracking entry point leaves `max_workers`
unset by default. Candidate grids, multistart fits, prefix searches, teacher
evaluation, and local artifact rendering then use the CPUs actually available
to the process: Linux affinity, the active cgroup CPU quota, and the number of
independent tasks are all respected. A deployment can set
`REGIME_CPU_WORKERS` to a deliberate benchmarked limit; an explicit
`max_workers` value remains available for a constrained run or test.

CPU-bound default evaluation work uses process pools, giving each worker an
independent interpreter and GIL. Nested numerical lanes are set to one
process-local lane, and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`MKL_NUM_THREADS` default to one in the cron wrappers to avoid native-library
oversubscription. The runtime also exposes physical-core and NUMA topology for
benchmarking, but does not pin workers to a NUMA node without measured benefit.

The non-statistical MLflow dossier/plot tail is independently bounded by
`REGIME_TRACKING_WORKERS` (default 16) because it is request/file-I/O bound;
set it explicitly when benchmarking a different MLflow service capacity. Set
`REGIME_PERFORMANCE_REPORT_PATH` to write an opt-in JSON stage report beside
the checkpoint state. Neither setting changes canonical statistical outputs.

Run the reproducible scheduler benchmark with:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_process_parallel.py
```

The benchmark reports wall time, throughput, speedup, child CPU utilization,
and peak RSS for 1/2/4/8/16/32 workers, physical cores, and logical CPUs.

The global Spearman distance stage also uses native array ranking and dot
products for its pairwise arithmetic. On the 52-feature/1,449-row fixture,
the measured distance stage fell from 11.671 s to 0.139 s (about 84x), while
retaining pairwise missing-value handling and the canonical result contract.

The repository test runner also uses all available CPUs by default through
`pytest-xdist` (`-n auto` in `pyproject.toml`). Local and CI test commands
therefore share the same parallel default; an explicit `-n` remains an
operator/test override.
