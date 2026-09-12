# Xetra v4 performance evidence

The uncached evaluation used the same immutable snapshot for both runs: 16,768
rows, 166 features, and 246 outer folds on the 88-logical-CPU/44-physical-core
host.

| Run | Wall time | Result |
| --- | ---: | --- |
| Baseline | 2,682.98 s | Statistics complete; tracking failed on the all-invalid selection edge case |
| Optimized | 1,954.60 s | Complete |

The optimized run reduced end-to-end wall time by 728.38 s (27.15%). The
parallel MLflow/dossier tail completed in 13.38 s with 16 bounded workers,
instead of the roughly 13-minute serial tail observed in the baseline. The
statistics stage remained the dominant cost at 1,926.14 s and was not claimed
as improved.

The baseline and optimized durable result root hashes and result hashes are
identical. The all-invalid result is represented by a deterministic summary
plot rather than failing during tracking. Process workers cap BLAS/OpenMP to
one native numerical lane, and durable ledgers enable WAL before workers start.

The full machine-readable report is
[`xetra_v4_performance.json`](xetra_v4_performance.json); the detailed runtime
report remains outside Git at `/home/dev_regime/perf-final-xetra-20260912/performance.json`.
