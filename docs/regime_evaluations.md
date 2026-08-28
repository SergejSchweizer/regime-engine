# Xetra v3 Regime Evaluations

Xetra v3 adds a canonical 61-feature policy, `xetra_semantic_medoid_v3`, without
changing the historical v1/v2 48-feature universe or its identities. The policy
uses eight economic semantic blocks. Stage 1 selects one preliminary medoid per
block from all 61 features using first-fold TRAIN data only. Stage 2 applies the
fixed cross-block pruning rule and freezes the ordered multivariate tuple before
any model evaluation.

## Immediate-change features

The 13 added PostgreSQL columns, in their canonical evaluation order, are:

```text
vix_delta_1obs
vix9d_delta_1obs
vix3m_delta_1obs
vix6m_delta_1obs
vix1y_delta_1obs
vstoxx_delta_1obs
move_delta_1obs
ciss_delta_1obs
euro_hy_oas_delta_1obs
us_2y_delta_1obs
us_10y_delta_1obs
estr_delta_1obs
usd_broad_delta_1obs
```

They belong respectively to US equity volatility spot; US equity volatility term
structure; Europe equity volatility; rates volatility; systemic stress; credit
stress; rates/yield curve; and USD FX. A delta feature may be selected as a
Stage-1 medoid.

## Three independent evaluations

`medoid_multivariate` evaluates the frozen Stage-2 tuple with the twelve v3
candidates and selects `medoid_multivariate_statistical_champion`. This is the
only production-eligible champion.

`medoid_univariate` independently evaluates each of the eight Stage-1 medoids,
twelve candidates per feature. Its diagnostic champion is
`medoid_univariate_evaluation_champion`.

`delta1_univariate` independently evaluates each feature in the ordered
13-delta tuple, also with twelve candidates per feature. Its diagnostic champion
is `delta1_univariate_evaluation_champion`.

The candidate universe is Gaussian K2-K5, two-mixture GMM-HMM K2-K5, and
Student-t K2-K5. The exact counts are 12 multivariate, 96 medoid-univariate, and
156 delta-univariate candidates: 264 attempts before invalid-fold rejection.

The medoid and delta univariate evaluations each create a complete-case clock
only from their own feature tuple. Their hashes and retained observations can
differ. A feature appearing in both namespaces is fitted again in each one; no
model, clock, MLflow run, or local evidence is shared across evaluations.

Within a feature, candidate selection uses the canonical statistical ranking.
Across univariate features, ranking uses only label-invariant dominant-state NMI
on shared valid OOS support, then shared timestamp count, then feature name.
Cross-feature PLL, BIC, AIC, confidence, entropy, and economic metrics are
forbidden. Both univariate champions are diagnostic-only: they never cause final
refit, OOS publication, registration, or alias mutation.

## Evidence and tracking

Every parent, feature, candidate, and failed MLflow run owns one immutable local
directory:

```text
./evaluations/<evaluation>/<mlflow_run_id>/
```

Each directory contains `statistics.json` and `statistics.md`. The JSON is
UTF-8, deterministically ordered, and contains finite numbers only. Its exact
finalized bytes are uploaded as `statistics/statistics.json`; the same SHA-256
is retained in local and MLflow metadata. A mismatch fails the run.

Statistics evidence is organized into `identity`, `lineage`, `input`, `model`,
`folds`, `states`, `aggregate`, `feature_selection`, `agreement`, `champion`,
and, for failures, `failure`. It records run identity/status/timing, source and
evaluation lineage, ordered inputs and missingness, model and multistart
configuration, planned/invalid folds, state and fitted-parameter diagnostics,
aggregate/ranking evidence, feature selection, agreement, and champion outcome.
Credentials, DSNs, raw source rows, and model-binary payloads are never written.

## Standalone execution

Run `scripts/run_xetra_v3_evaluations.py` from a checkout with the Xetra v3
profile and feature policy available, a readable feature PostgreSQL source, a
writable evaluation root, and `MLFLOW_TRACKING_URI` configured. The runner reads
one source snapshot, freezes selection from the first TRAIN fold, builds three
independent clocks, tracks all three hierarchies, and prints their identities and
counts. It does not perform production refit, publish OOS predictions, register
a model, or mutate aliases.