# Regime Engine Evaluation Contract

Status date: 2026-08-23

This document is authoritative for the statistical feature-selection, HMM fitting, walk-forward evaluation, state-alignment, candidate-ranking and final-production-refit semantics of `regime-engine`.

Consumer portfolio/economic metrics are outside this contract.

## 1. Identity

```text
profile_id=xetra
profile_config_version=1
registered_model=regime-xetra
production_alias=champion
feature_selection_policy=xetra_semantic_medoid_v1
```

The phrase *statistical champion* denotes the winning candidate family/K after evaluation. `champion` denotes the MLflow serving alias after a separate mandatory final production refit. `engine-champion` is not an alias.

## 2. Scientific claim boundary

Input time semantics are:

```text
data_time_semantics=current_vintage_observation_day
```

Evaluation is split-leak-free and causal relative to the current-vintage observation sequence. It is not claimed to be historical provider-release-time/vintage-safe or fully point-in-time tradable because upstream `timestamp_m1` is observation-day identity rather than release/availability time.

## 3. Model observation sequence

Upstream SQL NULLs are allowed. No fill/interpolation/carry is permitted.

After the final feature set is frozen, an HMM observation exists only at a timestamp where every selected feature is non-null and finite. Incomplete timestamps are excluded and retained as gap evidence.

One HMM transition occurs per consecutive retained observation. Calendar gaps do not apply extra powers of the transition matrix.

This identical observation clock is used for every fold, final refit, latest and replay.

## 4. Feature selection

Source universe: exactly 48 `regime-loader` feature-version-1 feature columns in the eight semantic blocks defined in `BACKLOG.md` and `configs/feature_selection/xetra_semantic_medoid_v1.yaml`.

Pinned policy:

```text
within_block_method=absolute_spearman_medoid
cross_block_method=absolute_spearman_prune
minimum_feature_coverage=0.90
minimum_nonzero_variance=1e-12
minimum_block_complete_observations=504
maximum_cross_block_abs_spearman=0.85
numeric_tie_abs_tolerance=1e-12
```

Selection uses only first-fold TRAIN source rows.

### Spearman definition

For every relevant complete-case matrix:

1. rank each feature column using average ranks for tied values;
2. compute ordinary Pearson correlation among rank columns;
3. all required correlations must be finite.

Greek symbol:

- **rho (ρ)** — pronounced *ROH*; Spearman rank correlation.

### Stage 1

For each block:

- coverage = non-null count / first-fold TRAIN source-row count;
- eligible iff coverage >=0.90, all non-null values finite, and population variance (`ddof=0`) > `1e-12`;
- block complete cases use all eligible candidates;
- require >=504 complete rows;
- distance `d(i,j)=1-abs(ρ(i,j))`;
- medoid score = arithmetic mean distance to the other eligible candidates; singleton score=0;
- rank lower score, higher coverage, earlier configured position; differences <=`1e-12` are ties;
- exactly one winner per block, yielding exactly eight preliminary medoids.

### Stage 2

- form one fixed complete-case 8x8 Spearman matrix over preliminary medoids;
- require >=504 rows;
- conflict iff `abs(ρ)>0.85`; exactly 0.85 is allowed;
- process highest absolute correlation first, then canonical pair order on ties;
- remove higher Stage-1 medoid score, then lower coverage, then later block;
- never recompute correlations;
- never search for replacement features;
- survivors stay in canonical block order;
- legal final dimension is `1<=d<=8`.

The cross-block comparison of Stage-1 medoid scores from blocks of different size is an intentional policy-v1 simplification and is not silently normalized.

No fitted-HMM metric, ETF return, portfolio statistic or trading target may affect feature selection.

### Selection hashes

`feature_selection_definition_hash` covers only first-fold-TRAIN-determined policy/evidence/final features. It deliberately excludes full-build identity and later rows.

`feature_selection_execution_hash` covers:

```text
feature_selection_definition_hash
source_build_id
data_sha256
evaluation_plan_hash
```

Changing rows strictly after first-fold `train_end` may change execution/source lineage but must not change the definition hash or selection evidence.

### Non-decision diagnostics

The frozen production/evaluation feature set is never changed after first-fold selection. Two diagnostics are permitted and must be labelled non-decision evidence:

1. first-fold Stage-2 threshold sensitivity at `0.80`, `0.85`, and `0.90`; canonical policy remains exactly `0.85`;
2. shadow reruns of the same Stage-1/Stage-2 selector on later fold TRAIN samples to measure selected-feature overlap/stability versus the frozen set.

These diagnostics may not change any fold input, definition hash, champion ranking, or registered model. They exist only to expose feature-selection fragility.

## 5. Walk-forward plan

```text
minimum_train_source_observations=1260
test_source_observations=63
step_source_observations=63
allow_partial_final_test=false
minimum_model_train_observations=504
minimum_model_test_observations=42
ranking_abs_tolerance=1e-12
```

Windows expand. TEST starts strictly after TRAIN. No synthetic dates. Every fold has stable one-based `fold_index`, deterministic `fold_id`, and UTC train/test bounds.

The evaluation cutoff is exactly the `test_end` of the final planned complete fold. No source observation after that timestamp may participate in walk-forward scoring or the subsequent final production refit for that evaluation run.

Source-row windowing occurs first; resolved-feature complete-case filtering then determines usable HMM observation counts.

## 6. Gaussian HMM candidates and pinned fitting policy

Candidates:

| Candidate | K | Covariance |
|---|---:|---|
| `gaussian_hmm_k2_full` | 2 | full |
| `gaussian_hmm_k3_full` | 3 | full |
| `gaussian_hmm_k4_full` | 4 | full |

Candidate identity is profile-versioned and exact. The public candidate universes are:

```text
v1 = gaussian_hmm_k2_full, gaussian_hmm_k3_full, gaussian_hmm_k4_full
v2 = gaussian_hmm_k2_full through gaussian_hmm_k5_full,
     gmm_hmm_k2_m2_full through gmm_hmm_k5_m2_full,
     student_t_hmm_k2_full through student_t_hmm_k5_full
```

No candidate universe is inferred from its length. A missing, reordered or unexpected candidate fails closed.

Backend/configuration:

```text
backend=hmmlearn==0.3.3
covariance_type=full
implementation=log
seeds=[11,23,37,53,71,89,107,131]
minimum_valid_starts=6
minimum_multistart_success_rate=0.75
n_iter=1000
tol=1e-4
min_covar=1e-6
startprob_prior=1.0
transmat_prior=1.0
means_prior=0.0
means_weight=0.0
covars_prior=0.01
covars_weight=1.0
params=stmc
init_params=stmc
```

Reduced covariance modes `diag`, `spherical`, `tied`, or any other non-`full` mode are unsupported and fail closed.

Each start records seed, convergence, iterations, TRAIN log likelihood, numerical validity and failure reason. The valid converged start with greatest TRAIN log likelihood wins that fold; numeric values within absolute tolerance `1e-12` are tied and lower seed wins. Fewer than 6 valid starts or success rate <0.75 invalidates the fold.

## 7. Causal forward filter

Greek symbol:

- **alpha (α)** — pronounced *AL-fa*; filtered state probability.

For the first retained observation `x_0`:

```text
u_0(k) = pi(k) * b_0(k)
c_0 = sum_k u_0(k)
alpha_0(k) = u_0(k) / c_0
loglik = ln(c_0)
```

For each later retained observation:

```text
prior_t = alpha_(t-1) @ A
u_t(k) = prior_t(k) * b_t(k)
c_t = sum_k u_t(k)
alpha_t(k) = u_t(k) / c_t
loglik += ln(c_t)
```

Implementation must be numerically stabilized/log-domain equivalent. Filtered values at time t depend only on retained observations through t.

Viterbi/smoothing are retrospective diagnostics only and cannot be used for OOS/production probabilities.

## 8. OOS predictive likelihood — TRAIN continuation is mandatory

A fold TEST sequence does not restart from `pi`.

1. filter retained TRAIN observations;
2. keep `alpha_train_end`;
3. first retained TEST prior is `alpha_train_end @ A`;
4. process TEST sequentially;
5. sum only TEST `ln(c_t)` terms;
6. divide by retained TEST observation count for fold per-observation OOS PLL.

A backend `score(X_test)` call that independently initializes TEST from `startprob_` is not this metric and may not be used.

Candidate aggregates:

```text
oos_predictive_loglik_mean = arithmetic mean of valid-fold per-observation values
oos_predictive_loglik_std = population std, ddof=0
oos_predictive_loglik_worst_fold = minimum valid-fold value
oos_predictive_loglik_best_fold = maximum valid-fold value
```

A separately named pooled observation-weighted diagnostic is permitted but is not a ranking substitute.

## 9. State signatures and persistent alignment

Persistent IDs are `state_0 ... state_(K-1)`.

For a fitted state k in standardized feature space:

```text
signature_k = concat(
    mean vector in exact feature order,
    log(sqrt(diag(full covariance))),
    upper off-diagonal triangle of the covariance-derived correlation matrix
)
```

All components finite.

Distance:

```text
RMS(s1,s2)=sqrt(mean((s1-s2)^2))
```

First valid fold:

- construct `signature_sort_key = tuple(round(component, 10) for component in signature)`;
- sort keys lexicographically ascending and assign `state_0...`;
- if any two rounded sort keys are identical, initial alignment is ambiguous and the fold is invalid.

Later folds:

- reference is previous valid fold's persistent signatures for same K;
- enumerate all K! one-to-one mappings (`K<=4`);
- total cost is sum of matched RMS distances;
- choose unique minimum;
- if best and second-best total costs differ by <=`1e-10`, mapping is ambiguous and fold invalid;
- record matched per-state and maximum drift.

Drift is diagnostic in profile v1. There is no maximum-drift hard threshold and no agent may invent one.

Final production refit aligns to the last valid evaluation fold of the winning K by the same rule.

## 10. Numerical/covariance validity

For every full covariance state matrix:

- shape `d x d`;
- finite;
- maximum absolute asymmetry <=`1e-10`;
- after that check only, `(S+S.T)/2` may be used for validation;
- minimum diagonal variance >=`1e-12`;
- Cholesky succeeds without unrecorded jitter.

Initial probabilities and every transition row must be finite, nonnegative and normalized within absolute tolerance `1e-10`; values outside that tolerance fail rather than being silently renormalized.

For a Gaussian HMM with K states and d features:

```text
p=(K-1)+K(K-1)+Kd+K*d*(d+1)/2
AIC=2p-2*TRAIN_loglik
BIC=p*ln(n_train)-2*TRAIN_loglik
```

## 11. Occupancy, persistence and uncertainty

TRAIN hard occupancy = fraction of retained TRAIN observations whose largest filtered probability is that state.

TRAIN soft occupancy for state k:

```text
mean_t alpha_t(k)
```

Fold hard gates:

```text
minimum_train_hard_occupancy=0.03
minimum_train_soft_occupancy=0.05
```

OOS occupancy is diagnostic only.

Dominant-state durations are counts of consecutive retained model observations, not calendar days.

`switches_per_year` uses actual UTC timestamp span:

```text
switch_count / elapsed_calendar_days * 365.2425
```

Undefined for zero elapsed span; never fabricated.

Confidence:

```text
confidence_t=max_k alpha_t(k)
```

Entropy uses natural logarithm:

```text
H_t=-sum_k alpha_t(k)*ln(alpha_t(k))
```

Low-confidence diagnostic threshold is exactly 0.60.

## 12. Fold and candidate hard gates

A fold is valid only if:

- retained TRAIN observations >=504;
- retained TEST observations >=42;
- >=6 of 8 starts valid/converged;
- multistart success rate >=0.75;
- all parameters finite/valid;
- full covariance checks pass;
- every TRAIN hard occupancy >=0.03;
- every TRAIN soft occupancy >=0.05;
- state alignment succeeds uniquely.

Candidate valid-fold rate must be >=0.80.

Invalid folds stay in evidence with failure reasons and missing unavailable metric values; they are not interpolated and do not enter valid-fold means.

## 13. Statistical champion ranking

After hard gates, deterministic order is:

1. highest `oos_predictive_loglik_mean`;
2. lower `oos_predictive_loglik_std`;
3. higher `oos_predictive_loglik_worst_fold`;
4. lower `bic_mean`;
5. lower `aic_mean`;
6. fewer states K;
7. lexicographically earlier canonical candidate ID.

For every numeric ranking stage, candidates whose values differ by <=`1e-12` are tied and comparison proceeds to the next stage. K and candidate ID are exact tie-breaks.

There is no weighted score. TRAIN likelihood alone cannot select the champion. Consumer economics never enter this ranking.

## 14. Mandatory final production refit

No walk-forward fold model is registered as production.

After statistical champion K is selected:

1. keep frozen selected features unchanged;
2. take all source rows from the same evaluation source snapshot through the exact evaluation cutoff defined in section 5;
3. apply identical resolved-feature complete-case observation mask;
4. require >=504 usable observations;
5. fit a fresh scaler over the full refit sample;
6. run the exact eight-seed multistart fit for winning K;
7. reapply numerical/covariance/multistart/TRAIN-occupancy gates;
8. align to the last valid evaluation fold for the winning K;
9. filter the entire final-refit sequence causally;
10. persist the final temporal/filter state.

Required production artifact fields:

```text
inference_origin_timestamp
trained_through_timestamp
terminal_filtered_probabilities
```

`trained_through_timestamp` is the final retained complete model observation at or before the evaluation cutoff; it need not equal cutoff if the cutoff source row is incomplete across final features.

Final refit never retroactively changes the OOS evaluation/ranking.

Only this final-refit artifact may be registered as a version of `regime-xetra`.

## 15. Latest and fixed-model replay initialization

For inference entirely after `trained_through_timestamp`:

- initialize continuation from stored terminal filtered probabilities;
- process every subsequent retained observation through requested end;
- return only requested interval/timestamp.

If replay includes a timestamp at or before `trained_through_timestamp`:

- filter from stored `inference_origin_timestamp` using model initial probabilities through requested end;
- return only predictions inside the inclusive interval `[start,end]`.

The client's arbitrary replay start never becomes a new HMM initial condition.

For same exact model version and same source build, overlapping returned replay timestamps must have identical probabilities even when requested replay starts differ.

Replay mode is exactly `fixed_model_replay` and uses the current serving-source vintage. It is never `walk_forward_oos`.

## 16. Required MLflow evaluation evidence

Parent evaluation run records profile/config/hash, source lineage, time semantics, split plan, both selection hashes, candidate count and statistical selection result.

Candidate runs record family/K/full covariance, feature order/hash, multistart settings, aggregate scorecard and candidate-run `fold_*` metric histories.

Every planned fold is represented in `fold_timeline.parquet` and `fold_metrics.parquet`.

Canonical fold-history keys include at least:

```text
fold_train_loglik
fold_oos_predictive_loglik
fold_oos_predictive_loglik_per_obs
fold_aic
fold_bic
fold_multistart_success_rate
fold_min_train_hard_occupancy
fold_min_train_soft_occupancy
fold_max_state_signature_drift
fold_mean_state_duration
fold_switches_per_year
fold_oos_entropy_mean
fold_oos_confidence_mean
```

Per-state histories use persistent state IDs.

Trend x-axis is actual TEST-end UTC. Invalid/missing folds appear as gaps/explicit invalid markers, never interpolation.

Per valid fold, retain machine-readable transition matrix and full covariance matrices plus transition/covariance heatmaps preserving persistent states, exact feature order and off-diagonals.

Required parent cross-candidate plots and all feature-selection visual-audit plots follow `PLOT_STYLE.md`.

Feature-selection non-decision diagnostics from section 4 are stored under `feature_selection/diagnostics/` and clearly labelled as diagnostics; they must never be used by the selector or champion code path.

`plots/manifest.json` deterministically maps plots to exact source artifacts/metrics/hashes.

## 17. Separation from downstream economics

The engine selects statistically robust regime models only.

Portfell or future consumers may evaluate immutable `walk_forward_oos` predictions using asset returns/portfolio metrics, but this cannot change engine feature selection or statistical champion rules.

Because source timestamps lack historical release/vintage semantics, downstream research must preserve the same current-vintage limitation unless/until the upstream source contract is versioned to provide availability/vintage information.
