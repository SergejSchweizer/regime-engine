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

Greek symbol: $\rho$ (Spearman rank correlation).

### Stage 1

For each block:

- $c = n_{\mathrm{non\text{-}null}} / n_{\mathrm{TRAIN}}$;
- eligible iff $c \ge 0.90$, all non-null values are finite, and population variance $\sigma^2$ (`ddof=0`) is $> 10^{-12}$;
- block complete cases use all eligible candidates;
- require $n \ge 504$ complete rows;
- distance $d(i,j) = 1 - |\rho(i,j)|$;
- medoid score $\overline{d}_i$ is the arithmetic mean distance to the other eligible candidates; for a singleton, $\overline{d}_i = 0$;
- rank lower $\overline{d}_i$, higher $c$, then earlier configured position; differences $\le 10^{-12}$ are ties;
- exactly one winner per block, yielding exactly eight preliminary medoids.

### Stage 2

- form one fixed complete-case $8 \times 8$ Spearman matrix over preliminary medoids;
- require $n \ge 504$ rows;
- conflict iff $|\rho| > 0.85$; exactly $0.85$ is allowed;
- process highest absolute correlation first, then canonical pair order on ties;
- remove higher Stage-1 medoid score, then lower coverage, then later block;
- never recompute correlations;
- never search for replacement features;
- survivors stay in canonical block order;
- legal final dimension is $1 \le d \le 8$.

The cross-block comparison of Stage-1 medoid scores from blocks of different size is an intentional policy-v1 simplification and is not silently normalized.

No fitted-HMM metric, ETF return, portfolio statistic or trading target may affect feature selection.

### Anchored numeric-tolerance semantics

The absolute numeric tolerance is exactly $\varepsilon = 10^{-12}$. It defines an anchored equivalence
set, never a pairwise comparator relation: pairwise tolerance chaining is forbidden.
For a maximize stage, compute the exact maximum of the current candidate set as the
anchor $a$ and retain every value $x \ge a - \varepsilon$. For a minimize stage, compute the
exact minimum as the anchor $a$ and retain every value $x \le a + \varepsilon$.

Every secondary feature-selection stage is evaluated only inside the anchored tied set
from the preceding stage. For example, Stage 1 first anchors the global minimum medoid
score, then anchors the global maximum coverage among only those tied features, then
uses configured position as the exact final tie-breaker.

The adversarial chain $a = 0$, $b = 0.75\varepsilon$, $c = 1.5\varepsilon$ demonstrates why pairwise
comparison is invalid: $a$ and $b$ may share an anchor tie, and $b$ and $c$ may share
another, but $a$ and $c$ must not be joined transitively.

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

$$
n_{\mathrm{TRAIN,source}} \ge 1260, \qquad
n_{\mathrm{TEST,source}} = 63, \qquad
\Delta n_{\mathrm{source}} = 63, \qquad
n_{\mathrm{TRAIN,model}} \ge 504, \qquad
n_{\mathrm{TEST,model}} \ge 42, \qquad
\varepsilon_{\mathrm{rank}} = 10^{-12}.
$$

Partial final TEST folds are forbidden.

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

Each start records seed, convergence, iterations, TRAIN log likelihood, numerical validity and failure reason. All valid converged starts are collected before winner selection. The exact global maximum TRAIN log likelihood is the anchor; starts within $\varepsilon = 10^{-12}$ of it form the tied set and the lowest seed in that set wins. Fewer than $6$ valid starts or success rate $< 0.75$ invalidates the fold.

## 7. Causal forward filter

Greek symbol: $\alpha$ (filtered state probability).

For the first retained observation $x_0$:

$$
u_0(k) = \pi(k) b_0(k), \qquad
c_0 = \sum_k u_0(k), \qquad
\alpha_0(k) = \frac{u_0(k)}{c_0}, \qquad
\log \mathcal{L} = \ln(c_0).
$$

For each later retained observation:

$$
\boldsymbol{\pi}_t = \boldsymbol{\alpha}_{t-1} A, \qquad
u_t(k) = \pi_t(k) b_t(k), \qquad
c_t = \sum_k u_t(k), \qquad
\alpha_t(k) = \frac{u_t(k)}{c_t}, \qquad
\log \mathcal{L} \mathrel{+}= \ln(c_t).
$$

Implementation must be numerically stabilized/log-domain equivalent. Filtered values at time t depend only on retained observations through t.

Viterbi/smoothing are retrospective diagnostics only and cannot be used for OOS/production probabilities.

## 8. OOS predictive likelihood — TRAIN continuation is mandatory

A fold TEST sequence does not restart from $\boldsymbol{\pi}$.

1. filter retained TRAIN observations;
2. keep $\boldsymbol{\alpha}_{\mathrm{TRAIN,end}}$;
3. first retained TEST prior is $\boldsymbol{\alpha}_{\mathrm{TRAIN,end}} A$;
4. process TEST sequentially;
5. sum only TEST $\ln(c_t)$ terms;
6. divide by retained TEST observation count for fold per-observation OOS PLL.

A backend `score(X_test)` call that independently initializes TEST from `startprob_` is not this metric and may not be used.

Candidate aggregates:

$$
\bar{\ell}_{\mathrm{OOS}} = \operatorname{mean}(\ell_f), \qquad
\sigma_{\mathrm{OOS}} = \operatorname{std}_{\mathrm{population}}(\ell_f), \qquad
\ell_{\mathrm{worst}} = \min_f \ell_f, \qquad
\ell_{\mathrm{best}} = \max_f \ell_f.
$$

A separately named pooled observation-weighted diagnostic is permitted but is not a ranking substitute.

After each candidate's independent valid-fold-rate gate passes, statistical ranking
uses only the intersection of valid fold IDs across all accepted candidates. The
common-valid-fold rate is the intersection count divided by planned fold count and
must be at least `0.80`, otherwise selection fails closed. OOS mean, population
standard deviation, worst fold, BIC mean, and AIC mean are recomputed from exactly
that common support. Per-candidate valid-fold aggregates remain diagnostics and
cannot improve a candidate's statistical rank by omitting difficult folds.

## 9. State signatures and persistent alignment

Persistent IDs are `state_0 ... state_(K-1)`.

For a fitted state k in the fixed alignment coordinate system:

$$
\mathbf{s}_k = \operatorname{concat}\!\left(
\boldsymbol{\mu}_k,
\log\!\sqrt{\operatorname{diag}(\Sigma_k)},
\operatorname{upper}\!\left(\operatorname{corr}(\Sigma_k)\right)
\right).
$$

All components finite.

The fixed alignment coordinate system is the scaler fitted only on retained TRAIN
observations of the first planned fold for the frozen final feature set. It is
evaluation evidence only: each fold still fits and filters with its own TRAIN-only
scaler. Direct RMS comparison of parameters standardized by different fold scalers
is forbidden. For a fold-local mean $\mu_f$, covariance $\Sigma_f$, scaler mean
$m_f$, scale $s_f$, and fixed reference scaler mean $m_r$, scale $s_r$:

$$
\boldsymbol{\mu}_r = \frac{\mathbf{m}_f + \mathbf{s}_f \odot \boldsymbol{\mu}_f - \mathbf{m}_r}{\mathbf{s}_r},
\qquad D = \operatorname{diag}\!\left(\frac{\mathbf{s}_f}{\mathbf{s}_r}\right),
\qquad \Sigma_r = D \Sigma_f D.
$$

No diagonal-only covariance approximation is permitted.

Distance:

$$
\operatorname{RMS}(\mathbf{s}_1, \mathbf{s}_2) = \sqrt{\operatorname{mean}\!\left((\mathbf{s}_1 - \mathbf{s}_2)^2\right)}.
$$

First valid fold:

- construct `signature_sort_key = tuple(round(component, 10) for component in signature)`;
- sort keys lexicographically ascending and assign `state_0...`;
- if any two rounded sort keys are identical, initial alignment is ambiguous and the fold is invalid.

Later folds:

- reference is previous valid fold's persistent signatures for same K;
- enumerate all $K!$ one-to-one mappings for $K \le 4$;
- total cost is sum of matched RMS distances;
- choose unique minimum;
- if best and second-best total costs differ by $\le 10^{-10}$, mapping is ambiguous and the fold is invalid;
- record matched per-state and maximum drift.

Drift is diagnostic in profile v1. There is no maximum-drift hard threshold and no agent may invent one.

Final production refit aligns to the last valid evaluation fold of the winning K by the same rule.

## 10. Numerical/covariance validity

For every full covariance state matrix:

- shape $d \times d$;
- finite;
- maximum absolute asymmetry $\le 10^{-10}$;
- after that check only, $(\Sigma + \Sigma^\mathsf{T}) / 2$ may be used for validation;
- minimum diagonal variance $\ge 10^{-12}$;
- Cholesky succeeds without unrecorded jitter.

Initial probabilities and every transition row must be finite, nonnegative and normalized within absolute tolerance $\varepsilon = 10^{-10}$; values outside that tolerance fail rather than being silently renormalized.

For a Gaussian HMM with K states and d features:

Before AIC/BIC use, the winning-start TRAIN likelihood is recomputed by the
causal forward/emission implementation. The fit-returned and causal values must
satisfy $|\mathcal{L}_{\mathrm{fit}} - \mathcal{L}_{\mathrm{filter}}| \le 10^{-10} \max(1, |\mathcal{L}_{\mathrm{fit}}|, |\mathcal{L}_{\mathrm{filter}}|)$.
A parity failure invalidates that fold; after a passing check, $\mathcal{L}_{\mathrm{filter}}$ is
the canonical stored TRAIN likelihood and the value used for AIC/BIC.

$$
p = (K - 1) + K(K - 1) + Kd + \frac{Kd(d + 1)}{2}, \qquad
\operatorname{AIC} = 2p - 2\mathcal{L}_{\mathrm{TRAIN}}, \qquad
\operatorname{BIC} = p \ln(n_{\mathrm{TRAIN}}) - 2\mathcal{L}_{\mathrm{TRAIN}}.
$$

## 11. Occupancy, persistence and uncertainty

TRAIN hard occupancy is the fraction of retained TRAIN observations whose largest filtered probability is that state.

TRAIN soft occupancy for state k:

$$
\bar{\alpha}(k) = \operatorname{mean}_t \alpha_t(k).
$$

Fold hard gates:

```text
minimum_train_hard_occupancy=0.03
minimum_train_soft_occupancy=0.05
```

OOS occupancy is diagnostic only.

Dominant-state durations are counts of consecutive retained model observations, not calendar days.

`switches_per_year` uses actual UTC timestamp span:

$$
\lambda_{\mathrm{switch}} = \frac{N_{\mathrm{switch}}}{D_{\mathrm{calendar}}} \cdot 365.2425.
$$

Undefined for zero elapsed span; never fabricated.

Confidence:

$$
\gamma_t = \max_k \alpha_t(k).
$$

Entropy uses natural logarithm:

$$
H_t = -\sum_k \alpha_t(k) \ln\!\left(\alpha_t(k)\right).
$$

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

At every numeric ranking stage, recursively partition the current group from the exact
best anchor using the anchored $\varepsilon = 10^{-12}$ rule. The tied partition proceeds to the next
numeric stage; later partitions remain after it. This makes ranking deterministic and
transitive. K and candidate ID are exact tie-breaks.

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

## 18. Non-normative MLflow evaluation snapshot

This section records an operational snapshot queried from the deployed MLflow backend on
2026-08-27. It is empirical evidence, not part of the selection contract above, and must be
updated or removed when the referenced source build or candidate universe changes.

### Comparable runs

MLflow contained 14 finished parent evaluation runs for source build
`20260823T063926Z`. All used evaluation-plan hash prefix `33e8f85db7e9` and
feature-selection-definition hash prefix `237a8b1fe699`. Five three-candidate runs repeated
one exact aggregate result signature, as did five four-candidate runs. This confirms
deterministic aggregate reproduction for those repeated inputs; repeated runs are not
independent statistical evidence.

The newest finished comparison was parent run `cd5a3e05e2824f4ba9c481de2e66e6df`.
It contained eight candidates and 62 planned folds. Its aggregate scorecard, ordered by the
primary ranking metric, was:

| Candidate | Valid folds | OOS loglik mean | OOS std | Worst fold | BIC mean | AIC mean |
|---|---:|---:|---:|---:|---:|---:|
| `gaussian_hmm_k5_full` | 62/62 | -7.2689 | 3.5307 | -20.4292 | 29150.45 | 28212.43 |
| `gaussian_hmm_k4_full` | 62/62 | -7.3444 | 3.7793 | -26.2415 | 30975.34 | 30249.70 |
| `gaussian_hmm_k3_full` | 62/62 | -7.4670 | 3.1924 | -21.6244 | 34067.79 | 33542.73 |
| `gmm_hmm_k3_m2_full` | 62/62 | -7.5851 | 3.5805 | -19.4449 | 30682.79 | 29662.17 |
| `gaussian_hmm_k2_full` | 62/62 | -8.0874 | 3.7954 | -30.4469 | 38850.01 | 38513.74 |
| `gmm_hmm_k4_m2_full` | 62/62 | -8.0922 | 4.3984 | -21.4038 | 28288.97 | 26902.59 |
| `gmm_hmm_k2_m2_full` | 62/62 | -8.0958 | 3.2797 | -19.4792 | 34889.95 | 34223.31 |
| `gmm_hmm_k5_m2_full` | 60/62 | -8.1638 | 5.2439 | -30.2428 | 26855.29 | 25088.71 |

### Interpretation

- `gaussian_hmm_k5_full` was the statistical selection result. Its mean OOS advantage was
    0.0755 over Gaussian K=4 and 0.3162 over the best GMM candidate, GMM K=3.
- Gaussian K=3 had the lowest OOS standard deviation among Gaussian candidates, and GMM K=3
    had the best worst-fold result overall. Neither can override the primary mean-OOS ranking
    stage defined in section 13.
- GMM K=4 and K=5 had lower mean AIC/BIC than Gaussian K=5, but their OOS means were worse.
    This is expected under the ranking contract, where information criteria are later
    tie-breakers rather than weighted objectives.
- GMM K=5 was the only candidate with invalid folds (two of 62), had the lowest valid-fold
    rate, and had the highest OOS dispersion. It still passed the 0.80 valid-fold-rate gate,
    but its stability evidence was weakest in this comparison.

### Operational limitations

The completed eight-candidate run predates the full 12-candidate universe. It contains all
Gaussian candidates and all two-mixture GMM candidates, but no Student-t candidates.
Therefore it cannot establish the champion of the complete v2 universe. A 12-candidate run
started from image revision `18edfebfacd12c91f942d203aff8d0695bd22cc2` was still running
at snapshot time and is deliberately excluded until MLflow records a finished parent run.

Three older parent runs remained marked `RUNNING` despite having no active evaluation
process. They use different plan/selection hashes and are excluded from every comparison;
their stale lifecycle status should be repaired operationally. Registered model versions
2 through 17 were `READY` but exposed no MLflow `run_id`, so direct registry-version to
evaluation-run linkage could not be verified from registry metadata alone.

The observed parent-run durations measure evidence rendering and MLflow persistence after
candidate evaluation, not total model-fitting time. The eight-candidate parent took 509.0
seconds; this value must not be used to estimate end-to-end evaluation runtime.


## Xetra v3 canonical 61-feature policy

Xetra profile configuration version 3 is a versioned extension of v2. Historical v1/v2 profile and feature-selection identities, hashes and behavior remain immutable.

Canonical identity:

 ```text
profile_id=xetra
profile_config_version=3
feature_selection_policy=xetra_semantic_medoid_v3
canonical_feature_universe_size=61
semantic_block_count=8
```

The ordered v3 feature universe is the ordered v2 48-feature universe plus these exact existing PostgreSQL columns, assigned to their economic blocks:

| Semantic block | Added v3 feature(s) |
|---|---|
| US equity volatility spot | `vix_delta_1obs` |
| US equity volatility term structure | `vix9d_delta_1obs`, `vix3m_delta_1obs`, `vix6m_delta_1obs`, `vix1y_delta_1obs` |
| Europe equity volatility | `vstoxx_delta_1obs` |
| Rates volatility | `move_delta_1obs` |
| Systemic stress | `ciss_delta_1obs` |
| Credit stress | `euro_hy_oas_delta_1obs` |
| Rates / yield curve | `us_2y_delta_1obs`, `us_10y_delta_1obs`, `estr_delta_1obs` |
| USD FX | `usd_broad_delta_1obs` |

The exact ordered added tuple is:

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

Stage 1 runs the existing first-fold-TRAIN-only absolute-Spearman medoid selector on all 61 canonical features within the same eight semantic blocks. A `*_delta_1obs` feature is a normal Stage-1 candidate and may become its block's `preliminary_medoid`. Stage 2 applies the unchanged cross-block absolute-Spearman pruning rule to the eight Stage-1 representatives and freezes the surviving ordered multivariate feature tuple. Coverage, variance, complete-observation gates, numeric tie semantics, the strict `>0.85` cross-block conflict threshold, missing-value semantics, no-HMM-feedback rule and no-economic-input rule are unchanged from v2.

Xetra v3 has exactly the same 12 model candidates, in the same order, as v2:

```text
gaussian_hmm_k2_full
gaussian_hmm_k3_full
gaussian_hmm_k4_full
gaussian_hmm_k5_full
gmm_hmm_k2_m2_full
gmm_hmm_k3_m2_full
gmm_hmm_k4_m2_full
gmm_hmm_k5_m2_full
student_t_hmm_k2_full
student_t_hmm_k3_full
student_t_hmm_k4_full
student_t_hmm_k5_full
```

Walk-forward windows, multistart seeds/gates, family-specific fit settings, occupancy gates, common-valid-fold comparison, anchored `1e-12` ranking semantics, causal TEST continuation, state alignment, likelihood-parity checks and final-refit rules remain unchanged unless a later versioned contract explicitly changes them.

## Xetra v3 first-class regime evaluations

Xetra v3 exposes exactly three evaluation identities:

```text
medoid_multivariate
medoid_univariate
delta1_univariate
```

`xetra_univariate_shadow_v1` is not a v3 runtime evaluation identity. The three evaluations share the same Xetra v3 model-family/K configuration where applicable but have separate input, clock, lineage and evidence contracts.

### `medoid_multivariate`

Input features are exactly the frozen ordered Stage-2 feature tuple produced by canonical first-fold TRAIN-only Xetra v3 feature selection. It evaluates exactly the 12 Xetra v3 model candidates on the canonical complete-case walk-forward observation clock for that frozen multivariate tuple. Statistical gates, common-valid-fold support and the seven-stage anchored ranking are unchanged. Its winner is `medoid_multivariate_statistical_champion`. This is the only evaluation champion eligible for final production refit, immutable OOS publication, challenger registration and a later explicit production alias promotion.

### `medoid_univariate`

Input features are exactly the eight Stage-1 `preliminary_medoids`, in canonical semantic-block order, before Stage-2 pruning. Each feature is evaluated alone against exactly the 12 Xetra v3 model candidates, giving exactly 96 candidate evaluations. One common diagnostic complete-case clock is built across exactly these eight medoid features. Every one-feature grid in this evaluation uses that same medoid clock. The within-feature statistical winner is `diagnostic_feature_model_winner`; the evaluation-level diagnostic champion is `medoid_univariate_evaluation_champion`.

### `delta1_univariate`

Input features are exactly the ordered 13 canonical one-observation delta features defined by the Xetra v3 feature policy. Each feature is evaluated alone against exactly the 12 Xetra v3 model candidates, giving exactly 156 candidate evaluations. One common diagnostic complete-case clock is built across exactly these 13 delta features. Every one-feature grid in this evaluation uses that same delta1 clock. The within-feature statistical winner is `diagnostic_feature_model_winner`; the evaluation-level diagnostic champion is `delta1_univariate_evaluation_champion`.

The complete three-evaluation execution therefore contains exactly 12 multivariate + 96 medoid-univariate + 156 delta1-univariate candidate evaluations before invalid-fold rejection.

### Evaluation-clock isolation

The three clocks are distinct contracts:

- `medoid_multivariate` uses the canonical complete-case clock for the frozen final Stage-2 tuple;
- `medoid_univariate` uses a diagnostic common clock over exactly the eight Stage-1 medoids;
- `delta1_univariate` uses a diagnostic common clock over exactly the ordered 13 delta1 features.

A combined 21-feature medoid+delta diagnostic clock is forbidden. A feature may occur in both univariate evaluations when a delta1 feature is also a Stage-1 medoid, but the two evaluations remain independent. Cross-evaluation fitted-model reuse is forbidden because the evaluation-specific retained observations, clock hash and execution lineage may differ even when the feature name and candidate ID are identical. No fill, interpolation, carry, synthesis, per-feature-clock fallback or source-row boundary mutation is allowed.

### Statistical and production boundaries

Within a single univariate feature, the 12 family/K candidates use the same hard gates,
common-valid-fold comparison and seven-stage statistical ranking as the canonical Xetra v3
candidate grid. This winner is that feature's `diagnostic_feature_model_winner`. A feature
with shared valid-fold support below $0.80$ is ineligible only for its evaluation-level
champion selection; its diagnostic grid evidence remains recorded.

The exact cross-feature selection rule for both univariate evaluations is: rank eligible
feature winners by dominant-state NMI descending using anchored
$\varepsilon = 10^{-12}$ ties, then shared OOS timestamp count descending, then feature
name ascending. No eligible feature produces explicit no-champion evidence. No fallback
metric is allowed. Raw OOS PLL, BIC, AIC and economic metrics are forbidden for ranking
different feature names.

`medoid_univariate_evaluation_champion` and `delta1_univariate_evaluation_champion` are
diagnostic-only and cannot trigger final refit, OOS publication, model registration,
challenger/champion alias mutation or economic decisions. Only
`medoid_multivariate_statistical_champion` is production-eligible.
