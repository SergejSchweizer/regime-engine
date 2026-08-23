# Market Regime Engine — Evaluation Contract

Status date: 2026-08-23

`EVALUATION.md` is the durable sidecar for model evaluation in `market-regime-engine`. It explains exactly which model variants are compared, how leak-free walk-forward evaluation is performed, which statistics are computed, which metrics are logged to MLflow, and how `engine-champion` is selected.

## Sidecar maintenance rule

This file must remain synchronized with implementation and configuration. Any PR that changes **candidate model families, state counts, covariance modes, preprocessing, walk-forward policy, inference semantics, diagnostic definitions, quality gates, selection/ranking rules, MLflow metric names, MLflow artifacts, or model lifecycle aliases** must update `EVALUATION.md` in the same PR. A PR that changes evaluation behavior without updating this sidecar is incomplete.

The sidecar describes the current intended evaluation contract. Model/profile-specific thresholds remain configuration values and must be versioned in the corresponding model profile.

## Purpose

The engine must answer two different questions without conflating them:

1. **Model quality:** Which candidate is the best statistically robust and causally valid market-regime estimator under the declared profile?
2. **Consumer value:** Does a particular validated model improve a downstream application such as Portfell?

`market-regime-engine` owns the first question. Portfell and future consumers own the second. Portfolio returns, portfolio optimization, Sharpe/Sortino ranking, transaction costs, and application-specific utility are therefore not `engine-champion` selection metrics.

## MVP candidate models

For `xetra_cross_asset_v1`, the MVP candidate grid is:

| Candidate | Family | States K | Covariance | Required in MVP |
|---|---|---:|---|---|
| `gaussian_hmm_k2_full` | Gaussian HMM | 2 | full | yes |
| `gaussian_hmm_k3_full` | Gaussian HMM | 3 | full | yes |
| `gaussian_hmm_k4_full` | Gaussian HMM | 4 | full | yes |

**Covariance policy:** every covariance-bearing HMM emission model uses a separate **full covariance matrix per hidden state**. `diag`, `spherical`, `tied`, or any other reduced covariance mode is unsupported and must fail profile/model validation rather than being silently converted. The MVP therefore compares state count `K`, not covariance structure.

Optional challengers, after the common model/evaluation protocol is stable:

| Candidate | Family | Typical initial configuration | Required in MVP |
|---|---|---|---|
| `student_t_hmm_k3_full` | Student-t HMM | K=3, full covariance | no |
| `hsmm_k3_full` | Hidden semi-Markov model | K=3, full-covariance emissions with explicit duration model | no |

All candidates must use the same feature profile and the same walk-forward split plan when they are compared in one experiment.

## Evaluation flow

```text
immutable causal feature build
          |
          v
validated model profile
          |
          v
expanding walk-forward folds
          |
          +--> fit preprocessing on TRAIN only
          |
          +--> multi-start fit candidate on TRAIN only
          |
          +--> align fitted states using TRAIN/prior reference only
          |
          +--> calculate TRAIN diagnostics
          |
          +--> causal filtered inference on TEST
          |
          +--> calculate TEST / OOS predictive metrics
          |
          v
aggregate fold metrics per candidate
          |
          v
hard quality gates
          |
          v
rank valid candidates
          |
          v
engine-champion
          |
          v
MLflow Model Registry
```

No future test row may influence preprocessing, fitting, state alignment, or an earlier filtered probability.

## Walk-forward semantics

The default evaluation uses expanding training windows. Each fold has a training interval strictly before its test interval. The exact minimum training observations, test-window size, partial-final-window policy, and overlap policy are declared in the model profile.

Every fold has a deterministic one-based `fold_index`, a stable `fold_id`, and explicit UTC `train_start`, `train_end`, `test_start`, and `test_end` bounds. The same fold identifiers and temporal bounds must be used by evaluation results, MLflow metric history, artifacts, and plots. Fold-to-fold visualizations use the real `test_end` date as the human-facing x-axis; `fold_index` is the machine ordering key.

Backtest-safe test inference uses **filtered probabilities** only.

Greek symbols used below:

- **alpha (α)** — pronounced *AL-fa*; filtered state probability.

For state `k` at time `t`:

\[
\alpha_t(k) = P(S_t=k \mid X_1,\ldots,X_t)
\]

The engine must not use a smoothed posterior of the form `P(S_t | X_1,...,X_T)` with `T > t` for OOS evaluation or production inference. Smoothing/Viterbi may exist only as explicitly labeled retrospective diagnostics.

## Multi-start fitting

EM-based HMM fitting can converge to local optima. Every fold/candidate therefore uses an explicit deterministic seed set.

For every start the engine records:

- seed;
- converged / failed;
- final training log likelihood;
- iteration count;
- finite-parameter validity;
- failure reason when invalid.

Only valid converged starts participate in winner selection for that fold. The fold fit is the valid converged start with the highest training log likelihood, using deterministic tie-breaking. A configurable minimum number/rate of valid starts is a hard gate.

Aggregates tracked per fold/candidate include:

- `multistart_total`;
- `multistart_converged`;
- `multistart_valid`;
- `multistart_success_rate`;
- `multistart_loglik_best`;
- `multistart_loglik_median`;
- `multistart_loglik_std`.

## In-sample fit metrics

### Training log likelihood

Greek symbols used below:

- **theta (θ)** — pronounced *THAY-ta*; model parameter vector.

\[
\log L(\theta)=\log P(X_1,\ldots,X_T\mid\theta)
\]

Higher is better for the same data, but raw training likelihood is not a champion criterion because more complex models can improve it mechanically.

MLflow metric: `train_loglik` per fold, plus candidate aggregates `train_loglik_mean`, `train_loglik_std`.

### AIC

\[
AIC = 2p - 2\log(\hat L)
\]

where `p` is the number of free parameters and `L-hat` is the maximized likelihood. Lower is better. AIC is diagnostic/secondary, never the sole champion criterion.

For a Gaussian HMM with `K` states, `d` observed features, and one full symmetric covariance matrix per state, the free-parameter count is

\[
p=(K-1)+K(K-1)+Kd+K\frac{d(d+1)}{2}.
\]

The four terms are respectively initial-state probabilities, transition probabilities, state means, and full state covariance matrices. This full-covariance parameter count must be used consistently by AIC and BIC diagnostics.

MLflow: `aic` per fold; `aic_mean`, `aic_std` per candidate.

### BIC

\[
BIC = p\log(n) - 2\log(\hat L)
\]

where `n` is the number of training observations. Lower is better. BIC penalizes model complexity more strongly than AIC. BIC is a secondary/tie-break diagnostic, not the primary ranking metric.

MLflow: `bic` per fold; `bic_mean`, `bic_std` per candidate.

## Primary OOS generalisation metric

### Mean OOS predictive log likelihood per observation

The primary statistical ranking criterion is **out-of-sample predictive log likelihood normalized per test observation**. It measures how much probability the fitted model assigns to genuinely unseen observations under the causal fold setup.

For a fold with `m` test observations:

\[
OOS\_PLL_{fold} = \frac{1}{m}\log P(X_{test}\mid X_{train},\hat\theta_{train})
\]

Higher is better. Normalization by test observations makes folds of different lengths comparable.

Required metrics:

- `oos_predictive_loglik_per_obs` per fold;
- `oos_predictive_loglik_mean` per candidate;
- `oos_predictive_loglik_std` per candidate;
- `oos_predictive_loglik_median` per candidate;
- `oos_predictive_loglik_worst_fold` = minimum fold value;
- `oos_predictive_loglik_best_fold` = maximum fold value;
- `oos_predictive_loglik_fold_count`.

The engine must also retain the unnormalized fold predictive log likelihood as an audit statistic, but it must not be compared directly across unequal test-window lengths.

## Fold stability

A candidate with a slightly better mean but severe fold instability may be less suitable for production than a slightly lower but stable candidate. Fold stability is therefore an explicit secondary ranking dimension.

Required statistics include:

- standard deviation of OOS predictive log likelihood;
- worst-fold OOS predictive log likelihood;
- valid-fold count and rate;
- convergence-failure fold count;
- alignment-failure fold count;
- hard-gate-failure fold count.

No hidden composite score is allowed. The model profile declares deterministic ranking and tie-break order.

## State occupancy

A model that creates tiny or empty states can fit noise rather than a reusable regime.

### Hard occupancy

Each timestamp is assigned to the state with maximum filtered probability. Hard occupancy is the fraction of observations assigned to each state.

Required per-state/fold statistics:

- `hard_occupancy_state_<id>`;
- candidate `min_hard_occupancy`.

### Soft/effective occupancy

Soft occupancy uses the filtered posterior mass instead of a hard assignment. For state `k`:

\[
soft\_occupancy_k = \frac{1}{T}\sum_{t=1}^{T} p_{t,k}
\]

Required statistics:

- `soft_occupancy_state_<id>`;
- `min_soft_occupancy`.

Minimum acceptable occupancy is profile-configured and is a hard gate.

## Transition and persistence statistics

The full fitted transition matrix is retained as an MLflow artifact for every selected fold/model representation.

Greek symbols used below:

- **A** — transition matrix; no Greek symbol.

For transition matrix element `A_ij`:

\[
A_{ij}=P(S_{t+1}=j\mid S_t=i)
\]

Required metrics/artifacts:

- complete `transition_matrix` artifact;
- `self_transition_state_<id>` for each persistent state;
- minimum/maximum/mean self-transition probability;
- empirical number of state switches;
- `switches_per_year` or frequency-normalized equivalent.

Persistence is diagnostic rather than a target to maximize. Extremely low persistence can indicate noisy regimes; extremely high persistence can indicate an unresponsive model.

## Regime duration statistics

Using the dominant filtered state sequence, compute contiguous run lengths for each persistent state.

Required per-state statistics:

- mean duration;
- median duration;
- 90th-percentile duration;
- maximum duration;
- number of observed runs.

Candidate summary includes `min_mean_state_duration`, `mean_state_duration`, and `max_mean_state_duration`. Duration is interpreted relative to the model frequency and downstream decision cadence; it is not maximized mechanically.

## State alignment and signature stability

Raw HMM state numbers are not persistent semantics. Every fold/model must generate normalized state signatures and align them deterministically to persistent state IDs.

Required diagnostics:

- state-to-reference assignment;
- alignment cost/distance per state;
- `max_state_signature_drift` per fold;
- candidate mean/max drift;
- ambiguous-mapping indicator;
- alignment failure count/rate;
- state-signature artifact and alignment artifact.

The profile declares maximum allowed drift. Exceeding it is a hard gate, not a warning that can be ignored during promotion.

## Prediction uncertainty

### Entropy

Greek symbols used below:

- **H** — entropy symbol; no Greek symbol.

For K state probabilities:

\[
H_t=-\sum_{k=1}^{K}p_{t,k}\log(p_{t,k})
\]

Required OOS statistics:

- `oos_entropy_mean`;
- `oos_entropy_median`;
- `oos_entropy_p90`;
- `oos_confidence_mean`;
- `oos_low_confidence_fraction`, using a profile-declared confidence threshold.

Lower entropy is **not** automatically better; an overconfident bad model is still bad. Entropy/confidence are diagnostics and cannot override poor OOS predictive likelihood or hard-gate failures.

## Parameter and numerical validity

Hard validation checks include:

- all initial-state probabilities finite and normalized;
- every transition row finite and normalized;
- emission means finite;
- covariance mode is exactly `full` for every covariance-bearing HMM candidate;
- every state covariance is a finite `d x d` symmetric matrix with valid full-covariance structure and successful positive-definiteness/Cholesky validation under the declared numerical policy;
- off-diagonal covariance terms are preserved in fitted artifacts, reconstruction, state signatures, and MLflow evidence;
- `diag`, `spherical`, `tied`, and other reduced covariance modes fail closed;
- model reconstruction round-trip valid;
- preprocessing parameters finite;
- feature order exactly matches profile/model artifact.

Any violation makes the fit/fold invalid.

## Engine-champion selection

Selection is deterministic and profile-driven. No weighted “magic score” is permitted.

The default Xetra cross-asset ordering is:

```text
all candidates
    |
    v
hard quality gates
    |-- convergence / minimum valid starts
    |-- finite valid parameters
    |-- full-covariance validity
    |-- minimum state occupancy
    |-- successful state alignment
    |-- maximum signature-drift limit
    |-- minimum valid-fold requirement
    v
valid candidates
    |
    v
PRIMARY: highest mean OOS predictive log likelihood per observation
    |
    v
SECONDARY: better fold stability
    |-- lower OOS log-likelihood standard deviation
    |-- better worst fold
    v
TERTIARY / TIE BREAK: lower BIC, then lower AIC
    |
    v
FINAL DETERMINISTIC TIE BREAK: fewer states K, then stable candidate ID
    |
    v
engine-champion
```

Because covariance structure is fixed to `full`, model-complexity tie breaking in the MVP is driven by state count `K`. A more complex model does not win merely because training likelihood improves. A candidate that fails a hard gate cannot win regardless of its OOS mean.

## MLflow experiment structure

Production tracking and registry use:

```text
MLFLOW_TRACKING_URI=http://10.10.1.3:5000
```

Required CI tests remain hermetic and use local/fake MLflow stores; the shared server is exercised only by the explicit external smoke test.

Recommended hierarchy:

```text
Experiment: market-regime-engine/<profile-id>

Parent evaluation run
  |
  +-- candidate run: gaussian_hmm_k2_full
  |      +-- fold runs / fold metrics
  |
  +-- candidate run: gaussian_hmm_k3_full
  |      +-- fold runs / fold metrics
  |
  +-- candidate run: gaussian_hmm_k4_full
```

### Parent-run parameters/tags

- profile ID/version/hash;
- engine version;
- Git SHA;
- feature version;
- source build ID;
- evaluation-plan hash;
- split policy;
- inference mode;
- candidate count;
- selection-policy version.

### Candidate-run parameters

- model family;
- K/state count;
- covariance mode, which must be exactly `full` for covariance-bearing HMMs;
- feature count/order hash;
- multi-start seed policy/count;
- convergence tolerance/max iterations;
- candidate ID.

### Candidate-run aggregate metrics

At minimum:

- `oos_predictive_loglik_mean`;
- `oos_predictive_loglik_std`;
- `oos_predictive_loglik_median`;
- `oos_predictive_loglik_worst_fold`;
- `train_loglik_mean`;
- `aic_mean`;
- `bic_mean`;
- `valid_fold_rate`;
- `multistart_success_rate_mean`;
- `min_hard_occupancy`;
- `min_soft_occupancy`;
- `max_state_signature_drift`;
- `alignment_failure_count`;
- `mean_state_duration`;
- `switches_per_year`;
- `oos_entropy_mean`;
- `oos_confidence_mean`;
- all hard-gate pass/fail indicators that are representable as scalar metrics/tags.

## Fold-to-fold MLflow observability

Fold runs remain the auditable source of individual fold results, but they are not sufficient for convenient trend analysis. Therefore every scalar fold diagnostic required by this contract must also be logged on its **candidate run as MLflow metric history**.

The required logging semantics are:

- `step` is the deterministic one-based `fold_index`;
- the metric timestamp is the UTC `test_end` instant where the MLflow client supports an explicit timestamp;
- `fold_timeline.parquet` is the authoritative mapping from `fold_index` and `fold_id` to train/test bounds;
- invalid folds remain present in `fold_timeline.parquet` and `fold_metrics.parquet`, but unavailable scalar values are not invented, imputed, forward-filled, or interpolated;
- line plots must show invalid/missing folds as gaps or explicit invalid markers, never as a continuous fabricated value;
- aggregate candidate metrics such as `oos_predictive_loglik_mean` keep their existing names and are distinct from fold-history names.

Canonical candidate-run fold-history keys include at minimum:

- `fold_train_loglik`;
- `fold_oos_predictive_loglik`;
- `fold_oos_predictive_loglik_per_obs`;
- `fold_aic`;
- `fold_bic`;
- `fold_multistart_success_rate`;
- `fold_min_hard_occupancy`;
- `fold_min_soft_occupancy`;
- `fold_max_state_signature_drift`;
- `fold_mean_state_duration`;
- `fold_switches_per_year`;
- `fold_oos_entropy_mean`;
- `fold_oos_confidence_mean`.

Per-state scalar histories use stable persistent-state identifiers, including:

- `fold_hard_occupancy_state_<id>`;
- `fold_soft_occupancy_state_<id>`;
- `fold_self_transition_state_<id>`;
- `fold_mean_duration_state_<id>`;
- `fold_state_signature_drift_state_<id>`.

Any additional scalar metric introduced into the fold scorecard must either receive an equivalent stable `fold_*` history key or be explicitly documented as non-trendable with a reason.

### Required candidate-run trend plots

MLflow metric history provides native line-chart data. In addition, each candidate run must store deterministic PNG trend artifacts with the real fold `test_end` date on the x-axis so the operator can immediately relate instability to market periods. Required plots are:

- `plots/oos_predictive_loglik_per_obs_by_fold.png`;
- `plots/train_loglik_by_fold.png`;
- `plots/aic_by_fold.png`;
- `plots/bic_by_fold.png`;
- `plots/multistart_success_rate_by_fold.png`;
- `plots/hard_occupancy_by_fold.png`;
- `plots/soft_occupancy_by_fold.png`;
- `plots/self_transition_by_fold.png`;
- `plots/state_signature_drift_by_fold.png`;
- `plots/state_duration_by_fold.png`;
- `plots/switches_per_year_by_fold.png`;
- `plots/oos_entropy_by_fold.png`;
- `plots/oos_confidence_by_fold.png`.

Occupancy, self-transition, duration, and signature-drift plots may contain one line per persistent state plus the candidate summary line where useful. Plots with metrics on incompatible numerical scales must remain separate; no hidden secondary axes are permitted.

### Matrix diagnostics by fold

Transition and covariance matrices are not scalar metric histories. For every valid fold they must be retained as machine-readable artifacts and visual heatmaps:

- `folds/<fold_id>/transition_matrix.json` or Parquet equivalent;
- `plots/folds/<fold_id>/transition_matrix.png`;
- full covariance matrix artifact for every persistent state;
- `plots/folds/<fold_id>/covariance_state_<id>.png` for every persistent state.

The covariance heatmaps must retain off-diagonal terms and exact feature order. Heatmaps are diagnostic evidence only and must not alter champion-selection rules.

### Parent-run cross-candidate plots

Because all MVP candidates use the same walk-forward split plan, the parent evaluation run must provide aligned comparison plots for the most important fold trends. At minimum:

- `plots/candidates/oos_predictive_loglik_per_obs_by_fold.png`;
- `plots/candidates/multistart_success_rate_by_fold.png`;
- `plots/candidates/min_hard_occupancy_by_fold.png`;
- `plots/candidates/max_state_signature_drift_by_fold.png`;
- `plots/candidates/oos_entropy_by_fold.png`;
- `plots/candidates/oos_confidence_by_fold.png`.

Each line is identified by stable candidate ID. The x-axis is the shared fold `test_end` date, not wall-clock logging time. A candidate with an invalid fold must show a gap/invalid marker for that fold instead of interpolation.

A deterministic `plots/manifest.json` records every generated plot path, source metric keys, candidate/fold IDs, x-axis field, and source artifact hash so plots remain auditable rather than presentation-only output.

### Fold-run metrics

At minimum:

- train/test bounds and observation counts;
- `train_loglik`;
- `oos_predictive_loglik`;
- `oos_predictive_loglik_per_obs`;
- `aic`;
- `bic`;
- multistart total/converged/valid/success rate;
- hard and soft occupancy by state;
- self-transition probability by state;
- duration summary by state;
- state-signature drift by state and maximum drift;
- entropy/confidence summaries;
- convergence/alignment/quality-gate status.

### Required MLflow artifacts

- `model_spec.json`;
- `evaluation_plan.json`;
- `fold_timeline.parquet`;
- `fold_metrics.parquet`;
- `candidate_scorecard.json`;
- `candidate_comparison.parquet` on the parent run;
- `multistart_metrics.parquet`;
- `transition_matrix.json` or Parquet equivalent;
- full covariance matrices/model-spec evidence;
- `state_signatures.json`;
- `state_alignment.json`;
- `occupancy_by_fold.parquet`;
- `duration_by_fold.parquet`;
- `oos_predictions.parquet` reference/build metadata;
- `plots/manifest.json`;
- all required candidate trend plots, parent cross-candidate plots, and per-fold transition/full-covariance heatmaps;
- feature order and preprocessing metadata;
- champion-selection result including rejected candidates and reasons.

## Candidate scorecard

Every candidate must expose one concise scorecard so an operator can understand the result without reconstructing fold data manually.

```text
MODEL
  family
  states
  covariance
  candidate_id

GENERALISATION
  oos_predictive_loglik_mean
  oos_predictive_loglik_std
  oos_predictive_loglik_worst_fold
  valid_fold_rate

IN-SAMPLE
  train_loglik_mean
  aic_mean
  bic_mean

FIT STABILITY
  multistart_success_rate_mean
  convergence_failure_count

STATE QUALITY
  min_hard_occupancy
  min_soft_occupancy
  max_state_signature_drift
  alignment_failure_count
  mean_state_duration
  switches_per_year

UNCERTAINTY
  oos_entropy_mean
  oos_confidence_mean

SELECTION
  hard_gates_passed
  rank
  selected_as_engine_champion
  rejection_reason_if_any
```

## How to interpret a typical comparison

A candidate may have the best training likelihood yet lose because it creates a tiny state, is unstable across folds, or generalizes worse OOS. For example, K=4 can fit training data better than K=3 but fail minimum occupancy or state-drift gates. Conversely, K=2 may be extremely stable yet underfit and have materially worse OOS predictive likelihood. The champion is the simplest valid model that wins under the declared deterministic OOS ranking, not the model with the prettiest in-sample fit.

## Separation from downstream Portfell evaluation

After the engine registers validated candidates/champion, Portfell may retrieve their leak-free `walk_forward_oos` predictions and evaluate regime-conditioned ETF statistics and portfolio performance. That application-level comparison can select a consumer-specific alias such as `portfell-production` even if it differs from `engine-champion`.

The engine must not change its statistical champion criteria to optimize Portfell Sharpe, return, drawdown, or transaction costs. This separation preserves model reuse across future consumers.