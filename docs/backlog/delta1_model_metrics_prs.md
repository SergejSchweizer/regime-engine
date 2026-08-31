# Delta1-univariate MLflow Model Metrics — backlog patch

This planning document contains the exact section intended to be appended to `BACKLOG.md`. It is isolated on the planning branch so the six implementation PRs can be reviewed atomically before touching the authoritative backlog.

## Scope contract

The first Model Metrics rollout is **only** for `delta1_univariate`. In this scope, a **dataset** means exactly one feature from the canonical ordered 13-delta tuple, and a **model** means one of the exact 12 canonical Xetra v3 candidate IDs. The existing MLflow hierarchy remains `delta1_univariate parent -> 13 feature/dataset runs -> 12 candidate/model runs per feature`; this rollout does not change candidate selection, agreement ranking, final refit, registry aliases, `medoid_multivariate`, or `medoid_univariate`.

For each delta feature run, Model Metrics must present the same deterministic model-centric structure:

```text
model_metrics/
  models/
    <candidate_id>/
      performance/
        train_loglik_per_obs
        oos_predictive_loglik_per_obs
        aic_per_train_obs
        bic_per_train_obs
        multistart_success_rate
      optimization/
        em_convergence
  comparisons/
    oos_predictive_loglik_per_obs_all_models
    em_convergence_all_models
  manifest.json
```

The first rollout uses the existing MLflow nested-run/artifact model and metric-history APIs; it must not fork or patch MLflow's React frontend. The `model_metrics/` namespace is the canonical dataset section inside each delta feature run.

The per-model EM convergence plot is an **optimization diagnostic only**. For each valid walk-forward fold it uses the winning multistart fit only, plots `TRAIN log likelihood / TRAIN model observation count` against EM iteration, never interpolates beyond a fold's recorded iterations, and overlays a deterministic across-fold median plus 25th/75th percentile band. A candidate with no usable convergence history gets explicit unavailable evidence instead of a fabricated curve. The all-model convergence comparison uses one median curve per canonical candidate and is explicitly labelled `optimization diagnostic only — not model selection`.

The canonical cross-model performance comparison remains OOS predictive log likelihood per observation on the common delta evaluation clock. Training likelihood, EM convergence, AIC/BIC, or any other optimization diagnostic must never become a cross-feature ranking criterion and must never change `diagnostic_feature_model_winner` or `delta1_univariate_evaluation_champion`.

### PR-163 — Pin delta1 Model Metrics diagnostic semantics

- **Branch:** `pr/PR-163-delta1-model-metrics-contract`
- **Depends on:** PR-152, PR-162
- **Allowed:** `EVALUATION.md`, `PLOT_STYLE.md`

Acceptance:

- [ ] Scope is explicitly limited to `delta1_univariate`; `medoid_multivariate` and `medoid_univariate` behavior and artifact layout are unchanged.
- [ ] Define `dataset` for this feature exactly as one of the canonical ordered 13 delta features and `model` exactly as one of the canonical 12 v3 candidate IDs.
- [ ] Pin the feature-run artifact namespace exactly to `model_metrics/models/<candidate_id>/...`, `model_metrics/comparisons/...`, and `model_metrics/manifest.json`.
- [ ] Pin the five core per-model performance histories exactly: TRAIN log likelihood per observation, OOS predictive log likelihood per observation, AIC per TRAIN observation, BIC per TRAIN observation, and multistart success rate.
- [ ] Pin one per-model `em_convergence` optimization plot and one `em_convergence_all_models` comparison plot.
- [ ] Define EM x-axis exactly as one-based completed EM iteration and y-axis exactly as `TRAIN log likelihood per observation`.
- [ ] Define one candidate plot as valid-fold winning-start trajectories plus deterministic across-fold median and 25th/75th percentile envelope; no failed/non-winning seed is mixed into the main candidate curve.
- [ ] Define varying iteration lengths with no extrapolation/interpolation beyond a fold's recorded history; aggregation at iteration `i` uses only valid winner histories that actually contain iteration `i`.
- [ ] Define the all-model EM comparison as one median curve per canonical candidate in canonical candidate order; no curve value is interpreted as statistical rank.
- [ ] Every EM plot title/legend contains `optimization diagnostic only — not model selection` or an equivalent unambiguous phrase.
- [ ] Canonical model comparison remains OOS predictive log likelihood per observation; no TRAIN/EM/AIC/BIC value is fed into cross-feature agreement ranking.
- [ ] A candidate with no valid convergence trace is represented as explicitly unavailable and must not be silently omitted from the dataset section.
- [ ] First rollout explicitly reuses MLflow nested runs/artifacts/metric histories and introduces no MLflow frontend fork, injected React bundle, or second web UI.
- [ ] No numerical gate, candidate universe, state alignment rule, evaluation clock, statistical-selection rule, or production eligibility rule changes in this PR.

### PR-164 — Preserve exact EM likelihood histories in HMM fit results

- **Branch:** `pr/PR-164-hmm-em-likelihood-history`
- **Depends on:** PR-163, PR-128, PR-129
- **Allowed:** `src/market_regime_engine/models/protocols.py`, `src/market_regime_engine/models/gaussian_hmm.py`, `src/market_regime_engine/models/student_t_hmm.py`, `src/market_regime_engine/training/multistart.py`, `tests/unit/models/test_gaussian_hmm.py`, `tests/unit/models/test_student_t_hmm.py`, `tests/unit/training/test_multistart.py`, direct `FitResult` fixture tests only

Acceptance:

- [ ] Extend `FitResult` with one immutable ordered `em_log_likelihood_history: tuple[float, ...]` containing the optimizer objective for each completed EM iteration.
- [ ] A production successful fit requires a non-empty finite history with `len(history) == iterations`; NaN/Inf or length mismatch fails closed.
- [ ] Gaussian HMM copies the exact `hmmlearn` convergence-monitor history after fit; no second fit or synthetic reconstruction is used.
- [ ] GMM-HMM copies the exact `hmmlearn` GMM convergence-monitor history under the same semantics.
- [ ] Student-t HMM records the exact likelihood value evaluated by its existing EM loop once per completed iteration; recording does not alter parameter updates or stopping tolerance.
- [ ] Existing material-likelihood-regression validation uses the same captured history rather than a separately reconstructed sequence.
- [ ] `train_log_likelihood` remains the canonical post-fit TRAIN likelihood used by PR-129 parity/AIC/BIC; the EM history is diagnostic and does not replace that scalar.
- [ ] `MultistartResult.winner` preserves the complete winning `FitResult` history without rerunning the winner.
- [ ] Multistart winner selection remains exact global TRAIN-likelihood maximum with anchored `1e-12` tie semantics then lowest seed; history shape never influences the winner.
- [ ] Reconstruction/inference paths do not invent an EM history for already persisted model artifacts.
- [ ] Tests prove exact history length/order for Gaussian, GMM and Student-t fits and prove multistart returns the winner's original history byte-for-byte/value-for-value.
- [ ] Tests prove adding the history does not change final fitted parameters, canonical TRAIN likelihood, OOS continuation likelihood, AIC/BIC, convergence gate, or winning seed for fixed fixtures.

### PR-165 — Add fail-closed pre-finalization payload logging for evaluation runs

- **Branch:** `pr/PR-165-evaluation-tracking-payload-hook`
- **Depends on:** PR-152
- **Allowed:** `src/market_regime_engine/mlflow_support/evaluation_tracking.py`, `tests/unit/mlflow_support/test_evaluation_tracking.py`

Acceptance:

- [ ] Refactor the existing statistics-run helper to support an optional injected payload emitter executed after MLflow run creation/local `RUNNING` evidence creation and before local/MLflow `FINISHED` finalization.
- [ ] The emitter receives only the current run ID and deterministic writable run-evidence directory/context needed to log metrics/artifacts; no global active-run state is required.
- [ ] With no emitter, behavior and exact existing parent/feature/candidate hierarchy remain byte-for-byte/semantically unchanged.
- [ ] Emitter success permits normal statistics finalization, SHA-256 parity logging, artifact logging, and `FINISHED` termination exactly once.
- [ ] Emitter failure marks the MLflow run `FAILED`, finalizes the local dossier as `FAILED` when safely possible, preserves already available safe evidence, and re-raises; a false `FINISHED` state is impossible.
- [ ] Statistics hash parity is computed only after the final lifecycle state is known.
- [ ] The helper does not know delta feature names, model families, plot semantics, or candidate ordering; it is generic tracking infrastructure only.
- [ ] No registry operation, final refit, OOS publication, alias mutation, HTTP/UI customization, or model computation is introduced.
- [ ] Unit tests cover no-emitter compatibility, successful metric/artifact emission, emitter exception, MLflow logging exception, local finalization exception, and exactly-once run termination.

### PR-166 — Render delta1 per-model and all-model EM convergence diagnostics

- **Branch:** `pr/PR-166-delta1-em-convergence-plots`
- **Depends on:** PR-163, PR-164
- **Allowed:** `src/market_regime_engine/mlflow_support/plots.py`, plot-focused unit tests only

Acceptance:

- [ ] Add a pure renderer for one `WalkForwardEvaluation` candidate that consumes only stored winning `FitResult.em_log_likelihood_history`; it never refits a model.
- [ ] Normalize every fold history point exactly as `em_log_likelihood_history[i] / train_model_observation_count` using that same fold's retained TRAIN model count.
- [ ] Candidate plot x-axis is one-based `EM iteration`; y-axis is exactly `TRAIN log likelihood per observation`.
- [ ] Valid fold winner trajectories are visible individually with deterministic fold identity while the candidate median is visually dominant and the 25th/75th percentile envelope is present.
- [ ] Median/quantiles at iteration `i` use only valid folds with an observed `i`; no padding value, forward fill, interpolation, extrapolation, or invented plateau is allowed.
- [ ] Invalid folds and valid folds lacking a trace are counted in explicit plot/manifest metadata; they do not contribute numeric points.
- [ ] A candidate with zero usable traces yields deterministic unavailable plot/evidence rather than raising an unrelated feature-wide failure or drawing a fake zero line.
- [ ] Add an all-candidate renderer for exactly the candidate evaluations supplied by one `UnivariateFeatureGrid`; each candidate contributes only its across-fold median history.
- [ ] All-model legend order follows the exact canonical 12-candidate order and does not reorder by final TRAIN likelihood, convergence speed, AIC/BIC, or OOS score.
- [ ] Both plot titles clearly state the delta feature and `optimization diagnostic only — not model selection`.
- [ ] Source hashes/manifest metadata include feature name, candidate ID(s), fold IDs, winning seeds, TRAIN counts, raw history values, aggregation rule and plot type so the rendering is reproducible.
- [ ] Generated artifacts satisfy the existing `PLOT_STYLE.md` accessibility/label/determinism requirements and include publication-quality vector output where that contract requires it.
- [ ] Renderer code performs no MLflow calls and changes no evaluation metric or selection result.

### PR-167 — Wire delta1 dataset Model Metrics sections into MLflow

- **Branch:** `pr/PR-167-delta1-model-metrics-tracking`
- **Depends on:** PR-164, PR-165, PR-166, PR-152, PR-162
- **Allowed:** `src/market_regime_engine/mlflow_support/evaluation_tracking.py`, `src/market_regime_engine/evaluation_statistics/contracts.py`, `src/market_regime_engine/evaluation_statistics/render.py`, `tests/unit/mlflow_support/test_evaluation_tracking.py`, `tests/unit/evaluation_statistics/test_writer.py`

Acceptance:

- [ ] The code path is activated only when `evaluation_id == delta1_univariate`; both other evaluation IDs retain their existing tracking behavior and artifact structure.
- [ ] Exactly 13 delta feature runs remain the dataset sections, in the canonical ordered delta tuple; no synthetic aggregate dataset run is introduced.
- [ ] Inside every delta feature run create exactly one deterministic `model_metrics/` artifact namespace matching PR-163.
- [ ] Every dataset section lists all 12 canonical candidate IDs even when a candidate has no accepted/valid folds; unavailable candidates retain explicit status/reason evidence.
- [ ] For every candidate, publish the five exact core performance plots/histories pinned by PR-163 under `model_metrics/models/<candidate_id>/performance/` by reusing existing metric definitions/rendering rather than recomputing alternate statistics.
- [ ] For every candidate, publish the PR-166 EM convergence artifact under `model_metrics/models/<candidate_id>/optimization/em_convergence`.
- [ ] Log candidate-level native MLflow metric histories for the across-fold EM median using one metric key with `step = EM iteration`; values exactly equal the plotted median series.
- [ ] Publish exactly one all-model OOS predictive-log-likelihood-per-observation comparison under `model_metrics/comparisons/oos_predictive_loglik_per_obs_all_models` using the existing canonical candidate comparison semantics.
- [ ] Publish exactly one PR-166 all-model EM comparison under `model_metrics/comparisons/em_convergence_all_models`.
- [ ] Write `model_metrics/manifest.json` containing the exact 12 candidate IDs/order, every expected artifact path, availability status, source hash, source metric keys, and comparison plot identities.
- [ ] Candidate local `statistics.json` gains the winning-fold optimization evidence needed to reproduce its EM plot: fold ID, winning seed, TRAIN model observation count, completed iteration count and raw EM history; no raw source row is stored.
- [ ] Local statistics and MLflow evidence remain finite-only and secret-free; raw feature values, DSNs, credentials and model binary payloads remain forbidden.
- [ ] Plot/metric emission occurs through PR-165 before the relevant run is finalized; any emission failure fails that run/evaluation instead of leaving a false `FINISHED` run with an incomplete Model Metrics section.
- [ ] No metric or plot produced here is passed into candidate selection, cross-feature NMI agreement, evaluation champion selection, final refit, registration or alias logic.
- [ ] No cross-feature PLL/AIC/BIC/EM comparison plot is created; comparisons are strictly within one delta dataset across its 12 models.
- [ ] No custom MLflow frontend code, injected JavaScript, second dashboard service or non-MLflow port is introduced.

### PR-168 — Prove delta1 Model Metrics hierarchy hermetically

- **Branch:** `pr/PR-168-delta1-model-metrics-proof`
- **Depends on:** PR-167
- **Allowed:** `tests/integration/mlflow_support/test_delta1_model_metrics.py`, delta1 Model Metrics fixtures only

Acceptance:

- [ ] Hermetic file-store MLflow fixture requires no NAS, feature PostgreSQL, network service, browser automation, or external MLflow instance.
- [ ] Fixture represents exactly the 13 canonical delta datasets and exact 12 canonical candidate identities per dataset.
- [ ] Prove the tracking hierarchy remains one `delta1_univariate` parent, 13 feature/dataset child runs and 156 candidate/model child runs.
- [ ] For each of all 13 feature runs, prove `model_metrics/manifest.json` exists and enumerates exactly 12 candidate sections in canonical order.
- [ ] For each candidate section, prove the five core performance entries and one EM-convergence entry exist or carry explicit unavailable status/reason; silent omission is forbidden.
- [ ] Prove each usable candidate EM plot/metric series equals the stored winning-fold histories normalized by exact per-fold TRAIN model counts and aggregated under the PR-163 missing-iteration rule.
- [ ] Include histories with different convergence lengths and prove no extrapolation, interpolation, padding, or last-value carry occurs.
- [ ] Include one candidate with no usable history and prove it remains listed with unavailable evidence while the other 11 models/dataset and other datasets are unaffected.
- [ ] Prove each dataset has exactly one OOS all-model comparison and one EM all-model comparison, both containing the exact 12 candidate identities/order.
- [ ] Prove the EM comparison is labelled optimization-only and that changing EM histories alone cannot change `diagnostic_feature_model_winner` or `delta1_univariate_evaluation_champion`.
- [ ] Prove no artifact or metric is emitted into `medoid_multivariate` or `medoid_univariate` Model Metrics namespaces by this rollout.
- [ ] Prove local statistics contain the exact optimization trace evidence and remain byte/hash-consistent with the MLflow `statistics/statistics.json` artifact.
- [ ] Inject one plot/logging failure and prove the affected MLflow/local run is `FAILED` with no false finished incomplete section.
- [ ] Include at least one small real Gaussian-HMM, GMM-HMM and Student-t HMM fit asserting non-empty EM histories reach the renderer without refitting.
- [ ] Required tests are deterministic, hermetic and satisfy the repository coverage gate.

## Delta1 Model Metrics execution graph

```text
M0 prerequisites: PR-152 PR-162
M1: PR-163
M2 parallel after PR-163: PR-164; PR-165 may start after PR-152
M3: PR-166 after PR-163+PR-164
M4: PR-167 after PR-164+PR-165+PR-166+PR-152+PR-162
M5: PR-168 after PR-167
```

PR-165 and PR-164 intentionally own disjoint tracking/model files and may run in parallel after their dependencies are merged. PR-166 is rendering-only. PR-167 is the only PR that composes the new delta1-only evidence into the MLflow hierarchy; PR-168 is proof-only.