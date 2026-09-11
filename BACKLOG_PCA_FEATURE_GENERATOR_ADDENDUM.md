# PCA Feature Generator Backlog Addendum

This addendum defines the implementation backlog for introducing Principal Component Analysis (PCA) as a leakage-safe feature generator in `regime-engine`. PCA-generated component time series are added to the ordinary feature universe and, after generation, are subject to the same downstream selection, freezing, train-only scaling, HMM fitting, walk-forward OOS evaluation, MLflow tracking, and production-serving contracts as all other candidate features.

## Design principles

- PCA is a **feature generator**, not a privileged alternative model input path.
- PCA source features are standardized using TRAIN-only statistics before PCA fitting.
- The retained PCA dimension is the smallest `k` whose cumulative explained-variance ratio is at least **0.90**.
- No additional eigenvalue cutoff (for example, `lambda > 1`) is imposed by default.
- PCA fitting, component-count selection, and transformations must be leakage-safe and reproducible.
- Generated components `PC1 ... PCk` are appended to the feature universe and subsequently compete with raw features under the ordinary feature-selection pipeline.
- PCA features that survive feature selection are normalized by the existing per-fold TRAIN-only `StandardScaler` exactly like retained raw features before HMM fitting.
- PCA metadata, source feature order, standardization parameters, loadings, eigenvalues, explained-variance ratios, cumulative explained variance, and selected `k` must be immutable and auditable.
- Missing or non-finite values must not be silently imputed.

---

## PR-PCA-01 — PCA contracts and deterministic TRAIN-only transformer

**Goal:** Introduce the core PCA domain contracts and deterministic numerical implementation without wiring PCA into feature discovery yet.

**Scope**
- Add a dedicated PCA preprocessing/feature-generation module.
- Define an immutable PCA artifact containing at minimum:
  - ordered PCA source feature names,
  - TRAIN-only source means and standard deviations,
  - component loading matrix,
  - eigenvalues,
  - explained-variance ratios,
  - cumulative explained-variance ratios,
  - retained component count `k`,
  - variance target (`0.90`),
  - deterministic serialization/hash identity.
- Fit source standardization exclusively on the supplied TRAIN matrix.
- Fit PCA exclusively on the standardized TRAIN matrix.
- Select the smallest `k` satisfying cumulative explained variance `>= 0.90`.
- Provide deterministic transform methods for TRAIN, TEST/OOS, replay, and latest observations using the frozen artifact.
- Reject empty matrices, duplicate feature names, NaN/inf values, zero/near-zero variance source features, and dimension/order mismatches.
- Resolve PCA sign indeterminacy deterministically so repeated fits to identical input produce the same serialized component orientation.

**Acceptance criteria**
- Unit tests verify exact TRAIN-only statistics and no use of OOS observations during fitting.
- Unit tests verify `k = min{m : cumulative_EVR[m] >= 0.90}`.
- Repeated fits on identical data produce byte-stable/canonical artifact serialization.
- Transforming OOS data never refits source scaling or PCA.
- Numerical reconstruction checks validate the component scores against the stored loadings.

---

## PR-PCA-02 — PCA source-universe policy and complete-case eligibility

**Goal:** Define exactly which raw features are allowed to feed PCA and make the policy consistent with existing data-quality and no-imputation rules.

**Scope**
- Add configuration for PCA generation, including `enabled`, `variance_target: 0.90`, and source-universe policy.
- Reuse existing feature-quality/eligibility evidence where appropriate rather than bypassing it.
- Define deterministic ordering of PCA source features.
- Apply PCA only to eligible numeric, finite TRAIN observations.
- Preserve the project's complete-case/no-silent-imputation behavior.
- Persist the rejected source features and reasons in the PCA artifact/evaluation evidence.
- Ensure feature discovery and PCA generation use only data available at the applicable TRAIN cutoff.

**Acceptance criteria**
- PCA cannot consume future observations or metadata unavailable at the TRAIN cutoff.
- PCA cannot silently fill missing values.
- Source-feature inclusion/exclusion is reproducible and auditable.
- Tests cover missingness, near-zero variance, duplicate columns, nonnumeric data, and unstable source ordering.

---

## PR-PCA-03 — Generate `PC1 ... PCk` and append them to the feature universe

**Goal:** Make PCA components first-class generated features in the same candidate universe as raw features.

**Scope**
- Materialize deterministic component feature names such as `pca_pc_001`, `pca_pc_002`, ... `pca_pc_k`.
- Attach provenance metadata to every generated PC:
  - PCA artifact/hash,
  - component index,
  - eigenvalue,
  - individual EVR,
  - cumulative EVR,
  - source feature set/hash,
  - fit cutoff.
- Append PCA feature time series to the feature universe rather than sending them directly to the HMM.
- Ensure generated PCs have the same observation clock/length semantics as ordinary features for every transformed row for which the PCA source vector is complete.
- Expose generated PCs through the existing feature contracts/catalog interfaces used downstream.

**Acceptance criteria**
- A TRAIN matrix with `T` usable rows yields `T x k` PCA scores.
- PC features are queryable/identifiable like ordinary candidate features.
- No downstream stage needs a PCA-specific shortcut merely to read component values.
- Provenance allows any PC value to be traced back to the frozen PCA artifact and source universe.

---

## PR-PCA-04 — Integrate PCA features into ordinary feature selection

**Goal:** Ensure generated PCs compete with raw features under the same downstream feature-selection principles.

**Scope**
- Add PCA features to the combined candidate feature universe after PCA generation.
- Do **not** create a privileged rule that automatically retains all PCs.
- Ensure raw semantic-block reduction does not collapse all PCA components into a single artificial `PCA` medoid.
- Refactor the selection flow, if necessary, so semantic raw-feature reduction and PCA generation feed a common downstream selection/correlation/stability stage.
- Apply existing cross-feature redundancy/correlation policy to raw-vs-PC and PC-vs-PC candidates where mathematically appropriate.
- Preserve final feature-dimension constraints and deterministic tie-breaking.
- Record whether each selected feature is raw or PCA-generated.

**Acceptance criteria**
- Multiple PCs can survive when justified; all PCs can also be rejected.
- PC inclusion is determined by the same selection evidence/constraints that govern the common candidate pool, not by explained variance alone.
- Tests cover raw-vs-PC redundancy, PC-vs-PC redundancy, tie-breaking, and final dimension caps.
- Existing raw-only behavior remains unchanged when PCA generation is disabled.

---

## PR-PCA-05 — Walk-forward PCA lifecycle and leakage guards

**Goal:** Make PCA generation fully compatible with the existing walk-forward OOS architecture.

**Scope**
- Define the PCA fit/refit policy explicitly for walk-forward folds.
- At each allowed fit point, estimate PCA source normalization and loadings from TRAIN only.
- Transform TEST/OOS using the frozen TRAIN PCA artifact.
- Prevent `fit_transform` or equivalent operations on combined TRAIN+TEST data.
- Ensure PCA component-count selection from the 90% EVR target is TRAIN-only.
- Add explicit leakage assertions/tests for means, variances, covariance structure, loadings, eigenvalues, and `k`.
- Preserve deterministic component identities/orientation within each frozen PCA artifact.

**Acceptance criteria**
- Perturbing future/OOS observations cannot change the PCA artifact fitted for an earlier TRAIN window.
- OOS scores change when OOS inputs change, but PCA loadings/source scaling do not.
- All fold outputs record the exact PCA artifact/hash used.
- Walk-forward runs are reproducible under identical source data/configuration.

---

## PR-PCA-06 — Apply existing HMM feature normalization to selected PCA features

**Goal:** Treat PCA-generated features exactly like retained raw features once they reach the HMM observation matrix.

**Scope**
- Keep the existing HMM `StandardScaler` behavior unchanged: fit per fold on retained TRAIN observations using population variance (`ddof=0`) and apply the same scaler to TEST/OOS.
- Ensure selected PCA features pass through that same scaler together with selected raw features.
- Clearly separate:
  1. PCA-source standardization used to construct PCs, and
  2. downstream HMM feature standardization applied to the final retained feature matrix.
- Persist both transformation layers in model/evaluation evidence.

**Acceptance criteria**
- A retained PC is not exempt from the normal HMM scaler.
- HMM scaling parameters are estimated only from fold TRAIN rows.
- Serving/replay reconstructs both preprocessing layers exactly.
- Tests prove identical downstream scaling semantics for raw and PCA-generated retained features.

---

## PR-PCA-07 — Model packaging, serving, replay, and latest inference

**Goal:** Make PCA-generated features production-safe and reproducible outside offline evaluation.

**Scope**
- Extend model package/artifact contracts to include the PCA artifact(s) required by a selected model.
- Ensure `latest`, replay, OOS serving, and production refit can regenerate required PC values from raw/source features.
- Validate exact source feature order and PCA artifact hash before inference.
- Fail closed when a required PCA source feature is missing/non-finite rather than silently substituting values.
- Preserve backward compatibility for models with no PCA-generated selected features.

**Acceptance criteria**
- Offline and serving transformations are numerically identical for identical source rows.
- A packaged model containing PC features can be loaded without refitting PCA.
- A model package contains all information necessary to reconstruct selected PCs deterministically.
- Legacy/raw-only models remain loadable and unchanged.

---

## PR-PCA-08 — MLflow tracking, diagnostics, and explainability

**Goal:** Make PCA generation transparent in experiments and production evidence.

**Scope**
- Log PCA configuration and artifact identity to MLflow.
- Log per-component:
  - eigenvalue,
  - EVR,
  - cumulative EVR,
  - retained/not-retained by the 90% generation threshold,
  - selected/not-selected by downstream feature selection.
- Log the PCA loading matrix as an artifact/table.
- Add plots for scree/eigenvalue curve and cumulative explained variance.
- Add a loading-contribution report showing the highest absolute raw-feature loadings per PC.
- Track source-universe size, generated PC count, and final selected PC count.

**Acceptance criteria**
- An MLflow run can answer: which raw features generated each PC, why `k` was chosen, and which PCs ultimately reached the HMM.
- Diagnostic plots/artifacts are reproducible from stored evidence.
- Tracking does not alter model behavior or introduce hidden recomputation.

---

## PR-PCA-09 — Comparative OOS evaluation: raw-only vs raw+PCA universe

**Goal:** Quantify whether PCA feature generation improves regime modeling rather than assuming that it does.

**Scope**
- Add an evaluation path that compares at least:
  - baseline raw-feature universe,
  - raw + PCA-generated feature universe using cumulative EVR `>= 0.90`.
- Keep walk-forward splits, model-family grid, random seeds, feature caps, and evaluation clocks comparable.
- Compare OOS predictive log-likelihood per observation, AIC/BIC where applicable, occupancy stability, state-duration diagnostics, switches/year, entropy/confidence, state-signature drift, and feature-selection stability.
- Report the number and identity of PCA features selected in each candidate/fold or frozen-selection definition.

**Acceptance criteria**
- PCA is not promoted by default merely because it exists.
- Evidence clearly shows whether PCA improves OOS regime quality/stability and at what dimensional cost.
- Results are tracked in MLflow and included in evaluation reports.

---

## PR-PCA-10 — Documentation, configuration, migration, and regression coverage

**Goal:** Finish the PCA feature-generator rollout with explicit contracts and operational documentation.

**Scope**
- Update `ARCHITECTURE.md`, `README.md`, evaluation documentation, profile/config examples, and feature-selection documentation.
- Document the two-stage standardization semantics and the 90% cumulative-EVR generation threshold.
- Document why generated PCs enter the ordinary feature universe instead of bypassing feature selection.
- Document leakage constraints and complete-case behavior.
- Add regression tests ensuring PCA-disabled profiles reproduce prior raw-only outputs.
- Add end-to-end coverage from source rows -> PCA generation -> feature universe -> selection -> HMM scaling -> HMM -> OOS inference -> packaging/serving.

**Acceptance criteria**
- PCA can be enabled/disabled by explicit configuration.
- PCA-disabled results remain backward compatible.
- Documentation matches executable contracts and tests.
- CI covers the complete PCA path and leakage-sensitive invariants.

## Recommended merge order

`PR-PCA-01 -> PR-PCA-02 -> PR-PCA-03 -> PR-PCA-04 -> PR-PCA-05 -> PR-PCA-06 -> PR-PCA-07 -> PR-PCA-08 -> PR-PCA-09 -> PR-PCA-10`

PRs 07 and 08 may proceed in parallel after PRs 01-06 are stable; PR-09 depends on the end-to-end evaluation path and PR-10 should close the rollout.
