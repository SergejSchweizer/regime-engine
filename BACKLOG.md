# Regime Engine — Canonical Backlog

Status date: 2026-09-18

This file is the **single authoritative backlog** for `regime-engine`.
Open and acceptance-pending work is kept at the top. Completed implementation and
historical acceptance work is condensed at the bottom. Separate backlog supplements
must not be created; new findings belong here.

Planning PR IDs such as `PR-449` are repository planning identities used in branch and
commit names. They do not need to equal the numeric GitHub pull-request number.

## Open / active work

### Execution policy

The target architecture is now **PCA-only HMM input** for a potentially very large raw
feature universe. The active plan is ordered so that small implementation changes are made first,
focused QA PRs immediately falsify each development step, and only after the canonical cutover do
the expensive full-system and external tests run.

Global rules for every active planning PR:

- one PR owns one primary behavior or one independent QA responsibility;
- implementation PRs must not be padded with unrelated cleanup;
- QA-only PRs must not add new production behavior;
- every active branch starts from the then-current `origin/main` and is rebased immediately
  before opening/updating its GitHub PR;
- every stochastic/numerical primitive has a source-controlled seed/configuration and canonical
  ordering/sign rules where relevant;
- no Outer-TEST row may influence raw eligibility, standardization, PCA fitting, L selection,
  model-family selection or K-slot fitting;
- external PostgreSQL/MLflow writes are forbidden unless the PR is explicitly marked external;
- complete end-to-end acceptance starts only after all implementation and focused QA work is green.

The intended order is:

```text
PR-448
  -> repository/CI correctness + focused QA
  -> PCA-only statistical contract + focused QA
  -> raw quality/common-support -> scaler/PCA -> PC-prefix L search + focused QA
  -> outer/deployment integration -> evidence -> cutover + focused QA
  -> complete hermetic/high-dimensional/system tests
  -> current-source external audit
  -> external MLflow/registry durability
  -> authorized publication/readback
```

---

### PR-448 — Consolidate and order the canonical backlog

**Status:** ACTIVE — planning/documentation only  
**GitHub:** #439

**Purpose:** keep `BACKLOG.md` as the single authoritative backlog, remove active work that
belongs to the superseded clustering/medoid/teacher architectures, and replace it with the
dependency-ordered PCA-only plan below.

#### Acceptance

- [ ] `BACKLOG.md` is the only active backlog file.
- [ ] Old open medoid/clustering/teacher/prefix-selection planning items are absent from active work.
- [ ] Every active implementation PR has an explicit dependency and one bounded responsibility.
- [ ] Every statistically risky implementation step has a separate focused QA PR.
- [ ] Full hermetic/system tests appear only after implementation/cutover work.
- [ ] External tests/publication appear only after local/system acceptance.
- [ ] Superseded planning is retained only as compact historical traceability, not as active work.
- [ ] No runtime, statistical, PostgreSQL, MLflow, registry or serving behavior changes in PR-448.

---

## Phase 1 — Repository and failure-semantics foundations

### PR-449 — Make 90% coverage the single CI authority

**Type:** implementation / CI correctness  
**Depends on:** PR-448

#### Acceptance

- [ ] `tool.coverage.report.fail_under = 90` remains the sole canonical coverage threshold.
- [ ] Merge and push workflows contain no lower command-line override.
- [ ] Merge and push use equivalent coverage commands and data-file handling.
- [ ] Existing multiprocessing coverage combine behavior is preserved.
- [ ] No cross-job coverage artifact plumbing is introduced.
- [ ] Test selection is unchanged in this PR.
- [ ] A local threshold mutation below 90 fails before merge.
- [ ] Ruff, formatting, strict mypy and the unit lane pass.

### PR-450 — QA: regression-proof the 90% coverage contract

**Type:** QA only  
**Depends on:** PR-449

#### Acceptance

- [ ] Read `pyproject.toml` and prove the threshold is exactly 90.
- [ ] Reject any explicit merge/push `--fail-under` lower than 90.
- [ ] Mutations to 89 and 80 fail QA.
- [ ] Prove merge/push coverage commands are semantically equivalent.
- [ ] Prove no cross-job coverage artifact transfer is required.
- [ ] QA is hermetic and uses no network service or secret.
- [ ] Production code is unchanged.

### PR-451 — Add hermetic integration lanes to merge and push gates

**Type:** implementation / CI completeness  
**Depends on:** PR-450

#### Acceptance

- [ ] Both workflows contain a dedicated integration job parallel to lint/type/unit.
- [ ] Selector is exactly `pytest -n auto tests -m "integration and not slow and not external"`.
- [ ] Python 3.14.7 and the repository-locked dependency bootstrap are used.
- [ ] Terminal gates require lint, type, unit and integration.
- [ ] Failed/cancelled integration makes the terminal gate fail.
- [ ] Slow/external/unrelated E2E tests remain excluded.
- [ ] No NAS PostgreSQL/MLflow credential or alias mutation is used.
- [ ] Unit coverage semantics remain unchanged.

### PR-452 — QA: prove integration gating is mandatory and hermetic

**Type:** QA only  
**Depends on:** PR-451

#### Acceptance

- [ ] Static QA proves both workflows declare the integration job.
- [ ] Terminal gates must depend on and inspect integration status.
- [ ] Selector must include `integration` and exclude `slow` and `external`.
- [ ] Removing integration from either terminal gate fails QA.
- [ ] Replacing the selector with bare `integration` fails QA.
- [ ] The selected integration suite completes without network/secrets.
- [ ] Production code is unchanged.

### PR-453 — Separate statistical invalidity from unexpected software failures

**Type:** implementation / correctness  
**Depends on:** PR-452

#### Acceptance

- [ ] Introduce one explicit recoverable evaluation-invalidity base exception.
- [ ] Only that typed family may become invalid fold/L/K evidence.
- [ ] Generic `RuntimeError`, `KeyError`, `AssertionError`, `TypeError`, unrelated
  `ValueError`, `KeyboardInterrupt` and `SystemExit` cannot become statistical invalidity.
- [ ] Expected data/statistical gates translate to the typed exception at their owning boundary.
- [ ] Process workers preserve failure classification in the parent.
- [ ] Unexpected failures can never merely reduce a valid-fold rate.
- [ ] Persisted recoverable reasons are deterministic and contain no traceback, address,
  credential or raw vector.
- [ ] Statistical thresholds/seeds/ranking semantics are otherwise unchanged.
- [ ] No blanket `except Exception` emits eligibility evidence.

### PR-454 — QA: adversarial failure-classification matrix

**Type:** QA only  
**Depends on:** PR-453

#### Acceptance

- [ ] Typed recoverable invalidity becomes stable invalid evidence at fold/L/K boundaries.
- [ ] Injected runtime/key/assertion/type/unrelated-value errors escape the evidence boundary.
- [ ] One unexpected fold failure cannot be hidden by enough successful folds to pass a rate gate.
- [ ] Serial/process execution classify the same injected failure identically.
- [ ] Recoverable-invalid evidence is completion-order independent.
- [ ] `KeyboardInterrupt` and `SystemExit` propagate.
- [ ] QA adds no new production behavior.

---

## Phase 2 — PCA-only statistical architecture

The new architecture intentionally removes global Spearman clustering, cluster-count selection,
medoids, provisional teacher HMMs, raw-feature regime scores and medoid/backward-elimination
selection from the canonical path.

The target data flow is:

```text
large raw feature universe
    -> TRAIN-only raw quality filter
    -> TRAIN-only complete-case PCA clock
    -> TRAIN-only standardization
    -> TRAIN-only PCA (top D_ref components)
    -> nested prefixes PC1..PCL
    -> causal same-K HMM preservation curve
    -> deterministic L*
    -> one common PC1..PCL* tuple
    -> same feature tuple for K=2,3,4,5 / all final families
    -> strict Outer OOS
```

Raw features are **inputs to PCA only**. A canonical HMM may never consume a raw source feature
directly.

### PR-476 — Define the PCA-only regime-selection contract

**Type:** architecture / contracts  
**Depends on:** PR-454

#### Acceptance

- [ ] Introduce a new explicit evaluation/profile version; historical v4 semantics remain
  unchanged until the controlled cutover.
- [ ] Define `N_raw` = raw catalog size, `N_eligible` = TRAIN-quality-eligible raw count,
  `D_ref` = fixed PCA reference dimension, `L*` = selected PC prefix length, and K separately.
- [ ] Raw quality filtering occurs before PCA on the current TRAIN partition only.
- [ ] Canonical HMM observations contain PCA scores only; raw feature columns are forbidden HMM
  inputs in the new profile.
- [ ] Standardization and PCA fit only on TRAIN rows; TEST is transform-only.
- [ ] PCA uses a single fold-local basis up to `D_ref`; all candidate L values are prefixes of
  that same basis.
- [ ] `D_ref` is derived from the strictest parameter-safety bound of the retained final
  model-family/K universe and is currently expected to resolve to 8; the derivation, not a magic
  hard-coded number, is canonical.
- [ ] Explained-variance ratios are diagnostics only; the historical 0.90 explained-variance
  threshold does not choose L in the new profile.
- [ ] L candidates are exactly `2..D_ref`.
- [ ] K candidates remain exactly 2,3,4,5 and must all consume the same selected PC prefix.
- [ ] No clustering, silhouette, medoid, provisional teacher, raw-feature regime score,
  state-information-ratio, eta-squared winner or medoid elimination participates in the new path.
- [ ] Missing values are never imputed, filled, interpolated or carried.
- [ ] No external service is mutated in this contract PR.

### PR-477 — QA: static PCA-only contract consistency

**Type:** QA only  
**Depends on:** PR-476

#### Acceptance

- [ ] Parse the new contract and prove all required identities and clocks are defined.
- [ ] Reject any contract allowing a raw feature directly into an HMM.
- [ ] Reject any new-profile reference to clustering/medoids/teacher/raw-feature scoring.
- [ ] Reject any L rule based directly on cumulative explained variance.
- [ ] Prove L and K are distinct and K2-K5 require one common PC prefix.
- [ ] Prove TEST data are transform-only for scaler/PCA.
- [ ] Prove no-imputation semantics are explicit.
- [ ] Mutation of `D_ref` derivation, common-prefix rule or TRAIN boundary fails QA.
- [ ] Production runtime is unchanged.

### PR-478 — Build TRAIN-only raw-quality and PCA common-support preflight

**Type:** implementation / input eligibility  
**Depends on:** PR-477

#### Acceptance

- [ ] Apply the existing raw quality rules to raw source features before any scaler/PCA fit.
- [ ] Coverage denominator is the complete TRAIN source-row count, not an instrument's own lifespan.
- [ ] Missing pre-history remains SQL-NULL/`None`; NaN/Inf remains a source-contract failure.
- [ ] Raw features below minimum coverage or variance are excluded with deterministic reasons.
- [ ] Build the PCA complete-case timestamp mask across the eligible raw tuple without imputation.
- [ ] Preserve all source timestamps in evidence; complete-case filtering creates a model clock,
  not a rewritten source dataset.
- [ ] Require at least the canonical minimum model TRAIN observations on the PCA complete-case clock.
- [ ] Require at least `D_ref + 1` complete TRAIN rows and at least `D_ref` non-degenerate raw
  dimensions so the requested PCA rank is mathematically possible.
- [ ] Insufficient common support is typed statistical invalidity, not a software exception.
- [ ] Persist raw catalog hash, eligible tuple/hash, excluded-feature reasons, complete-row count,
  first/last complete timestamp and complete-clock hash.
- [ ] No scaler/PCA/HMM fit occurs when the preflight is invalid.

### PR-479 — QA: raw-history filtering and common-support invariants

**Type:** QA only  
**Depends on:** PR-478

#### Acceptance

- [ ] A late-listed synthetic feature has its pre-listing rows represented as missing against the
  full TRAIN denominator and is rejected when coverage is below the threshold.
- [ ] Shortening the denominator to the feature's own lifespan fails QA.
- [ ] Features with adequate individual coverage but insufficient joint complete-case support make
  the PCA preflight invalid.
- [ ] Removing the sparse offender can make a later independent fixture valid only through the
  documented raw-quality rule; no hidden adaptive pruning is permitted.
- [ ] Internal gaps remain missing and are never forward/back-filled/interpolated.
- [ ] NaN/Inf causes source-contract failure rather than ordinary feature rejection.
- [ ] Row ordering and feature ordering mutations change/fail the expected clock/hash deterministically.
- [ ] QA introduces no production selection logic.

### PR-480 — Implement fold-local standardization and deterministic PCA

**Type:** implementation / numerical transform  
**Depends on:** PR-479

#### Acceptance

- [ ] Fit column means/scales on complete TRAIN rows only.
- [ ] Use the repository-pinned population-variance convention for scaling and reject zero/non-finite
  scale after quality filtering.
- [ ] Transform TRAIN and TEST with the frozen TRAIN scaler; TEST cannot refit or update moments.
- [ ] Fit exactly `D_ref` PCA components on standardized complete TRAIN rows.
- [ ] Use one source-controlled PCA/SVD solver configuration suitable for thousands of columns;
  all solver parameters and seed are profile identity.
- [ ] `whiten=False`; HMM receives ordinary PCA scores, not whitened scores.
- [ ] Canonicalize each component sign: the loading with largest absolute magnitude is positive;
  ties use the smallest canonical raw-feature ordinal.
- [ ] Component order is descending explained variance with deterministic tie handling.
- [ ] TEST PC scores are emitted only for timestamps where every frozen PCA-input raw feature is
  finite; missing TEST vectors remain gap evidence.
- [ ] Persist scaler parameters, PCA loadings, singular/eigen values, explained-variance ratios,
  component-sign evidence and transform identity hashes.
- [ ] Raw feature values are never emitted as HMM observation columns.

### PR-481 — QA: PCA leakage, sign and numerical-reference matrix

**Type:** QA only  
**Depends on:** PR-480

#### Acceptance

- [ ] Mutating future TEST rows cannot alter TRAIN means/scales/loadings/component ordering.
- [ ] A deliberate TEST-refit implementation fails QA.
- [ ] Independently standardize a small fixture and reproduce its covariance/SVD PCA subspace
  without calling the production PCA helper.
- [ ] Component sign flips are canonicalized to identical stored loadings/scores.
- [ ] Exact/near loading-magnitude ties exercise the canonical raw-ordinal sign rule.
- [ ] Repeat runs with the pinned solver seed/config produce the same canonical transform identity.
- [ ] Raw-feature column rescaling before TRAIN standardization leaves the PCA result equivalent
  within the repository's numerical tolerance.
- [ ] Missing TEST input yields no PC observation for that timestamp and never triggers imputation.
- [ ] QA adds no production behavior.

### PR-482 — Derive the safe PCA reference dimension and nested prefix plan

**Type:** implementation / dimension planning  
**Depends on:** PR-481

#### Acceptance

- [ ] Compute the largest common `D_ref` that satisfies parameter-safety bounds for every retained
  final family and K=2,3,4,5 at the canonical minimum TRAIN support.
- [ ] For the current Gaussian/GMM-HMM(m=2)/Student-t full-covariance universe, independently assert
  that the derived bound resolves to the expected current value.
- [ ] If a future model-family contract changes the bound, profile identity changes.
- [ ] Candidate prefixes are exactly `PC1..PC2`, ..., `PC1..PC_D_ref`.
- [ ] Prefixes are nested and share one fold-local PCA basis; PCA is never refit separately per L.
- [ ] Every prefix preserves canonical PC order.
- [ ] If a fold cannot produce `D_ref` valid components it is statistically invalid before HMM fit.
- [ ] Raw PLL/AIC/BIC are not compared across different L dimensions.
- [ ] Persist `D_ref`, its parameter-count proof, candidate L tuple and prefix hashes.

### PR-483 — QA: parameter-safety and prefix-plan oracle

**Type:** QA only  
**Depends on:** PR-482

#### Acceptance

- [ ] Independent formulas recompute Gaussian and GMM-HMM free-parameter counts for every K/L.
- [ ] The strictest family/K combination determines `D_ref`.
- [ ] Mutation allowing an unsafe L fails QA.
- [ ] Mutation refitting PCA independently per prefix fails QA.
- [ ] Every L prefix is a literal prefix of the same component identity tuple.
- [ ] Candidate order is deterministic under randomized task completion.
- [ ] A rank-deficient fold is rejected before any HMM fit.
- [ ] QA adds no production behavior.

### PR-484 — Build causal full-PCA K=2..5 reference evidence

**Type:** implementation / inner walk-forward HMM evidence  
**Depends on:** PR-483

#### Acceptance

- [ ] For every inner fold, run raw-quality/common-support/scaler/PCA strictly on inner TRAIN.
- [ ] Fit Gaussian full-covariance HMM K=2,3,4,5 on `PC1..PC_D_ref`.
- [ ] All K values in one fold consume the same scaler/PCA identity and component order.
- [ ] Reuse deterministic multistart, covariance, occupancy and validity gates.
- [ ] Persist only causal inner-TEST filtered probabilities as reference arrays.
- [ ] Smoothed probabilities and full-sample Viterbi labels are forbidden.
- [ ] Persist K, fold, transformed timestamps, PCA identity, component tuple, source identity and
  posterior-evidence hash.
- [ ] Reference HMMs do not select raw features or alter the PCA basis.
- [ ] Serial/process execution produces equivalent canonical reference evidence.

### PR-485 — QA: causal full-PCA reference isolation

**Type:** QA only  
**Depends on:** PR-484

#### Acceptance

- [ ] Future inner-TEST mutation cannot alter earlier filtered probabilities.
- [ ] Smoothed-posterior substitution fails QA.
- [ ] Viterbi-label substitution fails QA.
- [ ] K2-K5 bind the exact same fold-local PCA identity.
- [ ] Raw feature order/eligible-set/PCA identity mutation invalidates stale reference reuse.
- [ ] Timestamp mismatch fails closed.
- [ ] Real-HMM serial/process fixtures produce equivalent reference evidence under canonical
  numerical tolerances.
- [ ] QA introduces no new selection rule.

### PR-486 — Score PCA prefixes by causal same-K regime preservation

**Type:** implementation / L-search evidence  
**Depends on:** PR-485

For each L and K within the same inner fold:

```text
nmi(K,f,L) =
    soft_NMI(
        full_PC1..PC_Dref filtered posterior,
        PC1..PC_L filtered posterior
    )

q_K(L) = median over valid inner folds f of nmi(K,f,L)
Q_L    = min over K in {2,3,4,5} q_K(L)
```

#### Acceptance

- [ ] Fit each reduced prefix L=2..D_ref-1 with Gaussian K2-K5 on the same fold-local PCA scores.
- [ ] `L=D_ref` is the reference and has `Q_Dref=1.0` by identity.
- [ ] Compare only same-K full-vs-prefix filtered posteriors on exact shared timestamps.
- [ ] Reuse canonical soft NMI with finite/entropy/shared-support gates.
- [ ] Shared support must satisfy the canonical minimum.
- [ ] Every K must satisfy the canonical inner valid-fold-rate hard gate for L to be eligible.
- [ ] `q_K(L)` is the median of valid fold-local NMI values.
- [ ] `Q_L` is the minimum across K2-K5; mean/max aggregation is forbidden.
- [ ] Raw likelihood/AIC/BIC are never used for cross-L ranking.
- [ ] Persist every per-fold/per-K NMI, validity reason, `q_K`, `Q_L`, L and PCA-prefix hash.
- [ ] Independent L/K tasks may run in processes, but completion order cannot alter evidence.

### PR-487 — QA: independent PCA-prefix preservation oracle

**Type:** QA only  
**Depends on:** PR-486

#### Acceptance

- [ ] Independent soft-NMI math reproduces every golden full-vs-prefix value.
- [ ] Independent aggregation reproduces each `q_K` median and `Q_L=min_K(q_K)`.
- [ ] Mutation from minimum-K aggregation to mean/maximum fails QA.
- [ ] One K below the validity gate makes that L ineligible.
- [ ] State-label permutations leave NMI, `q_K` and `Q_L` unchanged.
- [ ] Moving/reference-per-L PCA bases fail QA; comparisons require one common fold-local basis.
- [ ] Randomized worker completion produces identical evidence hashes.
- [ ] QA adds no production behavior.

### PR-488 — Select L* with a deterministic PCA-prefix elbow

**Type:** implementation / model-dimension selection  
**Depends on:** PR-487

#### Acceptance

- [ ] Build the raw preservation curve `Q_L` for L=2..D_ref with `Q_Dref=1.0`.
- [ ] Persist the raw curve unchanged for audit.
- [ ] Build monotone decision curve `Qhat_L=max(Q_j for j<=L)`.
- [ ] If the smallest eligible L is already 1.0 within `1e-12`, select the smallest such L.
- [ ] Otherwise normalize eligible L and `Qhat_L` endpoints to x,y in [0,1].
- [ ] Compute the interior elbow score `E_L=(y_L-x_L)/sqrt(2)`.
- [ ] Maximum E wins; ties within `1e-12` choose the smaller L.
- [ ] If fewer than three eligible L values exist and no exact lower-L identity exists, choose the
  largest eligible L conservatively.
- [ ] Persist raw/monotone curves, normalized coordinates, E values, selected L* and final PC tuple.
- [ ] Outer TEST data cannot influence the curve or elbow.
- [ ] One L* is shared by K2-K5 and all retained final model families.

### PR-489 — QA: PCA-elbow boundary and mutation matrix

**Type:** QA only  
**Depends on:** PR-488

#### Acceptance

- [ ] Golden curves cover clear elbow, flat curve, noisy/non-monotone raw curve and insufficient
  eligible-L cases.
- [ ] Independently recompute the monotone envelope and normalized elbow score.
- [ ] Exact lower-L identity selects the smallest equivalent L.
- [ ] Tie within `1e-12` selects the smaller L.
- [ ] Removing the monotone envelope fails QA.
- [ ] Replacing the deterministic elbow with a visual/manual/nondeterministic choice fails QA.
- [ ] Future Outer-TEST mutation leaves L* unchanged.
- [ ] QA adds no production behavior.

---

## Phase 3 — Integrate PCA-only selection, evidence and cutover

### PR-490 — Integrate PCA-only selection into outer validation and deployment

**Type:** implementation / orchestration  
**Depends on:** PR-489

#### Acceptance

- [ ] Each Outer-TRAIN selection uses inner folds that independently run raw quality ->
  common-support preflight -> scaler -> PCA -> L-prefix evaluation.
- [ ] Freeze L* before Outer TEST.
- [ ] Refit raw quality/scaler/PCA from scratch on complete Outer TRAIN after L* is selected.
- [ ] Fit every final K/family candidate only on `PC1..PC_L*`.
- [ ] K2-K5 and Gaussian/GMM-HMM/Student-t candidates for one selection identity bind the same
  Outer-TRAIN scaler/PCA identity and final PC tuple.
- [ ] No final candidate receives raw source columns.
- [ ] Transform Outer TEST using the frozen Outer-TRAIN scaler/PCA only.
- [ ] Missing Outer-TEST raw vector -> no PC observation at that timestamp; preserve gap semantics.
- [ ] Deployment selection reruns the full TRAIN-only procedure through the deployment cutoff;
  it never copies the last Outer-fold PCA or L*.
- [ ] Final package identity binds source build, raw eligible set, scaler, PCA loadings, L*, PC tuple,
  K, family and cutoff.
- [ ] Serial/process orchestration yields equivalent canonical selection identities.

### PR-491 — QA: orchestration leakage and raw-input exclusion

**Type:** QA only  
**Depends on:** PR-490

#### Acceptance

- [ ] Mutating Outer TEST raw rows cannot alter Outer-TRAIN raw eligibility, scaler, PCA or L*.
- [ ] A spy proves final HMM fit calls receive only PC columns.
- [ ] Any raw-feature HMM input in the new profile fails QA.
- [ ] K2-K5/family candidates bind one identical final PC tuple/hash.
- [ ] Model-family changes cannot trigger PCA or L reselection.
- [ ] Deployment selection executes anew at deployment cutoff.
- [ ] Serial/process runs yield equivalent selection/package identities.
- [ ] Injected recoverable vs unexpected failures follow PR-453 semantics.
- [ ] QA adds no production behavior.

### PR-492 — Project PCA-only lineage and L-selection evidence into MLflow

**Type:** implementation / observability  
**Depends on:** PR-491

#### Acceptance

- [ ] Log raw catalog/eligible counts and quality exclusion summaries.
- [ ] Store the complete raw eligible tuple/hash as an artifact rather than exploding thousands of
  feature names into MLflow parameters.
- [ ] Store scaler means/scales and PCA loadings/component metadata as immutable artifacts with
  hashes and dimensions.
- [ ] Log explained-variance ratio and cumulative explained variance per PC as diagnostics only.
- [ ] Log `D_ref`, parameter-safety proof, per-K/per-fold prefix NMI, `q_K(L)`, `Q_L`,
  monotone curve, elbow score and L*.
- [ ] Log one common final PC tuple/hash and cross-check it across K2-K5.
- [ ] New-profile runs emit no clustering/silhouette/medoid/teacher/raw-feature-regime-score
  decision artifacts.
- [ ] Metric/artifact catalog versioning distinguishes the PCA-only profile from historical v4.
- [ ] Evidence serialization remains deterministic and completion-order independent.
- [ ] Hermetic tests require no NAS MLflow endpoint.

### PR-493 — QA: PCA/MLflow evidence completeness

**Type:** QA only  
**Depends on:** PR-492

#### Acceptance

- [ ] Independently enumerate every mandatory PCA/L/K evidence domain.
- [ ] Missing scaler/PCA/eligible-set/L evidence makes a run incomplete.
- [ ] K2-K5 final PC hashes must be identical for one selection identity.
- [ ] Historical clustering/teacher artifacts cannot satisfy new-profile completeness.
- [ ] New-profile runs containing clustering/medoid/teacher decision artifacts fail QA.
- [ ] Metric values and artifact hashes are independent of emission order.
- [ ] Local FileStore fixtures reproduce the complete evidence matrix without network access.
- [ ] QA adds no production selection behavior.

### PR-494 — Cut over canonical Xetra evaluation to PCA-only HMM inputs

**Type:** implementation / controlled migration  
**Depends on:** PR-493

#### Acceptance

- [ ] Promote the PCA-only profile/evaluation version to the sole canonical Xetra selection path.
- [ ] Retire canonical runtime use of absolute-Spearman clustering, silhouette M*, medoids,
  provisional teacher, raw-feature regime scoring and teacher/medoid prefix selection.
- [ ] No compatibility flag or silent fallback can reactivate the old statistical architecture.
- [ ] Historical v4 runs/packages remain historical evidence and are never relabelled.
- [ ] Update README, EVALUATION, lifecycle/source documentation and public identity constants.
- [ ] Canonical documentation shows raw quality -> scaler/PCA -> PC-prefix L* -> HMM K2-K5.
- [ ] Existing source lineage, no-imputation, non-resumable full-run, external service and manual
  champion-promotion boundaries remain explicit.
- [ ] No production alias mutation occurs in this cutover PR.

### PR-495 — QA: zero-legacy PCA-only cutover proof

**Type:** QA only  
**Depends on:** PR-494

#### Acceptance

- [ ] Public Xetra entry points resolve only the PCA-only profile/evaluation identity.
- [ ] Static/runtime scans reject canonical calls to clustering/medoid/teacher/raw-feature-scoring
  modules.
- [ ] HMM input schemas in the new profile contain only canonical PC names.
- [ ] Historical v4 package/evidence readers remain distinguishable and read-only where required.
- [ ] Documentation and runtime constants agree on the new canonical version.
- [ ] K2-K5 enforce one final PC tuple per selection identity.
- [ ] No external service is mutated.
- [ ] QA/documentation corrections only; no new selection behavior.

---

## Phase 4 — Full local/system acceptance after implementation is complete

No PR in this phase is acceptance evidence until PR-495 is green.

### PR-496 — Full hermetic PCA-only end-to-end acceptance

**Type:** QA / complete-system acceptance  
**Depends on:** PR-495

#### Acceptance

- [ ] Run a complete hermetic evaluation with real PCA and real HMM fits from immutable source
  snapshot through raw quality, scaler/PCA, L search, final K/family grids and Outer TEST.
- [ ] Independently reproduce the raw eligible set, complete-case clock, scaler moments, PCA
  loadings/subspace, prefix preservation curve, elbow and L*.
- [ ] Prove future Outer-TEST mutation cannot alter any TRAIN-side decision.
- [ ] Prove every final K2-K5 slot uses the exact same PCA identity and PC tuple/hash.
- [ ] Prove deployment reruns raw quality/scaler/PCA/L selection at deployment cutoff.
- [ ] Verify serial/process canonical parity for selection/model/evidence identities.
- [ ] Verify no clustering/medoid/teacher decision path is executed.
- [ ] Run lint, format, strict mypy, unit and hermetic integration gates.
- [ ] No external PostgreSQL/MLflow mutation.

### PR-497 — High-dimensional PCA scale and resource acceptance

**Type:** QA / performance and bounded-resource acceptance  
**Depends on:** PR-496

#### Acceptance

- [ ] Exercise synthetic universes of at least 2,000 and 5,000 raw features with the canonical
  TRAIN row counts.
- [ ] Prove no NxN raw-feature distance/correlation matrix is allocated by the PCA-only path.
- [ ] Record wall time, peak parent/child RSS, effective worker count and PCA solver timing.
- [ ] Prove memory grows with the source matrix/PCA workspace rather than O(N_raw^2) clustering
  artifacts.
- [ ] Verify PCA produces exactly D_ref components and identical L* semantics at scale.
- [ ] Repeat the same fixture and prove canonical statistical identity is stable.
- [ ] Explicit one-worker and adaptive-worker runs are statistically equivalent.
- [ ] No OOM fallback, feature truncation or silent sampling is permitted.
- [ ] This PR may tune only non-statistical scheduling/resource defaults; any statistical change
  requires a new implementation PR.

### PR-498 — Production K-slot callback acceptance on PCA-only inputs

**Type:** QA / integration acceptance  
**Depends on:** PR-497

#### Acceptance

- [ ] Exercise K2,K3,K4,K5 through the real production outer-validation callback.
- [ ] All four slots consume the exact same final PC tuple/hash.
- [ ] Preserve per-K eligibility gates and deterministic process assembly.
- [ ] Prove serial/process parity through the production callback.
- [ ] One ineligible K cannot contaminate another K slot.
- [ ] No callback can rerun raw quality/PCA/L selection after the tuple is frozen.
- [ ] No external service is required.

### PR-499 — Deployment/refit/package acceptance for every K slot

**Type:** QA / lifecycle acceptance  
**Depends on:** PR-498

#### Acceptance

- [ ] Exercise real deployment selection through final scaler/PCA refit and package assembly.
- [ ] Every eligible K package binds the same deployment PCA identity and final PC tuple/hash.
- [ ] Bind source build, deployment cutoff, raw eligible hash, scaler/PCA hashes, L*, K and family.
- [ ] Prove deployment reruns selection rather than copying the last validation-fold transform.
- [ ] Prove package round-trip restores scaler, PCA loadings, PC order and HMM exactly.
- [ ] Ineligible K creates no package.
- [ ] No external publication occurs.

### PR-500 — Model Metrics and diagnostic projection acceptance

**Type:** QA / observability acceptance  
**Depends on:** PR-499

#### Acceptance

- [ ] Project the complete metric/artifact/plot matrix for every eligible K slot.
- [ ] Include raw-quality, PCA explained-variance/loadings, L-curve/elbow and final PC lineage.
- [ ] Fail closed on incomplete lineage or missing required projection domains.
- [ ] Preserve serial/process completion-order parity.
- [ ] Unavailable K slots are explicit and have no alias/package artifact.
- [ ] Historical v4 clustering/teacher evidence cannot satisfy PCA-only completeness.
- [ ] No external publication occurs.

### PR-501 — Final independent mathematical acceptance closure

**Type:** QA / independent oracle  
**Depends on:** PR-500

#### Acceptance

- [ ] Independent math code recomputes scaling, PCA reference quantities on small golden fixtures,
  same-K soft NMI aggregation and elbow selection without production selection helpers.
- [ ] Cover sign permutations, state-label permutations, rank deficiency, missing-row clocks,
  L ties and invalid K cases.
- [ ] Recompute every final PC tuple and K-slot identity for the golden fixture.
- [ ] Mutation tests fail when common-PC, min-over-K, no-imputation, elbow or TRAIN-only rules change.
- [ ] Prove serial/process canonical parity.
- [ ] No external service is required.

### PR-502 — Production-shaped full PCA-only E2E closure

**Type:** QA / final local production-lineage acceptance  
**Depends on:** PR-501

#### Acceptance

- [ ] Run the complete Gaussian/GMM-HMM/Student-t K2-K5 path against a production-shaped immutable
  source fixture with thousands of raw features.
- [ ] Preserve one common PC prefix across K and families.
- [ ] Produce deployment packages, metrics and plots for eligible slots.
- [ ] Prove ineligible-slot fail-closed behavior and future-row invariance.
- [ ] Prove process/serial and independent-process identity parity.
- [ ] Preserve champion/challenger semantics without touching an external registry.
- [ ] This is the final local gate before current-source/external work.

---

## Phase 5 — Current-source and external acceptance

### PR-503 — Full current-Xetra PCA-only computation and independent audit

**Type:** external acceptance  
**Depends on:** PR-502 and a production-eligible upstream source snapshot

#### Acceptance

- [ ] Capture one immutable current source snapshot and record exact catalog/data lineage.
- [ ] Run the complete canonical PCA-only evaluation from raw quality through Outer OOS.
- [ ] Record exact raw-eligible, complete-clock, scaler, PCA, L*, final-PC and K-slot identities.
- [ ] Produce the complete independent math/audit dossier for every required fold.
- [ ] Independently reproduce all required small-matrix PCA and L-selection checks from persisted
  artifacts without trusting production helper results.
- [ ] Prove K2-K5 share the same final PC hash within each selection identity.
- [ ] Reconcile source rows, timestamps, package/model identities and audit hashes exactly.
- [ ] Record zero unexplained numerical-audit errors under the new contract.
- [ ] Historical v4/raw+PCA/teacher runs are not accepted as evidence.
- [ ] No model publication occurs in this PR.

### PR-504 — External MLflow PCA-only evidence completeness proof

**Type:** external QA  
**Depends on:** PR-503 and explicit namespace decision

#### Acceptance

- [ ] Resolve historical experiment namespace handling by explicit operator decision; no silent
  deletion.
- [ ] Verify the fresh PR-503 run against the PCA-only metric/artifact catalog.
- [ ] Verify eligible-set, scaler, loadings, explained-variance, L/elbow, K-slot and plot artifacts.
- [ ] Verify exact dataset/model/PC lineage and metric identity.
- [ ] Verify artifact hashes, sizes and freshness against the same evaluation identity.
- [ ] Prove historical clustering/teacher runs cannot satisfy PCA-only completeness.
- [ ] No production alias promotion occurs.

### PR-505 — External registry and compare-and-swap durability

**Type:** external QA / registry safety  
**Depends on:** PR-504

#### Acceptance

- [ ] Verify the complete registry naming/version matrix for every eligible K slot.
- [ ] Every K version binds the same PCA-only final PC hash for its selection identity.
- [ ] Preserve immutable registration and audited compare-and-swap promotion semantics.
- [ ] Failed registration/promotion cannot mutate existing aliases.
- [ ] Process-kill/retry leaves no partial alias/version state.
- [ ] Retry is idempotent for the same immutable package identity.
- [ ] Conflicting package identity is rejected rather than overwritten.
- [ ] Ineligible K creates no version or alias.
- [ ] Historical v4 versions cannot masquerade as PCA-only versions.

### PR-506 — Authorized external publication and readback

**Type:** final external gate  
**Depends on:** PR-505

#### Acceptance

- [ ] PR-503, PR-504 and PR-505 are green first.
- [ ] Publish only explicitly authorized PCA-only packages.
- [ ] Read back registered model names, versions, aliases, source lineage, scaler/PCA hashes,
  final PC tuple/hash, L*, K, family and package digest from NAS MLflow.
- [ ] Verify every published K slot binds the expected common final PC tuple/hash.
- [ ] Verify publication is idempotent for the same package identity.
- [ ] Failed readback/promotion leaves prior alias state unchanged.
- [ ] Champion promotion remains an explicit operator action.
- [ ] Archive final external acceptance evidence and exact MLflow identities.

---

### Active dependency graph

```text
PR-448
  -> 449 -> 450 -> 451 -> 452 -> 453 -> 454
  -> 476 -> 477 -> 478 -> 479 -> 480 -> 481 -> 482 -> 483
  -> 484 -> 485 -> 486 -> 487 -> 488 -> 489
  -> 490 -> 491 -> 492 -> 493 -> 494 -> 495
  -> 496 -> 497 -> 498 -> 499 -> 500 -> 501 -> 502
  -> 503 -> 504 -> 505 -> 506
```

Planning PRs PR-232, PR-250, PR-423..PR-430, PR-455..PR-475 are superseded by the PCA-only
plan and are not active execution items.

---

## Current architectural decisions and non-goals

- **Target selection redesign:** PR-476–PR-506 replace the raw+PCA/clustering/medoid/teacher architecture with a PCA-only HMM input path. Raw features are quality-filtered on TRAIN, standardized on TRAIN, compressed into a fixed safe reference PCA basis, and only PC prefixes may enter HMMs. L* is selected from causal same-K OOS regime-preservation evidence; K=2..5 share one final PC prefix. The current v4 runtime remains historical/current only until the controlled cutover in PR-494.
- Only **Xetra v4** is active. Legacy v1-v3 evaluation/package/serving compatibility is
  retired; Git history is the archive.
- Current v4 still uses the historical raw-plus-generated-PCA universe until PR-494. The target profile is PCA-only at the HMM boundary: raw features are PCA inputs, never HMM inputs.
- The target path runs raw TRAIN quality/common-support checks before scaler/PCA fit, retains strict no-imputation semantics, and treats insufficient joint complete-case support as typed statistical invalidity.
- Production packages are published to the external NAS MLflow service at
  `http://10.10.1.3:5000`; local FileStore registration is not a production fallback.
- Feature PostgreSQL and MLflow are external dependencies; Docker/Compose services do
  not belong to this repository.
- CPU-bound production work uses available affinity/cgroup-aware capacity by default;
  explicit worker counts are operator/test overrides.
- The full Xetra v4 evaluator is intentionally **non-resumable**. Interrupted full runs
  restart from the beginning. Retry/resume semantics remain limited to the dedicated
  metric-export harness.
- State identities are fold/model-version local according to the v4 contract; semantic
  labels must not leak into discovery or statistical ranking.

## Current external state snapshot

As observed on 2026-09-16:

- source relation: `macro_loader.macro_features_daily`
- source build: `20260915T212921Z`
- schema version: 6
- feature version: 5
- source rows: 16,768 through 2026-09-04
- external feature read-only smoke: passing
- MLflow health: `200 OK`
- experiment `macro-regime-evaluation`: 768 historical runs
- visible LoggedModels: 0
- registered model versions: 0
- deleted-LoggedModel inventory: unavailable through HTTP RestStore
- no current full evaluation running
- no production publication/alias mutation was performed by the failed 2026-09-16 run

---

# Completed / historical work

The entries below are intentionally condensed. Detailed implementation history remains
available in Git history and merged GitHub PR discussions; completed work should not
dominate the active backlog.

| Planning / GitHub PRs | Final state | Condensed result |
|---|---|---|
| PR-210–PR-230 | COMPLETE | Core global-regime v4 contracts, discovery, HMM evaluation, ranking, serving and lifecycle foundations implemented. |
| PR-231 / GitHub #361 | ACCEPTANCE COMPLETE | Four local deterministic proof bundles accepted; golden digest and independent likelihood evidence closed. |
| PR-233–PR-241 | IMPLEMENTED | Supporting source/runtime/release components merged; any remaining external release proof is delegated to PR-232/PR-250. |
| PR-242–PR-244 / GitHub #408 | ACCEPTANCE COMPLETE | Snapshot/process-kill/filesystem crash boundaries and three-family multistart interruption/golden parity covered. |
| PR-245–PR-270 | COMPLETE | Process-parallel evaluation, metric families, stage/checkpoint hardening, diagnostics, telemetry and audit handoff merged; external proof delegated where applicable. |
| GitHub #352, #354, #356 | MERGED | Hermetic proof, lease/race, retry, process and tracking hardening merged. |
| PR-329, PR-330 | COMPLETE | Process-parallel Spearman pair and hierarchy-cut silhouette kernels. |
| PR-333–PR-336, PR-338, PR-339, PR-342, PR-343 | COMPLETE | Plot/process/audit/MLflow/snapshot/checkpoint hardening merged. |
| PR-369–PR-374 | COMPLETE | CPU candidate lanes, strict MLflow completeness contract, PCA-bound audit, mandatory PCA and final-vs-teacher OOS evidence merged. |
| PR-378–PR-382 | COMPLETE | Removed stale PCA opt-in semantics, hardened coverage plumbing, plot/prefix parallelism and pre-PCA source/model-clock preflight. |
| PR-384, PR-386, PR-387 | COMPLETE | Removed GIL-bound multistart fallback; tightened current-Xetra audit; fail-closed deleted-LoggedModel visibility. |
| PR-390, PR-391, PR-393, PR-395–PR-398 | COMPLETE | Parallel tracking preparation, process-safe callback enforcement, PCA docs/profile contract, serial-prefix completeness and non-resumable full-run contract cleanup. |
| GitHub #398, #399 | MERGED | Strict math-audit dossier identity and exact MLflow metric-catalog/IEEE-754 verification. |
| PR-401–PR-404 | COMPLETE | MLflow acceptance, lifecycle/refit/alias QA, fold-local process parallelism and private child-fold result contract. |
| PR-406 / GitHub #408 | COMPLETE | Spawned process-kill/filesystem crash boundary plus three-family multistart interruption acceptance. |
| PR-408 / GitHub #406 | COMPLETE | Positive all-nine-family Model Metrics plot matrix and order-independent hashes. |
| PR-413 / GitHub #412 | COMPLETE | Deployment/lifecycle/package source identity, cutoff, alias immutability and schema acceptance. |
| PR-414 / GitHub #413 | COMPLETE | MLflow evidence artifact freshness/hash binding and Xetra source-identity cross-binding. |
| PR-420–PR-422 / implementation #418 | ACCEPTANCE COMPLETE | K-specific selection, TRAIN-only K orchestration and exact same-K Gaussian/GMM/Student-t family ranking accepted locally. |
| PR-431 / GitHub #415 | COMPLETE | Dimension-independent `cross_k_score.v1`, K=2..5 Model Metrics projection and bounded process-parallel scoring. |
| PR-442 / GitHub #430 | COMPLETE | Cross-K formula, eligibility, non-finite rejection, canonical hash and projection acceptance hardened. |
| PR-445–PR-447 | COMPLETE | Hermetic closure of K-slot selection/fixed-K family acceptance; PR-447 merged as GitHub #438. |
| PCA PR-255–PR-262 / GitHub #303–#311 | COMPLETE | Mandatory PCA rollout and associated metric/QA contracts closed. |

### Closed without merge / superseded

- Active planning PRs PR-232, PR-250, PR-423–PR-430 and PR-455–PR-475 were superseded by the PCA-only architecture plan PR-476–PR-506; they are retained only in Git/GitHub history and must not be implemented from this backlog.
- GitHub #277, #284 and #317 were closed without merge and are superseded by later
  merged work.
- Draft planning IDs PR-186–PR-206 are superseded and must not be implemented.
- Historical requirements for v1-v3 compatibility or full-run computation-position
  resume are superseded by the architectural decisions above.
