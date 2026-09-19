# Regime Engine — Canonical Backlog

Status date: 2026-09-19

This file is the **single authoritative backlog** for `regime-engine`.
Open and acceptance-pending work is kept at the top. Completed implementation and
historical acceptance work is condensed at the bottom. Separate backlog supplements
must not be created; new findings belong here.

Planning PR IDs such as `PR-449` are repository planning identities used in branch and
commit names. They do not need to equal the numeric GitHub pull-request number.

## Open / active work

### Execution policy

The target architecture is now a **scalable fold-local HMM feature-selection pipeline** for a
potentially multi-thousand-feature universe. Core features remain directly interpretable candidates;
generated transformations are compressed by family-local PCA; all surviving core features and family
PCs then pass deterministic global correlation-leader pruning before HMM-based SFFS selection and
final ablation. Feature-selection metadata is persisted locally in DuckDB, while every major stage is
auditable in MLflow through deterministic metrics, tables and plots. The plan is ordered so that small
implementation changes land first, focused QA PRs immediately falsify each risky step, and only after
the canonical cutover do expensive full-system and external tests run.

Global rules for every active planning PR:

- one PR owns one primary behavior or one independent QA responsibility;
- implementation PRs must not be padded with unrelated cleanup;
- QA-only PRs must not add new production behavior;
- every active branch starts from the then-current `origin/main` and is rebased immediately
  before opening/updating its GitHub PR;
- every stochastic/numerical primitive has a source-controlled seed/configuration and canonical
  ordering/sign rules where relevant;
- no Outer-TEST row may influence quality filtering, provenance classification, standardization,
  family PCA, correlation pruning, SFFS, ablation, model-family selection or K-slot fitting;
- external PostgreSQL/MLflow writes are forbidden unless the PR is explicitly marked external;
- complete end-to-end acceptance starts only after all implementation and focused QA work is green.

The intended order is:

~~~text
PR-448
  -> repository/CI correctness + focused QA
  -> calendar-month refit/model-clock implementation + focused QA
  -> canonical feature-selection contract + local metadata foundation
  -> provenance + TRAIN-only quality + family PCA + focused QA
  -> global correlation-leader pruning + MLflow auditability + focused QA
  -> parallel HMM SFFS + ablation + cumulative statistics + focused QA
  -> outer/deployment integration -> cutover -> zero-legacy QA
  -> complete hermetic/high-dimensional/system tests
  -> current-source external audit
  -> external MLflow/registry durability
  -> authorized publication/readback
~~~

---

### PR-448 — Consolidate and order the canonical backlog

**Status:** ACTIVE — planning/documentation only  
**GitHub:** #439

**Purpose:** keep BACKLOG.md as the single authoritative backlog, remove active work that
belongs to superseded clustering/medoid/teacher and PCA-only-prefix architectures, and replace it
with the dependency-ordered scalable feature-selection plan below.

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

## Phase 2 — Calendar-month refit and evaluation clock

The canonical evaluation clock must mirror the intended live pipeline: the complete adaptive
pipeline is refit **once after each calendar month closes**, then kept frozen for the immediately
following calendar month. There is no intramonth refit.

Canonical month semantics:

```text
month m closes
    -> TRAIN uses all admissible history through the last source observation in month m
    -> quality / family PCA / correlation pruning / SFFS / ablation / final HMM selection are rerun
    -> resulting frozen model becomes effective for month m+1
    -> every observation in month m+1 is causal OOS under that frozen model
    -> next refit occurs only after month m+1 closes
```

Calendar membership is determined from `timestamp_m1` in `Europe/Berlin`. A validation TEST
month must be a complete closed calendar month relative to the evaluation cutoff; the final partial
month is never used as TEST evidence.

### PR-507 — Implement the canonical calendar-month model clock

**Type:** implementation / temporal contract  
**Depends on:** PR-454

#### Acceptance

- [ ] Add one canonical `Europe/Berlin` calendar-month clock shared by outer evaluation, inner
  selection and deployment/refit orchestration.
- [ ] Convert each `timestamp_m1` to the canonical timezone only for month membership; preserve the
  original timezone-aware timestamp as data/evidence.
- [ ] Define a refit boundary as the last available source observation belonging to a closed
  calendar month.
- [ ] Outer TRAIN is expanding and must contain at least the existing canonical minimum TRAIN
  history before the first eligible refit boundary.
- [ ] Outer TEST is exactly the immediately following **complete calendar month**, not a fixed
  63-observation block.
- [ ] Outer STEP is exactly one calendar month.
- [ ] The final partial calendar month at the evaluation cutoff produces no Outer TEST fold.
- [ ] Inner selection mirrors the same cadence: expanding inner TRAIN, refit at closed month-end,
  TEST on the immediately following complete calendar month, step one month.
- [ ] Preserve the existing minimum inner TRAIN history; retire fixed 63-row/42-row TEST semantics
  from the new profile because monthly TEST length is calendar-driven.
- [ ] A monthly TEST fold is valid only when its source month is complete and the downstream
  transform/model evidence is non-empty and passes the existing finite/entropy/support/model gates;
  no synthetic rows may be added to reach a row count.
- [ ] No TRAIN observation may have a timestamp later than its fold's month-end cutoff.
- [ ] Persist for every fold: `train_through_month`, exact `train_cutoff_timestamp`,
  `test_calendar_month`, first/last TEST timestamp, TEST source-row count and a month-clock hash.
- [ ] Two folds may never overlap in TEST timestamps.
- [ ] Missing calendar months remain explicit gaps; the evaluator must not silently test the next
  non-empty month as though it were the immediately following month.
- [ ] DST transitions cannot change month assignment or duplicate/drop a source timestamp.
- [ ] This PR changes only clock construction/contracts; it does not change PCA/HMM math.

### PR-508 — QA: month-boundary, leakage and live-cadence clock matrix

**Type:** QA only  
**Depends on:** PR-507

#### Acceptance

- [ ] Golden fixtures cover 28-, 29-, 30- and 31-day months and year rollover December -> January.
- [ ] Cover CET/CEST transitions and prove every timestamp belongs to exactly one local calendar
  month.
- [ ] Prove a January refit uses data only through January month-end and February is the complete
  OOS TEST month.
- [ ] Prove the next fold refits through February month-end and tests March.
- [ ] A snapshot cut off in the middle of September may validate through August at most; partial
  September is never emitted as a TEST fold.
- [ ] Mutation of any future-month row cannot alter an earlier fold's TRAIN cutoff or membership.
- [ ] A fixed 63-row/63-step implementation fails QA.
- [ ] A 21-trading-day approximation to a calendar month fails QA.
- [ ] Skipping an empty/absent immediately-following month and jumping to a later month fails QA.
- [ ] Duplicate TEST timestamps across adjacent folds fail QA.
- [ ] Serial/process clock construction yields byte-identical fold plans and hashes.
- [ ] QA adds no production statistical behavior.

---

## Phase 3 — Canonical scalable feature-selection architecture

The target feature-selection path is intentionally simple and must remain auditable:

~~~text
PostgreSQL candidate universe: potentially thousands of features
    -> TRAIN-only quality filter
    -> canonical provenance split
         core features ------------------------------+
         generated transformations                   |
              -> family-local standardization        |
              -> family-local PCA, max 8 PCs/family  |
                                                     v
    -> core candidates + family PCs
    -> global absolute-Pearson correlation-leader pruning
    -> small representative universe
    -> HMM SFFS per configured K
    -> final one-feature-at-a-time ablation
    -> final HMM candidate tuple
    -> strict Outer OOS
~~~

Canonical separation of responsibilities:

- quality filtering removes invalid data, not weak HMM predictors;
- family PCA compresses generated transformations only; core features bypass PCA;
- global correlation pruning removes redundancy only; it does not score HMM usefulness;
- SFFS is the only feature-combination search;
- ablation measures marginal contribution of the selected final tuple;
- local DuckDB is the durable feature-selection memory;
- MLflow is the visual and metric audit trail, not the metadata source of truth;
- raw likelihood, AIC and BIC may not be compared across different feature dimensions.

Canonical defaults for this profile are fixed in PR-476 and may change only through a new
versioned contract:

~~~text
family_pca_max_components          = 8
correlation_abs_threshold          = 0.95
correlation_subwindow_abs_threshold= 0.90
correlation_subwindows             = 3
correlation_min_pair_rows          = 30
correlation_min_subwindow_rows     = 10
sffs_max_features                  = 10
~~~

The correlation step uses absolute Pearson correlation. Negative and positive correlations are
therefore equally eligible for redundancy removal. Correlation is computed from TRAIN rows only.
The three stability subwindows are contiguous chronological thirds of the TRAIN row positions;
their sizes may differ by at most one row.

### PR-476 — Define the canonical scalable feature-selection contract

**Type:** contract / configuration  
**Depends on:** PR-508

#### Acceptance

- [ ] Introduce one versioned feature-selection profile for the pipeline defined above.
- [ ] Persist the exact canonical defaults listed above; no hidden environment-specific threshold
  changes are allowed.
- [ ] Core features are direct HMM candidates and never forced through PCA.
- [ ] Generated transformation features may reach the HMM only through a family PC.
- [ ] Family PCA retains at most the first 8 non-zero-rank PCs; explained variance is diagnostic
  only and never decides the retained count.
- [ ] Global correlation pruning operates only on quality-eligible core features plus retained
  family PCs.
- [ ] Correlation pruning uses absolute Pearson correlation and the full-TRAIN plus three-subwindow
  stability rule defined above.
- [ ] Correlation pruning is explicitly redundancy-only; no target, HMM score, likelihood, AIC,
  BIC, future return or semantic label may influence representative choice.
- [ ] SFFS has a hard cap of 10 final features per configured K and starts from the best eligible
  singleton under the canonical feature-subset score.
- [ ] SFFS compares different feature dimensions only with a dimension-independent score; raw
  HMM likelihood, AIC and BIC comparisons across dimensions are forbidden.
- [ ] Final ablation removes exactly one selected feature at a time and refits/re-evaluates the
  same HMM selector contract.
- [ ] Outer TEST is evaluation-only and never influences any feature-selection step.
- [ ] The complete feature-selection profile hash is part of fold/model evidence.

### PR-477 — QA: static feature-selection contract consistency

**Type:** QA only  
**Depends on:** PR-476

#### Acceptance

- [ ] Assert every canonical default exactly, including 8 PCA components, 0.95/0.90 correlation
  thresholds, three subwindows, 30/10 support minima and 10-feature SFFS cap.
- [ ] Mutation of any default changes the profile hash.
- [ ] Reject a profile that routes a generated transformation directly to the HMM.
- [ ] Reject a profile that forces all core features through PCA.
- [ ] Reject Spearman, signed-only correlation or target-aware representative selection.
- [ ] Reject explained-variance-driven PC-count selection.
- [ ] Reject raw PLL/AIC/BIC as a cross-dimension SFFS objective.
- [ ] Prove Outer-TEST access is absent from the feature-selection contract.
- [ ] QA is hermetic and changes no production behavior.

### PR-478 — Add the local DuckDB feature-selection metadata store

**Type:** implementation / local metadata  
**Depends on:** PR-477

The store lives under the existing configured local evaluation state root as
feature_selection.duckdb. PostgreSQL remains the source feature store; MLflow remains tracking.

#### Acceptance

- [ ] Add a single local DuckDB database at state_root/feature_selection.duckdb.
- [ ] Create only these durable tables: feature_registry, fold_feature_stats, pca_loadings,
  correlation_mapping, sffs_steps and fold_model_stats.
- [ ] Create feature_global_stats as a deterministic SQL view, not a separately mutable table.
- [ ] feature_registry records immutable feature identity, source identity, role, family,
  transformation provenance, first_seen and lifecycle status.
- [ ] fold_feature_stats records eligibility, quality reason, direct/PC participation, PCA credit,
  representative status, SFFS participation, final selection and ablation loss.
- [ ] pca_loadings records fold, family, PC ordinal, source feature, loading, squared loading and
  explained-variance diagnostic.
- [ ] correlation_mapping records fold, candidate, chosen representative, full absolute correlation,
  three subwindow correlations, support counts and retained/rejected reason.
- [ ] sffs_steps records fold, K, step number, action, candidate, selected tuple hash and canonical
  score components.
- [ ] fold_model_stats records fold identity, selected tuple/hash, K, family, validity, diagnostics
  and MLflow run identity when tracking exists.
- [ ] One fold commit is transactional and idempotent under the same fold/profile/source identity.
- [ ] Failed or interrupted fold writes leave no partial committed fold result.
- [ ] No PostgreSQL schema mutation and no mandatory MLflow dependency is introduced.

### PR-479 — QA: DuckDB durability, idempotency and schema contract

**Type:** QA only  
**Depends on:** PR-478

#### Acceptance

- [ ] Fresh-state bootstrap creates exactly the six tables and one view declared by PR-478.
- [ ] Reopening the same state root preserves byte-equivalent logical rows.
- [ ] Replaying the same fold is idempotent and creates no duplicate logical records.
- [ ] Injected failure before transaction commit leaves zero partial fold rows.
- [ ] Conflicting immutable feature identity fails closed.
- [ ] Concurrent readers observe only committed fold states.
- [ ] feature_global_stats is reproduced independently from base tables.
- [ ] DuckDB state contains no raw time-series vectors or credentials.
- [ ] QA uses a temporary local state root only.

### PR-480 — Make feature provenance and family membership canonical

**Type:** implementation / provenance  
**Depends on:** PR-479

#### Acceptance

- [ ] Every discovered candidate receives one immutable role: core or transformation.
- [ ] Every transformation receives one immutable source family and structured transformation
  provenance from the existing generated-feature provenance contract.
- [ ] Feature role/family may not be inferred from ad-hoc string heuristics inside selection code.
- [ ] Missing or conflicting provenance fails closed with the exact feature identity.
- [ ] Core features preserve their source column identity.
- [ ] Transformation provenance includes source family, transform name and normalized parameters
  sufficient to distinguish windows/variants.
- [ ] Canonical feature identity is independent of discovery order and process completion order.
- [ ] The same source snapshot and provenance produce the same feature-registry rows and hashes.
- [ ] No statistical filtering or HMM fitting is added in this PR.

### PR-481 — QA: provenance, family and identity matrix

**Type:** QA only  
**Depends on:** PR-480

#### Acceptance

- [ ] Fixtures cover core features and multiple transformation families with multiple windows.
- [ ] Missing role, missing family, conflicting family and conflicting transformation provenance all
  fail closed.
- [ ] Discovery-order reversal preserves registry identities and hashes.
- [ ] Equivalent normalized transformation parameters map to one canonical identity.
- [ ] Distinct windows/parameters cannot collide.
- [ ] No name-pattern-only fallback is accepted.
- [ ] Production statistical behavior is unchanged.

### PR-482 — Implement TRAIN-only high-dimensional quality filtering

**Type:** implementation / statistical preprocessing  
**Depends on:** PR-481

#### Acceptance

- [ ] Reuse the current canonical v4 feature-quality thresholds exactly; this PR does not change
  their numeric values.
- [ ] Compute quality statistics from the current fold TRAIN partition only.
- [ ] Evaluate coverage, finite-value validity and variance in vectorized/chunked form suitable for
  several thousand columns.
- [ ] No pairwise NxN matrix is allocated in the quality stage.
- [ ] Every rejected feature receives one deterministic reason code.
- [ ] Every accepted feature remains traceable to feature_registry.
- [ ] No fill, interpolation, forward carry or synthetic observation is permitted.
- [ ] Outer TEST values cannot change any quality decision or quality statistic.
- [ ] Quality results are persisted to fold_feature_stats.
- [ ] Serial and process-enabled execution produce the same ordered result.

### PR-483 — QA: quality-filter leakage and high-dimensional parity

**Type:** QA only  
**Depends on:** PR-482

#### Acceptance

- [ ] Independent reference statistics reproduce coverage/finite/variance decisions exactly.
- [ ] Mutating any Outer-TEST value cannot alter TRAIN quality results.
- [ ] Fixtures cover all-null, near-constant, non-finite, short-history and fully valid columns.
- [ ] A synthetic matrix with at least 5,000 candidate features completes without an NxN quality
  allocation.
- [ ] Input-column order reversal preserves accepted identities and reasons.
- [ ] Serial/process results and hashes are identical.
- [ ] QA adds no production behavior.

### PR-484 — Implement family-local standardization and PCA for transformations

**Type:** implementation / dimensionality reduction  
**Depends on:** PR-483

#### Acceptance

- [ ] PCA is applied separately to each transformation family; no global PCA is used.
- [ ] Core features bypass PCA unchanged.
- [ ] Each family uses its own TRAIN-only complete-case model clock; no cross-family complete-case
  intersection is required.
- [ ] Standardization parameters are fitted from that family TRAIN clock only using the repository
  population-variance convention.
- [ ] Retain PC1 through PCm where m=min(8, numerical rank); explained variance does not choose m.
- [ ] PC order is descending explained variance with deterministic tie handling.
- [ ] Canonicalize every PC sign by making its largest-absolute loading positive; loading ties use
  canonical source-feature identity.
- [ ] PC semantic identity is family plus ordinal, independent of fold-specific loading values.
- [ ] TEST and later timestamps are transformed only with frozen TRAIN scaler/loadings.
- [ ] Persist scaler/PCA identities, explained-variance diagnostics and all loadings needed for
  feature-credit attribution.
- [ ] A family with zero usable numerical rank is statistically invalid for that family only; it
  does not crash unrelated families.

### PR-485 — QA: family-PCA math, leakage and deterministic identity

**Type:** QA only  
**Depends on:** PR-484

#### Acceptance

- [ ] Independent NumPy/SVD reference reproduces scaling, rank, PC scores and loadings on fixtures.
- [ ] Mutation of Outer-TEST rows cannot alter TRAIN scaler, rank, loading or PC identity.
- [ ] Families with 1, 2, 8, 9 and more than 100 transformation features cover the rank/cap
  boundaries.
- [ ] No family emits more than eight PCs.
- [ ] Explained-variance threshold mutations cannot change retained PC count.
- [ ] Column-order and SVD sign reversals preserve canonical PC identities and hashes.
- [ ] Two independent families never share a complete-case mask or PCA fit.
- [ ] Core features are byte-identical before and after the PCA stage.
- [ ] QA adds no production behavior.

### PR-486 — Implement global stable correlation-leader pruning

**Type:** implementation / redundancy reduction  
**Depends on:** PR-485

The input universe is exactly quality-eligible core features plus family PCs.

#### Acceptance

- [ ] Compute pairwise absolute Pearson correlations from TRAIN rows only.
- [ ] Use blockwise/vectorized correlation work and retain only threshold-relevant edges/support
  metadata; do not require a persistent dense NxN artifact.
- [ ] A pair is redundant only when full-TRAIN absolute correlation is at least 0.95, pair support
  is at least 30, median absolute correlation across the three chronological TRAIN thirds is at
  least 0.90, and each third has at least 10 paired finite rows.
- [ ] If support is insufficient, the pair is not treated as redundant.
- [ ] Leader selection is lexicographic: largest number of directly redundant unassigned neighbors;
  then highest median full absolute correlation to those neighbors; then higher TRAIN coverage;
  then core before PC; then canonical candidate identity.
- [ ] After selecting a leader, remove only candidates directly redundant with that leader.
- [ ] Transitive graph connectivity alone never removes a candidate: A-B and B-C cannot remove C
  through A when A-C fails the redundancy rule.
- [ ] Negative and positive correlation use the same absolute threshold.
- [ ] Repeat until no unassigned candidate remains; every input maps to exactly one retained leader.
- [ ] Persist the full candidate-to-leader mapping and all supporting correlations/counts.
- [ ] No HMM fit, target variable or future/OOS information is used.

### PR-487 — QA: independent correlation-leader oracle and adversarial chains

**Type:** QA only  
**Depends on:** PR-486

#### Acceptance

- [ ] Independent implementation reproduces all retained leaders and mappings.
- [ ] Cover +0.95, -0.95, just-below-threshold and insufficient-support boundaries.
- [ ] Cover a chain where A-B and B-C pass but A-C fails; C must survive when A is leader.
- [ ] Cover a case with high crisis-only full-sample correlation but unstable thirds; both features
  must survive.
- [ ] Cover exact leader tie-breaks including core-versus-PC and canonical-name fallback.
- [ ] Row/column/process completion order cannot alter the result or hash.
- [ ] Outer-TEST mutation cannot alter any correlation result.
- [ ] QA adds no production behavior.

### PR-488 — Add MLflow audit artifacts for quality, PCA and correlation stages

**Type:** implementation / observability  
**Depends on:** PR-487

#### Acceptance

- [ ] Log one feature_funnel plot with counts for discovered, quality-eligible, family-PC/core,
  correlation-representative and later final stages when available.
- [ ] Log one family_survival plot with source count, quality count, retained PCs and representative
  count per family.
- [ ] Log one explained-variance curve per PCA family under a deterministic pca/ artifact path.
- [ ] Log top-loading plots for every family PC that survives correlation pruning; top 20 absolute
  loadings are shown, with the full loading table persisted separately.
- [ ] Log correlation group-size ranking for every retained leader.
- [ ] Log representative correlation heatmap for at most 80 representatives, chosen by descending
  covered-group size then canonical identity; the complete correlation mapping remains available
  as a table artifact regardless of plot truncation.
- [ ] Log exact profile/source/fold hashes beside every artifact bundle.
- [ ] Plot generation never changes selection results.
- [ ] QA/local tests use MLflow FileStore; production runs use the configured external tracking URI.
- [ ] DuckDB remains the metadata source of truth if MLflow logging is disabled or fails before an
  authorized external run.

### PR-489 — QA: MLflow preprocessing evidence completeness and plot determinism

**Type:** QA only  
**Depends on:** PR-488

#### Acceptance

- [ ] A hermetic FileStore run contains every required plot/table for a multi-family fixture.
- [ ] Artifact names and payload hashes are deterministic under process completion-order reversal.
- [ ] Full loading and correlation tables contain every underlying row even when plots show top-N.
- [ ] Heatmap selection obeys the exact 80-representative rule.
- [ ] Missing required preprocessing artifacts fail the completeness verifier.
- [ ] Plot rendering failures cannot silently alter statistical results.
- [ ] No NAS MLflow write occurs in this QA PR.

### PR-490 — Implement process-parallel HMM SFFS on correlation representatives

**Type:** implementation / feature subset search  
**Depends on:** PR-489

#### Acceptance

- [ ] SFFS input is exactly the retained correlation representatives for the current outer TRAIN
  fold; no removed candidate may re-enter directly.
- [ ] Run SFFS independently for each configured K slot; current legal K values remain 2,3,4,5.
- [ ] Use the Gaussian full-covariance HMM as the feature selector for each K; the selected tuple is
  then reused by all configured emission families at that K.
- [ ] Candidate subset scoring uses a new immutable feature_subset_score.v1 built from the existing
  dimension-independent forecast, calibration, stability, support and valid-fold components used
  by cross_k_score.v1, but with no K-complexity penalty.
- [ ] feature_subset_score.v1 uses only monthly inner folds contained inside outer TRAIN and applies
  the same eligibility gates: valid-fold rate at least 0.80, at least three valid inner folds and a
  valid latest inner fold.
- [ ] The score is tuning evidence only; Outer TEST remains the sole unbiased fold performance
  evidence.
- [ ] Start from the best eligible singleton; after every forward add, perform backward removals
  while the canonical score strictly improves by more than 1e-12.
- [ ] Stop when no forward addition improves the score by more than 1e-12 or 10 features are
  selected.
- [ ] Ranking ties use total score, forecast score, worst-fold forecast, calibration, stability,
  robustness, fewer features and finally canonical tuple identity.
- [ ] All candidate additions/removals within one SFFS step run in bounded process workers with
  deterministic result assembly.
- [ ] Reuse existing affinity/cgroup-aware worker sizing; cap native BLAS/OpenMP threads to one per
  worker and forbid nested process-pool oversubscription.
- [ ] Persist every evaluated SFFS step/action/score to sffs_steps.
- [ ] Raw HMM likelihood, AIC and BIC are logged only as diagnostics and never choose between
  different feature dimensions.

### PR-491 — QA: SFFS score, floating search and CPU determinism

**Type:** QA only  
**Depends on:** PR-490

#### Acceptance

- [ ] Independently recompute feature_subset_score.v1 component by component.
- [ ] Cover no-winner, eligibility-boundary, singleton, forward-add and backward-remove paths.
- [ ] A synthetic example proves floating backward removal can remove an earlier selected feature.
- [ ] No result may exceed the 10-feature cap.
- [ ] Raw PLL/AIC/BIC improvements cannot override a worse dimension-independent score.
- [ ] Worker counts 1, 8, 32 and auto produce identical selected tuples, steps and hashes.
- [ ] Native thread-pool inspection proves one BLAS/OpenMP thread per process during the QA run.
- [ ] Outer-TEST mutation cannot alter SFFS selection.
- [ ] QA adds no production behavior.

### PR-492 — Implement final one-at-a-time feature ablation

**Type:** implementation / marginal contribution  
**Depends on:** PR-491

#### Acceptance

- [ ] For each final SFFS tuple, refit/re-evaluate exactly one model per selected feature removed.
- [ ] Use the same K, selector family, inner-fold plan, seeds and feature_subset_score.v1 contract as
  the final SFFS model.
- [ ] Define ablation_loss as final_total_score minus removed_feature_total_score.
- [ ] Persist negative, zero and positive ablation_loss values without clipping.
- [ ] An invalid ablated model receives an explicit invalid reason and cannot fabricate a numeric
  total score.
- [ ] Ablations run in bounded process workers and assemble results in canonical feature order.
- [ ] Persist every result in fold_feature_stats.
- [ ] Do not automatically remove an SFFS-selected feature merely because its measured ablation
  loss is non-positive in one fold.

### PR-493 — QA: ablation completeness and independent marginal-loss oracle

**Type:** QA only  
**Depends on:** PR-492

#### Acceptance

- [ ] Exactly N ablation evaluations exist for an N-feature final tuple.
- [ ] Independent recomputation reproduces every finite ablation_loss.
- [ ] Cover positive, zero, negative and invalid-ablation cases.
- [ ] Completion-order reversal preserves rows and hashes.
- [ ] Outer-TEST mutation cannot alter ablation results.
- [ ] QA adds no production behavior.

### PR-494 — Persist PCA credit and cumulative cross-fold feature statistics

**Type:** implementation / cumulative metadata  
**Depends on:** PR-493

#### Acceptance

- [ ] For each selected family PC, attribute its fold contribution to source transformations by
  squared loading weight.
- [ ] Define source_feature_pca_credit as the sum over selected PCs of
  abs(selected_pc_ablation_loss) multiplied by loading_squared.
- [ ] Directly selected core features receive direct_selection_count and their own ablation_loss;
  they do not receive synthetic PCA credit.
- [ ] feature_global_stats exposes eligible_folds, quality_pass_folds, representative_folds,
  selected_folds, selection_rate, mean/median ablation_loss, total/mean PCA credit,
  last_selected_fold and consecutive_unused_folds.
- [ ] Aggregation is deterministic and derived only from committed fold rows.
- [ ] Replaying a fold replaces/reuses the same logical fold contribution and never double counts.
- [ ] MLflow logs cumulative selection-frequency, mean-ablation and PCA-credit plots after each
  completed fold.
- [ ] DuckDB remains authoritative; MLflow plots are projections of committed local statistics.

### PR-495 — QA: cumulative statistics and PCA-credit conservation

**Type:** QA only  
**Depends on:** PR-494

#### Acceptance

- [ ] Independent SQL/Python aggregation reproduces feature_global_stats exactly.
- [ ] Squared loadings for each normalized selected PC distribute exactly its absolute ablation
  contribution within numerical tolerance.
- [ ] Fold replay cannot change cumulative counts.
- [ ] Failed/uncommitted folds contribute zero cumulative statistics.
- [ ] Selection-frequency, ablation and PCA-credit MLflow plots match DuckDB source rows exactly.
- [ ] Process/order reversal preserves cumulative hashes.
- [ ] QA adds no production behavior.

### PR-496 — Add semiautomatic feature lifecycle recommendations

**Type:** implementation / governance  
**Depends on:** PR-495

No production PostgreSQL column is dropped by this PR.

#### Acceptance

- [ ] Lifecycle states are ACTIVE, DEPRECATED_CANDIDATE, DEPRECATED and DROPPABLE.
- [ ] Raw/core source features are never automatically advanced beyond DEPRECATED_CANDIDATE.
- [ ] Generated transformations may become DEPRECATED_CANDIDATE only after at least 20 eligible
  folds, zero final selections in the last 20 eligible folds, zero representative selections in the
  last 20 eligible folds, and cumulative PCA credit below the versioned lifecycle epsilon.
- [ ] The lifecycle epsilon is explicit in configuration and stored in the profile hash.
- [ ] DEPRECATED and DROPPABLE transitions require explicit operator approval; the pipeline only
  recommends them.
- [ ] Emit a deterministic lifecycle report with the exact evidence supporting every recommendation.
- [ ] PostgreSQL DROP/ALTER statements are never executed by the automated selection pipeline.
- [ ] Re-activated evidence automatically returns an unapproved candidate to ACTIVE.

### PR-497 — QA: lifecycle safety and no-automatic-drop proof

**Type:** QA only  
**Depends on:** PR-496

#### Acceptance

- [ ] Boundary fixtures cover 19 versus 20 eligible folds and every lifecycle predicate.
- [ ] Any recent selection, representative use or material PCA credit blocks deprecation candidacy.
- [ ] Core/raw features cannot become automatically DROPPABLE.
- [ ] Search the production feature-selection path and prove no PostgreSQL DROP/ALTER action exists.
- [ ] Operator-approved state changes are explicit, auditable and reversible before physical DB
  maintenance occurs outside this pipeline.
- [ ] QA adds no production behavior.

---

## Phase 4 — Integrate, visualize and cut over the canonical pipeline

### PR-498 — Integrate the complete feature-selection pipeline into monthly outer refit

**Type:** implementation / orchestration  
**Depends on:** PR-497

#### Acceptance

- [ ] The monthly fold flow is exactly quality -> provenance split -> family PCA -> global
  correlation leaders -> per-K SFFS -> ablation -> final HMM fit -> Outer TEST.
- [ ] Every stage consumes only rows at or before the outer TRAIN cutoff.
- [ ] The entire feature-selection pipeline is rerun after each closed calendar month; no intramonth
  feature reselection occurs.
- [ ] The resulting feature/PCA/HMM package is frozen for the complete following calendar month.
- [ ] Final package identity contains source, month clock, feature-selection profile, provenance,
  PCA, representative mapping, selected tuple, K/family and model hashes.
- [ ] Failed selection yields explicit invalid fold evidence and no partial package.
- [ ] DuckDB fold transaction commits only after the fold's statistical artifacts are complete.
- [ ] MLflow parent/child runs link every stage artifact to the same fold/package identity.
- [ ] Deployment/refit uses the latest closed-month TRAIN cutoff and the same pipeline implementation.
- [ ] No historical PCA-only-prefix selector remains on this new profile.

### PR-499 — QA: orchestration leakage, month cadence and stage parity

**Type:** QA only  
**Depends on:** PR-498

#### Acceptance

- [ ] Golden test exercises two consecutive month-end refits and two complete OOS months.
- [ ] Mutating month m+1 cannot alter the package fitted through month m.
- [ ] A midmonth request resolves to the latest closed-month package without refitting.
- [ ] Stage identities in DuckDB, package metadata and MLflow are mutually consistent.
- [ ] Inject failure after each stage and prove no later stage is presented as complete.
- [ ] Serial/process orchestration produces identical canonical package/statistical hashes.
- [ ] No generated raw transformation column reaches the HMM directly.

### PR-500 — Cut over Xetra v4 and remove superseded active selectors

**Type:** implementation / controlled cutover  
**Depends on:** PR-499

#### Acceptance

- [ ] Promote the scalable feature-selection profile to the sole canonical Xetra v4 evaluation path.
- [ ] Remove the active PCA-only-prefix L* selector and old clustering/medoid/teacher selector from
  public Xetra entry points.
- [ ] Historical artifact readers may remain only where required to inspect old runs; they cannot
  construct a new canonical evaluation.
- [ ] Public evaluation/refit/serving package schema requires the new feature-selection profile hash.
- [ ] Missing DuckDB is allowed for inference from an already frozen package, but new evaluation/refit
  creates/uses the local metadata store.
- [ ] No compatibility fallback silently re-enables a superseded selector.
- [ ] Documentation and examples show only the canonical new pipeline.

### PR-501 — QA: zero-legacy and full hermetic multi-fold pipeline proof

**Type:** QA only / full local acceptance  
**Depends on:** PR-500

#### Acceptance

- [ ] Static import/config scan proves no canonical entry point can select the old PCA-only-prefix,
  clustering, medoid or teacher selector.
- [ ] Run a complete hermetic multi-fold evaluation from thousands-feature input through final HMM
  and Outer TEST with real PCA, correlation, HMM SFFS and ablation computation.
- [ ] Verify every required DuckDB row family and every required MLflow plot/table exists.
- [ ] Independently recompute one fold's quality decisions, PCA, correlation leaders, SFFS score,
  final selected tuple and ablation losses.
- [ ] Repeat the same pinned source and prove identical canonical statistical hashes.
- [ ] Perturb only Outer TEST and prove all TRAIN-side feature-selection artifacts remain unchanged.
- [ ] No network service or production mutation is required.

### PR-502 — QA: 10,000-feature scale, CPU and memory acceptance

**Type:** QA only / resource acceptance  
**Depends on:** PR-501

Target host class: approximately 86 vCPUs and 256 GiB RAM.

#### Acceptance

- [ ] Run a production-shaped synthetic fixture with at least 10,000 discovered features across
  multiple core and transformation families.
- [ ] Quality filtering and family PCA complete without materializing a global 10,000x10,000
  correlation matrix.
- [ ] Global correlation work is blockwise and persists only required edge/mapping evidence.
- [ ] HMM SFFS receives only post-PCA/post-correlation representatives, never the original
  10,000-feature matrix as an HMM observation vector.
- [ ] Candidate HMM fits parallelize through the existing process backend; worker counts 1, 8, 32
  and auto remain statistically identical.
- [ ] Native numerical thread pools remain one thread per worker.
- [ ] Full feature matrices are shared/read-only or memory-mapped where worker fan-out would
  otherwise copy them.
- [ ] Peak resident memory for the acceptance run stays below 64 GiB.
- [ ] The run records wall time, peak RSS, candidate counts per stage and effective worker count in
  MLflow and the local fold metadata.
- [ ] No Spark, Ray or external distributed-compute dependency is introduced.

---

## Phase 5 — External/current-source acceptance and publication

### PR-503 — Run the current-Xetra read-only feature-selection audit

**Type:** external QA / read-only  
**Depends on:** PR-502 and a production-eligible upstream source snapshot

#### Acceptance

- [ ] Query the current PostgreSQL source read-only and capture exact source/catalog identities.
- [ ] Report discovered feature count, core/transformation counts, quality survivors, PCs by family,
  correlation representatives and planned SFFS candidate counts.
- [ ] Prove every feature used by the run has canonical provenance.
- [ ] Run at least three consecutive closed-month outer folds when the source history permits.
- [ ] Do not mutate PostgreSQL, MLflow registry aliases or production model versions.
- [ ] Archive exact DuckDB local-state digest and feature-selection profile hash.

### PR-504 — Verify external MLflow feature-selection evidence completeness

**Type:** external QA / tracking  
**Depends on:** PR-503 and explicit MLflow namespace authorization

#### Acceptance

- [ ] Run the canonical pipeline in the explicitly authorized MLflow experiment/namespace.
- [ ] Verify every preprocessing, correlation, SFFS, ablation, cumulative-statistics and HMM
  diagnostic artifact required by the contract.
- [ ] Read back all metric/table/plot artifact hashes and bind them to source/fold/package identity.
- [ ] Compare MLflow cumulative plots against the authoritative local DuckDB statistics.
- [ ] Historical runs from superseded selectors cannot satisfy new-profile completeness.
- [ ] No registry alias mutation occurs.

### PR-505 — External package, registry and compare-and-swap durability

**Type:** external QA / lifecycle  
**Depends on:** PR-504

#### Acceptance

- [ ] Package each eligible K/family slot with its exact selected semantic feature tuple,
  fold-fitted core scalers/family PCA state and profile hash.
- [ ] Readback reconstructs the same observation vector order and package digest.
- [ ] Registry writes are idempotent for identical package identity.
- [ ] Failed compare-and-swap leaves all existing aliases unchanged.
- [ ] Lifecycle recommendations remain metadata only and cannot trigger PostgreSQL feature drops.
- [ ] Historical package versions cannot masquerade as the new profile.

### PR-506 — Authorized publication and independent readback

**Type:** external publication / final acceptance  
**Depends on:** PR-505  
**Requires:** explicit operator authorization

#### Acceptance

- [ ] PR-503, PR-504 and PR-505 are green first.
- [ ] Publish only explicitly authorized packages produced by the canonical new pipeline.
- [ ] Independently read back model version, aliases, source identity, closed-month cutoff,
  feature-selection profile, selected tuple, PCA state, representative evidence, K/family and
  package digest.
- [ ] Verify publication is idempotent for the same package identity.
- [ ] Failed readback/promotion leaves prior alias state unchanged.
- [ ] Champion promotion remains an explicit operator action.
- [ ] Archive final external acceptance evidence and exact MLflow identities.

---

### Active dependency graph

~~~text
PR-448
  -> 449 -> 450 -> 451 -> 452 -> 453 -> 454
  -> 507 -> 508
  -> 476 -> 477 -> 478 -> 479 -> 480 -> 481 -> 482 -> 483
  -> 484 -> 485 -> 486 -> 487 -> 488 -> 489
  -> 490 -> 491 -> 492 -> 493 -> 494 -> 495 -> 496 -> 497
  -> 498 -> 499 -> 500 -> 501 -> 502
  -> 503 -> 504 -> 505 -> 506
~~~

Planning PRs PR-232, PR-250, PR-423..PR-430, PR-455..PR-475 and the previous contents
of PR-476..PR-506 describing the PCA-only-prefix architecture are superseded and are not active
execution items.

---

## Current architectural decisions and non-goals

- **Target selection redesign:** PR-507/PR-508 plus the rewritten PR-476–PR-506 define the
  canonical scalable feature pipeline: TRAIN-only quality -> provenance split -> family PCA for
  transformations -> global stable absolute-Pearson correlation leaders across core+PCs -> per-K
  HMM SFFS -> final ablation -> frozen next-month package.
- **Core vs transformations:** core features remain direct interpretable candidates. Generated
  transformations never enter the HMM directly; they are represented only by family PCs.
- **PCA semantics:** PCA is family-local, TRAIN-only, rank-limited and capped at eight PCs per
  family. Explained variance is diagnostic only.
- **Correlation semantics:** correlation pruning removes redundancy only. It is global across all
  surviving core features and family PCs, uses absolute Pearson 0.95 plus three-subwindow stability
  0.90, and uses direct leader-neighbor removal rather than transitive connected components.
- **HMM feature search:** SFFS is the only combinatorial feature-subset search. It is capped at ten
  inputs per K, uses process-parallel candidate fits and a dimension-independent score; raw PLL/AIC/
  BIC never compare different feature dimensions.
- **Metadata:** local DuckDB under the configured state root is the durable feature-selection memory.
  MLflow is the visual/metric audit trail and may be reconstructed from committed metadata.
- **Feature cleanup:** the automated pipeline only recommends lifecycle states. It never drops or
  alters PostgreSQL feature columns. Physical deletion remains a separate operator-approved
  maintenance action.
- **Canonical live/evaluation cadence:** the complete pipeline refits once after each Europe/Berlin
  calendar month closes and is frozen for the immediately following complete calendar month.
- Only **Xetra v4** is active. Legacy v1-v3 evaluation/package/serving compatibility is retired;
  Git history is the archive.
- Production packages are published to the external NAS MLflow service at
  http://10.10.1.3:5000; local FileStore is used for hermetic QA.
- Feature PostgreSQL and MLflow are external dependencies; Docker/Compose services do not belong to
  this repository.
- CPU-bound production work uses the existing affinity/cgroup-aware process backend. Native
  BLAS/OpenMP threads remain capped at one per worker. No Spark/Ray layer is introduced.
- The full Xetra v4 evaluator remains intentionally non-resumable. Interrupted full runs restart
  from the beginning; local DuckDB fold metadata is committed transactionally only for completed
  fold-selection results.
- State identities are fold/model-version local; semantic labels must not leak into discovery,
  correlation pruning or statistical ranking.

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

- Planning PRs PR-232, PR-250, PR-423–PR-430, PR-455–PR-475 and the former PCA-only-prefix contents of PR-476–PR-506 are superseded by the rewritten scalable feature-selection plan; superseded definitions survive only in Git/GitHub history and must not be implemented from old text.
- GitHub #277, #284 and #317 were closed without merge and are superseded by later
  merged work.
- Draft planning IDs PR-186–PR-206 are superseded and must not be implemented.
- Historical requirements for v1-v3 compatibility or full-run computation-position
  resume are superseded by the architectural decisions above.
