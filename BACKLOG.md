# Regime Engine — Canonical Backlog

Status date: 2026-09-20

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
generated transformations first pass conservative family-internal near-duplicate pruning and are
then compressed by family-local PCA; all surviving core features and family PCs pass deterministic
global correlation-leader pruning before HMM-based SFFS selection and final ablation. Feature-selection metadata is persisted locally in DuckDB, while every major stage is
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
- no Outer-TEST row may influence quality filtering, provenance classification, family
  near-duplicate pruning, standardization, family PCA, global correlation pruning, SFFS, ablation,
  model-family selection or K-slot fitting;
- external PostgreSQL/MLflow writes are forbidden unless the PR is explicitly marked external;
- complete end-to-end acceptance starts only after all implementation and focused QA work is green.

The intended order is:

~~~text
PR-448
  -> repository/CI correctness + focused QA
  -> calendar-month refit/model-clock implementation + focused QA
  -> canonical feature-selection contract
  -> exact macro_loader.macro_features materialized-view source + QA
  -> local DuckDB metadata + provenance + TRAIN-only quality
  -> unified parallel runtime for the 86-vCPU / 256-GiB host + QA
  -> family near-duplicate pruning -> family PCA -> family-stage parallelism + QA
  -> global correlation leaders -> parallel correlation tiling + QA
  -> MLflow preprocessing evidence + QA
  -> HMM SFFS -> ablation -> flattened HMM task frontier + QA
  -> cumulative statistics -> parallel outer-fold coordinator + QA
  -> lifecycle recommendations
  -> monthly orchestration -> canonical cutover
  -> remove legacy statistical/source code + QA
  -> remove compatibility/package/serving code + QA
  -> refactor the canonical-only codebase + QA
  -> documentation consolidation/onboarding + documentation QA
  -> full hermetic/high-dimensional/system tests
  -> target-host parallel benchmark
  -> current-source external audit
  -> external MLflow/registry durability
  -> authorized publication/readback
~~~

### Current repository and external state

- `origin/main` is `9165249`; local working branch is
  `pr/PR-476-feature-role-selection-contract` at the current pushed `HEAD`;
  the working tree is clean before this backlog update.
- The previous K-slot implementation/QA closures are preserved in Git history
  and their local acceptance evidence is complete; this cutover intentionally
  supersedes their old planning text with the scalable PR-449–PR-531 chain.
- The authorized full evaluation was attempted on 2026-09-20 and stopped
  before HMM/PCA/MLflow writes because the canonical NAS source was then
  unavailable. The source is now reachable read-only as `macro-loader`, exposes
  the canonical 168-column materialized view, and has a verified
  `dataset_id='macro_features'` lineage row (4,426 rows, 2010-01-01 through
  2026-09-18, schema version 6, feature version 5). NAS MLflow health is `OK`
  and no `regime-xetra` model exists.
- An authorized NAS PostgreSQL metadata INSERT published the missing canonical
  lineage row; the existing `macro_features_daily` row was not changed. No
  MLflow, model, registry or alias mutation has been performed. The full
  evaluation remains intentionally not run until the remaining HMM-ablation and
  QA/provenance acceptance criteria are complete.
- PR-449 is merged as GitHub PR #448 at `56885cb`; PR-450 is merged as
  GitHub PR #449 at `b0856c7`; PR-451 is merged as GitHub PR #450 at
  `758c5a7`; PR-452 is merged as GitHub PR #451 at `026b3a2`. Their
  implementation branches are retained
  only as rebased pointers to `origin/main` for the current branch-retention
  policy.
- **Current active implementation:** PR-476 is GitHub PR #456 on branch
  `pr/PR-476-feature-role-selection-contract`; GitHub Git-Policy, Lint, Type,
  Unit and Merge-Gate checks are green. Full evaluation is intentionally not
  run.

---

### PR-448 — Consolidate and order the canonical backlog

**Status:** ACCEPTANCE COMPLETE — planning/documentation only; closure follows in PR-455
**GitHub:** #439 (rebased closure candidate in PR-455)

**Purpose:** keep BACKLOG.md as the single authoritative backlog, remove active work that
belongs to superseded clustering/medoid/teacher and PCA-only-prefix architectures, and replace it
with the dependency-ordered scalable feature-selection plan below.

#### Acceptance

- [x] `BACKLOG.md` is the only active backlog file.
- [x] Old open medoid/clustering/teacher/prefix-selection planning items are absent from active work.
- [x] Every active implementation PR has an explicit dependency and one bounded responsibility.
- [x] Every statistically risky implementation step has a separate focused QA PR.
- [x] Full hermetic/system tests appear only after implementation/cutover work.
- [x] External tests/publication appear only after local/system acceptance.
- [x] Superseded planning is retained only as compact historical traceability, not as active work.
- [x] No runtime, statistical, PostgreSQL, MLflow, registry or serving behavior changes in PR-448.

---

## Phase 1 — Repository and failure-semantics foundations

### PR-449 — Make 85% coverage the single CI authority

**Status:** ACCEPTANCE COMPLETE — merged as GitHub PR #448 at `56885cb`; GitHub CI green

**Type:** implementation / CI correctness
**Depends on:** PR-448

#### Acceptance

- [x] `tool.coverage.report.fail_under = 85` remains the sole canonical coverage threshold.
- [x] Merge and push workflows contain no lower command-line override.
- [x] Merge and push use equivalent coverage commands and data-file handling.
- [x] Existing multiprocessing coverage combine behavior is preserved.
- [x] No cross-job coverage artifact plumbing is introduced.
- [x] Test selection is unchanged in this PR.
- [x] The CI contract asserts that a threshold mutation below 85 fails before merge.
- [x] Ruff, formatting, strict mypy and the unit lane pass; GitHub Merge Gate is green.

### PR-450 — QA: regression-proof the 85% coverage contract

**Status:** ACCEPTANCE COMPLETE — merged as GitHub PR #449 at `b0856c7`; all required gates green

**Branch:** `pr/PR-450-coverage-contract-qa`

**Type:** QA only
**Depends on:** PR-449

#### Acceptance

- [x] Read `pyproject.toml` and prove the threshold is exactly 85.
- [x] Reject any explicit merge/push `--fail-under` lower than 85.
- [x] Mutations to 84 and 80 fail QA.
- [x] Prove merge/push coverage commands are semantically equivalent.
- [x] Prove no cross-job coverage artifact transfer is required.
- [x] QA is hermetic and uses no network service or secret.
- [x] Production code is unchanged.

### PR-451 — Add hermetic integration lanes to merge and push gates

**Status:** ACCEPTANCE COMPLETE — merged as GitHub PR #450 at `758c5a7`; all gates green

**Branch:** `pr/PR-451-hermetic-integration-gates`

**Type:** implementation / CI completeness
**Depends on:** PR-450

#### Acceptance

- [x] Both workflows contain a dedicated integration job parallel to lint/type/unit.
- [x] Selector is exactly `pytest -n auto tests -m "integration and not slow and not external"`.
- [x] Python 3.14.7 and the repository-locked dependency bootstrap are used.
- [x] Terminal gates require lint, type, unit and integration.
- [x] Failed/cancelled integration makes the terminal gate fail.
- [x] Slow/external/unrelated E2E tests remain excluded.
- [x] No NAS PostgreSQL/MLflow credential or alias mutation is used.
- [x] Unit coverage semantics remain unchanged.

### PR-452 — QA: prove integration gating is mandatory and hermetic

**Status:** ACCEPTANCE COMPLETE — merged as GitHub PR #451 at `026b3a2`; all gates green

**Branch:** `pr/PR-452-integration-gating-qa`

**Type:** QA only
**Depends on:** PR-451

#### Acceptance

- [x] Static QA proves both workflows declare the integration job.
- [x] Terminal gates must depend on and inspect integration status.
- [x] Selector must include `integration` and exclude `slow` and `external`.
- [x] Removing integration from either terminal gate fails QA.
- [x] Replacing the selector with bare `integration` fails QA.
- [x] The selected integration suite completes without network/secrets.
- [x] Production code is unchanged.

### PR-453 — Separate statistical invalidity from unexpected software failures

**Status:** ACCEPTANCE COMPLETE — merged as GitHub PR #452 at `dd882fc`; all required local and GitHub gates green

**Branch:** `pr/PR-453-evaluation-invalidity-boundary`

**Type:** implementation / correctness
**Depends on:** PR-452

#### Acceptance

- [x] Introduce one explicit recoverable evaluation-invalidity base exception.
- [x] Only that typed family may become invalid fold/L/K evidence.
- [x] Generic `RuntimeError`, `KeyError`, `AssertionError`, `TypeError`, unrelated
  `ValueError`, `KeyboardInterrupt` and `SystemExit` cannot become statistical invalidity.
- [x] Expected data/statistical gates translate to the typed exception at their owning boundary.
- [x] Process workers preserve failure classification in the parent.
- [x] Unexpected failures can never merely reduce a valid-fold rate.
- [x] Persisted recoverable reasons are deterministic and contain no traceback, address,
  credential or raw vector.
- [x] Statistical thresholds/seeds/ranking semantics are otherwise unchanged.
- [x] No blanket `except Exception` emits eligibility evidence.

### PR-454 — QA: adversarial failure-classification matrix

**Status:** OPEN — acceptance pending; the implementation was merged as GitHub PR #453 at `ce41867`, but the acceptance checklist below is not complete

**Branch:** `pr/PR-454-adversarial-failure-classification`

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

**Status:** ACCEPTANCE COMPLETE — merged as GitHub PR #454 at `6eb1f6b`; PR branch rebased to `origin/main` and deleted; 961 unit tests, Ruff, and 83 hermetic integration tests green; full evaluation not run

**Branch:** `pr/PR-507-calendar-month-model-clock`

**Type:** implementation / temporal contract
**Depends on:** PR-454

#### Acceptance

- [x] Add one canonical `Europe/Berlin` calendar-month clock shared by outer evaluation, inner
  selection and deployment/refit orchestration.
- [x] Convert each `timestamp_m1` to the canonical timezone only for month membership; preserve the
  original timezone-aware timestamp as data/evidence.
- [x] Define a refit boundary as the last available source observation belonging to a closed
  calendar month.
- [x] Outer TRAIN is expanding and must contain at least the existing canonical minimum TRAIN
  history before the first eligible refit boundary.
- [x] Outer TEST is exactly the immediately following **complete calendar month**, not a fixed
  63-observation block.
- [x] Outer STEP is exactly one calendar month.
- [x] The final partial calendar month at the evaluation cutoff produces no Outer TEST fold.
- [x] Inner selection mirrors the same cadence: expanding inner TRAIN, refit at closed month-end,
  TEST on the immediately following complete calendar month, step one month.
- [x] Preserve the existing minimum inner TRAIN history; retire fixed 63-row/42-row TEST semantics
  from the new profile because monthly TEST length is calendar-driven.
- [x] A monthly TEST fold is valid only when its source month is complete and the downstream
  transform/model evidence is non-empty and passes the existing finite/entropy/support/model gates;
  no synthetic rows may be added to reach a row count.
- [x] No TRAIN observation may have a timestamp later than its fold's month-end cutoff.
- [x] Persist for every fold: `train_through_month`, exact `train_cutoff_timestamp`,
  `test_calendar_month`, first/last TEST timestamp, TEST source-row count and a month-clock hash.
- [x] Two folds may never overlap in TEST timestamps.
- [x] Missing calendar months remain explicit gaps; the evaluator must not silently test the next
  non-empty month as though it were the immediately following month.
- [x] DST transitions cannot change month assignment or duplicate/drop a source timestamp.
- [x] This PR changes only clock construction/contracts; it does not change PCA/HMM math.

### PR-508 — QA: month-boundary, leakage and live-cadence clock matrix

**Status:** ACCEPTANCE COMPLETE — merged as GitHub PR #455 at `9165249`; PR branch rebased to `origin/main` and deleted; 968 unit tests, Ruff, Mypy and targeted PR-508 QA green; full evaluation not run

**Branch:** `pr/PR-508-calendar-month-qa`

**Type:** QA only
**Depends on:** PR-507

#### Acceptance

- [x] Golden fixtures cover 28-, 29-, 30- and 31-day months and year rollover December -> January.
- [x] Cover CET/CEST transitions and prove every timestamp belongs to exactly one local calendar
  month.
- [x] Prove a January refit uses data only through January month-end and February is the complete
  OOS TEST month.
- [x] Prove the next fold refits through February month-end and tests March.
- [x] A snapshot cut off in the middle of September may validate through August at most; partial
  September is never emitted as a TEST fold.
- [x] Mutation of any future-month row cannot alter an earlier fold's TRAIN cutoff or membership.
- [x] A fixed 63-row/63-step implementation fails QA.
- [x] A 21-trading-day approximation to a calendar month fails QA.
- [x] Skipping an empty/absent immediately-following month and jumping to a later month fails QA.
- [x] Duplicate TEST timestamps across adjacent folds fail QA.
- [x] Serial/process clock construction yields byte-identical fold plans and hashes.
- [x] QA adds no production statistical behavior.

---

## Phase 3 — Canonical scalable feature-selection architecture

The target feature-selection path is intentionally simple and must remain auditable.

The sole production feature relation for this profile is the macro-loader materialized view
`macro_loader.macro_features`. The candidate universe is every non-`timestamp_m1`
`DOUBLE PRECISION` column in that one validated view. The historical
`macro_loader.macro_features_daily` relation and schema-wide wildcard discovery are not valid
sources for the new profile.

~~~mermaid
flowchart TD
    A[macro_loader.macro_features<br/>thousands of columns] --> B[TRAIN-only quality filter]
    B --> C{Canonical role}
    C -->|Core| D[Core candidates]
    C -->|Transformation| E[Family-internal near-duplicate pruning]
    E --> F[Family-local scaling + PCA<br/>max 8 PCs/family]
    D --> G[Core + family PCs]
    F --> G
    G --> H[Global stable absolute-Pearson<br/>correlation-leader pruning]
    H --> I[Small representative universe]
    I --> J[HMM SFFS per K]
    J --> K[One-at-a-time ablation]
    K --> L[Final HMM tuple]
    L --> M[Strict Outer OOS month]
~~~

Canonical separation of responsibilities:

- quality filtering removes invalid data, not weak HMM predictors;
- family near-duplicate pruning removes only effectively duplicated transformations before PCA;
- family PCA compresses generated transformations only; core features bypass PCA;
- global correlation pruning removes redundancy among core features and family PCs only; it does not
  score HMM usefulness;
- SFFS is the only feature-combination search;
- ablation measures marginal contribution of the selected final tuple;
- local DuckDB is the durable feature-selection memory;
- MLflow is the visual and metric audit trail, not the metadata source of truth;
- raw likelihood, AIC and BIC may not be compared across different feature dimensions.

Canonical defaults for this profile are fixed in PR-476 and may change only through a new
versioned contract:

~~~text
family_near_duplicate_abs_threshold           = 0.995
family_near_duplicate_subwindow_abs_threshold = 0.99
family_pca_max_components                     = 8
correlation_abs_threshold                     = 0.95
correlation_subwindow_abs_threshold           = 0.90
correlation_subwindows                        = 3
correlation_min_pair_rows                     = 30
correlation_min_subwindow_rows                = 10
sffs_max_features                             = 10
~~~

Both correlation stages use absolute Pearson correlation and TRAIN rows only. The three stability
subwindows are contiguous chronological thirds of TRAIN row positions; their sizes may differ by at
most one row. Family near-duplicate pruning is deliberately much stricter than global redundancy
pruning so PCA still receives economically meaningful within-family covariance structure.

### PR-476 — Define canonical feature roles and the scalable selection contract

**Status:** IMPLEMENTATION COMPLETE — branch `pr/PR-476-feature-role-selection-contract` is at the current pushed `HEAD` with a clean working tree; GitHub PR #456 remains open pending final merge after the green gates. The canonical role/family module is publicly exported, validates the complete temporal/core catalog identity, and includes fail-closed stage boundaries for quality, family PCA, correlation, SFFS, and HMM inputs. Deterministic TRAIN-only family near-duplicate reduction, family-local standardization/PCA, global stable absolute-Pearson redundancy pruning, capped dimension-independent SFFS, one-feature-at-a-time ablation, explicit role/profile evidence metadata, a sequential composition pipeline, and `build_feature_role_contract_from_catalog()` for discovered source catalogs are implemented; PCA retains the first non-zero-rank components up to eight and stores explained variance diagnostically only. The current read-only NAS catalog contains 168 `macro_loader.macro_features` columns (20 CORE, 147 transformations, 1 temporal key), and the repository role contract classifies all 167 non-temporal columns across all 13 families. Its canonical lineage row is now published and independently verified: source build `20260920T080048Z`, 4,426 rows, schema version 6, feature version 5, and bounds 2010-01-01 through 2026-09-18. The production composition and one-step evaluation script now instantiate the sole `macro_features` materialized-view adapter; schema-wide relation enumeration, `PostgresFeatureSource`, compatibility aliases, caller-selected schemas and legacy `macro_features_daily` test identities were removed. The source contract is `read_with_catalog()` only and rejects raw-source substitution. Global V4 evidence and MLflow parent/fold model dossiers now carry the role-contract and selection-profile hashes when the canonical catalog path is used. The pipeline runs quality boundary → family reduction → family PCA → global redundancy → SFFS → HMM-backed ablation without TEST inputs; final ablation requires same-contract/model fresh-fit SHA-256 evidence and rejects reused fits. Hermetic role-contract QA now covers all 13 families, canonical defaults, forbidden policies and fail-closed mutation paths. Eager heavy-module exports were removed from the package initializer to keep multiprocessing spawn imports hermetic. Full unit tests (1002), Ruff and Mypy pass; six affected HMM/source integration tests pass in 8:31 with 24 workers. No full evaluation has run. Production K-slot HMM integration remains intentionally assigned to PR-490 and is not a PR-476 acceptance criterion.

**Branch:** `pr/PR-476-feature-role-selection-contract`

**Acceptance note:** Runtime source-universe integration, raw-source exclusion in the canonical
adapter, and role/profile hash transport into fold/model evidence are implemented and tested.
HMM-backed ablation and independent QA/provenance proofs remain open for the subsequent PR-490
and QA work; the PR-476 statistical stage contracts marked below are implemented and tested.
Production HMM
ablation is intentionally not wired to a surrogate valid-fold-rate score: it must consume the
canonical `feature_subset_score.v1` contract introduced by PR-490, otherwise the implementation
would silently violate the dimension-independent SFFS semantics.

**External source audit:** read-only login as `macro-loader` succeeds against database
`macro_loader`; `macro_loader.macro_features` is a materialized view with 168 columns and
`macro-loader` has `SELECT` but no write privilege. `macro_loader_sync.gold_sync_state` now
contains the canonical `dataset_id='macro_features'` row with the matching current view
fingerprint (`4ceea44bd95232abd972b4af6f12dbc037f01a82cd6a6c04943e2136e614ba76`) and bounds;
the real `MacroFeaturesPostgresSource` read reproduced 167 catalog entries and 4,426 rows with
the same digest. The legacy `macro_features_daily` row remains present but is never used as a
fallback. `scripts/verify_feature_postgres.sh` passes 1/1.

**Type:** contract / configuration
**Depends on:** PR-508

This is the first statistical implementation contract. Feature roles are fixed before source
integration, quality filtering, PCA, correlation pruning or any HMM fit so later stages cannot
silently reinterpret the same PostgreSQL column.

The canonical source-view role split is:

~~~mermaid
flowchart TD
    A[macro_loader.macro_features] --> T[timestamp_m1]
    A --> C[20 CORE candidates]
    A --> X[all remaining feature columns = TRANSFORMATIONS]
    T --> TK[temporal key only]
    C --> CG[global correlation pruning]
    X --> F[13 source-series families]
    F --> ND[family near-duplicate pruning]
    ND --> P[family PCA]
    P --> CG
    CG --> S[HMM SFFS]
~~~

The exact 20 CORE candidates are:

~~~text
vix_log_level
vix9d_log_level
vix3m_log_level
vix6m_log_level
vix1y_log_level
vstoxx_log_level
move_log_level
ciss_log_level
euro_hy_oas_log_level
us_2y_log_level
us_10y_log_level
estr_log_level
usd_broad_log_level
vix9d_vix_ratio
vix_vix3m_ratio
vix9d_vix3m_log_ratio
vix3m_minus_vix
vix6m_minus_vix
vix1y_minus_vix
us_10y_minus_us_2y
~~~

Semantics are intentional:

- the 13 `*_log_level` columns are current-state level descriptors;
- the six VIX same-timestamp ratios/spreads are current volatility-curve structure;
- `us_10y_minus_us_2y` is current yield-curve structure;
- `timestamp_m1` is never a feature;
- every current `delta`, `zscore`, momentum-autocorrelation, geometric-return and
  `usd_broad_log_return_20obs` column is a TRANSFORMATION;
- unchanged source values in `macro_loader.macro_raw` are outside the regime-engine feature
  universe and are never substituted for the canonical log-level columns.

Current TRANSFORMATION families are exactly the 13 source series:

~~~text
vix
vix9d
vix3m
vix6m
vix1y
vstoxx
move
ciss
euro_hy_oas
us_2y
us_10y
estr
usd_broad
~~~

A transformation belongs to exactly one source family. The role/family mapping must be represented
as canonical structured provenance; downstream statistical code may not make ad-hoc role decisions
from suffix tests. Future materialized-view columns are still discovered from PostgreSQL, but a new
column may proceed beyond discovery only if the versioned provenance contract can classify it
unambiguously as CORE or as exactly one transformation family. Otherwise the new profile fails
closed and requires an explicit contract update.

#### Acceptance

- [x] Introduce one versioned feature-selection profile for the pipeline defined above.
- [x] Encode the exact 20 CORE identities above in one canonical role contract; there is no second
  core allowlist elsewhere in the codebase.
- [x] Classify `timestamp_m1` as temporal key only and prove it can never enter quality ranking,
  family PCA, correlation candidates, SFFS or an HMM observation vector.
- [x] Classify every currently catalogued non-core feature in `macro_loader.macro_features` as a
  TRANSFORMATION assigned to exactly one of the 13 source families above.
- [x] Explicitly classify `usd_broad_log_return_20obs` as a USD_BROAD transformation, not as CORE.
- [x] Treat all `*_delta_*`, `*_zscore_*`, `*_momentum_autocorr_*` and
  `*_return_geom_*` columns in the current view as transformations, never direct HMM candidates.
- [x] Do not read or substitute unchanged raw source levels from `macro_loader.macro_raw`; the
  canonical current-state level inputs are the 13 log-level columns listed above.
- [x] Role classification is semantic only and does not waive later TRAIN-only finite/coverage/
  variance validation; in particular, no assumption about how an upstream log-level was constructed
  is invented by regime-engine.
- [x] Persist the exact canonical defaults listed above, including the 0.995/0.99 family
  near-duplicate thresholds; no hidden environment-specific threshold changes are allowed.
- [x] Core features are direct HMM candidates and never forced through PCA.
- [x] Generated transformation features may reach the HMM only through a family PC.
- [x] Family near-duplicate pruning occurs before family PCA and may remove only direct stable
  near-duplicates under the canonical 0.995/0.99 rule.
- [x] Family PCA retains at most the first 8 non-zero-rank PCs; explained variance is diagnostic
  only and never decides the retained count.
- [x] Global correlation pruning operates only on quality-eligible core features plus retained
  family PCs.
- [x] Correlation pruning uses absolute Pearson correlation and the full-TRAIN plus three-subwindow
  stability rule defined above.
- [x] Correlation pruning is explicitly redundancy-only; no target, HMM score, likelihood, AIC,
  BIC, future return or semantic label may influence representative choice.
- [x] SFFS has a hard cap of 10 final features per configured K and starts from the best eligible
  singleton under the canonical feature-subset score.
- [x] SFFS compares different feature dimensions only with a dimension-independent score; raw
  HMM likelihood, AIC and BIC comparisons across dimensions are forbidden.
- [x] Final ablation removes exactly one selected feature at a time and refits/re-evaluates the
  same HMM selector contract.
- [x] Outer TEST is evaluation-only and never influences any feature-selection step.
- [x] The complete feature-role/family contract and feature-selection profile hash are part of
  fold/model evidence.

### PR-477 — QA: static feature-role and selection-contract consistency

**Type:** QA only
**Depends on:** PR-476

**Status:** ACCEPTANCE COMPLETE — QA implementation is present in PR-476 and covered by hermetic
unit tests. The current NAS catalog inventory was verified read-only: 168 columns total (20 CORE,
147 transformations, and 1 temporal key), with all 13 transformation families represented. No
production behavior was changed.

#### Acceptance

- [x] Assert the CORE set contains exactly the 20 identities declared by PR-476 and no additional
  column.
- [x] Assert `timestamp_m1` has only the temporal-key role.
- [x] Assert every current delta, z-score, momentum-autocorrelation, geometric-return and
  `usd_broad_log_return_20obs` feature is a transformation in exactly one of the 13 families.
- [x] Assert every currently exposed `macro_loader.macro_features` column is accounted for by
  temporal-key, CORE or TRANSFORMATION classification with no overlap.
- [x] Adding an unclassifiable future feature column fails closed rather than defaulting it to CORE
  or an arbitrary family.
- [x] Mutation of a role, family, CORE identity or canonical default changes the profile hash.
- [x] Assert every canonical default exactly, including 0.995/0.99 family near-duplicate thresholds,
  8 PCA components, 0.95/0.90 global-correlation thresholds, three subwindows, 30/10 support minima
  and the 10-feature SFFS cap.
- [x] Reject a profile that routes a generated transformation directly to the HMM.
- [x] Reject a profile that forces all core features through PCA.
- [x] Reject Spearman, signed-only correlation or target-aware representative selection.
- [x] Reject explained-variance-driven PC-count selection.
- [x] Reject raw PLL/AIC/BIC as a cross-dimension SFFS objective.
- [x] Prove Outer-TEST access is absent from the feature-selection contract.
- [x] QA is hermetic and changes no production behavior.

### PR-509 — Cut the canonical feature source over to macro_loader.macro_features

**Type:** implementation / data-source contract
**Depends on:** PR-477

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-509-macro-features-source-contract`; the
canonical adapter, source lineage, catalog validation and production source wiring were merged
with PR-476 and are now independently audited by this PR. No legacy relation fallback remains.

The source is the exact PostgreSQL materialized view `macro_loader.macro_features`, produced and
versioned by `macro-loader`.

#### Acceptance

- [x] Replace the canonical Xetra-v4 candidate source with exactly
  `macro_loader.macro_features`.
- [x] Reject `macro_loader.macro_features_daily`, `macro_loader.macro_raw`, arbitrary tables,
  schema-wide relation discovery and caller-supplied feature-table overrides on the new profile.
- [x] Read `timestamp_m1` plus every non-timestamp `DOUBLE PRECISION` column in PostgreSQL ordinal
  order from the one materialized view.
- [x] Validate relation kind is materialized view and fail closed on missing view, wrong timestamp
  type, wrong feature type, duplicate column identity or unsupported extra column type.
- [x] Capture the upstream macro-feature view version/fingerprint exposed by macro-loader and bind
  it into source/evaluation identity.
- [x] Keep `macro_loader_sync.gold_sync_state` as lineage/control evidence only; it is not a
  candidate-feature relation.
- [x] Snapshot acquisition remains REPEATABLE READ / READ ONLY and closes before PCA/HMM work.
- [x] No feature-name allowlist narrows valid materialized-view columns before TRAIN-only quality.
- [x] Production inference from an already frozen package continues to request only that package's
  exact feature dependencies.

### PR-510 — QA: macro_features source, lineage and zero-legacy relation proof

**Type:** QA only
**Depends on:** PR-509

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-510-macro-features-source-qa`; the
hermetic QA matrix is independently implemented and passes without external writes.

#### Acceptance

- [x] Real-PostgreSQL-shaped fixture proves only `macro_loader.macro_features` is accepted.
- [x] A fixture exposing both `macro_features` and historical `macro_features_daily` proves the
  latter is ignored/rejected as a candidate source.
- [x] Add a valid `DOUBLE PRECISION` column to the materialized-view fixture with no regime-engine
  feature-list change; the next snapshot includes it before quality filtering.
- [x] Wrong relation kind, wrong timestamp type and non-double feature columns fail closed.
- [x] View-version/fingerprint mutation changes dataset/evaluation identity.
- [x] Catalog column order reversal at the database layer is reflected only through PostgreSQL
  ordinal order and produces the expected identity change.
- [x] Static search proves the new canonical path contains no
  `macro_loader.macro_features_daily` or schema-wide wildcard source fallback.
- [x] QA performs no external write.

### PR-478 — Add the local DuckDB feature-selection metadata store

**Type:** implementation / local metadata
**Depends on:** PR-510

The store lives under the existing configured local evaluation state root as
feature_selection.duckdb. PostgreSQL remains the source feature store; MLflow remains tracking.

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-478-duckdb-feature-selection-store`; targeted
DuckDB contract tests pass locally. PR-479 remains the separate durability/QA proof.

#### Acceptance

- [x] Add a single local DuckDB database at state_root/feature_selection.duckdb.
- [x] Create only these durable tables: feature_registry, fold_feature_stats, pca_loadings,
  correlation_mapping, sffs_steps and fold_model_stats.
- [x] Create feature_global_stats as a deterministic SQL view, not a separately mutable table.
- [x] feature_registry records immutable feature identity, source identity, role, family,
  transformation provenance, first_seen and lifecycle status.
- [x] fold_feature_stats records eligibility, quality reason, direct/PC participation, PCA credit,
  representative status, SFFS participation, final selection and ablation loss.
- [x] pca_loadings records fold, family, PC ordinal, source feature, loading, squared loading and
  explained-variance diagnostic.
- [x] correlation_mapping records fold, candidate, chosen representative, full absolute correlation,
  three subwindow correlations, support counts and retained/rejected reason.
- [x] sffs_steps records fold, K, step number, action, candidate, selected tuple hash and canonical
  score components.
- [x] fold_model_stats records fold identity, selected tuple/hash, K, family, validity, diagnostics
  and MLflow run identity when tracking exists.
- [x] One fold commit is transactional and idempotent under the same fold/profile/source identity.
- [x] Failed or interrupted fold writes leave no partial committed fold result.
- [x] No PostgreSQL schema mutation and no mandatory MLflow dependency is introduced.

### PR-479 — QA: DuckDB durability, idempotency and schema contract

**Type:** QA only
**Depends on:** PR-478

**Status:** QA COMPLETE on branch `pr/PR-479-duckdb-durability-qa`; targeted durability and
schema-contract tests pass locally.

#### Acceptance

- [x] Fresh-state bootstrap creates exactly the six tables and one view declared by PR-478.
- [x] Reopening the same state root preserves byte-equivalent logical rows.
- [x] Replaying the same fold is idempotent and creates no duplicate logical records.
- [x] Injected failure before transaction commit leaves zero partial fold rows.
- [x] Conflicting immutable feature identity fails closed.
- [x] Concurrent readers observe only committed fold states.
- [x] feature_global_stats is reproduced independently from base tables.
- [x] DuckDB state contains no raw time-series vectors or credentials.
- [x] QA uses a temporary local state root only.

### PR-480 — Materialize the PR-476 role/family contract as canonical provenance

**Type:** implementation / provenance
**Depends on:** PR-479

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-480-canonical-feature-provenance`; targeted
provenance identity tests pass locally.

This PR implements provenance transport/storage for the already-fixed PR-476 classification; it
must not redefine which features are CORE or which transformation family owns a feature.

#### Acceptance

- [x] Every discovered candidate receives the immutable PR-476 role: core or transformation.
- [x] Every transformation receives one immutable source family and structured transformation
  provenance from the existing generated-feature provenance contract.
- [x] Feature role/family may not be inferred from ad-hoc string heuristics inside selection code.
- [x] Missing or conflicting provenance fails closed with the exact feature identity.
- [x] Core features preserve their source column identity.
- [x] Transformation provenance includes source family, transform name and normalized parameters
  sufficient to distinguish windows/variants.
- [x] Canonical feature identity is independent of discovery order and process completion order.
- [x] The same source snapshot and provenance produce the same feature-registry rows and hashes.
- [x] No statistical filtering or HMM fitting is added in this PR.

### PR-481 — QA: provenance, family and identity matrix

**Type:** QA only
**Depends on:** PR-480

**Status:** QA COMPLETE on branch `pr/PR-481-provenance-identity-qa`; provenance matrix and
regression tests pass locally.

#### Acceptance

- [x] Fixtures cover core features and multiple transformation families with multiple windows.
- [x] Missing role, missing family, conflicting family and conflicting transformation provenance all
  fail closed.
- [x] Discovery-order reversal preserves registry identities and hashes.
- [x] Equivalent normalized transformation parameters map to one canonical identity.
- [x] Distinct windows/parameters cannot collide.
- [x] No name-pattern-only fallback is accepted.
- [x] Production statistical behavior is unchanged.

### PR-482 — Implement TRAIN-only high-dimensional quality filtering

**Type:** implementation / statistical preprocessing
**Depends on:** PR-481

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-482-train-quality-filter`; targeted quality
tests pass locally. PR-483 remains the separate leakage/high-dimensional QA proof.

#### Acceptance

- [x] Reuse the current canonical v4 feature-quality thresholds exactly; this PR does not change
  their numeric values.
- [x] Compute quality statistics from the current fold TRAIN partition only.
- [x] Evaluate coverage, finite-value validity and variance in vectorized/chunked form suitable for
  several thousand columns.
- [x] No pairwise NxN matrix is allocated in the quality stage.
- [x] Every rejected feature receives one deterministic reason code.
- [x] Every accepted feature remains traceable to feature_registry.
- [x] No fill, interpolation, forward carry or synthetic observation is permitted.
- [x] Outer TEST values cannot change any quality decision or quality statistic.
- [x] Quality results are persisted to fold_feature_stats.
- [x] Serial and process-enabled execution produce the same ordered result.

### PR-483 — QA: quality-filter leakage and high-dimensional parity

**Type:** QA only
**Depends on:** PR-482

**Status:** QA COMPLETE on branch `pr/PR-483-quality-filter-qa`; targeted quality QA passes locally.

#### Acceptance

- [x] Independent reference statistics reproduce coverage/finite/variance decisions exactly.
- [x] Mutating any Outer-TEST value cannot alter TRAIN quality results.
- [x] Fixtures cover all-null, near-constant, non-finite, short-history and fully valid columns.
- [x] A synthetic matrix with at least 5,000 candidate features completes without an NxN quality
  allocation.
- [x] Input-column order reversal preserves accepted identities and reasons.
- [x] Serial/process results and hashes are identical.
- [x] QA adds no production behavior.

### PR-513 — Add one shared fold-local parallel execution planner

**Type:** implementation / performance infrastructure
**Depends on:** PR-483

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-513-shared-parallel-planner`; targeted
planner, worker-cap, shared-matrix, quality, tracking, and performance tests pass locally.

The goal is maximum useful CPU concurrency on the target host class (86 vCPUs, 256 GiB RAM)
without nested process pools, oversubscribed BLAS threads or repeated matrix copies.

#### Acceptance

- [x] Introduce one immutable ParallelExecutionPlan derived from process affinity, cgroup CPU quota,
  runnable task count and explicit operator override.
- [x] `auto` has no arbitrary worker cap below the available CPU budget; effective workers are
  `min(available_cpu_budget, runnable_tasks)` unless a measured memory-safety bound is lower.
- [x] Create at most one CPU process pool per evaluation process and reuse it across eligible stages.
- [x] Process workers may never create child process pools.
- [x] Set/verify OMP, MKL, OpenBLAS and NumExpr native thread pools to one thread per worker for
  process-parallel numerical stages.
- [x] Materialize each fold's numeric candidate matrix once as immutable shared/read-only memory or
  memory-mapped storage; worker tasks receive column/row indices and small metadata, not full
  DataFrame copies.
- [x] Centralize bounded task submission/backpressure and deterministic result reordering.
- [x] Preserve explicit serial mode for reference tests and debugging.
- [x] Parallel runtime policy changes no statistical formula, seed, ordering rule or threshold.
- [x] Persist effective CPU budget, worker count, task count and shared-matrix identity in fold
  runtime metadata and MLflow runtime metrics.

### PR-514 — QA: parallel planner, shared-memory and thread-cap determinism

**Type:** QA only
**Depends on:** PR-513

**Status:** QA COMPLETE on branch `pr/PR-514-parallel-planner-qa`; targeted planner QA passes locally.

#### Acceptance

- [x] Worker counts 1, 8, 32, 64 and auto yield identical canonical results for deterministic test
  tasks.
- [x] On a host exposing at least 64 CPUs and at least 64 runnable tasks, auto schedules more than
  32 workers unless an explicit tested resource bound applies.
- [x] Inspect child environments/native thread pools and prove one numerical native thread per
  process.
- [x] Instrument serialization/copies and prove the large fold matrix is not serialized once per
  task.
- [x] Process completion-order reversal preserves canonical result order and hashes.
- [x] Worker failure, cancellation and KeyboardInterrupt propagate through the existing typed failure
  semantics without deadlock or orphan workers.
- [x] Serial mode remains functional and statistically identical.
- [x] QA adds no production statistical behavior.

### PR-515 — Implement family-internal stable near-duplicate pruning

**Type:** implementation / redundancy preprocessing
**Depends on:** PR-514

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-515-family-near-duplicate-pruning`; targeted
family-reduction tests pass locally. PR-516 remains the independent oracle/edge-case QA proof.

#### Acceptance

- [x] Run only within one transformation family and only on outer-TRAIN data.
- [x] A pair is a family near-duplicate only when full-TRAIN absolute Pearson correlation is at
  least 0.995, full pair support is at least 30, median absolute correlation across the three TRAIN
  thirds is at least 0.99, and every third has at least 10 paired finite rows.
- [x] Insufficient support never removes a pair.
- [x] Use direct-leader removal, not transitive connected components.
- [x] Family leader ranking is deterministic: largest direct near-duplicate neighborhood, then
  highest median full absolute correlation, then higher TRAIN coverage, then canonical feature
  identity.
- [x] Persist every source-feature-to-family-leader mapping and support/correlation evidence.
- [x] No HMM, target, explained variance, semantic score or future/OOS row influences this stage.
- [x] Core features bypass this stage unchanged.

### PR-516 — QA: family near-duplicate oracle and PCA-input preservation

**Type:** QA only
**Depends on:** PR-515

**Status:** QA COMPLETE on branch `pr/PR-516-family-reduction-qa`; independent boundary/chain/PCA
preservation tests pass locally.

#### Acceptance

- [x] Independent reference implementation reproduces every family leader/mapping.
- [x] Cover +0.995, -0.995, 0.994999..., 0.99 subwindow and support boundaries.
- [x] Cover an A-B-C chain where A-B and B-C pass but A-C fails; A and C cannot be collapsed through
  B.
- [x] Prove a 0.95-correlated but not near-duplicate transformation pair remains available to PCA.
- [x] Outer-TEST mutation cannot change any family mapping.
- [x] Row/column/process order cannot alter result or hash.
- [x] QA adds no production behavior.

### PR-484 — Implement family-local standardization and PCA for transformations

**Type:** implementation / dimensionality reduction
**Depends on:** PR-516

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-484-family-local-pca`; targeted PCA and
pipeline tests pass locally. PR-485 remains the independent mathematical/leakage QA proof.

#### Acceptance

- [x] PCA is applied separately to each transformation family; no global PCA is used.
- [x] Core features bypass PCA unchanged.
- [x] Each family uses its own TRAIN-only complete-case model clock; no cross-family complete-case
  intersection is required.
- [x] Standardization parameters are fitted from that family TRAIN clock only using the repository
  population-variance convention.
- [x] Retain PC1 through PCm where m=min(8, numerical rank); explained variance does not choose m.
- [x] PC order is descending explained variance with deterministic tie handling.
- [x] Canonicalize every PC sign by making its largest-absolute loading positive; loading ties use
  canonical source-feature identity.
- [x] PC semantic identity is family plus ordinal, independent of fold-specific loading values.
- [x] TEST and later timestamps are transformed only with frozen TRAIN scaler/loadings.
- [x] Persist scaler/PCA identities, explained-variance diagnostics and all loadings needed for
  feature-credit attribution.
- [x] A family with zero usable numerical rank is statistically invalid for that family only; it
  does not crash unrelated families.

### PR-485 — QA: family-PCA math, leakage and deterministic identity

**Type:** QA only
**Depends on:** PR-484

**Status:** QA COMPLETE on branch `pr/PR-485-family-pca-qa`; independent SVD/rank/leakage QA passes
locally.

#### Acceptance

- [x] Independent NumPy/SVD reference reproduces scaling, rank, PC scores and loadings on fixtures.
- [x] Mutation of Outer-TEST rows cannot alter TRAIN scaler, rank, loading or PC identity.
- [x] Families with 1, 2, 8, 9 and more than 100 transformation features cover the rank/cap
  boundaries.
- [x] No family emits more than eight PCs.
- [x] Explained-variance threshold mutations cannot change retained PC count.
- [x] Column-order and SVD sign reversals preserve canonical PC identities and hashes.
- [x] Two independent families never share a complete-case mask or PCA fit.
- [x] Core features are byte-identical before and after the PCA stage.
- [x] QA adds no production behavior.

### PR-517 — Parallelize independent family pruning/scaling/PCA work

**Type:** implementation / performance
**Depends on:** PR-485

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-517-parallel-family-stage`; targeted pipeline
parity tests pass locally. PR-518 remains the separate family-stage concurrency QA proof.

#### Acceptance

- [x] Execute independent transformation families concurrently through the shared PR-513 process
  pool.
- [x] One family task owns near-duplicate pruning, scaler fit/transform and PCA for that family only;
  core candidates remain coordinator data.
- [x] Submit all eligible family tasks before blocking for completion.
- [x] Reassemble family outputs strictly by canonical family identity, never completion order.
- [x] Reuse the shared fold matrix and pass only family column indices/metadata to workers.
- [x] No nested process/native-thread parallelism is introduced.
- [x] Families with typed statistical invalidity do not cancel unrelated family tasks; unexpected
  software failures abort the stage.
- [x] Serial execution remains available and byte/canonical-hash equivalent.

### PR-518 — QA: family-stage concurrency, utilization and deterministic parity

**Type:** QA only
**Depends on:** PR-517

**Status:** QA COMPLETE on branch `pr/PR-518-family-stage-qa`; canonical and 32-family synthetic
concurrency/parity tests pass locally.

#### Acceptance

- [x] A fixture with at least 32 independent families proves concurrent task execution when CPU
  capacity is available.
- [x] Worker counts 1, 8, 32, 64 and auto produce identical family mappings, scaler/PCA outputs and
  hashes.
- [x] Deliberately delay random family tasks and prove completion order cannot alter output order.
- [x] Prove family matrices are read from shared/memory-mapped storage rather than copied per task.
- [x] Typed invalid family and unexpected-failure fixtures follow the required isolation semantics.
- [x] QA adds no new statistical behavior.

### PR-486 — Implement global stable correlation-leader pruning

**Type:** implementation / redundancy reduction
**Depends on:** PR-518

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-486-global-correlation-leaders`; targeted
global-reduction tests pass locally. PR-487 is the active independent oracle/adversarial-chain QA.

The input universe is exactly quality-eligible core features plus family PCs.

#### Acceptance

- [x] Compute pairwise absolute Pearson correlations from TRAIN rows only.
- [x] Use blockwise/vectorized correlation work and retain only threshold-relevant edges/support
  metadata; do not require a persistent dense NxN artifact.
- [x] A pair is redundant only when full-TRAIN absolute correlation is at least 0.95, pair support
  is at least 30, median absolute correlation across the three chronological TRAIN thirds is at
  least 0.90, and each third has at least 10 paired finite rows.
- [x] If support is insufficient, the pair is not treated as redundant.
- [x] Leader selection is lexicographic: largest number of directly redundant unassigned neighbors;
  then highest median full absolute correlation to those neighbors; then higher TRAIN coverage;
  then core before PC; then canonical candidate identity.
- [x] After selecting a leader, remove only candidates directly redundant with that leader.
- [x] Transitive graph connectivity alone never removes a candidate: A-B and B-C cannot remove C
  through A when A-C fails the redundancy rule.
- [x] Negative and positive correlation use the same absolute threshold.
- [x] Repeat until no unassigned candidate remains; every input maps to exactly one retained leader.
- [x] Persist the full candidate-to-leader mapping and all supporting correlations/counts.
- [x] No HMM fit, target variable or future/OOS information is used.

### PR-487 — QA: independent correlation-leader oracle and adversarial chains

**Type:** QA only
**Depends on:** PR-486

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-487-global-correlation-qa`; independent
oracle and adversarial-chain tests pass locally. No full evaluation was run.

#### Acceptance

- [x] Independent implementation reproduces all retained leaders and mappings.
- [x] Cover +0.95, -0.95, just-below-threshold and insufficient-support boundaries.
- [x] Cover a chain where A-B and B-C pass but A-C fails; C survives when A is leader.
- [x] Cover a case with high crisis-only full-sample correlation but unstable thirds; both features
  must survive.
- [x] Cover exact leader tie-breaks including core-versus-PC and canonical-name fallback.
- [x] Row/column/process completion order cannot alter the result or hash.
- [x] Outer-TEST mutation cannot alter any correlation result.
- [x] QA adds no production behavior.

### PR-519 — Parallelize global correlation as upper-triangle tiles

**Type:** implementation / performance
**Depends on:** PR-487

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-519-parallel-correlation-tiles`; targeted
serial/process parity and global-correlation tests pass locally. No full evaluation was run.

#### Acceptance

- [x] Partition the representative-candidate upper triangle into independent pair tiles using the
  shared PR-513 executor.
- [x] Tile planning creates enough runnable work to consume the available CPU budget when pair count
  permits; no fixed low worker ceiling is introduced.
- [x] Workers receive shared-matrix row/column indices and emit only threshold-relevant edges plus
  support/subwindow statistics.
- [x] The coordinator deterministically merges tile outputs before leader selection.
- [x] No persistent dense NxN correlation artifact is required for production selection.
- [x] Global leader results are bit/canonical-hash equivalent to the serial PR-486 algorithm.
- [x] Native numerical threads remain one per process and workers never create child pools.
- [x] Task backpressure bounds in-flight result memory.

### PR-520 — QA: tiled-correlation scale, parity and failure isolation

**Type:** QA only
**Depends on:** PR-519

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-520-correlation-parallel-qa`; targeted
multi-worker parity and independent-edge tests pass locally. No full evaluation was run.

#### Acceptance

- [x] Independent serial oracle reproduces every qualifying edge and final leader on fixtures.
- [x] Worker counts 1, 8, 32, 64 and auto produce identical mappings/hashes.
- [x] A high-dimensional fixture generates at least four times as many tiles as available workers
  when pair count permits, proving the pool can stay fed.
- [x] Delayed/reordered tile completion cannot alter leader selection.
- [x] Peak correlation-stage memory stays bounded without a dense persisted 10,000x10,000 matrix.
- [x] Injected worker failure aborts unexpected-error runs and leaves no partial committed mapping.
- [x] QA adds no production statistical behavior.

### PR-488 — Add MLflow audit artifacts for quality, PCA and correlation stages

**Type:** implementation / observability
**Depends on:** PR-520

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-488-mlflow-preprocessing-evidence`; the
deterministic evidence bundle, FileStore upload test, artifact identity binding, and MLflow
failure-isolation contract pass locally. No full evaluation was run.

#### Acceptance

- [x] Log one feature_funnel plot with counts for discovered, quality-eligible, family-PC/core,
  correlation-representative and later final stages when available.
- [x] Log one family_survival plot with source count, quality count, retained PCs and representative
  count per family.
- [x] Log one explained-variance curve per PCA family under a deterministic pca/ artifact path.
- [x] Log top-loading plots for every family PC that survives correlation pruning; top 20 absolute
  loadings are shown, with the full loading table persisted separately.
- [x] Log correlation group-size ranking for every retained leader.
- [x] Log representative correlation heatmap for at most 80 representatives, chosen by descending
  covered-group size then canonical identity; the complete correlation mapping remains available
  as a table artifact regardless of plot truncation.
- [x] Log exact profile/source/fold hashes beside every artifact bundle.
- [x] Plot generation never changes selection results.
- [x] QA/local tests use MLflow FileStore; production runs use the configured external tracking URI.
- [x] DuckDB remains the metadata source of truth if MLflow logging is disabled or fails before an
  authorized external run.

### PR-489 — QA: MLflow preprocessing evidence completeness and plot determinism

**Type:** QA only
**Depends on:** PR-488

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-489-mlflow-preprocessing-evidence-qa`;
hermetic FileStore, hash, completeness, missing-artifact, and MLflow-failure-isolation tests pass
locally. No NAS MLflow write and no full evaluation were performed.

#### Acceptance

- [x] A hermetic FileStore run contains every required plot/table for a multi-family fixture.
- [x] Artifact names and payload hashes are deterministic under process completion-order reversal.
- [x] Full loading and correlation tables contain every underlying row even when plots show top-N.
- [x] Heatmap selection obeys the exact 80-representative rule.
- [x] Missing required preprocessing artifacts fail the completeness verifier.
- [x] Plot rendering failures cannot silently alter statistical results.
- [x] No NAS MLflow write occurs in this QA PR.

### PR-490 — Implement process-parallel HMM SFFS on correlation representatives

**Type:** implementation / feature subset search
**Depends on:** PR-489

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-490-process-parallel-hmm-sffs`; all acceptance
criteria are checked and the pushed PR is awaiting merge after the green GitHub gates. The SFFS coordinator now
enforces the strict `1e-12` forward and backward improvement rules, carries all canonical score
tie-break components, forwards the configured worker budget, and exposes every evaluated
candidate for durable metadata conversion. A fixed-K Gaussian selector boundary for K=2,3,4,5
is covered by focused tests. The canonical pipeline now runs the fixed-K Gaussian selector for
K=2,3,4,5, exposes the selected tuple for reuse by every emission family at that K, and commits
each slot's `sffs_steps` atomically. A nested prefix worker no longer creates a child process pool:
candidate work is bounded by worker-local threads with one native numerical lane. No full
evaluation was run.

**Current implementation note:** the immutable `feature_subset_score.v1` data contract and pure
scoring/ranking implementation now exist on the pushed PR-476 branch and are covered by 8 focused
unit tests plus the 154-test feature-discovery/source regression slice. The SFFS coordinator now
uses the existing GIL-independent process pool for pickle-safe score evaluators, with deterministic
serial fallback for non-pickleable test callbacks. The production evaluator remains an injected
TRAIN-only Gaussian-HMM callback; raw likelihood and information criteria cannot enter SFFS ranking.

#### Acceptance

- [x] SFFS input is exactly the retained correlation representatives for the current outer TRAIN
  fold; no removed candidate may re-enter directly.
- [x] Run SFFS independently for each configured K slot; current legal K values remain 2,3,4,5.
- [x] Use the Gaussian full-covariance HMM as the feature selector for each K; the selected tuple is
  then reused by all configured emission families at that K.
- [x] Candidate subset scoring uses a new immutable feature_subset_score.v1 built from the existing
  dimension-independent forecast, calibration, stability, support and valid-fold components used
  by cross_k_score.v1, but with no K-complexity penalty.
- [x] feature_subset_score.v1 uses only monthly inner folds contained inside outer TRAIN and applies
  the same eligibility gates: valid-fold rate at least 0.80, at least three valid inner folds and a
  valid latest inner fold.
- [x] The score is tuning evidence only; Outer TEST remains the sole unbiased fold performance
  evidence.
- [x] Start from the best eligible singleton; after every forward add, perform backward removals
  while the canonical score strictly improves by more than 1e-12.
- [x] Stop when no forward addition improves the score by more than 1e-12 or 10 features are
  selected.
- [x] Ranking ties use total score, forecast score, worst-fold forecast, calibration, stability,
  robustness, fewer features and finally canonical tuple identity.
- [x] All candidate additions/removals within one SFFS step run in bounded process workers with
  deterministic result assembly.
- [x] Reuse existing affinity/cgroup-aware worker sizing; cap native BLAS/OpenMP threads to one per
  worker and forbid nested process-pool oversubscription.
- [x] Persist every evaluated SFFS step/action/score to sffs_steps.
- [x] Raw HMM likelihood, AIC and BIC are logged only as diagnostics and never choose between
  different feature dimensions.

### PR-491 — QA: SFFS score, floating search and CPU determinism

**Type:** QA only
**Depends on:** PR-490

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-491-qa-sffs-score-floating`; QA is
production-code-free and the acceptance suite is implemented locally. No full evaluation was run.

#### Acceptance

- [x] Independently recompute feature_subset_score.v1 component by component.
- [x] Cover no-winner, eligibility-boundary, singleton, forward-add and backward-remove paths.
- [x] A synthetic example proves floating backward removal can remove an earlier selected feature.
- [x] No result may exceed the 10-feature cap.
- [x] Raw PLL/AIC/BIC improvements cannot override a worse dimension-independent score.
- [x] Worker counts 1, 8, 32 and auto produce identical selected tuples, steps and hashes.
- [x] Native thread-pool inspection proves one BLAS/OpenMP thread per process during the QA run.
- [x] Outer-TEST mutation cannot alter SFFS selection.
- [x] QA adds no production behavior.

### PR-492 — Implement final one-at-a-time feature ablation

**Type:** implementation / marginal contribution
**Depends on:** PR-491

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-492-final-feature-ablation`; signed
ablation-loss computation, bounded process execution, invalid-fit evidence and feature-stat
persistence are implemented and under focused verification. No full evaluation was run.

#### Acceptance

- [x] For each final SFFS tuple, refit/re-evaluate exactly one model per selected feature removed.
- [x] Use the same K, selector family, inner-fold plan, seeds and feature_subset_score.v1 contract as
  the final SFFS model.
- [x] Define ablation_loss as final_total_score minus removed_feature_total_score.
- [x] Persist negative, zero and positive ablation_loss values without clipping.
- [x] An invalid ablated model receives an explicit invalid reason and cannot fabricate a numeric
  total score.
- [x] Ablations run in bounded process workers and assemble results in canonical feature order.
- [x] Persist every result in fold_feature_stats.
- [x] Do not automatically remove an SFFS-selected feature merely because its measured ablation
  loss is non-positive in one fold.

### PR-493 — QA: ablation completeness and independent marginal-loss oracle

**Type:** QA only
**Depends on:** PR-492

**Status:** IMPLEMENTATION COMPLETE on branch `pr/PR-493-qa-ablation-oracle`; the independent
oracle and completion-order QA are production-code-free. No full evaluation was run.

#### Acceptance

- [x] Exactly N ablation evaluations exist for an N-feature final tuple.
- [x] Independent recomputation reproduces every finite ablation_loss.
- [x] Cover positive, zero, negative and invalid-ablation cases.
- [x] Completion-order reversal preserves rows and hashes.
- [x] Outer-TEST mutation cannot alter ablation results.
- [x] QA adds no production behavior.

### PR-521 — Flatten HMM selection work into one shared parallel task frontier

**Type:** implementation / performance
**Depends on:** PR-493

**Status:** MERGED as PR #483 into `origin/main`; the
persistent `SharedTaskFrontier` boundary now provides canonical task ordering, completion-order
independence, fail-fast worker errors and queue/runnable/utilization metrics. K-slot SFFS now
reuses one caller-owned frontier across all K slots and SFFS steps, and the canonical selection
pipeline reuses that same frontier for final ablation when both evaluator seams are pickleable.
Multistart and walk-forward now accept the same caller-owned frontier; direct multistart uses the
same frontier boundary instead of a private process-pool path, and parallel fold workers no
longer attempt nested multistart pools. Multistart frontier tasks open one immutable memmapped
TRAIN matrix per batch and carry only its path/shape/dtype identity. Frontier results are restored
to submission identity before coordinator aggregation; the frontier submits only a bounded
in-flight window and replenishes it as workers finish. A batch API now flattens all jobs' eight
seeds into one frontier and aggregates each job canonically. The canonical provisional-teacher
candidate coordinator now owns one Frontier across all K candidates; each candidate batches its
  inner folds and eight seeds through that same pool, and the old candidate-local process-pool path
  is removed. The production provisional-teacher path is therefore flattened through
  `candidate × inner-fold × seed`; the canonical SFFS API now also exposes the explicit
  `FrontierFeatureSubsetEvaluator` for candidate-subset fit evidence. SFFS and final ablation no
  longer create direct candidate-local process pools. The remaining child-pool audit covers the
  candidate-grid/fold orchestration boundaries are now closed: candidate-grid and direct
  walk-forward execution both use one caller-owned shared frontier, with no candidate-local,
  fold-local or multistart-local process pool in the HMM path. Serial parity QA is closed by
  exact SFFS frontier parity and walk-forward fold/model parity tests. No full evaluation was
  run.

SFFS control remains sequential where mathematically dependent, but every independent HMM fit below
that control boundary is flattened onto the one shared process pool.

#### Acceptance

- [x] Decompose candidate scoring into independent fit tasks over the ready
  `K x inner-fold x candidate-subset x seed/start` frontier through the explicit
  `FrontierFeatureSubsetEvaluator`/`FrontierFoldJob` contract.
- [x] Keep SFFS add/remove decisions in the deterministic coordinator; process workers return fit
  evidence only.
- [x] Reuse persistent workers across SFFS steps, K slots and final ablations when the canonical
  evaluator seams are pickleable; the serial/non-pickleable reference path remains explicit.
- [x] Never create candidate-local or multistart-local child pools.
- [x] Submit ready tasks from all K slots fairly so one slow K cannot starve other runnable work;
  independent K coordinators submit concurrently through the shared frontier.
- [x] Aggregate seeds -> inner-fold candidate score -> SFFS decision in canonical identity order,
  independent of task completion order.
- [x] Reuse shared/memory-mapped fold matrices; HMM tasks receive row/column indices and frozen
  preprocessing identities rather than copied feature arrays where backend contracts permit.
- [x] Cache only immutable fit-independent slices/indices; never cache a model result across a
  different feature tuple, fold, seed, K or profile identity.
- [x] Serial reference mode remains statistically identical.
- [x] Record queue depth, runnable tasks, worker utilization proxy, fit count and stage wall time.

Current git status at closure: local `main` and `origin/main` point to the merged PR; the
PR-521 source branch was deleted locally and remotely. The local Hermetic integration hook passed,
no full evaluation was run, and no integration test ran as a GitHub merge gate.

### PR-522 — QA: HMM frontier saturation, nested-pool prohibition and parity

**Type:** QA only
**Depends on:** PR-521

**Status:** MERGED as GitHub PR #484 into `origin/main`; source branch was deleted locally and
remotely. No full evaluation was run, and integration tests were not part of the GitHub merge
gate.

#### Acceptance

- [x] A synthetic selection fixture exposes at least 2x the available CPU count of independent HMM
  fit tasks and proves auto keeps the shared pool supplied until the frontier contracts.
- [x] Worker counts 1, 8, 32, 64 and auto produce identical SFFS paths, selected tuples, ablations
  and canonical hashes.
- [x] Static/runtime inspection proves no HMM worker creates a child process pool or >1 native
  numerical thread.
- [x] Randomized task delays and completion order cannot alter selected tuples.
- [x] Unexpected worker failure aborts the run; typed invalid fits remain candidate evidence only.
- [x] Matrix-transfer instrumentation proves task fan-out does not copy the full fold matrix per fit.
- [x] QA adds no new statistical behavior.

Current git status at QA closure: local `main` and `origin/main` point to the merged PR; only
those branches remain after pruning. Targeted QA tests passed, no full evaluation was run, and
no integration test ran as a GitHub merge gate.

### PR-494 — Persist PCA credit and cumulative cross-fold feature statistics

**Type:** implementation / cumulative metadata
**Depends on:** PR-522

**Status:** MERGED as GitHub PR #485 into `origin/main` at squash commit `9274458`; source branch
was deleted locally and remotely.
PCA-loading persistence, deterministic PCA-credit attribution, cumulative DuckDB view fields,
replay-safe upserts, and cumulative MLflow plot projection are implemented locally. No full
evaluation has been run; integration tests remain local-only and are not a GitHub merge gate.

Current implementation evidence: targeted PR tests 22 passed; mypy, Ruff and formatting passed;
the local Hermetic integration hook passed. The repository-wide non-integration run passed 1,216
tests and exposed six unrelated pre-existing E2E failures in the synthetic PCA-universe and
K-Champion cutoff fixtures. PR-494 remains open until its GitHub unit/type/lint/policy checks are
green.

#### Acceptance

- [x] For each selected family PC, attribute its fold contribution to source transformations by
  squared loading weight.
- [x] Define source_feature_pca_credit as the sum over selected PCs of
  abs(selected_pc_ablation_loss) multiplied by loading_squared.
- [x] Directly selected core features receive direct_selection_count and their own ablation_loss;
  they do not receive synthetic PCA credit.
- [x] feature_global_stats exposes eligible_folds, quality_pass_folds, representative_folds,
  selected_folds, selection_rate, mean/median ablation_loss, total/mean PCA credit,
  last_selected_fold and consecutive_unused_folds.
- [x] Aggregation is deterministic and derived only from committed fold rows.
- [x] Replaying a fold replaces/reuses the same logical fold contribution and never double counts.
- [x] MLflow logs cumulative selection-frequency, mean-ablation and PCA-credit plots after each
  completed fold.
- [x] DuckDB remains authoritative; MLflow plots are projections of committed local statistics.

Current git status at closure before the protected-branch status-only follow-up: local `main`
contained `9274458` and its source branch was deleted locally/remotely; the status-only commit is
being carried into the dependent PR-495 branch. Targeted tests, mypy, Ruff, local Hermetic
integration, and GitHub unit/type/lint/policy gates passed. The repository-wide non-integration
run had six pre-existing E2E fixture failures; no full evaluation was run.

### PR-495 — QA: cumulative statistics and PCA-credit conservation

**Type:** QA only
**Depends on:** PR-494

**Status:** MERGED as GitHub PR #486 into `origin/main` at squash commit `66ddcda`; source branch
was deleted locally and remotely. No full evaluation was run.

#### Acceptance

- [x] Independent SQL/Python aggregation reproduces feature_global_stats exactly.
- [x] Squared loadings for each normalized selected PC distribute exactly its absolute ablation
  contribution within numerical tolerance.
- [x] Fold replay cannot change cumulative counts.
- [x] Failed/uncommitted folds contribute zero cumulative statistics.
- [x] Selection-frequency, ablation and PCA-credit MLflow plots match DuckDB source rows exactly.
- [x] Process/order reversal preserves cumulative hashes.
- [x] QA adds no production behavior.

Current git status at closure: local `main` and `origin/main` are clean at `05e1830`; only
`main`/`origin/main` remained after pruning. QA tests, local Hermetic integration, and GitHub
unit/type/lint/policy gates passed. No full evaluation was run.

### PR-524 — Parallelize independent outer-fold controllers over the shared executor

**Type:** implementation / performance orchestration
**Depends on:** PR-495

**Status:** IMPLEMENTATION MERGED as GitHub PR #487 at squash commit `879d0e9`. The source branch
was deleted locally and remotely. The implementation closes the outer-pool/no-nested-pool
boundary; global frontier/ordered-commit follow-up evidence remains tracked by PR-525 and a
follow-up implementation if required. No full evaluation was run.

#### Acceptance

- [ ] Permit multiple independent historical outer-fold state machines to advance concurrently while
  all CPU-heavy work still uses the single shared PR-513 process pool.
- [ ] Outer-fold controllers themselves are lightweight coordinator tasks and may not spawn process
  pools.
- [ ] Ready work from different folds shares one global task frontier with fair scheduling.
- [ ] Each fold writes an isolated in-memory/local result bundle; DuckDB commits and cumulative
  aggregation occur in canonical fold order.
- [ ] Out-of-order fold completion cannot change cumulative statistics, MLflow identities or hashes.
- [ ] An unexpected software failure in any fold aborts the full evaluation; typed statistical
  invalidity remains fold-local evidence.
- [ ] Production single-month refit can execute one fold without paying multi-fold orchestration
  overhead.
- [ ] Serial outer-fold mode remains available and canonical-hash equivalent.

### PR-525 — QA: multi-fold concurrency, ordered commit and failure semantics

**Type:** QA only
**Depends on:** PR-524

**Status:** QA MERGED as GitHub PR #488 into `origin/main` at squash commit `05e1830`; source
branch was deleted locally and remotely. No full evaluation was run.

#### Acceptance

- [x] At least eight outer folds with deliberately varied runtimes execute concurrently when CPU
  capacity is available.
- [x] Force reverse completion order and prove DuckDB/cumulative outputs equal canonical serial fold
  order.
- [x] Worker counts 1, 8, 32, 64 and auto produce identical final statistical results.
- [x] Unexpected fold failure prevents later committed aggregate state; typed invalid fold does not.
- [x] Single-month deployment/refit path remains unchanged and does not create unnecessary fold
  controllers.
- [x] No nested process pool exists at fold, K, candidate or seed level.
- [x] QA adds no new statistical behavior.

Current git status at QA closure: local `main` and `origin/main` are clean at `05e1830`; only
`main`/`origin/main` remain after pruning. QA tests, local Hermetic integration, and GitHub
policy/lint/type/unit/merge gates passed. No full evaluation was run.

### PR-496 — Add semiautomatic feature lifecycle recommendations

**Type:** implementation / governance
**Depends on:** PR-525

**Status:** IMPLEMENTATION MERGED as GitHub PR #489 into `origin/main` at squash commit `550ecd1`;
source branch was deleted locally and remotely. No production PostgreSQL column is dropped by
this PR, and no full evaluation was run.

Current git status at closure: local `main` and `origin/main` are clean at `550ecd1`; only
`main`/`origin/main` remain after pruning. Lifecycle tests, mypy, Ruff, local Hermetic integration,
and GitHub policy/lint/type/unit/merge gates passed. No full evaluation was run.

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

**Status:** QA MERGED as GitHub PR #490 into `origin/main` at squash commit `ea5c924`; source
branch was deleted locally and remotely. All acceptance evidence passed and no full evaluation
was run.

Current git status at closure: local `main` and `origin/main` were clean at `ea5c924`; only
`main`/`origin/main` remained after pruning. Focused QA, local Hermetic integration, and GitHub
policy/lint/type/unit/merge gates passed. No full evaluation was run.

#### Acceptance

- [x] Boundary fixtures cover 19 versus 20 eligible folds and every lifecycle predicate.
- [x] Any recent selection, representative use or material PCA credit blocks deprecation candidacy.
- [x] Core/raw features cannot become automatically DROPPABLE.
- [x] Search the production feature-selection path and prove no PostgreSQL DROP/ALTER action exists.
- [x] Operator-approved state changes are explicit, auditable and reversible before physical DB
  maintenance occurs outside this pipeline.
- [x] QA adds no production behavior.

---

## Phase 4 — Integrate, visualize and cut over the canonical pipeline

### PR-498 — Integrate the complete feature-selection pipeline into monthly outer refit

**Type:** implementation / orchestration
**Depends on:** PR-497

**Status:** IMPLEMENTATION MERGED as GitHub PR #491 into `origin/main` at squash commit `6aee1b3`;
source branch was deleted locally and remotely. All
acceptance evidence passed and the
canonical closed-month orchestration, immutable package identity, post-fit DuckDB bundle commit,
MLflow parent/child stage manifests, four per-K SFFS slots, explicit provenance stage, and
following-month Outer TEST digest are implemented in `feature_discovery/monthly_refit.py`.
No full evaluation is being run.

#### Acceptance

- [x] The monthly fold flow is exactly quality -> provenance split -> family near-duplicate
  pruning -> family PCA -> global correlation leaders -> per-K SFFS -> ablation -> final HMM fit
  -> Outer TEST.
- [x] Every stage consumes only rows at or before the outer TRAIN cutoff.
- [x] The entire feature-selection pipeline is rerun after each closed calendar month; no intramonth
  feature reselection occurs.
- [x] The resulting feature/PCA/HMM package is frozen for the complete following calendar month.
- [x] Final package identity contains source, month clock, feature-selection profile, provenance,
  PCA, representative mapping, selected tuple, K/family and model hashes.
- [x] Failed selection yields explicit invalid fold evidence and no partial package.
- [x] DuckDB fold transaction commits only after the fold's statistical artifacts are complete.
- [x] MLflow parent/child runs link every stage artifact to the same fold/package identity.
- [x] Historical multi-fold evaluation uses the shared parallel executor/frontier; production
  single-month refit uses the same statistical stages without nested parallelism.
- [x] Deployment/refit uses the latest closed-month TRAIN cutoff and the same pipeline implementation.
- [x] No historical PCA-only-prefix selector remains on this new profile.

Current closure evidence: focused PR-498 QA (2), feature-discovery/calendar unit slice (231),
Ruff and strict Mypy passed; local Hermetic integration passed on `3ab8fe5`; GitHub
policy/lint/type/unit/merge gates passed. Local `main`/`origin/main` are clean at `6aee1b3` after
pruning. No full evaluation was run.

### PR-499 — QA: orchestration leakage, month cadence and stage parity

**Type:** QA only
**Depends on:** PR-498

**Status:** QA IMPLEMENTATION COMPLETE on branch `pr/PR-499-orchestration-cadence-qa` from
`origin/main` at `6aee1b3`; all acceptance evidence is present and no full evaluation is being
run. The QA is ready to push.

#### Acceptance

- [x] Golden test exercises two consecutive month-end refits and two complete OOS months.
- [x] Mutating month m+1 cannot alter the package fitted through month m.
- [x] A midmonth request resolves to the latest closed-month package without refitting.
- [x] Stage identities in DuckDB, package metadata and MLflow are mutually consistent.
- [x] Inject failure after each stage and prove no later stage is presented as complete.
- [x] Serial/process orchestration produces identical canonical package/statistical hashes.
- [x] No generated raw transformation column reaches the HMM directly.

Current evidence: 12 focused QA tests, Ruff and strict Mypy pass; no full evaluation is being
run. Acceptance remains open until the pushed local/GitHub gates pass.

### PR-500 — Cut over Xetra to the scalable canonical pipeline

**Type:** implementation / controlled cutover
**Depends on:** PR-499

**Status:** ACCEPTANCE COMPLETE — branch `pr/PR-500-canonical-xetra-cutover` is
pushed at `d18dbef`; GitHub PR #493 has green Git Policy, lint, strict Mypy,
unit, and merge-gate checks. Based on `origin/main` at `f8a83a4`. The public
canonical orchestration boundary is now
implemented in `commands/canonical_xetra.py` and is covered by focused unit
tests. Per-fold callback factories now bind model evaluation to each closed
month's TRAIN/TEST boundary. The backend now routes evaluation, canonical
package-based refit and OOS publication through that boundary. Raw-source/
implemented, including binding generated Family-PCA TRAIN values into the
process-safe SFFS callbacks. The production refit/serving package now also
supports serialized Family-PCA artifacts and refits the canonical family
transform on the deployment TRAIN window; focused round-trip and backend
package tests cover the new path. Final refit now consumes the canonical
monthly validation identity instead of invoking the superseded generic
walk-forward evaluator. Production-path call-graph QA and complete
acceptance evidence is complete. No full evaluation or NAS data run is part of
this implementation PR; those are owned by the later external QA PRs.
Current evidence: 19 backend/boundary tests plus 58 affected command and
feature-discovery tests, one production call-graph contract test, four
Family-PCA/two-stage tests, Ruff and strict Mypy pass, and the local Hermetic
integration hook passed on the previous commit. No full evaluation and no NAS
data run.

This PR changes authority, not repository cleanup. It makes the new pipeline the only supported
runtime path first; the following PRs then delete now-unreachable legacy and compatibility code.

#### Acceptance

- [x] Promote the scalable feature-selection profile to the sole canonical Xetra evaluation/refit
  path.
- [x] Public evaluation, refit and serving package creation require the new feature-selection profile
  hash and the exact selected semantic feature tuple.
- [x] Public entry points call only the new monthly quality -> family duplicate pruning -> family PCA
  -> global correlation -> SFFS -> ablation pipeline.
- [x] Missing DuckDB is allowed for inference from an already frozen canonical package, but every new
  evaluation/refit creates or uses the local metadata store.
- [x] Legacy selectors/readers may still physically exist only until PR-526/PR-528, but no public or
  production path may invoke them after this cutover.
- [x] No fallback, feature flag, environment toggle, config key or exception path can reactivate a
  superseded selector/source/package builder.
- [x] A production-path call graph captured in QA evidence contains only canonical implementations.
- [x] This PR does not perform broad file/module deletion or structural refactoring; those are owned
  by PR-526 through PR-531.

### PR-526 — Delete legacy statistical, discovery and source runtime code

**Type:** implementation / legacy removal
**Depends on:** PR-500

**Status:** ACCEPTANCE COMPLETE as GitHub PR #494 on branch
`pr/PR-526-legacy-runtime-removal`, pushed at `00f9816` and based on
`origin/main` at `47bf60b`. The obsolete candidate checkpoint store and
multistart seed-resume hooks are removed from the runtime path. The canonical
backend now uses a dedicated MLflow adapter that does not import the retired
evaluation hierarchy. CPU process-pool primitives used by canonical
discovery/training stages now live under `runtime.processes` rather than the
retired evaluation namespace. The obsolete `evaluation_runs`, statistical K/prefix/
teacher selectors, fixed-step walk-forward stack, compatibility MLflow evidence
adapters, and their sole-purpose tests are now removed from the working tree.
The shared task frontier is also now under `runtime.task_frontier`; the former
`evaluations` package has no active implementation left. Canonical import/bootstrap,
Ruff, strict Mypy, and 29 focused backend/refit tests plus 24 runtime-frontier tests
pass locally. The local zero-legacy verifier and GitHub gates are green; no full
NAS evaluation was run. PR-526 is ready to merge.

Git history is the archive. The repository must not retain executable implementations for
superseded statistical/source paths merely to preserve backwards compatibility.

#### Acceptance

- [x] Delete the superseded PCA-only-prefix/L* selector implementation and every production wrapper,
  adapter, registry entry and feature flag used only by it.
- [x] Delete the old clustering, hierarchy-cut, medoid, teacher-HMM and backward-elimination
  discovery/selection implementations when they have no canonical use after PR-500.
- [x] Delete raw-plus-generated-PCA and caller-supplied PCA-switch compatibility paths that are not
  part of the new family-PCA pipeline.
- [x] Delete canonical-source fallbacks for `macro_loader.macro_features_daily`,
  `macro_loader.macro_raw`, schema-wide relation enumeration and caller-selected feature relations.
- [x] Delete fixed-row/fixed-step walk-forward constructors that are superseded by the canonical
  calendar-month clock and are not used by generic test utilities with an explicit nonproduction
  purpose.
- [x] Delete deprecated computation-position/resume/checkpoint compatibility code from the full
  Xetra evaluator where the canonical contract is one-shot/non-resumable; keep only infrastructure
  that is still used by an explicitly current component.
- [x] Remove production imports, exports, dependency-injection bindings, factories, registries and
  CLI/config switches that reference any deleted implementation.
- [x] Remove tests and fixtures whose sole purpose is to validate deleted behavior; do not weaken
  tests for shared/current primitives.
- [x] Do not leave forwarding stubs, deprecation shims, aliases or `NotImplemented` placeholders for
  removed runtime paths.
- [x] Package import and bootstrap succeed after deletion with no optional import of deleted modules.
- [x] Canonical statistical behavior and hashes from PR-500 remain unchanged.

### PR-527 — QA: prove zero legacy statistical/source runtime remains

**Type:** QA only
**Depends on:** PR-526

**Status:** ACCEPTANCE COMPLETE — GitHub PR #495 passed all policy, lint, strict-mypy, unit and
merge gates and is ready to merge. The QA is local and synthetic-only; no NAS or MLflow
evaluation has been run. It distinguishes retired selector/source fallback symbols from
canonical feature-discovery terminology that remains part of the current profile contract.

#### Acceptance

- [x] Static repository scan finds no production symbol/import/config key for PCA-only-prefix/L*,
  medoid, teacher-HMM, clustering selector, raw-plus-PCA selector or old source fallbacks.
- [x] Static import-graph traversal proves no canonical module has an optional/dynamic import path to
  a deleted implementation.
- [x] CLI/config mutation tests prove historical selector/source switches are rejected as unknown
  rather than silently ignored.
- [x] A fixture exposing `macro_features_daily`, `macro_raw` and `macro_features` proves only
  `macro_loader.macro_features` can be used.
- [x] Month-clock tests prove no fixed 63-row, 42-row or 21-trading-day production constructor remains
  reachable.
- [x] One canonical evaluation fixture reproduces the pre-deletion PR-500 statistical hashes.
- [x] Python package build/import, Ruff, strict mypy, unit and hermetic integration suites pass.
- [x] QA adds no production behavior.

### PR-528 — Delete compatibility-only profile, package, artifact and serving code

**Type:** implementation / compatibility removal
**Depends on:** PR-527

**Status:** ACCEPTANCE COMPLETE locally — branch `pr/PR-528-delete-compatibility-boundaries` is
based on `origin/main` at `16a6596`; final GitHub validation and merge remain pending. No NAS,
PostgreSQL, MLflow or full evaluation run has been performed.

The current canonical profile/package/serving contract is the only supported contract. Historical
model packages and run artifacts remain available in MLflow/Git history as data, but this repository
does not keep compatibility code to execute or reinterpret superseded contracts.

#### Acceptance

- [x] Delete v1-v3 Xetra profile/config compatibility models, parsers, upgrade adapters and runtime
  branches that are not part of the sole current profile.
- [x] Delete legacy feature-order/package schema readers, deserializers, migration adapters and
  fallback package loaders that accept packages lacking the canonical feature-selection profile
  hash, selected semantic tuple or required current lineage.
- [x] Delete historical artifact readers whose only purpose is to normalize superseded evaluation
  evidence into current structures.
- [x] Delete deprecated CLI arguments, config aliases, environment-variable aliases and API request
  fields retained only for backwards compatibility.
- [x] Delete serving/refit fallback branches that infer missing current package fields or reconstruct
  historical feature-selection state.
- [x] Reject an old package/config/API payload explicitly at the current validation boundary; do not
  auto-upgrade it.
- [x] Do not mutate, delete or rewrite historical MLflow runs, registered model versions, artifacts
  or PostgreSQL data as part of compatibility-code removal.
- [x] Keep only current public API/profile/package types and their exact validators/serializers.
- [x] Remove tests/fixtures that assert acceptance of historical contracts and replace them with
  rejection tests at the current boundary.
- [x] No compatibility layer remains solely because an old artifact exists remotely.

### PR-529 — QA: current-contract-only package and serving boundary

**Type:** QA only
**Depends on:** PR-528

**Status:** ACCEPTANCE COMPLETE — branch `pr/PR-529-current-contract-package-serving-qa` is
validated locally and by GitHub PR #497; QA was local/synthetic-only and did not access NAS,
PostgreSQL, MLflow, or run the full evaluation. Merge to `origin/main` is the remaining gate.

#### Acceptance

- [x] Historical v1-v3 profile/config fixtures are rejected.
- [x] Historical package fixtures missing current profile hash, semantic feature tuple, PCA state or
  required lineage are rejected without migration.
- [x] Deprecated CLI/config/environment/API names fail closed as unknown/invalid.
- [x] Static scan finds no `legacy`, `compat`, `deprecated`, old-schema upgrader or historical
  package-reader module on a production import path; legitimate terminology in migration/history
  comments is excluded by an explicit allowlist.
- [x] Current package round-trip, registry readback and serving invocation remain byte/canonical-hash
  stable.
- [x] No test depends on a historical artifact to make current code import or execute.
- [x] QA adds no production behavior.

### PR-530 — Refactor the canonical-only implementation after legacy deletion

**Type:** implementation / structural refactor
**Depends on:** PR-529

**Status:** ACCEPTANCE COMPLETE — branch `pr/PR-530-canonical-only-structural-refactor` is
validated locally and by GitHub PR #498; structural work was local/synthetic-only, with no NAS,
PostgreSQL, MLflow, or full evaluation run. Merge to `origin/main` is the remaining gate.

Legacy deletion is expected to expose abstractions, wrappers and version-specific names that no
longer serve a second implementation. This PR simplifies them without changing statistical
behavior.

#### Acceptance

- [x] Remove interfaces, adapters, factories, strategy branches and dependency-injection bindings
  that have only one implementation and no independently useful test seam after PR-528.
- [x] Collapse duplicate canonical orchestration helpers so evaluation and deployment/refit call the
  same stage implementations rather than parallel copies.
- [x] Remove dead configuration fields, unused dataclasses, unreachable branches, stale metrics and
  unused serialization fields revealed by legacy deletion.
- [x] Rename internal modules/functions whose names describe superseded mechanics rather than their
  canonical responsibility; update all imports atomically.
- [x] Keep externally intentional identities such as repository/package name, `profile_id=xetra`,
  registered model identity and current API contract unchanged unless another active backlog item
  explicitly owns that change.
- [x] Establish one directional dependency flow:
  source/contracts -> preprocessing/selection -> HMM evaluation -> packaging/lifecycle -> serving.
- [x] No circular imports are introduced; package import must not execute network/database work.
- [x] Shared parallel-execution, DuckDB and MLflow abstractions remain single-source and are not
  duplicated during refactor.
- [x] Refactor produces identical canonical statistical/package hashes on pinned fixtures.
- [x] Ruff, formatting, strict mypy and required unit/integration tests pass at 85% coverage.

### PR-531 — QA: canonical-only import graph, dead-code and refactor proof

**Type:** QA only / structural acceptance
**Depends on:** PR-530

**Status:** IN PROGRESS — branch `pr/PR-531-canonical-only-import-graph-qa` is based on
`origin/main` at `26cf90f`; QA is local/synthetic-only and does not access NAS, PostgreSQL,
MLflow, or run the full evaluation.

#### Acceptance

- [ ] Build an import/dependency graph and prove the canonical layer direction required by PR-530.
- [ ] Reject circular imports and forbidden reverse dependencies.
- [ ] Static dead-code scan plus explicit import/export inventory finds no orphan production modules,
  public exports or configuration fields left by removed legacy paths.
- [ ] Repository-wide search verifies no compatibility shim, forwarding alias or deprecated runtime
  switch can recreate deleted behavior.
- [ ] Canonical CLI/evaluation/refit/serving smoke tests all exercise the same source and stage
  implementations.
- [ ] Pre-refactor pinned canonical fixtures reproduce identical statistical, package and evidence
  hashes.
- [ ] Clean environment package build/install/import succeeds without historical modules.
- [ ] Required lint/type/unit/integration/coverage gates pass.
- [ ] QA adds no production behavior.

### PR-511 — Consolidate repository documentation into one guided onboarding path

**Type:** documentation / architecture cleanup
**Depends on:** PR-531

The final repository-authored contract/onboarding Markdown set is intentionally small:

~~~text
README.md
ARCHITECTURE.md
EVALUATION.md
OPERATIONS.md
CONTRIBUTING.md
BACKLOG.md
~~~

Other legal/license files are unaffected.

#### Acceptance

- [ ] README.md is the single entry point and onboards a new user step by step: purpose -> data source
  -> bootstrap -> local test -> evaluation -> MLflow inspection -> production/refit -> where to read
  next.
- [ ] ARCHITECTURE.md owns system boundaries, package/component structure, data flow, local DuckDB,
  MLflow ownership and the shared parallel-execution architecture; it does not restate statistical
  formulas.
- [ ] EVALUATION.md owns monthly fold semantics, core/transformation roles, family near-duplicate
  pruning, family PCA, global correlation leaders, SFFS, ablation, HMM/K/family evaluation and
  statistical acceptance rules; it does not restate deployment commands.
- [ ] OPERATIONS.md is created and owns the exact `macro_loader.macro_features` materialized-view
  source contract, configuration/secrets, dataset pinning, one-shot execution, MLflow/registry
  operations, deployment/refit/readback and failure/recovery runbooks.
- [ ] CONTRIBUTING.md owns only developer setup, PR naming, tests/gates, code-quality rules and
  atomic-PR workflow.
- [ ] BACKLOG.md remains the only planning/acceptance document and does not duplicate normative
  prose from the five user/developer docs.
- [ ] Migrate still-valid content from DATA_SOURCE.md and EVALUATION_EXECUTION.md into OPERATIONS.md,
  then delete those two files.
- [ ] Migrate still-valid rendering/MLflow plot rules from PLOT_STYLE.md into the diagnostics section
  of EVALUATION.md or ARCHITECTURE.md as appropriate, then delete PLOT_STYLE.md.
- [ ] Remove stale/superseded architecture text and all compatibility instructions rather than
  documenting removed code paths; Git history and the condensed historical backlog are the archive.
- [ ] Every structural/process explanation uses Mermaid where a diagram is clearer than prose:
  README onboarding, system architecture, feature-selection flow, monthly evaluation flow,
  parallel task graph, operations/deployment flow and contributing/CI flow.
- [ ] Mermaid diagrams render with valid GitHub Mermaid syntax and contain no secrets/internal
  credentials.
- [ ] Cross-links form one acyclic reading path from README to the specialized owner document; no
  two files claim authority for the same contract.
- [ ] Documentation explicitly names `macro_loader.macro_features`; no active documentation names
  `macro_features_daily` as the feature source.

### PR-512 — QA: documentation completeness, uniqueness and Mermaid validation

**Type:** QA only / documentation
**Depends on:** PR-511

#### Acceptance

- [ ] Repository contract/onboarding Markdown inventory is exactly README.md, ARCHITECTURE.md,
  EVALUATION.md, OPERATIONS.md, CONTRIBUTING.md and BACKLOG.md, excluding legal/license metadata.
- [ ] DATA_SOURCE.md, EVALUATION_EXECUTION.md and PLOT_STYLE.md no longer exist.
- [ ] Link checker resolves every internal documentation link.
- [ ] Static owner-keyword checks prove source/operations, statistical evaluation, architecture,
  contributing and backlog contracts have exactly one owning document.
- [ ] Active docs contain no `macro_loader.macro_features_daily`, old PCA-only-prefix selector,
  clustering/medoid/teacher canonical path or contradictory monthly cadence.
- [ ] Parse every Mermaid fenced block with the repository's pinned Mermaid validation tool or a
  deterministic syntax validator.
- [ ] README walkthrough commands are executed in a hermetic smoke test where possible and clearly
  mark external-only commands.
- [ ] No required information disappears when the retired docs are removed; a migration checklist
  maps every retained old section to its new owner or marks it intentionally superseded.
- [ ] QA changes no production behavior.

### PR-501 — QA: zero-legacy and full hermetic multi-fold pipeline proof

**Type:** QA only / full local acceptance
**Depends on:** PR-512

#### Acceptance

- [ ] Static import/config/package scan proves no canonical entry point can select or load the old
  PCA-only-prefix, clustering, medoid, teacher, legacy-source, historical-profile or compatibility
  package paths.
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
- [ ] Candidate HMM fits use the shared process frontier; worker counts 1, 8, 32, 64 and auto
  remain statistically identical.
- [ ] Multi-fold evaluation uses concurrent fold controllers and the same global worker pool; no
  nested process pools are created.
- [ ] Native numerical thread pools remain one thread per worker.
- [ ] Full feature matrices are shared/read-only or memory-mapped where worker fan-out would
  otherwise copy them.
- [ ] Peak resident memory for the acceptance run stays below 64 GiB.
- [ ] The run records wall time, peak RSS, candidate counts per stage and effective worker count in
  MLflow and the local fold metadata.
- [ ] No Spark, Ray or external distributed-compute dependency is introduced.

---

### PR-523 — Benchmark and tune parallel throughput on the target 86-vCPU host

**Type:** manual/performance QA
**Depends on:** PR-502
**Runs on:** authorized target host class with approximately 86 vCPUs and 256 GiB RAM

#### Acceptance

- [ ] Run the same fixed production-shaped workload with worker budgets 1, 8, 16, 32, 48, 64, 80
  and auto, bounded by the host's actual affinity/cgroup capacity.
- [ ] Benchmark family preprocessing, global correlation tiles, HMM task frontier and complete
  multi-fold evaluation separately.
- [ ] Prove every worker-budget run produces identical canonical statistical hashes.
- [ ] Record wall time, CPU time, peak RSS, task throughput, queue starvation time and effective
  worker count for each stage.
- [ ] During stages with at least 2x as many runnable tasks as available CPUs, explain any sustained
  worker idleness greater than 10%; eliminate scheduler starvation before acceptance.
- [ ] Verify no native-thread oversubscription and no nested process pools on the real host.
- [ ] Select/document the fastest statistically identical safe runtime setting for this host; do not
  assume that the numerically largest worker count is fastest.
- [ ] Peak RSS remains below the PR-502 resource bound and the host remains responsive.
- [ ] Store the benchmark report in MLflow/local acceptance artifacts; do not mutate registry aliases
  or PostgreSQL features.

---

## Phase 5 — External/current-source acceptance and publication

### PR-503 — Run the current-Xetra read-only feature-selection audit

**Type:** external QA / read-only
**Depends on:** PR-523 and a production-eligible upstream source snapshot

#### Acceptance

- [ ] Query exactly `macro_loader.macro_features` read-only and capture materialized-view
  version/fingerprint plus exact source/catalog identities.
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

~~~mermaid
flowchart TD
    P448[448 backlog consolidation] --> P449[449 CI coverage]
    P449 --> P450[450 coverage QA]
    P450 --> P451[451 integration gates]
    P451 --> P452[452 gate QA]
    P452 --> P453[453 failure semantics]
    P453 --> P454[454 failure QA]
    P454 --> P507[507 monthly clock]
    P507 --> P508[508 clock QA]
    P508 --> P476[476 feature-selection contract]
    P476 --> P477[477 contract QA]
    P477 --> P509[509 macro_features source]
    P509 --> P510[510 source QA]
    P510 --> P478[478 DuckDB metadata]
    P478 --> P479[479 metadata QA]
    P479 --> P480[480 provenance]
    P480 --> P481[481 provenance QA]
    P481 --> P482[482 quality filter]
    P482 --> P483[483 quality QA]
    P483 --> P513[513 shared parallel planner]
    P513 --> P514[514 planner QA]
    P514 --> P515[515 family near-duplicates]
    P515 --> P516[516 family duplicate QA]
    P516 --> P484[484 family PCA]
    P484 --> P485[485 PCA QA]
    P485 --> P517[517 parallel family stages]
    P517 --> P518[518 family parallel QA]
    P518 --> P486[486 global correlation leaders]
    P486 --> P487[487 correlation oracle QA]
    P487 --> P519[519 parallel correlation tiles]
    P519 --> P520[520 correlation parallel QA]
    P520 --> P488[488 MLflow preprocessing evidence]
    P488 --> P489[489 MLflow QA]
    P489 --> P490[490 HMM SFFS]
    P490 --> P491[491 SFFS QA]
    P491 --> P492[492 ablation]
    P492 --> P493[493 ablation QA]
    P493 --> P521[521 flattened HMM frontier]
    P521 --> P522[522 HMM parallel QA]
    P522 --> P494[494 cumulative statistics]
    P494 --> P495[495 cumulative QA]
    P495 --> P524[524 concurrent outer folds]
    P524 --> P525[525 outer-fold parallel QA]
    P525 --> P496[496 lifecycle]
    P496 --> P497[497 lifecycle QA]
    P497 --> P498[498 monthly orchestration]
    P498 --> P499[499 orchestration QA]
    P499 --> P500[500 canonical cutover]
    P500 --> P526[526 delete legacy stats/source]
    P526 --> P527[527 zero-legacy QA]
    P527 --> P528[528 delete compatibility code]
    P528 --> P529[529 current-contract QA]
    P529 --> P530[530 canonical-only refactor]
    P530 --> P531[531 refactor/dead-code QA]
    P531 --> P511[511 docs consolidation]
    P511 --> P512[512 docs QA]
    P512 --> P501[501 full hermetic proof]
    P501 --> P502[502 10k resource QA]
    P502 --> P523[523 target-host benchmark]
    P523 --> P503[503 current-source audit]
    P503 --> P504[504 external MLflow QA]
    P504 --> P505[505 registry durability]
    P505 --> P506[506 authorized publication]
~~~

Planning PRs PR-232, PR-250, PR-423..PR-430, PR-455..PR-475 and the previous contents
of PR-476..PR-506 describing the PCA-only-prefix architecture are superseded and are not active
execution items. PR-509..PR-531 are new planning identities introduced by the scalable-source,
parallel-runtime, zero-legacy/compatibility, refactor and documentation redesign.

---

## Current architectural decisions and non-goals

- **Target selection redesign:** PR-507/PR-508 plus the rewritten PR-476–PR-506 define the
  canonical scalable feature pipeline: exact `macro_loader.macro_features` source -> TRAIN-only
  quality -> provenance split -> family near-duplicate pruning -> family PCA for transformations ->
  global stable absolute-Pearson correlation leaders across core+PCs -> per-K HMM SFFS -> final
  ablation -> frozen next-month package.
- **Core vs transformations:** core features remain direct interpretable candidates. Generated
  transformations never enter the HMM directly; they are represented only by family PCs.
- **Family preprocessing:** transformations first pass stable near-duplicate pruning at 0.995/0.99,
  then family-local TRAIN-only rank-limited PCA capped at eight PCs. Explained variance is diagnostic
  only.
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
- **Zero legacy/compatibility policy:** after PR-526–PR-531, only the current Xetra pipeline,
  source contract, package schema and serving contract exist in executable code. v1-v3 compatibility,
  old selectors, old source adapters, historical package readers and deprecated flags are deleted,
  not hidden behind shims. Historical artifacts remain immutable external data; Git history is the
  code archive.
- Production packages are published to the external NAS MLflow service at
  http://10.10.1.3:5000; local FileStore is used for hermetic QA.
- Feature PostgreSQL and MLflow are external dependencies; Docker/Compose services do not belong to
  this repository.
- **Parallel runtime:** one shared affinity/cgroup-aware process pool is reused across family work,
  correlation tiles and flattened HMM fit tasks; historical outer folds advance concurrently through
  coordinator state machines. Workers never spawn child pools, native BLAS/OpenMP/NumExpr threads are
  one per process, large fold matrices are shared/read-only or memory-mapped, and no Spark/Ray layer
  is introduced.
- The full Xetra v4 evaluator remains intentionally non-resumable. Interrupted full runs restart
  from the beginning; local DuckDB fold metadata is committed transactionally only for completed
  fold-selection results.
- State identities are fold/model-version local; semantic labels must not leak into discovery,
  correlation pruning or statistical ranking.

## Current external dependency state

As of the 2026-09-20 planning cutover:

- canonical feature relation for the new profile: `macro_loader.macro_features` materialized view;
- historical `macro_loader.macro_features_daily` counts/build evidence are superseded and are not
  valid acceptance evidence for this pipeline;
- exact live row count, feature count, materialized-view version/fingerprint and source lineage must
  be re-read by PR-503 before external acceptance;
- production MLflow remains the external service at `http://10.10.1.3:5000`;
- no production feature deletion, model publication or alias mutation is authorized by this backlog
  rewrite.

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

- Planning PRs PR-232, PR-250, PR-423–PR-430, PR-455–PR-475 and the former PCA-only-prefix contents of PR-476–PR-506 are superseded by the rewritten scalable feature-selection plan plus PR-509–PR-531; superseded definitions survive only in Git/GitHub history and must not be implemented from old text.
- GitHub #277, #284 and #317 were closed without merge and are superseded by later
  merged work.
- Draft planning IDs PR-186–PR-206 are superseded and must not be implemented.
- Historical requirements for v1-v3 compatibility or full-run computation-position
  resume are superseded by the architectural decisions above.
