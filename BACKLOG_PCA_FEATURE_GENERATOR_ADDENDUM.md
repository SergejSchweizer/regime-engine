# Regime Engine — PCA Feature Generator Backlog Addendum

Status date: 2026-09-12

This addendum extends `BACKLOG.md` with a PCA feature-generation wave for the active Xetra v4 architecture. PCA does **not** replace raw-feature discovery. It creates additional component-score time series, appends them to the same non-semantic feature universe as raw features, and then lets the existing v4 discovery/selection/HMM pipeline decide whether any PC survives.

The planning IDs continue the repository namespace after PR-254.

---

# 1. Weak-agent execution rules

These PRs are intentionally small, atomic and strictly sequential.

1. Start a PCA PR only after its immediately preceding PCA PR is merged to clean current `main`.
2. Use the exact branch name listed below.
3. Implement only the stated single purpose. No opportunistic refactor, rename, dependency upgrade or unrelated cleanup.
4. Touch only **Allowed** files plus the minimum package export required for the new symbol. If another file is needed, stop and amend the backlog first.
5. Do not change pinned constants, formulas, clocks, names, ordering rules, tolerances or fallback semantics.
6. Do not invent a fallback. Fail explicitly when a contract cannot be satisfied.
7. Every PR adds focused tests for its own behavior; later PRs may integrate earlier behavior but may not redefine it.
8. Every PR must pass `ruff`, strict `mypy`, focused tests and the repository merge gate required at that stage.
9. PR text must state the planning ID and must not claim behavior scheduled for a later PR.
10. A weak agent should be able to implement one PR by reading only this addendum, the immediately preceding merged PR, the listed files and their tests.

No PCA PR may be implemented in parallel with another PCA PR. The linear chain is deliberate so every change is independently reviewable and revertible.

---

# 2. Canonical PCA contract

```text
validated raw feature universe
        |
        +--> raw features -----------------------------------------+
        |                                                          |
        +--> resolve PCA sources on pinned earliest inner TRAIN    |
              -> TRAIN-only StandardScaler                         |
              -> deterministic PCA                                 |
              -> smallest k with cumulative EVR >= 0.90            |
              -> frozen PCA artifact                               |
              -> pca_pc_001 ... pca_pc_k --------------------------+
                                                                   |
                                                        combined feature universe
                                                                   |
                                                        existing v4 discovery /
                                                        scoring / selection
                                                                   |
                                                        retained raw + PC tuple
                                                                   |
                                                        existing per-fold TRAIN-only
                                                        StandardScaler
                                                                   |
                                                        HMM / OOS inference
```

Pinned rules:

- PCA is a **feature generator**, not a privileged HMM path.
- PCA source order is canonical raw-feature ordinal order.
- A raw feature is PCA-eligible only if it is structurally valid, finite/non-null on every row of the pinned PCA fit sample, and has population variance strictly `> 1e-12` there. This stricter completeness rule exists because imputation is forbidden.
- PCA requires at least `3` eligible sources and at least `504` fit rows.
- PCA source standardization reuses the existing `StandardScalerArtifact` semantics: TRAIN-only mean, population variance `ddof=0`, scale equal to square root of population variance.
- Stored PCA eigenvalues use population covariance: squared singular value divided by fit-row count.
- EVR is `eigenvalue_i / sum(eigenvalues)`.
- Retain exactly the smallest `k` with cumulative EVR `>= 0.90`.
- No Kaiser/eigenvalue rule such as `lambda > 1` is applied.
- Canonical sign: locate the largest absolute loading in a component; ties use smaller source ordinal; force that loading positive.
- Generated names are exactly `pca_pc_001`, `pca_pc_002`, ... with one-based three-digit indexes.
- Generated PCs are appended after all raw features in component order; raw ordinals never change.
- PCs receive no ranking bonus or exemption. Existing v4 quality, distance, clustering, scoring, cluster winner, prefix `L*`, final grid and HMM rules apply normally.
- A selected PC receives the existing per-fold TRAIN-only HMM `StandardScaler` exactly like a selected raw feature.
- No imputation, interpolation, carry-forward, backfill or OOS-informed normalization is allowed.
- Within one adaptive selection run, fit exactly one PCA artifact on the **earliest admissible inner TRAIN** and freeze it for all later timestamps in that selection run.
- Across outer folds, PCA artifacts may differ because the complete selection policy reruns TRAIN-only in every outer fold.
- Deployment selection creates its own frozen PCA artifact through the same selection function; production has no separate PCA fitting rule.

Exact unavailable reason codes:

```text
pca_disabled
pca_too_few_sources
pca_too_few_fit_rows
pca_nonfinite_fit_matrix
pca_decomposition_failed
```

`PCA unavailable` means zero PCs are generated and the raw universe continues unchanged. It is not by itself an evaluation failure. No other fallback or reason code is allowed without a contract PR.

---

# 3. Sequential implementation plan

| PR | Single purpose | Depends on |
|---|---|---|
| 255 | authoritative statistical contract | PR-241, PR-254 |
| 256 | immutable artifact/evidence types | 255 |
| 257 | numerical PCA fit kernel | 256 |
| 258 | frozen PCA transform kernel | 257 |
| 259 | PCA source resolver | 258 |
| 260 | one-shot PCA generator orchestration | 259 |
| 261 | PC time-series materialization | 260 |
| 262 | combined raw+PC candidate universe | 261 |
| 263 | distance/clustering integration | 262 |
| 264 | regime scoring/winner integration | 263 |
| 265 | prefix/final-feature selection integration | 264 |
| 266 | inner walk-forward freeze + HMM scaling | 265 |
| 267 | outer TEST integration | 266 |
| 268 | deployment-selection integration | 267 |
| 269 | final-refit/package persistence | 268 |
| 270 | latest/replay serving | 269 |
| 271 | MLflow PCA evidence | 270 |
| 272 | matched raw-only vs raw+PCA evaluation | 271 |
| 273 | explicit profile/config switch | 272 |
| 274 | hermetic end-to-end proof | 273 |
| 275 | documentation closure | 274 |

## PR-255 — Freeze the authoritative PCA statistical contract

- **Branch:** `pr/PR-255-pca-contract`
- **Depends on:** PR-241, PR-254
- **Allowed:** `EVALUATION.md`, directly linked documentation-contract tests only.
- **Purpose:** documentation contract only; no runtime code.

Acceptance:
- [ ] `EVALUATION.md` states every rule in Section 2 exactly, including `0.90`, `504`, source completeness, sign rule, names, fit clock and unavailable codes.
- [ ] It distinguishes PCA source scaling from final HMM observation scaling.
- [ ] It states that PCs join the same universe and receive no selection bonus.
- [ ] No executable behavior changes.

Not in scope: Python implementation, config, MLflow, package, serving.

## PR-256 — Add immutable PCA artifact/evidence types

- **Branch:** `pr/PR-256-pca-artifact-contracts`
- **Depends on:** PR-255
- **Allowed:** new `src/market_regime_engine/feature_generation/__init__.py`, new `src/market_regime_engine/feature_generation/contracts.py`, new `tests/unit/feature_generation/test_contracts.py`.
- **Purpose:** validated data structures only; no PCA math.

Acceptance:
- [ ] `PcaArtifact` stores ordered sources, source `StandardScalerArtifact`, retained loadings/eigenvalues/EVRs/cumulative EVRs, `k`, fit row count, fit bounds, `variance_target=0.90`, sign-rule version and canonical SHA-256.
- [ ] `PcaGenerationEvidence` represents either available artifact hash or one exact unavailable reason code.
- [ ] Validate all dimensions, duplicate names, finite/nonnegative EVRs, monotone cumulative EVR and final cumulative EVR `>=0.90`.
- [ ] Canonical serialization/hash is deterministic.

Not in scope: fitting, source selection, transformation.

## PR-257 — Implement deterministic PCA fit kernel

- **Branch:** `pr/PR-257-pca-fit-kernel`
- **Depends on:** PR-256
- **Allowed:** new `src/market_regime_engine/feature_generation/pca_fit.py`, new `tests/unit/feature_generation/test_pca_fit.py`.
- **Purpose:** complete finite TRAIN matrix -> `PcaArtifact`.

Acceptance:
- [ ] Input is only finite 2-D matrix, exact ordered source tuple and fit timestamps/bounds.
- [ ] Require rows `>=504`, columns `>=3`; fit existing `StandardScalerArtifact` on this matrix only.
- [ ] Use one deterministic full decomposition path; eigenvalues are squared singular values divided by `n`.
- [ ] Compute EVR/cumulative EVR; retain smallest `k` reaching `0.90`; apply canonical sign; no randomized solver/Kaiser cutoff.
- [ ] Identical input bytes/order produce identical artifact hash in the pinned runtime.

QA: independent covariance/eigenvalue fixture, exact `k` boundary test, sign-canonicalization fixture.

## PR-258 — Implement frozen PCA transform kernel

- **Branch:** `pr/PR-258-pca-transform-kernel`
- **Depends on:** PR-257
- **Allowed:** new `src/market_regime_engine/feature_generation/pca_transform.py`, new `tests/unit/feature_generation/test_pca_transform.py`.
- **Purpose:** frozen artifact + complete source rows -> PC scores; never fit.

Acceptance:
- [ ] Require exact source order/dimension.
- [ ] Apply stored source scaler then stored retained loading matrix.
- [ ] Return exactly `T x k` finite scores for complete finite input.
- [ ] No fitting path and no artifact mutation.

Not in scope: DataFrame timestamps, missing-row policy, universe integration.

## PR-259 — Resolve PCA source features and pinned fit matrix

- **Branch:** `pr/PR-259-pca-source-resolver`
- **Depends on:** PR-258
- **Allowed:** new `src/market_regime_engine/feature_generation/source.py`, new `tests/unit/feature_generation/test_source.py`.
- **Purpose:** earliest-inner-TRAIN frame -> deterministic PCA source tuple/fit matrix.

Acceptance:
- [ ] Consume only explicit earliest admissible inner-TRAIN frame plus canonical raw catalog.
- [ ] Eligible source must be numeric, finite/non-null on every fit row and population variance `>1e-12`; preserve canonical ordinal order.
- [ ] Return exact source tuple, fit matrix, timestamps and exclusion reasons.
- [ ] `<3` sources -> `pca_too_few_sources`; `<504` fit rows -> `pca_too_few_fit_rows`; no substitute selection or imputation.

QA: later-row perturbation cannot change resolver output.

## PR-260 — Compose one-shot PCA generator

- **Branch:** `pr/PR-260-pca-generation-run`
- **Depends on:** PR-259
- **Allowed:** new `src/market_regime_engine/feature_generation/generator.py`, new `tests/unit/feature_generation/test_generator.py`.
- **Purpose:** one orchestration call that returns one frozen artifact or explicit unavailable evidence.

Acceptance:
- [ ] Call PR-259 resolver then PR-257 fit at most once.
- [ ] Success returns one artifact + available evidence.
- [ ] Pinned precondition failures return zero artifact + exact reason; decomposition exception maps only to `pca_decomposition_failed`.
- [ ] No alternate solver/fallback and no mutation of raw input.

## PR-261 — Materialize PC time-series columns

- **Branch:** `pr/PR-261-pca-materialization`
- **Depends on:** PR-260
- **Allowed:** new `src/market_regime_engine/feature_generation/materialize.py`, new `tests/unit/feature_generation/test_materialize.py`.
- **Purpose:** frozen artifact + timestamped raw frame -> named PC columns.

Acceptance:
- [ ] Names exactly `pca_pc_001 ... pca_pc_k`; preserve timestamp order and row count.
- [ ] Complete source row -> PR-258 transform; any required source null/non-finite -> null for all PCs on that timestamp; never impute.
- [ ] Attach component index, eigenvalue, EVR, cumulative EVR, artifact hash and fit cutoff provenance.
- [ ] Unavailable evidence creates zero PC columns.

## PR-262 — Append PCs to the v4 candidate universe

- **Branch:** `pr/PR-262-pca-candidate-universe`
- **Depends on:** PR-261
- **Allowed:** only the v4 candidate-universe/catalog adapter immediately before discovery and its focused tests.
- **Purpose:** raw candidates + materialized PCs -> one universe.

Acceptance:
- [ ] Preserve every raw candidate and raw ordinal unchanged; append PCs in component order.
- [ ] Metadata records `origin=raw` or `origin=pca`; origin has no ranking weight.
- [ ] Disabled/unavailable PCA produces candidate universe exactly equal to raw-only input.
- [ ] Duplicate names fail closed.

Not in scope: distance, clustering or scoring.

## PR-263 — Admit PCs into v4 distance/clustering

- **Branch:** `pr/PR-263-pca-distance-clustering`
- **Depends on:** PR-262
- **Allowed:** `src/market_regime_engine/feature_discovery/distance.py`, `src/market_regime_engine/feature_discovery/clustering.py`, corresponding tests.
- **Purpose:** prove existing redundancy logic works on combined candidates.

Acceptance:
- [ ] Same absolute-Spearman support rules for raw/raw, raw/PC and PC/PC.
- [ ] Same average-linkage hierarchy/silhouette search and canonical combined ordinal.
- [ ] Existing `M <= min(12,N-1)` remains unchanged.
- [ ] No PCA group, special medoid or origin bonus.

## PR-264 — Admit PCs into regime scoring and cluster winners

- **Branch:** `pr/PR-264-pca-score-winners`
- **Depends on:** PR-263
- **Allowed:** `src/market_regime_engine/feature_discovery/scoring.py`, `src/market_regime_engine/feature_discovery/winners.py`, corresponding tests.
- **Purpose:** PCs compete with raw features under existing regime evidence.

Acceptance:
- [ ] Exact existing state-information-ratio/eta diagnostics and support rules apply to PCs.
- [ ] Same cluster-winner tie-break chain regardless of origin.
- [ ] Tests cover raw wins, PC wins and all PCs lose.
- [ ] Raw-only scoring/winner results remain unchanged when no PCs exist.

## PR-265 — Admit PCs into prefix and final-feature selection

- **Branch:** `pr/PR-265-pca-final-feature-selection`
- **Depends on:** PR-264
- **Allowed:** v4 prefix-length/final-grid selection modules and focused tests.
- **Purpose:** allow final tuples to mix raw and PC winners.

Acceptance:
- [ ] Existing `L=2..min(M*,8)` bound unchanged.
- [ ] Soft-NMI `L*` and final 12-candidate grid do not branch on feature origin.
- [ ] Final tuple preserves PC provenance.
- [ ] No cross-dimension PLL comparison is introduced.

## PR-266 — Freeze PCA inside inner walk-forward and apply normal HMM scaling

- **Branch:** `pr/PR-266-pca-inner-walk-forward`
- **Depends on:** PR-265
- **Allowed:** v4 TRAIN-selection orchestration plus `src/market_regime_engine/evaluation/walk_forward.py` integration points and tests only.
- **Purpose:** one PCA artifact per enclosing selection run; selected PCs use existing HMM scaler.

Acceptance:
- [ ] Determine earliest admissible inner TRAIN from existing plan; fit PR-260 exactly once there.
- [ ] Materialize later inner TRAIN/TEST PCs only with frozen artifact.
- [ ] Selected raw+PC observations pass through existing fold-specific TRAIN-only `fit_standard_scaler` unchanged.
- [ ] PCA source scaler and final HMM scaler remain separate artifacts.
- [ ] Inner-TEST perturbation cannot change PCA source tuple, `k`, loadings or hash.

## PR-267 — Carry frozen PCA through outer TEST evaluation

- **Branch:** `pr/PR-267-pca-outer-evaluation`
- **Depends on:** PR-266
- **Allowed:** `src/market_regime_engine/evaluations/global_regime_v4.py`, corresponding outer-evaluation tests.
- **Purpose:** outer TRAIN may fit/select PCA; outer TEST may only transform.

Acceptance:
- [ ] At most one outer-fold-local PCA artifact from that outer TRAIN selection run.
- [ ] Outer TEST never fits/refits/reselects PCA sources or `k`.
- [ ] Missing PCA source rows follow PR-261 null semantics and existing model complete-case rules.
- [ ] Fold evidence stores PCA availability/reason and artifact hash when available.

## PR-268 — Reuse the same PCA policy in deployment selection

- **Branch:** `pr/PR-268-pca-deployment-selection`
- **Depends on:** PR-267
- **Allowed:** `src/market_regime_engine/evaluations/deployment_selection.py`, corresponding tests.
- **Purpose:** no production-specific PCA fitting path.

Acceptance:
- [ ] Call the same v4 selection function used by outer TRAIN.
- [ ] Store deployment-selection PCA artifact/evidence with frozen selected configuration.
- [ ] Do not copy the last outer-fold PCA artifact.
- [ ] No duplicate PCA resolver/fitter implementation.

## PR-269 — Persist PCA through final refit and production package

- **Branch:** `pr/PR-269-pca-production-package`
- **Depends on:** PR-268
- **Allowed:** `src/market_regime_engine/training/final_refit.py`, `src/market_regime_engine/models/production_artifact.py`, `src/market_regime_engine/mlflow_support/model_package.py`, corresponding tests.
- **Purpose:** package enough frozen PCA state to reproduce selected PCs; never refit PCA.

Acceptance:
- [ ] Final refit consumes deployment-selected PCA artifact without fitting a new basis.
- [ ] Mixed raw+PC package contains exact PCA artifact/hash; raw-only package may omit it and remains backward compatible.
- [ ] Validate every selected PC index exists in stored artifact.
- [ ] Packaged state reproduces offline PC values exactly within pinned tolerance.

## PR-270 — Reproduce PCA in latest/replay serving

- **Branch:** `pr/PR-270-pca-serving`
- **Depends on:** PR-269
- **Allowed:** `src/market_regime_engine/serving/latest_handler.py`, `src/market_regime_engine/serving/replay_handler.py`, resolver/cache only if exact field plumbing is required, corresponding tests.
- **Purpose:** serving uses packaged PCA state only.

Acceptance:
- [ ] Mixed request path: raw PCA sources -> packaged PCA transform -> packaged HMM scaler -> HMM inference.
- [ ] No serving-time fit or source reselection; exact source order validated.
- [ ] Missing/non-finite required source values follow existing unavailable-observation behavior; never impute.
- [ ] Raw-only serving path unchanged.

## PR-271 — Add MLflow PCA evidence only

- **Branch:** `pr/PR-271-pca-mlflow-evidence`
- **Depends on:** PR-270
- **Allowed:** PCA-specific additions under `src/market_regime_engine/mlflow_support/*`, corresponding tracking/plot tests.
- **Purpose:** observability only; no model/selection changes.

Acceptance:
- [ ] Log availability/reason, source count, fit rows, retained `k`, `0.90`, fit cutoff and artifact hash.
- [ ] Log per-PC eigenvalue, EVR, cumulative EVR and downstream-selected boolean.
- [ ] Store loading matrix and top absolute source contributors; add scree and cumulative-EVR plots.
- [ ] Tracking reads existing evidence only and never recomputes PCA.

## PR-272 — Add matched raw-only vs raw+PCA evaluation

- **Branch:** `pr/PR-272-pca-policy-comparison`
- **Depends on:** PR-271
- **Allowed:** one new comparison orchestration module, one CLI/script entrypoint if required, corresponding tests.
- **Purpose:** controlled evidence comparison; no promotion behavior.

Acceptance:
- [ ] Exactly two policies: `raw_only` and `raw_plus_pca_90`.
- [ ] Same source snapshot, outer/inner plans, model grid, seeds and gates.
- [ ] Report outer valid-fold rate, soft-NMI mean/std/worst, selection stability, occupancy, duration, switches/year, entropy/confidence, signature drift and fold-local OOS PLL diagnostics.
- [ ] Never rank different final feature spaces by pooled/raw PLL.
- [ ] Report PC selection frequency by component index across outer folds.
- [ ] Command cannot mutate registry aliases.

## PR-273 — Add explicit PCA profile/config switch

- **Branch:** `pr/PR-273-pca-profile-config`
- **Depends on:** PR-272
- **Allowed:** `src/market_regime_engine/profiles/config.py`, active v4 profile YAML, config example and corresponding tests.
- **Purpose:** expose only the already-implemented policy.

Acceptance:
- [ ] Exact modes: `disabled` and `pca_90` only.
- [ ] `pca_90` maps to pinned `0.90`; no arbitrary user threshold in this PR.
- [ ] `disabled` reproduces raw-only behavior; unknown mode fails closed.
- [ ] One explicit documented default; no environment-variable shadow default.

## PR-274 — Add one hermetic end-to-end PCA proof

- **Branch:** `pr/PR-274-pca-e2e-proof`
- **Depends on:** PR-273
- **Allowed:** new PCA E2E/integration fixtures/tests only; no production source changes except a separately approved testability hook.
- **Purpose:** prove the completed path; add no behavior.

Acceptance:
- [ ] Execute source snapshot -> PCA fit -> PCs -> combined universe -> v4 selection -> mixed final tuple -> HMM scaler -> HMM -> outer OOS -> deployment selection -> final package -> latest -> replay.
- [ ] Fixture selects at least one PC and at least one raw feature.
- [ ] Identical rerun reproduces PCA artifact hash and outputs within existing numeric tolerances.
- [ ] Separate `disabled` run proves raw-only regression.
- [ ] Full merge gate and configured coverage threshold pass.

## PR-275 — Close PCA documentation and operations guidance

- **Branch:** `pr/PR-275-pca-docs-closure`
- **Depends on:** PR-274
- **Allowed:** `README.md`, `ARCHITECTURE.md`, `EVALUATION.md`, relevant operations/evaluation docs only.
- **Purpose:** document behavior already proven by code/E2E; no runtime change.

Acceptance:
- [ ] Document PCA as optional feature generator, exact `0.90`, `504`, source completeness and no Kaiser cutoff.
- [ ] Document earliest-inner-TRAIN fit, outer-fold-local refits, names/order/provenance and unavailable codes.
- [ ] Document the two scaling layers and normal raw+PC competition under v4.
- [ ] Link PR-272 comparison evidence and PR-274 E2E evidence.
- [ ] Mermaid/commands match executable contracts exactly; doc/link/command smoke checks pass.

---

# 4. Exact execution graph

```text
PR-255 -> PR-256 -> PR-257 -> PR-258 -> PR-259 -> PR-260 -> PR-261
       -> PR-262 -> PR-263 -> PR-264 -> PR-265 -> PR-266 -> PR-267
       -> PR-268 -> PR-269 -> PR-270 -> PR-271 -> PR-272 -> PR-273
       -> PR-274 -> PR-275
```

The invariant across every PR is: **PCA creates additional candidate features. Once generated, `pca_pc_*` features enter the same v4 feature universe and are treated like the rest of the features.**
