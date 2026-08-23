# Market Regime Engine — Feature Selection Backlog Extension

Status date: 2026-08-23

This file is the authoritative backlog extension for the statistical feature-selection path of `xetra_cross_asset_v1`. It deliberately keeps the method simple enough for weak parallel agents: first select one representative per semantic block, then remove cross-block representatives that are too strongly correlated. There is no replacement search, no feature-subset optimizer, and no downstream ETF/portfolio target.

For PR-045 through PR-050 and for the explicit addenda to PR-021, PR-022, PR-024, PR-035, and PR-036 below, this file is authoritative for feature-selection scope, dependencies, allowed files, and acceptance criteria. `BACKLOG.md`, `DATA_SOURCE.md`, `EVALUATION.md`, `PLOT_STYLE.md`, and `CONTRIBUTING.md` retain their existing areas of authority.

This planning extension does not itself change implemented evaluation behavior. PR-048 is the integration point that updates `EVALUATION.md` with the final selection/freeze semantics. PR-050 updates `EVALUATION.md` only for the MLflow evidence contract.

---

# Statistical feature-selection contract

## Boundary

The engine must reduce the 48 upstream `regime-loader` Gold features to a compact, reproducible HMM input set using statistics only.

The selector must never use:

- Xetra ETF returns;
- portfolio weights or portfolio returns;
- Sharpe, Sortino, Calmar, drawdown, Expected Shortfall, or transaction costs;
- trading labels, regime profitability, or execution data;
- HMM OOS likelihood, AIC, BIC, or any other fitted-HMM metric to search across feature subsets.

Portfell owns the later economic/Xetra ETF evaluation of regime predictions.

## Source universe

The source universe is exactly the 48 feature columns from `regime-loader` `regime_features_daily` feature version 1. `timestamp_m1` is the temporal key and is never a candidate feature.

The eight semantic blocks are exhaustive and mutually exclusive.

### Block 1 — `us_equity_volatility_spot`

1. `vix_level`
2. `vix_delta_5obs`
3. `vix_delta_20obs`
4. `vix_zscore_60obs`

### Block 2 — `us_equity_volatility_term_structure`

1. `vix9d_level`
2. `vix9d_delta_5obs`
3. `vix9d_delta_20obs`
4. `vix9d_zscore_60obs`
5. `vix3m_level`
6. `vix3m_delta_5obs`
7. `vix3m_delta_20obs`
8. `vix3m_zscore_60obs`
9. `vix6m_level`
10. `vix6m_delta_5obs`
11. `vix6m_delta_20obs`
12. `vix6m_zscore_60obs`
13. `vix1y_level`
14. `vix1y_delta_5obs`
15. `vix1y_delta_20obs`
16. `vix1y_zscore_60obs`
17. `vix9d_vix_ratio`
18. `vix_vix3m_ratio`
19. `vix3m_minus_vix`
20. `vix6m_minus_vix`
21. `vix1y_minus_vix`

### Block 3 — `europe_equity_volatility`

1. `vstoxx_level`
2. `vstoxx_delta_5obs`
3. `vstoxx_delta_20obs`
4. `vstoxx_zscore_60obs`

### Block 4 — `rates_volatility`

1. `move_level`
2. `move_delta_5obs`
3. `move_delta_20obs`
4. `move_zscore_60obs`

### Block 5 — `systemic_stress`

1. `ciss_level`
2. `ciss_delta_5obs`
3. `ciss_delta_20obs`

### Block 6 — `credit_stress`

1. `euro_hy_oas_level`
2. `euro_hy_oas_delta_5obs`
3. `euro_hy_oas_delta_20obs`

### Block 7 — `rates_yield_curve`

1. `us_2y_level`
2. `us_2y_delta_20obs`
3. `us_10y_level`
4. `us_10y_delta_20obs`
5. `estr_level`
6. `estr_delta_20obs`
7. `us_10y_minus_us_2y`

### Block 8 — `usd_fx`

1. `usd_broad_level`
2. `usd_broad_delta_20obs`

Canonical block order is Block 1 through Block 8. The preliminary stage therefore produces at most one representative per block and exactly eight representatives before cross-block pruning.

## Pinned Xetra policy

The Xetra policy has no hidden defaults:

```text
policy_id = xetra_semantic_medoid_v1
within_block_method = absolute_spearman_medoid
cross_block_method = absolute_spearman_prune
minimum_feature_coverage = 0.90
minimum_block_complete_observations = 504
maximum_cross_block_abs_spearman = 0.85
```

Changing any value requires a versioned policy/profile and the normal evaluation-sidecar update.

## Greek letter

- **rho (ρ)** — pronounced *ROH*; Spearman rank-correlation coefficient.

---

# Exact deterministic algorithm

Selection is executed once using only the TRAIN interval of the first planned expanding walk-forward fold.

## Stage 1 — one medoid per semantic block

For each block independently:

1. Start with the ordered candidates declared above.
2. Compute non-null coverage on first-fold TRAIN rows only.
3. A candidate is eligible only when:
   - coverage is at least `0.90`;
   - every non-null value is finite;
   - non-null variance is greater than zero.
4. No forward fill, backward fill, interpolation, imputation, or synthetic calendar row is allowed.
5. Keep only TRAIN rows complete across all eligible candidates of that block.
6. The block must have at least `504` complete rows or selection fails closed before any HMM fit.
7. Compute the Spearman correlation matrix on those complete rows.
8. Every required correlation must be finite; there is no Pearson fallback.
9. Define redundancy distance:

```text
d(i,j) = 1 - abs(ρ(i,j))
```

10. For each eligible candidate, compute the arithmetic mean distance to all other eligible candidates in the same block.
11. If exactly one candidate is eligible, its medoid score is `0.0`.
12. Select the candidate with the lowest medoid score.
13. Tie-break order is exactly:
    1. lower medoid score;
    2. higher first-fold TRAIN coverage;
    3. earlier candidate position in the block configuration.
14. Ranking uses full precision; do not round before ranking.
15. Exactly one medoid must be produced for every block or Stage 1 fails closed.

Stage 1 therefore produces exactly eight preliminary medoids.

## Stage 2 — simple cross-block correlation pruning

There is deliberately **no replacement search**.

1. Take the eight Stage-1 medoids in canonical block order.
2. Keep only first-fold TRAIN rows complete across all eight medoids.
3. At least `504` complete rows are required or selection fails closed.
4. Compute one fixed Spearman correlation matrix across the eight preliminary medoids on those complete rows.
5. Define a conflict when:

```text
abs(ρ(i,j)) > 0.85
```

6. If there is no conflict, keep all eight medoids.
7. If conflicts exist, process them greedily and deterministically using the fixed matrix:
   - choose the remaining conflicting pair with the highest `abs(ρ)`;
   - if multiple pairs have the same value, choose the pair whose earlier block appears first in canonical block order; if still tied, choose the pair whose later block appears first;
   - remove exactly one feature from that pair using the removal rule below;
   - continue until no remaining pair exceeds `0.85`.
8. Never recompute Stage-1 medoids and never search the removed feature's block for a replacement.
9. Never recompute the Stage-2 correlation matrix after a removal; the first-fold TRAIN complete-row matrix is the single auditable pruning basis.

### Exact removal rule

For a conflicting pair, remove the feature that is the weaker representative of its own semantic block:

1. higher Stage-1 medoid score is removed;
2. if medoid scores are exactly equal, lower first-fold TRAIN coverage is removed;
3. if coverage is also exactly equal, remove the feature from the later block in canonical block order.

This rule makes “remove the one that makes less sense to keep” deterministic and auditable without introducing a second optimization problem.

The final feature count is therefore:

```text
1 <= d <= 8
```

No target metric is used to choose `d`; `d` is solely the result of the fixed semantic-medoid and cross-block-correlation rules.

## Freeze rule

After Stage 2, the surviving features are ordered by their original canonical block order and frozen for the complete evaluation.

```text
first walk-forward TRAIN
        -> Stage 1: 8 semantic medoids
        -> Stage 2: prune |Spearman| > 0.85
        -> freeze d <= 8 features
        -> K=2 full / K=3 full / K=4 full
        -> every fold uses identical frozen features
```

Later folds never rerun either selection stage. TEST rows never affect selection. Appending or mutating rows strictly after first-fold `train_end` must not change the preliminary medoids, Stage-2 matrix, pruning decisions, final ordered features, or result hash.

All K=2/K=3/K=4 candidates in one comparison must use the identical frozen feature order and identical feature-selection hash. Candidate comparison across mixed feature-selection hashes or different feature dimensions is invalid.

## Required evidence

The frozen result must preserve:

- policy ID/version/hash;
- source dataset/build, schema version, and feature version;
- evaluation-plan hash;
- first fold ID and exact UTC TRAIN bounds;
- canonical blocks/candidates;
- per-feature coverage and eligibility/rejection reason;
- per-block complete-row count;
- within-block Spearman correlations/distances;
- Stage-1 medoid score and winner per block;
- Stage-2 common complete-row count;
- complete Stage-2 eight-medoid Spearman matrix;
- every cross-block conflict considered;
- deterministic removal reason for each pruned feature;
- preliminary eight-feature tuple;
- final ordered `d <= 8` feature tuple;
- deterministic result hash.

No downstream ETF/portfolio value is present in this evidence.

---

# Atomic implementation PRs

## PR-045 — Define feature-selection contracts and profile schema extension

- **Status:** BLOCKED by PR-007
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-045-feature-selection-contracts`
- **Depends on:** PR-007
- **Allowed files:** `src/market_regime_engine/feature_selection/__init__.py`, `src/market_regime_engine/feature_selection/contracts.py`, `src/market_regime_engine/profiles/schema.py`, `tests/unit/feature_selection/test_contracts.py`, `tests/unit/profiles/test_feature_selection_schema.py`

### Task

Introduce immutable contracts/schema only. Do not calculate correlations, load data, slice folds, fit HMMs, resolve profiles, or log MLflow data.

### Acceptance criteria

- [ ] Add immutable `FeatureBlock` with stable block ID and ordered unique candidates.
- [ ] Add immutable `FeatureSelectionPolicy` containing policy ID/version, within-block method, cross-block method, ordered blocks, coverage threshold, complete-row threshold, and cross-block correlation threshold.
- [ ] Accepted methods are exactly `absolute_spearman_medoid` and `absolute_spearman_prune` in their corresponding fields.
- [ ] Validate `0 < minimum_feature_coverage <= 1`, `minimum_block_complete_observations >= 2`, and `0 < maximum_cross_block_abs_spearman < 1`.
- [ ] Reject duplicate/empty blocks, duplicate candidates, and the same feature in multiple blocks.
- [ ] Evidence contracts support Stage-1 coverage/eligibility/correlation/medoid evidence plus Stage-2 matrix/conflict/removal evidence.
- [ ] `FeatureSelectionResult` contains lineage, first-fold bounds, eight preliminary medoids, zero-or-more removals, final ordered features, and deterministic result hash.
- [ ] Result validation requires exactly one preliminary medoid per configured block and `1 <= len(final_features) <= len(blocks)`.
- [ ] Final features must be an order-preserving subset of the preliminary medoids; replacement features are invalid.
- [ ] Extend profile schema so a profile declares either static exact model features or a feature-selection policy/source universe, never both and never neither.
- [ ] Unresolved feature-selection profiles cannot masquerade as resolved fitted-model feature order.
- [ ] Serialization round trips are deterministic.
- [ ] No model-library, MLflow, PostgreSQL, filesystem, FastAPI, or portfolio dependency is introduced.

## PR-046 — Define the exact Xetra eight-block policy

- **Status:** BLOCKED by PR-045
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-046-xetra-feature-blocks`
- **Depends on:** PR-045
- **Allowed files:** `configs/feature_selection/xetra_semantic_medoid_v1.yaml`, `docs/profiles/xetra_feature_selection_v1.md`, `tests/unit/feature_selection/test_xetra_feature_blocks.py`

### Task

Encode only the exact policy constants and 48-to-8 semantic block membership. Do not calculate correlations or select/prune features.

### Acceptance criteria

- [ ] Policy ID is exactly `xetra_semantic_medoid_v1`.
- [ ] `within_block_method = absolute_spearman_medoid`.
- [ ] `cross_block_method = absolute_spearman_prune`.
- [ ] `minimum_feature_coverage = 0.90`.
- [ ] `minimum_block_complete_observations = 504`.
- [ ] `maximum_cross_block_abs_spearman = 0.85`.
- [ ] Exactly eight canonical blocks are present in canonical order.
- [ ] Exactly 48 canonical `regime-loader` features are present exactly once.
- [ ] Block sizes are exactly `4, 21, 4, 4, 3, 3, 7, 2`.
- [ ] No `timestamp_m1`, ETF/portfolio target, HMM state, Sharpe/Sortino, or trading field is present.
- [ ] Documentation explains Stage 1 produces eight preliminary medoids and Stage 2 may only remove them; it never searches for replacements.
- [ ] Config validation reuses PR-045 contracts.

## PR-047 — Implement pure Stage-1 absolute-Spearman medoid selection

- **Status:** BLOCKED by PR-045
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-047-spearman-medoid-selector`
- **Depends on:** PR-045
- **Allowed files:** `src/market_regime_engine/feature_selection/selector.py`, `tests/unit/feature_selection/test_selector.py`, `tests/fixtures/feature_selection/*`

### Task

Implement Stage 1 only as a pure in-memory function. It receives an already-bounded TRAIN frame and validated policy and returns eight preliminary medoids plus Stage-1 evidence. Do not implement cross-block pruning here.

### Acceptance criteria

- [ ] Reads only supplied rows/columns and cannot fetch data.
- [ ] Missing configured columns fail closed with explicit names.
- [ ] Coverage is non-null count divided by supplied TRAIN row count.
- [ ] Coverage below threshold, non-finite values, and zero variance are handled exactly by the contract with explicit reasons.
- [ ] No fill/imputation occurs.
- [ ] Block-complete rows use all eligible features of that block.
- [ ] Fewer than the configured complete-row minimum fails selection.
- [ ] Spearman correlations are finite; no Pearson fallback.
- [ ] Distance is exactly `1 - abs(rho)`.
- [ ] Medoid score is exact arithmetic mean distance to all other eligible block candidates.
- [ ] One eligible feature receives score `0.0`.
- [ ] Ranking is lower medoid score, higher coverage, earlier candidate order.
- [ ] Ranking uses full precision.
- [ ] Output contains exactly eight preliminary medoids in canonical block order.
- [ ] Tests cover positive/negative perfect correlation, monotonic nonlinear ranks, coverage exclusion, zero variance, insufficient rows, undefined correlation, one-candidate block, tie-breaks, and repeat determinism.
- [ ] No Stage-2 pruning, HMM, MLflow, PostgreSQL, API, ETF, or portfolio logic exists in the module.

## PR-048 — Prune cross-block correlation and freeze from first TRAIN only

- **Status:** BLOCKED by PR-020, PR-046, PR-047
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-048-prune-freeze-first-train-features`
- **Depends on:** PR-020, PR-046, PR-047
- **Allowed files:** `src/market_regime_engine/feature_selection/freeze.py`, `tests/unit/feature_selection/test_freeze.py`, `tests/integration/test_feature_selection_freeze.py`, `EVALUATION.md`

### Task

Integrate first-fold TRAIN slicing, Stage-1 execution, the simple Stage-2 prune rule, and final freeze semantics. Do not fit HMMs or log MLflow data.

### Acceptance criteria

- [ ] Input to Stage 1 is exactly the TRAIN interval of the first planned walk-forward fold.
- [ ] No TEST/later row reaches Stage 1 or Stage 2.
- [ ] Stage 1 returns exactly eight preliminary medoids.
- [ ] Stage 2 uses first-fold TRAIN rows complete across all eight preliminary medoids.
- [ ] Stage 2 requires at least `504` complete rows.
- [ ] Stage 2 computes exactly one fixed eight-medoid Spearman matrix.
- [ ] Conflict threshold is exactly `abs(rho) > 0.85`; `0.85` itself is not a conflict.
- [ ] Conflict processing chooses highest `abs(rho)` first with canonical pair-order tie-break.
- [ ] Removal rule is higher Stage-1 medoid score, then lower coverage, then later canonical block.
- [ ] Removed blocks are not searched for replacement features.
- [ ] Stage-2 correlations are not recomputed after removals.
- [ ] Final ordered features are the surviving preliminary medoids in canonical block order.
- [ ] Selection is executed once per evaluation plan/source snapshot, not per candidate or fold.
- [ ] Appending/mutating rows after first-fold `train_end` cannot change any Stage-1/Stage-2 evidence, final features, or result hash.
- [ ] A positive-control mutation inside first-fold TRAIN is allowed to change the result.
- [ ] Selection failure prevents HMM evaluation from starting.
- [ ] `EVALUATION.md` documents both stages, threshold `0.85`, exact removal rule, no replacement search, first-TRAIN-only semantics, and freeze semantics.
- [ ] `EVALUATION.md` states the engine uses no ETF/portfolio metric for feature selection and Portfell owns downstream economic validation.
- [ ] `EVALUATION.md` states HMM likelihood/AIC/BIC are not used to search feature subsets of different dimensions.

## PR-049 — Resolve model profiles with the frozen pruned feature set

- **Status:** BLOCKED by PR-021, PR-048
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-049-resolve-selected-feature-profile`
- **Depends on:** PR-021, PR-048
- **Allowed files:** `src/market_regime_engine/profiles/resolution.py`, `tests/unit/profiles/test_resolution.py`, `tests/integration/test_xetra_profile_resolution.py`

### Task

Resolve the Xetra base profile with one compatible frozen `FeatureSelectionResult`. Do not calculate correlations, rerun selection, fit HMMs, or log MLflow data.

### Acceptance criteria

- [ ] Requires a validated base profile referencing `xetra_semantic_medoid_v1` and a compatible frozen result.
- [ ] Rejects source/build, feature-version, policy/hash, or evaluation-plan mismatches.
- [ ] Resolved model feature order is exactly the final surviving frozen features in canonical block order.
- [ ] Preserves the original 48-feature source universe and the preliminary eight medoids separately from final model features.
- [ ] Resolved profile hash includes the feature-selection result hash.
- [ ] K=2 full, K=3 full, and K=4 full receive identical final feature order and identical selection hash.
- [ ] Resolver accepts only `1..8` final features and rejects duplicates or replacements not present in preliminary medoids.
- [ ] Serialization/reload preserves source universe, preliminary medoids, removals, final features, and hash exactly.
- [ ] Integration test proves all three Gaussian candidates share exactly the same final feature dimension/order.
- [ ] No HMM fitting, MLflow, PostgreSQL, FastAPI, or portfolio logic exists here.

## PR-050 — Log auditable feature-selection evidence to MLflow

- **Status:** BLOCKED by PR-023, PR-048, PR-049
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-050-mlflow-feature-selection-evidence`
- **Depends on:** PR-023, PR-048, PR-049
- **Allowed files:** `src/market_regime_engine/mlflow_support/feature_selection_tracking.py`, `tests/unit/mlflow_support/test_feature_selection_tracking.py`, `tests/integration/test_mlflow_feature_selection_tracking.py`, `EVALUATION.md`

### Task

Log final feature-selection evidence to the parent MLflow evaluation run. Do not change selection mathematics or candidate ranking.

### Acceptance criteria

- [ ] Parent run records policy ID/version/hash, selection hash, preliminary feature count `8`, final feature count `d`, first fold ID, and selection TRAIN bounds.
- [ ] Candidate runs record the same `feature_selection_hash` and final feature count/order reference.
- [ ] Parent stores `feature_selection/selection.json` with lineage, policy, eight preliminary medoids, pruning decisions, final ordered features, and result hash.
- [ ] Parent stores `feature_selection/scores.parquet` with one row per source candidate and at least block ID, candidate order, coverage, eligibility reason, medoid score, and preliminary-selected flag.
- [ ] Parent stores `feature_selection/within_block_correlations.parquet` in deterministic long form.
- [ ] Parent stores `feature_selection/cross_block_correlations.parquet` with the fixed eight-medoid Stage-2 matrix in deterministic long form.
- [ ] Parent stores `feature_selection/pruning.parquet` with conflict order, feature pair, `abs_rho`, removed feature/block, kept feature/block, and exact removal reason.
- [ ] Artifact ordering is deterministic.
- [ ] No ETF/portfolio/trading metric is logged by this module.
- [ ] Hermetic local-file MLflow integration round-trips all evidence.
- [ ] Integration test proves K=2/K=3/K=4 reference exactly the same selection hash and final feature dimension/order.
- [ ] `EVALUATION.md` documents the exact MLflow tags/artifact paths.

---

# Addenda to existing backlog PRs

## PR-021 addendum — Xetra base profile

### Dependency override

PR-021 additionally depends on PR-045 and PR-046.

### Additional/replacement acceptance criteria

- [ ] `xetra_cross_asset_v1` declares the exact 48-feature source universe instead of hard-coding final HMM features.
- [ ] It references policy `xetra_semantic_medoid_v1`.
- [ ] The base profile is unresolved until PR-048/PR-049 provide the frozen pruned result.
- [ ] It keeps exactly `gaussian_hmm_k2_full`, `gaussian_hmm_k3_full`, and `gaussian_hmm_k4_full`.
- [ ] No downstream ETF/portfolio target is added.

## PR-022 addendum — Walk-forward runner

### Dependency override

PR-022 additionally depends on PR-049.

### Additional acceptance criteria

- [ ] Runner accepts only a resolved profile with frozen feature-selection hash.
- [ ] Preprocessing/HMM fitting uses exactly the resolved final feature order.
- [ ] Feature selection/pruning is never called inside the fold loop.
- [ ] Every fold and K candidate uses the same final features and selection hash.
- [ ] Fold evidence records selection hash and final feature count.
- [ ] Mixed feature orders/hashes fail before candidate comparison.

## PR-024 addendum — Candidate grid

### Additional acceptance criteria

- [ ] All compared candidates must have identical feature-selection hash, final feature order, and final feature count.
- [ ] Candidate comparison table records selection hash and feature count.
- [ ] K=2/K=3/K=4 therefore remain dimensionally comparable.

## PR-035 addendum — Hermetic engine E2E

### Dependency override

PR-035 additionally depends on PR-050.

### Additional acceptance criteria

- [ ] Fixture exposes all 48 canonical source features.
- [ ] Stage 1 selects exactly eight medoids once from first-fold TRAIN.
- [ ] Stage 2 prunes every pair above `0.85` using the exact deterministic removal rule and performs no replacement search.
- [ ] Final `d <= 8` features are frozen and identical for K=2/K=3/K=4 and every fold.
- [ ] Future TEST mutations leave selection and hash unchanged.
- [ ] MLflow contains Stage-1 and Stage-2 evidence and all candidates reference the same hash.
- [ ] No ETF/portfolio data is required.

## PR-036 addendum — Final documentation

### Additional acceptance criteria

- [ ] Documentation explains `48 source features -> 8 semantic blocks -> 8 Stage-1 medoids -> prune cross-block |Spearman| > 0.85 -> frozen d <= 8 features -> K=2/K=3/K=4 full-covariance walk-forward evaluation`.
- [ ] Documentation states there is no replacement search after Stage-2 pruning.
- [ ] Documentation states selection uses only first-fold TRAIN and is never rerun later.
- [ ] Documentation states feature/champion selection in the engine is statistical only and Portfell owns downstream Xetra ETF economic evaluation.
- [ ] Documentation links MLflow evidence and explains why each removed feature was pruned.

---

# Revised statistical evaluation flow

```text
regime-loader PostgreSQL snapshot
          |
          v
48 validated source features
          |
          v
FIRST FOLD TRAIN ONLY
          |
          +--> Stage 1
          |      8 semantic blocks
          |      -> one absolute-Spearman medoid each
          |      -> exactly 8 preliminary medoids
          |
          +--> Stage 2
                 fixed 8-medoid Spearman matrix
                 -> prune abs(rho) > 0.85
                 -> no replacements
                 -> final d <= 8 features
          |
          v
freeze feature-selection result/hash
          |
          +---------+------------------+
          |         |                  |
          v         v                  v
      K=2 full   K=3 full          K=4 full
          |         |                  |
          +---------+------------------+
                    |
                    v
       same expanding walk-forward folds
                    |
                    +--> TRAIN-only scaler
                    +--> deterministic multi-start fit
                    +--> TRAIN/prior-only state alignment
                    +--> TEST causal filtering only
                    +--> fold diagnostics / OOS PLL
                    |
                    v
             hard quality gates
                    |
                    v
       mean OOS PLL / fold stability
           -> BIC/AIC -> fewer K
                    |
                    v
              engine-champion
                    |
                    v
          MLflow registry / OOS output
                    |
                    v
        downstream Portfell evaluation
```

The selected feature dimension is determined before any HMM candidate is fitted. K=2/K=3/K=4 are therefore always compared in the same final `d`-dimensional observation space.

---

# Parallel execution plan

```text
After PR-007:
  PR-045

After PR-045, parallel:
  PR-046   PR-047

Independent existing work may continue:
  PR-020 walk-forward split planner

When existing dependencies plus PR-045/046 are ready:
  PR-021 Xetra base profile

After PR-020 + PR-046 + PR-047:
  PR-048

After PR-021 + PR-048:
  PR-049

Then PR-022 may start with its existing dependencies plus PR-049.

After PR-023 + PR-048 + PR-049:
  PR-050

PR-024 and PR-050 may proceed in parallel once their own dependencies are met.
PR-035 waits for PR-050 in addition to its existing dependencies.
```

## Weak-agent packet rule

For PR-045 through PR-050, an orchestrator gives a weak agent only:

1. the assigned PR section from this file;
2. the shared statistical feature-selection contract above;
3. `CONTRIBUTING.md`;
4. only the exact upstream contract/doc needed by that PR.

The agent must stop rather than broaden scope if it needs an unmerged dependency, a file outside `Allowed files`, a different threshold or method, a replacement search, per-fold re-selection, HMM-based feature-subset optimization, or downstream ETF/portfolio data.