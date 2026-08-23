# Market Regime Engine — Feature Selection Backlog Extension

Status date: 2026-08-23

This file is an additive implementation-backlog extension for the statistical feature-selection path used by `xetra_cross_asset_v1`. It exists so the feature-selection work can be delegated to weak parallel agents as small, exact, non-overlapping PRs without making the already-large `BACKLOG.md` PR sections broader.

For PR-045 through PR-050 and for the explicit addenda to PR-021, PR-022, PR-024, PR-035, and PR-036 below, this file is authoritative for scope, dependencies, allowed files, and acceptance criteria. All other backlog rules continue to come from `BACKLOG.md`, while `DATA_SOURCE.md`, `EVALUATION.md`, `PLOT_STYLE.md`, and `CONTRIBUTING.md` retain their existing areas of authority.

This planning extension does **not** by itself change the currently implemented evaluation behavior. PR-048 is the integration point that makes the feature-selection method part of the evaluation contract and therefore must update `EVALUATION.md` in the same PR. PR-050 updates `EVALUATION.md` again only for the new MLflow feature-selection evidence names/artifacts.

---

# Statistical feature-selection contract

## Purpose and boundary

The engine must reduce the upstream `regime-loader` feature universe to a compact, reproducible, statistically representative HMM input set without using any downstream Xetra ETF return, portfolio, Sharpe/Sortino, drawdown, transaction-cost, allocation, or trading target.

Feature selection is intentionally **not** an HMM wrapper search. It does not compare feature subsets by HMM OOS likelihood, AIC, BIC, state returns, or portfolio performance. This avoids creating a second combinatorial model-selection layer and avoids comparing likelihoods across different observation dimensions.

The engine feature-selection question is:

> Which single feature is the most representative, non-redundant member of each predefined market-information block using only the initial historical training information set?

Portfell remains responsible for the later economic/Xetra ETF evaluation of the resulting regime predictions.

## Upstream feature universe

The source universe is exactly the 48 feature columns from `regime-loader` `regime_features_daily` feature version 1. `timestamp_m1` is the temporal key and is not a candidate feature.

The eight semantic blocks below are exhaustive and mutually exclusive. Their union must contain exactly 48 unique feature names.

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

Block order above is canonical. The final HMM feature order is the selected representative from Block 1, then Block 2, through Block 8. Therefore a successful selection always contains exactly **8 ordered features**.

## Xetra selection-policy constants

The Xetra feature-selection profile pins these values explicitly; the selector implementation has no hidden defaults:

```text
policy_id = xetra_semantic_medoid_v1
method = absolute_spearman_medoid
selected_features = 8
minimum_feature_coverage = 0.90
minimum_block_complete_observations = 504
```

No implementation PR may change these values silently. A future change requires a versioned feature-selection policy/profile and the normal evaluation-sidecar update.

## Exact deterministic selection algorithm

Selection is executed **once**, using only the TRAIN interval of the first planned expanding walk-forward fold.

For each semantic block independently:

1. Start with the ordered candidate features declared for that block.
2. Compute each candidate's non-null coverage fraction using only first-fold TRAIN rows.
3. A feature is eligible only if:
   - its coverage is at least `minimum_feature_coverage`;
   - every non-null value is finite;
   - its non-null values have non-zero variance.
4. No forward fill, backward fill, interpolation, target-based imputation, or synthetic calendar row is allowed.
5. Build block-complete rows over **all eligible features in that block** by retaining only first-fold TRAIN rows where every eligible feature is non-null.
6. The block must have at least `minimum_block_complete_observations` block-complete rows. Otherwise selection fails closed before any HMM fit.
7. Compute the Spearman rank-correlation matrix on those block-complete rows.
8. Correlations must be finite. Undefined/non-finite correlation makes the block selection invalid; no Pearson fallback is allowed.
9. Convert correlation to redundancy distance.

Greek letter used below:

- **rho (ρ)** — pronounced *ROH*; Spearman rank-correlation coefficient.

For eligible features `i` and `j`:

```text
d(i,j) = 1 - abs(ρ(i,j))
```

10. For each eligible feature, compute its medoid score as the arithmetic mean of `d(i,j)` to every other eligible feature in the same block.
11. If exactly one feature is eligible, select it and record medoid score `0.0`.
12. Select the feature with the lowest medoid score.
13. Exact deterministic tie-break order is:
    1. lower medoid score;
    2. higher first-fold TRAIN coverage;
    3. earlier candidate position in the canonical block configuration.
14. Never round a medoid score before ranking/tie-breaking.
15. Exactly one feature must be selected from every block. Zero selected features, more than one selected feature, or a missing block fails closed.

The selector is therefore deterministic for the same input rows, policy, feature order, and dependency versions.

## Freeze rule

After all eight block representatives are selected, the ordered feature set is immutable for that evaluation:

```text
first walk-forward TRAIN
        -> select 8 features once
        -> freeze FeatureSelectionResult
        -> K=2 full / K=3 full / K=4 full
        -> every walk-forward fold uses the same 8 features
```

Later folds never rerun feature selection. TEST rows never affect feature selection. Appending or mutating rows strictly after the first TRAIN end must not change the selected features, medoid evidence, selection hash, or resolved model feature order.

All K=2/K=3/K=4 candidates in one comparison must use the identical frozen selection hash and identical ordered 8-feature set. Candidate comparison across mixed feature-selection hashes is invalid.

## Required selection evidence

A frozen selection result must preserve enough evidence to reproduce why each representative was chosen:

- policy ID/version/hash;
- source dataset/build, schema version, feature version;
- evaluation-plan hash;
- first fold ID;
- exact selection `train_start` and `train_end`;
- canonical block order and candidates;
- per-feature coverage and eligibility/rejection reason;
- per-block complete-row count;
- Spearman correlation values;
- redundancy distances;
- medoid score for every eligible candidate;
- selected feature for every block;
- final ordered eight-feature tuple;
- deterministic feature-selection result hash.

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

Introduce only the immutable contracts/schema needed to describe a feature-selection policy and result. Do not implement Spearman calculations, data loading, first-fold slicing, HMM fitting, profile resolution, MLflow logging, or Xetra block membership in this PR.

### Acceptance criteria

- [ ] Add immutable `FeatureBlock` contract with stable block ID and ordered, unique candidate feature names.
- [ ] Add immutable `FeatureSelectionPolicy` contract with policy ID/version, method, ordered blocks, `minimum_feature_coverage`, and `minimum_block_complete_observations`.
- [ ] Policy validation accepts method exactly `absolute_spearman_medoid`; unknown methods fail closed.
- [ ] Policy validation requires `0 < minimum_feature_coverage <= 1` and `minimum_block_complete_observations >= 2`.
- [ ] Policy validation rejects duplicate block IDs, empty blocks, duplicate feature names inside a block, and the same feature assigned to multiple blocks.
- [ ] Add immutable per-feature/per-block evidence contracts sufficient for coverage, eligibility reason, block-complete count, Spearman correlation/distance evidence, medoid score, and selected flag.
- [ ] Add immutable `FeatureSelectionResult` containing source/evaluation lineage, first-fold selection bounds, ordered block selections, exactly one selected feature per block, ordered final feature tuple, and deterministic result hash.
- [ ] Result validation rejects duplicate selected features and mismatched selected-feature/block order.
- [ ] Extend model-profile schema so a profile may declare either a static exact feature list **or** a feature-selection policy reference/source universe, never both and never neither.
- [ ] A feature-selection profile remains unresolved until a `FeatureSelectionResult` supplies the final ordered model features; unresolved profiles cannot expose themselves as a fitted-model feature order.
- [ ] Serialization round trips are deterministic.
- [ ] Unit tests cover every validation/fail-closed case above.
- [ ] No model-library, MLflow, PostgreSQL, filesystem, FastAPI, or portfolio dependency is introduced.

## PR-046 — Define the exact Xetra eight-block feature-selection profile

- **Status:** BLOCKED by PR-045
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-046-xetra-feature-blocks`
- **Depends on:** PR-045
- **Allowed files:** `configs/feature_selection/xetra_semantic_medoid_v1.yaml`, `docs/profiles/xetra_feature_selection_v1.md`, `tests/unit/feature_selection/test_xetra_feature_blocks.py`

### Task

Encode only the Xetra selection policy and exact semantic block membership. Do not calculate correlations, select representatives, load runtime data, modify HMM code, or log MLflow evidence in this PR.

### Acceptance criteria

- [ ] Config policy ID is exactly `xetra_semantic_medoid_v1`.
- [ ] Method is exactly `absolute_spearman_medoid`.
- [ ] `minimum_feature_coverage` is exactly `0.90`.
- [ ] `minimum_block_complete_observations` is exactly `504`.
- [ ] Config contains exactly the eight canonical blocks in the order specified in this extension.
- [ ] Config contains exactly the 48 canonical `regime-loader` feature names specified in this extension.
- [ ] Every canonical source feature appears exactly once; there are no duplicates or omissions.
- [ ] The test asserts block sizes exactly `4, 21, 4, 4, 3, 3, 7, 2` and total exactly `48`.
- [ ] No `timestamp_m1`, ETF return, portfolio, label, state, target, Sharpe/Sortino, or trading feature is present.
- [ ] Documentation explains why semantic blocks prevent redundant VIX-family features from dominating the full-covariance HMM input dimension.
- [ ] Documentation states successful selection returns exactly one representative per block and exactly eight ordered model features.
- [ ] Config validation uses PR-045 contracts; no duplicated schema implementation is introduced.

## PR-047 — Implement deterministic absolute-Spearman medoid selection

- **Status:** BLOCKED by PR-045
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-047-spearman-medoid-selector`
- **Depends on:** PR-045
- **Allowed files:** `src/market_regime_engine/feature_selection/selector.py`, `tests/unit/feature_selection/test_selector.py`, `tests/fixtures/feature_selection/*`

### Task

Implement a pure in-memory selector that receives an already-bounded training frame plus a validated policy and returns a `FeatureSelectionResult`/selection evidence. It must not know about walk-forward split planning, MLflow, HMMs, PostgreSQL, source transport, APIs, or portfolios.

### Acceptance criteria

- [ ] Selector reads only rows and columns supplied by its caller; it cannot fetch more data.
- [ ] Missing configured candidate columns fail closed with explicit names.
- [ ] Per-feature coverage is computed as non-null count divided by supplied training-row count.
- [ ] Features below policy coverage threshold are ineligible with explicit reason.
- [ ] Features with non-finite non-null values are ineligible/fail according to the contract with explicit reason; no replacement/imputation occurs.
- [ ] Features with zero non-null variance are ineligible with explicit reason.
- [ ] Block-complete rows are constructed only after feature eligibility is known and use all eligible features in that block.
- [ ] Fewer than `minimum_block_complete_observations` block-complete rows fails that block and the whole selection.
- [ ] Spearman rank correlations are computed only on block-complete rows and are finite.
- [ ] Redundancy distance is exactly `1 - abs(rho)`.
- [ ] Medoid score is exactly the arithmetic mean distance to all other eligible features in the same block.
- [ ] A one-eligible-feature block selects that feature with score `0.0`.
- [ ] Winner ranking is exactly lower medoid score, then higher coverage, then earlier configured candidate order.
- [ ] Ranking uses full computed precision; no pre-ranking rounding.
- [ ] Output order follows canonical block order, never alphabetical discovery order.
- [ ] Exactly one feature is selected per block; otherwise selection fails closed.
- [ ] Same frame/policy produces byte-equivalent serialized result evidence and the same result hash.
- [ ] Tests cover perfect positive correlation, perfect negative correlation, monotonic nonlinear ranks, coverage exclusion, zero variance, insufficient complete rows, undefined correlation, one-candidate block, deterministic tie-breaks, and repeated-run determinism.
- [ ] Tests prove selector accepts no ETF/target/label input and uses no HMM metric.

## PR-048 — Freeze feature selection from the first walk-forward TRAIN only

- **Status:** BLOCKED by PR-020, PR-046, PR-047
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-048-freeze-first-train-features`
- **Depends on:** PR-020, PR-046, PR-047
- **Allowed files:** `src/market_regime_engine/feature_selection/freeze.py`, `tests/unit/feature_selection/test_freeze.py`, `tests/integration/test_feature_selection_freeze.py`, `EVALUATION.md`

### Task

Integrate the pure selector with the deterministic walk-forward plan. This PR owns the leak-free selection/freeze semantics and is the first PR that makes the feature-selection method part of the evaluation contract. Do not fit an HMM or log MLflow data here.

### Acceptance criteria

- [ ] The selection slice is exactly the TRAIN interval of the first planned walk-forward fold.
- [ ] No first-fold TEST row or later-fold row is passed into the selector.
- [ ] The selection uses the same source dataset/build lineage and evaluation-plan hash as the later evaluation.
- [ ] The Xetra policy from PR-046 must resolve exactly eight ordered selected features or fail before HMM work begins.
- [ ] The frozen result records first fold ID, exact UTC train bounds, source lineage, policy hash, evidence, selected feature tuple, and deterministic result hash.
- [ ] Appending rows strictly after first-fold `train_end` cannot change any selection evidence, selected feature, selected order, or selection hash.
- [ ] Mutating any first-fold TEST/later row cannot change the frozen result.
- [ ] Mutating a first-fold TRAIN value is allowed to change the result and is covered by a positive-control test.
- [ ] Selection is executed once per evaluation plan/source snapshot, not once per candidate and not once per fold.
- [ ] No re-selection API exists inside the fold loop.
- [ ] No imputation/fill is introduced by the freeze layer.
- [ ] Selection failure is explicit and prevents model evaluation from starting.
- [ ] `EVALUATION.md` gains a normative feature-selection section matching the exact eight-block, absolute-Spearman medoid, first-TRAIN-only, frozen-feature semantics.
- [ ] `EVALUATION.md` explicitly states feature selection uses no ETF/portfolio metric and Portfell owns downstream economic validation.
- [ ] `EVALUATION.md` explicitly states raw HMM OOS likelihood is not used to search across feature subsets of different dimensions.

## PR-049 — Resolve model profiles with the frozen eight-feature result

- **Status:** BLOCKED by PR-021, PR-048
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-049-resolve-selected-feature-profile`
- **Depends on:** PR-021, PR-048
- **Allowed files:** `src/market_regime_engine/profiles/resolution.py`, `tests/unit/profiles/test_resolution.py`, `tests/integration/test_xetra_profile_resolution.py`

### Task

Create the narrow resolver that turns the validated Xetra base profile plus one frozen `FeatureSelectionResult` into the exact resolved model profile used for preprocessing/HMM fitting. Do not calculate correlations, rerun selection, fit a model, or log MLflow data.

### Acceptance criteria

- [ ] Resolver requires a validated base profile that references `xetra_semantic_medoid_v1` and a compatible frozen result.
- [ ] Resolver rejects source dataset/build, feature version, policy ID/version/hash, or evaluation-plan mismatches.
- [ ] Resolved model feature order is exactly the eight selected features in canonical block order.
- [ ] Resolved profile preserves the original 48-feature source universe separately from the final eight model features.
- [ ] Resolved profile hash includes the frozen feature-selection result hash.
- [ ] K=2 full, K=3 full, and K=4 full candidate specs all receive the identical eight-feature order and identical feature-selection hash.
- [ ] Resolver rejects zero/more-than-eight selected features for `xetra_cross_asset_v1`.
- [ ] Resolver rejects duplicate selected features.
- [ ] Unresolved base profiles cannot be passed through the resolver as if they already had a fitted-model feature order.
- [ ] Serialization/reload preserves source universe, resolved model features, and selection hash exactly.
- [ ] Integration test resolves the checked-in Xetra profile with a deterministic frozen fixture and proves all three Gaussian candidates share the same eight features.
- [ ] Module has no HMM fitting, MLflow, PostgreSQL, FastAPI, or portfolio logic.

## PR-050 — Log auditable feature-selection evidence to MLflow

- **Status:** BLOCKED by PR-023, PR-048, PR-049
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/PR-050-mlflow-feature-selection-evidence`
- **Depends on:** PR-023, PR-048, PR-049
- **Allowed files:** `src/market_regime_engine/mlflow_support/feature_selection_tracking.py`, `tests/unit/mlflow_support/test_feature_selection_tracking.py`, `tests/integration/test_mlflow_feature_selection_tracking.py`, `EVALUATION.md`

### Task

Log feature-selection provenance/evidence to the existing MLflow parent evaluation run. Do not change selection mathematics, rerun selection, alter candidate ranking, or add downstream ETF/portfolio metrics.

### Acceptance criteria

- [ ] Parent evaluation run records stable tags/parameters for policy ID/version/hash, selection hash, selected-feature count `8`, first fold ID, and selection train bounds.
- [ ] Each candidate run records only the same `feature_selection_hash` reference needed to prove all candidates use the same frozen feature set; detailed evidence is not duplicated per candidate.
- [ ] Parent run stores `feature_selection/selection.json` containing source/evaluation lineage, blocks, selected features/order, policy metadata, and result hash.
- [ ] Parent run stores `feature_selection/scores.parquet` with one row per candidate feature and at least block ID, candidate order, coverage, eligibility, rejection reason, medoid score, and selected flag.
- [ ] Parent run stores `feature_selection/correlations.parquet` in deterministic long form with block ID, feature A, feature B, Spearman rho, redundancy distance, and block-complete observation count.
- [ ] Artifact row/column ordering is deterministic.
- [ ] No ETF return, portfolio metric, Sharpe/Sortino, drawdown, allocation, transaction cost, or trading target is logged by this module.
- [ ] Local-file MLflow integration round-trips all feature-selection tags/artifacts without external network access.
- [ ] Integration test proves K=2/K=3/K=4 candidate runs reference exactly the same selection hash.
- [ ] `EVALUATION.md` documents the exact MLflow tags/artifact paths introduced by this PR.
- [ ] No new required test contacts the shared NAS MLflow server.

---

# Addenda to existing backlog PRs

These addenda are mandatory when the corresponding existing PR is delegated. They supplement the PR section in `BACKLOG.md` and take precedence if an old acceptance criterion implies a static final feature list or per-fold feature re-selection.

## PR-021 addendum — Xetra base profile

### Dependency override

PR-021 additionally depends on **PR-045 and PR-046**.

### Additional/replacement acceptance criteria

- [ ] `xetra_cross_asset_v1` declares the exact 48-feature source universe from the Xetra semantic block profile rather than hard-coding the final eight HMM features.
- [ ] It references selection policy exactly `xetra_semantic_medoid_v1`.
- [ ] The base profile is intentionally unresolved with respect to final HMM feature order until PR-048/PR-049 supply the frozen result.
- [ ] The profile still contains exactly the three Gaussian HMM candidates `gaussian_hmm_k2_full`, `gaussian_hmm_k3_full`, and `gaussian_hmm_k4_full`.
- [ ] No ETF/portfolio target or downstream performance field is added to the engine profile.
- [ ] Profile/hash tests include the feature-selection policy reference and 48-feature source-universe identity.

## PR-022 addendum — Walk-forward runner

### Dependency override

PR-022 additionally depends on **PR-049** and must not start until the resolved-profile path is merged.

### Additional acceptance criteria

- [ ] Runner accepts only a resolved model profile with a frozen feature-selection result/hash.
- [ ] Preprocessing and HMM fitting use exactly the resolved eight-feature order.
- [ ] Feature selection is not called from inside the fold loop.
- [ ] Every fold and every K=2/K=3/K=4 candidate uses the same eight features and same selection hash.
- [ ] Fold evidence records the selection hash used for the fit.
- [ ] Mutating future/test rows cannot change the selected feature set in addition to the existing no-leakage guarantees.
- [ ] Mixed/unresolved feature-selection hashes fail before candidate comparison.

## PR-024 addendum — Candidate grid

### Additional acceptance criteria

- [ ] Candidate-grid input contains one resolved profile/feature-selection hash shared by all candidates.
- [ ] Candidate comparison rejects candidates with different feature-selection hashes or feature orders.
- [ ] Candidate comparison table includes `feature_selection_hash` and resolved feature-count evidence.
- [ ] K=2/K=3/K=4 comparison therefore remains dimensionally comparable because all candidates model the identical eight-dimensional observation vector.

## PR-035 addendum — Hermetic engine E2E

### Dependency override

PR-035 additionally depends on **PR-050**.

### Additional acceptance criteria

- [ ] E2E fixture exposes all 48 canonical source features.
- [ ] Exactly eight features are selected once from the first fold TRAIN interval.
- [ ] Final selected order follows the eight canonical block order.
- [ ] All K=2/K=3/K=4 candidates use exactly the same selected features and selection hash.
- [ ] Appending/mutating future test rows leaves feature selection and its hash unchanged.
- [ ] MLflow parent run contains the required feature-selection evidence artifacts and candidate runs reference the same hash.
- [ ] No ETF/portfolio performance data is required anywhere in the E2E evaluation.

## PR-036 addendum — Final documentation

### Additional acceptance criteria

- [ ] README/operations/evaluation documentation explains `48 source features -> 8 semantic blocks -> one absolute-Spearman medoid per block -> frozen eight-feature profile -> K=2/K=3/K=4 full-covariance walk-forward evaluation`.
- [ ] Documentation states selection uses only first-fold TRAIN and is never rerun in later folds.
- [ ] Documentation states the engine's feature selection and champion selection are statistical only.
- [ ] Documentation states Portfell, not the engine, owns downstream Xetra ETF economic/portfolio evaluation.
- [ ] Documentation links the MLflow feature-selection evidence artifacts and explains how to audit why each representative was chosen.

---

# Revised statistical evaluation flow

```text
regime-loader PostgreSQL snapshot
          |
          v
48 validated source features
          |
          v
validated xetra_semantic_medoid_v1 policy
          |
          v
walk-forward plan
          |
          +--> FIRST FOLD TRAIN ONLY
                    |
                    +--> quality eligibility per semantic block
                    +--> block-complete TRAIN rows
                    +--> absolute Spearman redundancy distances
                    +--> one deterministic medoid per block
                    |
                    v
             frozen 8-feature result/hash
                    |
                    v
             resolved model profile
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

The feature set is selected once and held constant; the model complexity comparison is therefore only K=2 versus K=3 versus K=4 under the same eight-dimensional input space.

---

# Parallel execution plan for feature selection

```text
After PR-007:
  PR-045

After PR-045, parallel:
  PR-046   PR-047

Independent existing work can continue in parallel:
  PR-020 walk-forward split planner

When its existing dependencies plus PR-045/046 are ready:
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

For PR-045 through PR-050, an orchestrator gives a weak agent exactly:

1. the relevant single PR section from this file;
2. the `Statistical feature-selection contract` section above;
3. `CONTRIBUTING.md`;
4. only the upstream contract/doc explicitly needed by that PR.

The agent must not receive authority to implement adjacent PRs. If its task requires a file outside `Allowed files`, an unmerged dependency, an unspecified threshold, a different selection method, per-fold re-selection, HMM-based feature search, or downstream ETF data, it stops rather than broadening scope.
