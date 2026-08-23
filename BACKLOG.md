# Market Regime Engine — Implementation Backlog

Status date: 2026-08-23

This backlog defines the complete implementation plan for `market-regime-engine` as a reusable market-regime model platform.

The repository owns model training, causal HMM inference, leak-free walk-forward validation, statistical model comparison, MLflow tracking/registry integration, immutable prediction artifacts, and batch/realtime inference APIs. It does **not** own market-data acquisition or portfolio optimization.

```text
regime-loader
    -> immutable causal feature data
    -> market-regime-engine
    -> MLflow Model Registry + RegimePrediction.v1 + OOS prediction artifacts + API
    -> portfell and future consumers
```

`regime-loader` remains the reusable data product. `portfell` remains the Xetra ETF portfolio application. `market-regime-engine` stays consumer-agnostic so registered HMM models can later be reused by BTC, equity, rates, covered-call, or other projects.

## Repository bootstrap facts

- Default branch: `main`.
- Stable Python feature release selected for this repository: **Python 3.14.7**.
- Local development environment: repository-local `.venv`; `.venv/` is never committed.
- Python compatibility target for MVP: `>=3.14,<3.15`.
- Shared production MLflow Tracking Server / Model Registry: **`http://10.10.1.3:5000`**.
- Production/NAS examples use `MLFLOW_TRACKING_URI=http://10.10.1.3:5000`.
- Required CI remains hermetic and never depends on the shared NAS service.
- The shared MLflow/PostgreSQL deployment is external infrastructure and is not owned by this repository.
- `EVALUATION.md` is the durable sidecar for evaluation methodology, metric definitions, MLflow scorecards, and champion-selection rules.
- All implementation PRs start from a clean, up-to-date `main` after declared dependencies are merged.
- Agents never edit `BACKLOG.md`; backlog state is maintained by the orchestrator.
- `BACKLOG.md` is the single authoritative backlog file; secondary `BACKLOG*.md` planning files are forbidden.

## Non-negotiable architecture rules

1. No provider HTTP clients, CBOE/FRED/ECB/STOXX acquisition logic, or EODHD portfolio-data acquisition in this repository.
2. No portfolio optimizer, ETF weighting, Sharpe/Sortino application selection, transaction-cost model, or trading logic in this repository.
3. Model training may use only observations available inside the declared training window.
4. Backtest-safe inference uses filtered probabilities only. Retrospective smoothing/Viterbi is diagnostic-only.
5. All preprocessing parameters are fit on training data only.
6. Raw HMM state numbers are not stable consumer semantics; persistent state identity requires deterministic state alignment.
7. Every registered model carries exact feature order, preprocessing state, model parameters, state mapping, source/build lineage, package version, and Git commit.
8. Historical `walk_forward_oos` predictions and `fixed_model_replay` are different products and must never be conflated.
9. Application source code reads MLflow through `MLFLOW_TRACKING_URI`; production/NAS value is `http://10.10.1.3:5000`.
10. MLflow credentials/tokens are never committed.
11. Public consumer contracts are versioned. Initial prediction contract is `RegimePrediction.v1`.
12. MLflow Model Registry is authoritative for promoted fitted models; Git is authoritative for source/configuration, not binary fitted artifacts.
13. **Evaluation sidecar rule:** any PR that changes candidate model families, K/state counts, covariance modes, preprocessing, walk-forward semantics, inference semantics, diagnostic definitions, quality gates, ranking/tie-break rules, MLflow metric names/artifacts, or model lifecycle aliases must update `EVALUATION.md` in the same PR. Such a PR is incomplete if the sidecar is stale.
14. Engine champion selection must use explicit hard gates and deterministic ranking. No hidden or weighted “magic score” is permitted.
15. For `xetra_cross_asset_v1`, the default primary ranking statistic is **mean OOS predictive log likelihood per observation**; fold stability is secondary and BIC/AIC are tertiary/tie-break diagnostics.
16. **Full-covariance-only HMM rule:** every covariance-bearing HMM emission model uses one full covariance matrix per hidden state. `diag`, `spherical`, `tied`, or any other reduced covariance mode is unsupported and must fail validation; no implementation, profile, test, or documentation may silently fall back to a reduced covariance structure.
17. **Fold-observability rule:** every scalar fold diagnostic required by `EVALUATION.md` is logged both on its fold run and as ordered candidate-run MLflow metric history using deterministic `fold_index`; human-facing plots use the real fold `test_end` date. Missing/invalid folds are never interpolated. Non-scalar transition/full-covariance matrices remain machine-readable fold artifacts and are also rendered as diagnostic heatmaps.

## Git discipline for every PR

Every PR below has an explicit branch and Git status.

```text
Before work:
  git switch main
  git pull --ff-only
  git status --short

Required before branch creation:
  <empty output>

Create exactly the branch declared by the PR.

Before final push:
  git status --short

Required after all intended files are committed:
  <empty output>
```

An agent stops if the working tree is not clean, a dependency is not merged, or work requires files outside the PR's allowed-file scope.

## CI and merge policy target

Two independent GitHub Actions workflows are required.

### Push gate

Trigger: every branch push. Required parallel jobs:

```text
lint
 type
 unit
 integration
```

A final job named exactly `push-gate` depends on all four.

### Merge gate

Trigger: pull requests targeting `main`. Required parallel jobs:

```text
lint
 type
 unit
 integration
```

A final job named exactly `merge-gate` depends on all four.

Required CI must remain hermetic: neither push nor merge gates may require access to `10.10.1.3`, MLflow, market-data providers, or any other external service. External MLflow verification is an explicit opt-in smoke test.

### Protected `main`

After `merge-gate` exists on `main`, governance must enforce:

- changes to `main` only through pull requests;
- required status check `merge-gate`, strict/up-to-date;
- force pushes disabled;
- branch deletion disabled;
- conversation resolution required;
- administrators included in protection;
- repository auto-merge enabled;
- squash merge as merge method;
- merged feature branches deleted;
- auto-merge enabled on implementation PRs;
- merge only after successful `merge-gate`.

The initial bootstrap/governance sequence is the only temporary exception while the required check is being established.

---

# Wave 0 — Bootstrap and governance

Only PR-001 starts immediately. PR-002 and PR-003 start in parallel after PR-001. PR-004 follows PR-003.

## PR-001 — Bootstrap Python 3.14.7 project and local `.venv`

- **Status:** TODO
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/001-bootstrap-python314`
- **Depends on:** none
- **Allowed files:** `.python-version`, `.gitignore`, `pyproject.toml`, `README.md`, `src/market_regime_engine/__init__.py`, `tests/unit/test_package_smoke.py`, `tests/conftest.py`, `scripts/bootstrap_venv.sh`, `scripts/bootstrap_venv.ps1`

### Acceptance criteria

- [ ] `.python-version` contains `3.14.7`.
- [ ] `pyproject.toml` defines `market-regime-engine` and `requires-python = ">=3.14,<3.15"`.
- [ ] Source layout is `src/market_regime_engine`.
- [ ] Runtime dependencies include Pydantic, NumPy, SciPy, scikit-learn, Polars, PyArrow, FastAPI, Uvicorn, HTTPX, MLflow, Matplotlib, and a Python-3.14-compatible Gaussian-HMM implementation with full-covariance support.
- [ ] Dev dependencies include pytest, pytest-cov, pytest-xdist, Ruff, mypy, and build tooling.
- [ ] Ruff targets Python 3.14; mypy is strict.
- [ ] pytest markers include `unit`, `integration`, and `external_service`.
- [ ] `.gitignore` excludes `.venv/`, caches, coverage/build output, local MLflow state, local prediction artifacts, and IDE files.
- [ ] Shell and PowerShell bootstrap scripts create `.venv` with Python 3.14, upgrade packaging tools, and install editable dev dependencies.
- [ ] Bootstrap scripts fail on the wrong interpreter and print activation commands without activating implicitly.
- [ ] README documents exact Linux/macOS and Windows bootstrap commands.
- [ ] Smoke test verifies import and `__version__`.
- [ ] Clean-checkout `.venv` bootstrap and smoke test pass.
- [ ] `.venv/` is never tracked.

## PR-002 — Add parallel push quality gate

- **Status:** BLOCKED by PR-001
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/002-push-quality-gate`
- **Depends on:** PR-001
- **Allowed files:** `.github/workflows/push-gate.yml`

### Acceptance criteria

- [ ] Runs on every branch push.
- [ ] `lint`, `type`, `unit`, `integration` are independent parallel jobs.
- [ ] All jobs use Python 3.14.7.
- [ ] `lint` runs Ruff check and format-check.
- [ ] `type` runs strict mypy.
- [ ] `unit` excludes `integration` and `external_service`.
- [ ] `integration` includes only `integration` and excludes `external_service`.
- [ ] No external network/MLflow dependency is required.
- [ ] Final job is exactly `push-gate` with all four jobs in `needs`.
- [ ] Superseded runs on the same ref are cancelled.

## PR-003 — Add parallel merge quality gate

- **Status:** BLOCKED by PR-001
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/003-merge-quality-gate`
- **Depends on:** PR-001
- **Allowed files:** `.github/workflows/merge-gate.yml`

### Acceptance criteria

- [ ] Runs only for pull requests targeting `main`.
- [ ] `lint`, `type`, `unit`, `integration` are independent parallel jobs.
- [ ] All jobs use Python 3.14.7.
- [ ] Commands/marker policy match push gate.
- [ ] Final job is exactly `merge-gate` and depends on all four jobs.
- [ ] Failure/cancellation/unexpected skip of a required job prevents success.
- [ ] Superseded runs for the same PR are cancelled.
- [ ] No external network, MLflow, NAS, provider, deployment, or admin secret is required.

## PR-004 — Configure protected `main` and repository auto-merge

- **Status:** BLOCKED by PR-003
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/004-repository-governance`
- **Depends on:** PR-003
- **Allowed files:** `scripts/configure_github_governance.sh`, `docs/repository_governance.md`

### Acceptance criteria

- [ ] Script requires authenticated `gh` and fails without admin permission.
- [ ] Script targets exactly `SergejSchweizer/market-regime-engine` / `main`.
- [ ] Repository auto-merge is enabled.
- [ ] Squash merge and delete-branch-after-merge are configured.
- [ ] `main` requires pull requests and required check exactly `merge-gate` with strict/up-to-date branches.
- [ ] Force pushes and deletion of `main` are disabled.
- [ ] Conversation resolution is required and admins are included.
- [ ] Documentation states the script is executed after this PR merges.
- [ ] Exact verification commands are documented.
- [ ] Post-merge verification confirms every target setting.

---

# Wave 1 — Public contracts and boundaries

After PR-001, PR-005 and PR-006 may run in parallel. After PR-006, PR-007 through PR-013 may run in parallel.

## PR-005 — Document durable architecture and MLflow/evaluation ownership boundaries

- **Status:** BLOCKED by PR-001
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/005-architecture-contract`
- **Depends on:** PR-001
- **Allowed files:** `ARCHITECTURE.md`, `docs/model_lifecycle.md`, `README.md`, `EVALUATION.md`

### Acceptance criteria

- [ ] Architecture defines the engine as reusable model platform, not data loader or portfolio optimizer.
- [ ] Documents ports/adapters around feature input, model adapters, evaluation, MLflow, prediction persistence, and API.
- [ ] Documents shared production MLflow Tracking Server / Registry as `http://10.10.1.3:5000`.
- [ ] States MLflow/PostgreSQL deployment is external infrastructure.
- [ ] Source code resolves MLflow through `MLFLOW_TRACKING_URI`.
- [ ] Separates `fixed_model_replay` from `walk_forward_oos`.
- [ ] Requires filtered probabilities for causal inference; smoothing/Viterbi are diagnostic-only.
- [ ] Requires persistent state alignment for consumer-facing predictions.
- [ ] Defines `regime-loader -> engine -> consumers` dependency direction without package coupling.
- [ ] Documents the full-covariance-only HMM rule and explicitly excludes reduced covariance modes.
- [ ] Documents the MLflow observability hierarchy: parent evaluation run -> candidate run with fold metric histories -> auditable fold runs/artifacts.
- [ ] Documents that fold trend plots use actual TEST-end dates and that invalid folds are shown as gaps/invalid markers rather than interpolated.
- [ ] Model lifecycle defines `candidate`, `validated`, `engine-champion`, `challenger`, and consumer-specific aliases such as `portfell-production`.
- [ ] `EVALUATION.md` is linked as the normative evaluation sidecar and its maintenance rule is documented.
- [ ] README includes system diagram and links.

## PR-006 — Define versioned core domain contracts

- **Status:** BLOCKED by PR-001
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/006-core-domain-contracts`
- **Depends on:** PR-001
- **Allowed files:** `src/market_regime_engine/contracts/__init__.py`, `src/market_regime_engine/contracts/features.py`, `src/market_regime_engine/contracts/models.py`, `src/market_regime_engine/contracts/predictions.py`, `src/market_regime_engine/contracts/lineage.py`, `tests/unit/contracts/*`

### Acceptance criteria

- [ ] Immutable feature reference includes source dataset/build, as-of range, feature version, and ordered feature names.
- [ ] Immutable `ModelSpec` includes family, state count, covariance mode, profile/version, seeds/multi-start, and training-window policy.
- [ ] For every covariance-bearing HMM `ModelSpec`, covariance mode must be exactly `full`; `diag`, `spherical`, `tied`, and unknown covariance modes fail validation.
- [ ] Lineage includes engine version, Git SHA, training interval, source build, preprocessing version, and profile hash.
- [ ] `RegimePredictionV1` includes as-of, profile, model/version, persistent state IDs, probabilities, dominant state, entropy, confidence, lineage, and data-quality status.
- [ ] Validation rejects invalid/non-finite probabilities, duplicates, and non-normalized vectors.
- [ ] Raw library state labels cannot become consumer semantics without alignment ID.
- [ ] Serialization round-trip tests cover all public contracts, including preservation of `covariance_mode="full"`.
- [ ] Contract layer has no model-library, MLflow, filesystem, FastAPI, or provider dependency.

## PR-007 — Add model-profile configuration schema and loader

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/007-model-profile-config`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/profiles/__init__.py`, `src/market_regime_engine/profiles/schema.py`, `src/market_regime_engine/profiles/loader.py`, `tests/unit/profiles/*`, `EVALUATION.md`

### Acceptance criteria

- [ ] Schema declares profile ID/version, frequency, exact ordered features, candidate specs, training policy, inference policy, quality thresholds, and selection policy.
- [ ] Selection policy can explicitly declare primary metric, secondary fold-stability ordering, tertiary/tie-break metrics, and final deterministic tie-break.
- [ ] YAML loading is deterministic and validated before model work.
- [ ] Unknown keys and duplicate features fail closed.
- [ ] Unsupported family/state count/metric identifier fails clearly.
- [ ] Covariance-bearing HMM candidates accept only `covariance: full`; `diag`, `spherical`, `tied`, omitted/ambiguous covariance settings, and unknown covariance modes fail closed.
- [ ] Profile has deterministic content hash.
- [ ] Unit tests cover valid, malformed, unknown, duplicate, unsupported, full-covariance-only, ranking-policy, and hash cases.
- [ ] `EVALUATION.md` remains consistent with supported selection-policy fields and full-covariance policy.

## PR-008 — Add generic Parquet feature-source port and adapter

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/008-parquet-feature-source`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/features/__init__.py`, `src/market_regime_engine/features/ports.py`, `src/market_regime_engine/features/parquet_source.py`, `tests/unit/features/*`, `tests/integration/test_parquet_feature_source.py`

### Acceptance criteria

- [ ] Narrow `FeatureSource` protocol has no `regime-loader` import.
- [ ] Parquet adapter reads data plus explicit lineage/manifest input.
- [ ] Selects features in exact profile order.
- [ ] Duplicate/non-monotonic timestamps, missing/duplicate features, and non-finite values fail.
- [ ] No forward-fill, backward-fill, interpolation, or silent imputation.
- [ ] Source build and feature version are preserved.
- [ ] Integration fixture proves loader-shaped Gold data can be consumed without upstream package dependency.

## PR-009 — Add train-only preprocessing pipeline

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/009-preprocessing-pipeline`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/preprocessing/__init__.py`, `src/market_regime_engine/preprocessing/scaler.py`, `src/market_regime_engine/preprocessing/contracts.py`, `tests/unit/preprocessing/*`, `EVALUATION.md`

### Acceptance criteria

- [ ] Serializable preprocessing contract.
- [ ] Standard scaling fit only on provided training slice.
- [ ] Exact feature order is preserved.
- [ ] Zero variance and non-finite values fail explicitly.
- [ ] Deterministic serialize/deserialize.
- [ ] Test proves future rows cannot alter training parameters.
- [ ] `EVALUATION.md` documents any preprocessing semantics that affect evaluation.

## PR-010 — Define model-adapter and fitted-model artifact protocols

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/010-model-adapter-protocol`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/models/__init__.py`, `src/market_regime_engine/models/protocols.py`, `src/market_regime_engine/models/artifacts.py`, `tests/unit/models/test_protocols.py`, `tests/unit/models/test_artifacts.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Adapter protocol defines fit, score, predictive-score, parameter extraction, reconstruction, and filtered-inference capability boundaries.
- [ ] Predictive-score interface can return fold OOS predictive log likelihood and observation count without fitting on test data.
- [ ] Fitted artifact contains initial probabilities, transition matrix, emissions, feature order, K, family, covariance mode, preprocessing reference, and convergence metadata.
- [ ] Covariance-bearing HMM artifacts preserve complete per-state full covariance matrices including off-diagonal terms.
- [ ] Artifact validation checks shapes, finite values, normalized rows, full-covariance validity, and feature/model consistency.
- [ ] Protocol has no MLflow, HTTP, filesystem, or portfolio dependency.
- [ ] Deterministic dummy adapter proves serialization/reconstruction and predictive-score contracts.
- [ ] `EVALUATION.md` matches the public scoring semantics.

## PR-011 — Add immutable prediction-store port and local Parquet adapter

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/011-prediction-store`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/predictions/__init__.py`, `src/market_regime_engine/predictions/ports.py`, `src/market_regime_engine/predictions/parquet_store.py`, `tests/unit/predictions/*`, `tests/integration/test_prediction_store.py`

### Acceptance criteria

- [ ] Prediction metadata distinguishes `walk_forward_oos` and `fixed_model_replay`.
- [ ] Includes profile/version, model/version, range, lineage, and build ID.
- [ ] Parquet writes are atomic and versioned.
- [ ] Immutable builds cannot be overwritten.
- [ ] Manifest/data metadata agree.
- [ ] Research reader requires explicit build ID; no silent latest.
- [ ] Integration test covers write/reload/checksum and overwrite failure.

## PR-012 — Bind MLflow client boundary to shared NAS Tracking Server / Registry

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/012-mlflow-client-boundary`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/mlflow_support/__init__.py`, `src/market_regime_engine/mlflow_support/settings.py`, `src/market_regime_engine/mlflow_support/ports.py`, `tests/unit/mlflow_support/*`, `.env.example`

### Acceptance criteria

- [ ] Production MLflow setting is supplied through `MLFLOW_TRACKING_URI`.
- [ ] `.env.example` contains `MLFLOW_TRACKING_URI=http://10.10.1.3:5000` and no secrets.
- [ ] Settings expose the shared endpoint as documented production/NAS default while allowing local-file mode.
- [ ] Production validation rejects missing/blank tracking URI with actionable error.
- [ ] Narrow tracking and registry ports are defined for application code.
- [ ] Tracking port supports repeated metric logging with explicit metric key, numeric value, deterministic `step`, and optional timestamp so candidate-run fold histories can be represented without direct MLflow coupling in evaluation code.
- [ ] Artifact logging boundary supports deterministic PNG/JSON/Parquet diagnostic artifacts and nested artifact paths.
- [ ] No experiment ID, registered-model version, token, username, or password is hard-coded.
- [ ] Unit/local-file mode never contacts `10.10.1.3`.
- [ ] Unit tests cover production URI parsing, local-file URI, invalid URI, no-network construction, repeated metric-history logging contract, and nested artifact logging contract.

## PR-013 — Add FastAPI skeleton and health routes

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/013-api-skeleton`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/api/__init__.py`, `src/market_regime_engine/api/app.py`, `src/market_regime_engine/api/dependencies.py`, `src/market_regime_engine/api/routes/health.py`, `src/market_regime_engine/api/routes/latest.py`, `src/market_regime_engine/api/routes/batch.py`, `src/market_regime_engine/api/routes/evaluations.py`, `tests/unit/api/test_health.py`

### Acceptance criteria

- [ ] App factory has no MLflow/filesystem/network call at import time.
- [ ] `/health/live` returns liveness.
- [ ] `/health/ready` delegates to injected readiness checks.
- [ ] Placeholder route modules exist for latest, batch, and evaluations.
- [ ] No model/business logic in routes.
- [ ] OpenAPI generation unit test passes.

---

# Wave 2 — HMM core and causal inference

PR-014 follows PR-009/010. After PR-014, PR-015, PR-016, PR-018, and PR-019 may run in parallel. PR-017 follows PR-016. PR-020/021 run when their own dependencies are met.

## PR-014 — Implement full-covariance Gaussian HMM adapter

- **Status:** BLOCKED by PR-009, PR-010
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/014-gaussian-hmm-adapter`
- **Depends on:** PR-009, PR-010
- **Allowed files:** `src/market_regime_engine/models/gaussian_hmm.py`, `tests/unit/models/test_gaussian_hmm.py`, `tests/fixtures/hmm/*`, `EVALUATION.md`

### Acceptance criteria

- [ ] Implements model protocol including predictive scoring on unseen observations.
- [ ] K configurable; tests K=2/3/4.
- [ ] Gaussian emissions use covariance type exactly `full`; reduced covariance modes are not configurable alternatives.
- [ ] Explicit construction/configuration with `diag`, `spherical`, `tied`, or any non-`full` covariance mode fails closed with an actionable validation error.
- [ ] Every hidden state owns a `d x d` full covariance matrix; matrices are finite, symmetric, and pass the declared positive-definiteness/Cholesky validity check.
- [ ] Off-diagonal covariance terms are fitted, serialized, reconstructed, and verified by deterministic tests.
- [ ] Seed, max iterations, and tolerance explicit.
- [ ] Fit returns convergence, iterations, training log likelihood, initial probabilities, transition matrix, means, and full covariance matrices.
- [ ] Predictive score returns finite test log likelihood and observation count without refitting on test data.
- [ ] Invalid/non-converged fits are explicit and non-promotable.
- [ ] Fitted artifact reconstructs an equivalent full-covariance model.
- [ ] Deterministic synthetic tests verify normalization, shapes, reproducibility, off-diagonal covariance preservation, and predictive scoring for K=2/3/4.
- [ ] `EVALUATION.md` remains accurate for the three full-covariance Gaussian candidates.

## PR-015 — Add deterministic multi-start HMM fitting and stability metrics

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/015-hmm-multistart`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/training/__init__.py`, `src/market_regime_engine/training/multistart.py`, `tests/unit/training/test_multistart.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Explicit ordered seed list/count.
- [ ] Every start records seed, convergence, validity, iteration count, final training likelihood, and failure reason.
- [ ] Failed/non-finite fits are excluded from winner selection but retained diagnostically.
- [ ] Highest-likelihood valid converged fit wins with deterministic tie break.
- [ ] Minimum valid converged starts and/or success rate are configurable hard gates.
- [ ] Aggregates include `multistart_total`, `multistart_converged`, `multistart_valid`, `multistart_success_rate`, `multistart_loglik_best`, `multistart_loglik_median`, `multistart_loglik_std`.
- [ ] Insufficient stable starts fail closed.
- [ ] Tests cover mixed failures, ties, aggregate calculations, and insufficient valid starts.
- [ ] `EVALUATION.md` documents exactly the implemented multi-start metrics.

## PR-016 — Implement causal forward filtering

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/016-causal-forward-filter`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/inference/__init__.py`, `src/market_regime_engine/inference/filtering.py`, `tests/unit/inference/test_filtering.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Forward-only filtered state probabilities from fitted parameters.
- [ ] Output at `t` depends only on observations `<=t`.
- [ ] Probabilities finite/normalized.
- [ ] Numerical stabilization tested on long series.
- [ ] Appending future observations leaves historical filtered probabilities unchanged.
- [ ] Tests distinguish filtering from smoothing/Viterbi.
- [ ] Output can construct `RegimePredictionV1` after state alignment.
- [ ] `EVALUATION.md` states the exact causal inference semantics.

## PR-017 — Add transition-horizon probability forecasts

- **Status:** BLOCKED by PR-016
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/017-transition-forecasts`
- **Depends on:** PR-016
- **Allowed files:** `src/market_regime_engine/inference/forecasting.py`, `tests/unit/inference/test_forecasting.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Forecasts state distribution for integer horizons from transition matrix.
- [ ] Horizon zero equals current filtered distribution.
- [ ] Outputs finite/normalized.
- [ ] Invalid matrices/horizons fail.
- [ ] Tests compare 1-step/multi-step against direct matrix powers.
- [ ] `EVALUATION.md` is updated if forecast metrics become part of evaluation; otherwise it explicitly labels forecasts as inference outputs, not champion metrics.

## PR-018 — Add persistent state alignment and drift diagnostics

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/018-state-alignment`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/states/__init__.py`, `src/market_regime_engine/states/signatures.py`, `src/market_regime_engine/states/alignment.py`, `tests/unit/states/*`, `EVALUATION.md`

### Acceptance criteria

- [ ] Normalized state signatures from emissions in exact feature order.
- [ ] Full-covariance Gaussian state signatures preserve cross-feature covariance information; off-diagonal covariance terms must not be dropped or reduced to per-feature variances.
- [ ] Deterministic one-to-one mapping to reference states.
- [ ] Pure label permutations map correctly.
- [ ] Persistent IDs are independent of raw labels.
- [ ] Per-state alignment distance/drift is recorded.
- [ ] Maximum signature drift configurable and fail-closed.
- [ ] Ambiguous mapping is explicit and non-promotable.
- [ ] Alignment artifact has deterministic hash/version.
- [ ] Tests cover permutation, small/excessive drift, ambiguity, full-covariance signature preservation, and max-drift aggregation.
- [ ] `EVALUATION.md` documents alignment/drift metrics and hard-gate semantics.

## PR-019 — Add complete model-quality diagnostics and metric definitions

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/019-model-diagnostics`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/evaluation/__init__.py`, `src/market_regime_engine/evaluation/diagnostics.py`, `tests/unit/evaluation/test_diagnostics.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Compute training log likelihood.
- [ ] Compute AIC using explicit free-parameter count.
- [ ] Compute BIC using explicit free-parameter count and observation count.
- [ ] For Gaussian HMMs with K states and d features, AIC/BIC parameter counting uses the full-covariance term `K*d*(d+1)/2` in addition to initial probabilities, transitions, and means.
- [ ] Compute hard occupancy per state and `min_hard_occupancy`.
- [ ] Compute soft/effective occupancy per state and `min_soft_occupancy`.
- [ ] Compute transition/self-transition probabilities and candidate summary min/max/mean self-transition.
- [ ] Compute empirical dominant-state switch count and frequency-normalized `switches_per_year` (or declared frequency equivalent).
- [ ] Compute per-state duration mean, median, p90, maximum, and run count.
- [ ] Compute candidate duration summaries `min_mean_state_duration`, `mean_state_duration`, `max_mean_state_duration`.
- [ ] Compute OOS entropy mean/median/p90 and confidence mean when supplied filtered OOS predictions.
- [ ] Compute `oos_low_confidence_fraction` using a profile-declared threshold.
- [ ] Detect empty/near-empty states with configurable thresholds.
- [ ] Detect non-finite parameters, non-symmetric/non-positive-definite full covariance matrices, and any non-`full` covariance mode.
- [ ] Metric result schema uses stable explicit names matching `EVALUATION.md`.
- [ ] Every formula/metric definition is documented in `EVALUATION.md` and unit-tested on deterministic full-covariance fixtures.

## PR-020 — Add expanding walk-forward split planner

- **Status:** BLOCKED by PR-007, PR-008, PR-009
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/020-walk-forward-splits`
- **Depends on:** PR-007, PR-008, PR-009
- **Allowed files:** `src/market_regime_engine/evaluation/walk_forward_splits.py`, `tests/unit/evaluation/test_walk_forward_splits.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Expanding windows with explicit minimum train observations/test size.
- [ ] Every test row is strictly after its training interval.
- [ ] No test row leaks into same fold's train data.
- [ ] Calendar gaps create no synthetic rows.
- [ ] Every split has deterministic one-based `fold_index`, stable `fold_id`, and explicit UTC `train_start`, `train_end`, `test_start`, and `test_end`.
- [ ] Split plan exposes a deterministic fold-timeline representation suitable for MLflow metric-history ordering and real-date plot axes.
- [ ] Split plan deterministic/serializable and has deterministic hash.
- [ ] Tests cover normal, short, gaps, boundaries, overlap rejection, partial final window, stable fold IDs, and monotonic fold indices/test-end dates.
- [ ] `EVALUATION.md` documents walk-forward semantics, fold timeline semantics, and leakage constraints.

## PR-021 — Add reusable Xetra cross-asset model profile and explicit selection policy

- **Status:** BLOCKED by PR-007, PR-008
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/021-xetra-cross-asset-profile`
- **Depends on:** PR-007, PR-008
- **Allowed files:** `configs/xetra_cross_asset_v1.yaml`, `docs/profiles/xetra_cross_asset_v1.md`, `tests/unit/profiles/test_xetra_profile.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Profile ID/version is `xetra_cross_asset_v1`.
- [ ] Features come only from reusable `regime-loader` Gold contract; no ETF-return/portfolio feature is embedded.
- [ ] Candidate grid includes exactly three MVP Gaussian candidates: K=2 full, K=3 full, and K=4 full.
- [ ] Stable candidate IDs are exactly `gaussian_hmm_k2_full`, `gaussian_hmm_k3_full`, and `gaussian_hmm_k4_full`.
- [ ] No diagonal, spherical, tied, or other reduced-covariance Gaussian candidate is present or accepted by profile validation.
- [ ] Multi-start, convergence, minimum valid-start rate, minimum occupancy, maximum drift, minimum valid-fold rate, and walk-forward settings are explicit.
- [ ] Inference mode is filtered.
- [ ] Primary ranking metric is explicitly `oos_predictive_loglik_mean` where fold scores are normalized per test observation.
- [ ] Secondary ranking is explicit fold stability: lower `oos_predictive_loglik_std`, then better `oos_predictive_loglik_worst_fold`.
- [ ] Tertiary tie breaks are lower `bic_mean`, then lower `aic_mean`, then fewer states K, then stable candidate ID.
- [ ] No weighted composite score is configured.
- [ ] Documentation explains Xetra is downstream application universe while profile models reusable cross-asset market state.
- [ ] Profile validation/hash test passes and explicitly rejects a diagonal-covariance candidate fixture.
- [ ] `EVALUATION.md` candidate table and selection order exactly match this profile.

---

# Wave 3 — Walk-forward evaluation, MLflow, champion selection

PR-022 is the convergence point. PR-023 and PR-027 may run in parallel after it. PR-024 -> PR-025 -> PR-026 complete the promotion path.

## PR-022 — Implement leak-free walk-forward evaluation runner with OOS predictive scoring

- **Status:** BLOCKED by PR-015, PR-016, PR-018, PR-019, PR-020
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/022-walk-forward-runner`
- **Depends on:** PR-015, PR-016, PR-018, PR-019, PR-020
- **Allowed files:** `src/market_regime_engine/evaluation/walk_forward.py`, `tests/unit/evaluation/test_walk_forward.py`, `tests/integration/test_walk_forward_runner.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Each fold fits preprocessing only on training rows.
- [ ] Each fold fits HMM only on training rows.
- [ ] Alignment uses only current/prior training information.
- [ ] Test rows use causal filtering only.
- [ ] Candidate predictive scoring uses the fitted training model without test refit.
- [ ] Each fold result preserves deterministic `fold_index`, `fold_id`, train/test UTC bounds, observation counts, and validity status from the split plan.
- [ ] Each fold records unnormalized OOS predictive log likelihood, test observation count, and normalized `oos_predictive_loglik_per_obs`.
- [ ] Fold result exposes every scalar diagnostic required for MLflow fold-history logging, including train/OOS likelihood, AIC/BIC, multistart success, occupancy, self-transition, duration, signature drift, entropy/confidence, and applicable candidate-summary fold values.
- [ ] `fold_metrics` output has exactly one row per planned fold, including invalid folds with explicit failure reason; unavailable values remain missing and are never imputed/interpolated.
- [ ] One OOS prediction row per eligible test timestamp with fold ID/lineage.
- [ ] Duplicate OOS timestamps fail unless an explicit non-overlap policy prevents them before execution.
- [ ] Fold diagnostics include fit, multistart, occupancy, transition, duration, entropy/confidence, alignment, date bounds, and quality status.
- [ ] Fold validity is explicit; invalid folds carry reasons and cannot silently contribute to candidate means.
- [ ] Mutating future rows cannot change earlier OOS predictions or earlier fold metrics.
- [ ] Integration test verifies normalized predictive log likelihood using deterministic synthetic full-covariance data.
- [ ] Integration test verifies fold indices and TEST-end dates are strictly ordered and identical to the declared split plan.
- [ ] `EVALUATION.md` matches runner outputs, fold timeline semantics, and exact metric names.

## PR-023 — Add MLflow experiment tracking, fold metric histories, diagnostic plots, and candidate scorecards

- **Status:** BLOCKED by PR-012, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/023-mlflow-experiment-tracking`
- **Depends on:** PR-012, PR-022
- **Allowed files:** `src/market_regime_engine/mlflow_support/tracking.py`, `src/market_regime_engine/mlflow_support/plots.py`, `tests/unit/mlflow_support/test_tracking.py`, `tests/unit/mlflow_support/test_plots.py`, `tests/integration/test_mlflow_file_tracking.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Adapter uses configured `MLFLOW_TRACKING_URI`; production points to `http://10.10.1.3:5000`.
- [ ] Experiment is created/reused from explicit profile/experiment configuration.
- [ ] Parent run records profile ID/version/hash, engine version, Git SHA, feature version, source build, evaluation-plan hash, split policy, inference mode, candidate count, and selection-policy version.
- [ ] Candidate runs log family, K, covariance=`full`, candidate ID, seed policy, convergence settings, feature/order hash, and all aggregate scorecard metrics.
- [ ] Candidate aggregate metrics include `oos_predictive_loglik_mean`, `oos_predictive_loglik_std`, `oos_predictive_loglik_median`, `oos_predictive_loglik_worst_fold`, `oos_predictive_loglik_best_fold`, `valid_fold_rate`, `train_loglik_mean`, `aic_mean`, `bic_mean`, `multistart_success_rate_mean`, `min_hard_occupancy`, `min_soft_occupancy`, `max_state_signature_drift`, `alignment_failure_count`, `mean_state_duration`, `switches_per_year`, `oos_entropy_mean`, and `oos_confidence_mean`.
- [ ] Fold runs log train/test bounds/counts, train likelihood, unnormalized and normalized OOS predictive likelihood, AIC/BIC, multistart metrics, occupancy by state, self-transition by state, duration by state, state drift, entropy/confidence, convergence/alignment/gate status.
- [ ] Every scalar fold metric required by `EVALUATION.md` is also logged on the corresponding candidate run as MLflow metric history with deterministic `step=fold_index`.
- [ ] Canonical history keys include at minimum `fold_train_loglik`, `fold_oos_predictive_loglik`, `fold_oos_predictive_loglik_per_obs`, `fold_aic`, `fold_bic`, `fold_multistart_success_rate`, `fold_min_hard_occupancy`, `fold_min_soft_occupancy`, `fold_max_state_signature_drift`, `fold_mean_state_duration`, `fold_switches_per_year`, `fold_oos_entropy_mean`, and `fold_oos_confidence_mean`.
- [ ] Per-state histories use persistent IDs and stable keys including `fold_hard_occupancy_state_<id>`, `fold_soft_occupancy_state_<id>`, `fold_self_transition_state_<id>`, `fold_mean_duration_state_<id>`, and `fold_state_signature_drift_state_<id>`.
- [ ] Where supported by MLflow, history points use fold `test_end` UTC as their explicit metric timestamp; `fold_timeline.parquet` remains the authoritative `fold_index`/`fold_id` -> train/test-bound mapping.
- [ ] Invalid folds remain visible in timeline/metrics artifacts; missing metric values are not invented or logged as interpolated points.
- [ ] Candidate-run deterministic PNG trend artifacts use actual fold TEST-end dates on the x-axis and include exactly: `plots/oos_predictive_loglik_per_obs_by_fold.png`, `plots/train_loglik_by_fold.png`, `plots/aic_by_fold.png`, `plots/bic_by_fold.png`, `plots/multistart_success_rate_by_fold.png`, `plots/hard_occupancy_by_fold.png`, `plots/soft_occupancy_by_fold.png`, `plots/self_transition_by_fold.png`, `plots/state_signature_drift_by_fold.png`, `plots/state_duration_by_fold.png`, `plots/switches_per_year_by_fold.png`, `plots/oos_entropy_by_fold.png`, and `plots/oos_confidence_by_fold.png`.
- [ ] Invalid/missing folds appear as gaps or explicit invalid markers in trend plots; no line is silently interpolated across unavailable data.
- [ ] Metrics with incompatible numerical scales are plotted separately; hidden secondary axes are forbidden.
- [ ] Every valid fold stores machine-readable transition/full-covariance artifacts plus `plots/folds/<fold_id>/transition_matrix.png` and `plots/folds/<fold_id>/covariance_state_<id>.png` heatmaps for every persistent state.
- [ ] Covariance heatmaps retain exact feature order and all off-diagonal terms.
- [ ] `plots/manifest.json` deterministically records every generated plot path, source metric keys, candidate/fold IDs, x-axis field, and source artifact hash.
- [ ] Required artifacts include `evaluation_plan.json`, `fold_timeline.parquet`, `fold_metrics.parquet`, `candidate_scorecard.json`, `multistart_metrics.parquet`, transition matrix, full covariance matrices/model spec, state signatures, state alignment, occupancy-by-fold, duration-by-fold, OOS prediction reference, feature/preprocessing metadata, and plot manifest/plots.
- [ ] Parent run receives `candidate_comparison.parquet` and later champion-selection artifact.
- [ ] Metric/tag/history names and artifact paths are stable and documented in `EVALUATION.md`.
- [ ] OOS prediction artifact reference is logged; large data is not silently embedded as ad-hoc untracked output.
- [ ] Unit tests verify deterministic history-key generation, step ordering, persistent-state naming, plot paths, no interpolation, and plot-manifest content.
- [ ] Required integration with local file-backed MLflow verifies history point count/steps equal valid fold metrics, explicit fold timeline round-trip, and all required candidate plot/heatmap artifacts exist.
- [ ] Unit tests use fake port; required integration uses local file-backed MLflow and no external service.
- [ ] No required test attempts the shared NAS endpoint.

## PR-024 — Add candidate-grid orchestration, aggregate fold statistics, and parent comparison plots

- **Status:** BLOCKED by PR-007, PR-015, PR-022, PR-023
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/024-candidate-grid-orchestrator`
- **Depends on:** PR-007, PR-015, PR-022, PR-023
- **Allowed files:** `src/market_regime_engine/training/candidate_grid.py`, `tests/unit/training/test_candidate_grid.py`, `tests/integration/test_candidate_grid.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Expands only validated profile candidates.
- [ ] Deterministic candidate IDs.
- [ ] All candidates use the same declared walk-forward plan and source build.
- [ ] Candidate expansion rejects any covariance-bearing HMM whose covariance mode is not `full`.
- [ ] One candidate failure does not corrupt completed candidates.
- [ ] Candidate aggregation excludes invalid folds from means but records invalid count/rate and reasons.
- [ ] Aggregates include OOS predictive mean/std/median/min/max, valid-fold count/rate, train likelihood mean/std, AIC/BIC mean/std, multistart success aggregates, occupancy minima, alignment drift mean/max, duration summaries, switch frequency, entropy/confidence summaries.
- [ ] Output includes complete aggregate diagnostics plus OOS reference and candidate scorecard.
- [ ] Candidate comparison table has one row per candidate and all fields required by selection policy.
- [ ] Candidate fold histories are aligned by the shared `fold_index`/TEST-end timeline before cross-candidate visualization; no candidate may silently shift fold dates.
- [ ] Parent run stores at minimum `plots/candidates/oos_predictive_loglik_per_obs_by_fold.png`, `plots/candidates/multistart_success_rate_by_fold.png`, `plots/candidates/min_hard_occupancy_by_fold.png`, `plots/candidates/max_state_signature_drift_by_fold.png`, `plots/candidates/oos_entropy_by_fold.png`, and `plots/candidates/oos_confidence_by_fold.png`.
- [ ] Parent comparison plots use one line per stable candidate ID and actual TEST-end dates; invalid candidate folds appear as gaps/invalid markers rather than interpolation.
- [ ] Parent plot entries are included in the deterministic `plots/manifest.json` and link back to the same fold metrics used for selection.
- [ ] Deterministic for fixed input/profile/seeds.
- [ ] Integration covers at least Gaussian K=2 full and K=3 full end-to-end, verifies aligned cross-candidate plot data, and verifies required parent plot artifacts.
- [ ] `EVALUATION.md` matches aggregation and cross-candidate visualization definitions.

## PR-025 — Add hard validation gates and deterministic engine-champion selection

- **Status:** BLOCKED by PR-019, PR-024
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/025-engine-champion-selection`
- **Depends on:** PR-019, PR-024
- **Allowed files:** `src/market_regime_engine/evaluation/selection.py`, `tests/unit/evaluation/test_selection.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] Selection uses explicit profile policy; no hidden/weighted composite score.
- [ ] Hard gates include minimum stable multistart success, finite/valid parameters, valid full covariance matrices, minimum hard/soft occupancy, successful alignment, maximum state-signature drift, and minimum valid-fold rate.
- [ ] Candidate failing any hard gate cannot win regardless of likelihood.
- [ ] Primary ranking for Xetra profile is highest `oos_predictive_loglik_mean` based on per-observation fold scores.
- [ ] Secondary ranking is lower `oos_predictive_loglik_std`, then higher `oos_predictive_loglik_worst_fold`.
- [ ] Tertiary/tie-break ranking is lower `bic_mean`, then lower `aic_mean`, then fewer states K, then stable candidate ID; covariance mode never acts as a tie-break because it is fixed to `full`.
- [ ] Training likelihood alone can never promote a candidate.
- [ ] Selection output records rank, all hard-gate pass/fail results, rejected candidates, rejection reasons, and complete deterministic comparison chain.
- [ ] Tests cover zero-valid candidates, each hard-gate failure, invalid/reduced covariance rejection, primary-metric winner, stability tie-break, BIC/AIC tie-break, K-complexity tie-break, and deterministic final tie.
- [ ] `EVALUATION.md` exactly matches implemented selection order and gate semantics.

## PR-026 — Package fitted model in shared MLflow Model Registry and manage aliases

- **Status:** BLOCKED by PR-012, PR-016, PR-018, PR-025
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/026-mlflow-model-registry`
- **Depends on:** PR-012, PR-016, PR-018, PR-025
- **Allowed files:** `src/market_regime_engine/mlflow_support/model_package.py`, `src/market_regime_engine/mlflow_support/registry.py`, `tests/unit/mlflow_support/test_model_package.py`, `tests/unit/mlflow_support/test_registry.py`, `tests/integration/test_mlflow_registry_local.py`, `EVALUATION.md`

### Acceptance criteria

- [ ] MLflow package contains preprocessing, fitted HMM, complete per-state full covariance matrices, feature order, persistent state mapping/signature, profile hash, lineage, and inference-contract version.
- [ ] Model package records covariance mode exactly `full`; a package containing a reduced covariance mode fails validation and cannot be registered/promoted.
- [ ] Model version links back to evaluation parent/candidate run and champion-selection evidence.
- [ ] Artifact round-trip yields identical filtered prediction and identical full covariance matrices on deterministic fixture.
- [ ] Registry uses explicit registered model versions.
- [ ] Registry can set/move `engine-champion`, `challenger`, and arbitrary consumer aliases such as `portfell-production`.
- [ ] Alias movement is logged with source/destination version and reason.
- [ ] `engine-champion` can only come from validated deterministic selection result.
- [ ] Production registry operations use configured shared MLflow endpoint `http://10.10.1.3:5000`.
- [ ] Required integration tests remain local-file/no-network; shared-server behavior is verified by PR-034.
- [ ] `EVALUATION.md` documents alias semantics relevant to evaluation.

## PR-027 — Publish immutable walk-forward OOS prediction builds

- **Status:** BLOCKED by PR-011, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/027-oos-prediction-publication`
- **Depends on:** PR-011, PR-022
- **Allowed files:** `src/market_regime_engine/predictions/oos_publication.py`, `tests/unit/predictions/test_oos_publication.py`, `tests/integration/test_oos_prediction_publication.py`

### Acceptance criteria

- [ ] Mode exactly `walk_forward_oos`.
- [ ] Manifest includes profile, model/candidate, fold-plan hash, source build, feature version, Git SHA, and row/date counts.
- [ ] Rows contain filtered probabilities, persistent IDs, dominant state, entropy/confidence, fold ID, lineage.
- [ ] Same content is deterministic/idempotent; conflicting immutable build fails.
- [ ] Reloaded rows validate against `RegimePredictionV1`.

---

# Wave 4 — Batch/realtime services and API

After PR-026, PR-028/029 may run in parallel. PR-030 depends on PR-027 and API skeleton. PR-031 follows API use cases.

## PR-028 — Add fixed-model batch inference API

- **Status:** BLOCKED by PR-008, PR-013, PR-016, PR-018, PR-026
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/028-batch-inference-api`
- **Depends on:** PR-008, PR-013, PR-016, PR-018, PR-026
- **Allowed files:** `src/market_regime_engine/inference/batch.py`, `src/market_regime_engine/api/routes/batch.py`, `tests/unit/inference/test_batch.py`, `tests/unit/api/test_batch_route.py`, `tests/integration/test_batch_api.py`

### Acceptance criteria

- [ ] Loads explicit model version or registry alias from configured MLflow registry.
- [ ] Loaded covariance-bearing HMM model validates covariance mode exactly `full` before inference.
- [ ] Input is explicit feature-source/build reference and contract is validated.
- [ ] Output uses filtered inference and persistent state mapping.
- [ ] Metadata mode is `fixed_model_replay`.
- [ ] Never labels replay as walk-forward OOS.
- [ ] Date bounds deterministic/validated.
- [ ] Large results return immutable prediction-build reference rather than unbounded JSON.
- [ ] Local integration covers registered full-covariance model + Parquet input without network.

## PR-029 — Add realtime/latest inference API

- **Status:** BLOCKED by PR-008, PR-013, PR-016, PR-018, PR-026
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/029-realtime-latest-api`
- **Depends on:** PR-008, PR-013, PR-016, PR-018, PR-026
- **Allowed files:** `src/market_regime_engine/inference/realtime.py`, `src/market_regime_engine/api/routes/latest.py`, `tests/unit/inference/test_realtime.py`, `tests/unit/api/test_latest_route.py`, `tests/integration/test_latest_api.py`

### Acceptance criteria

- [ ] Resolves explicit profile + model alias/version from configured MLflow registry.
- [ ] Production default resolves registry through `http://10.10.1.3:5000`.
- [ ] Loaded covariance-bearing HMM model validates covariance mode exactly `full` before inference.
- [ ] Loads only feature data available up to as-of.
- [ ] Feature contract matches registered model exactly.
- [ ] Response validates `RegimePredictionV1` and includes resolved model version/lineage.
- [ ] MLflow unavailable, missing data, incompatible features, invalid covariance contract, or quality failure returns explicit non-200; no stale/invented prediction.
- [ ] Local integration proves deterministic latest prediction without external network.

## PR-030 — Add walk-forward OOS prediction retrieval API

- **Status:** BLOCKED by PR-013, PR-027
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/030-oos-prediction-api`
- **Depends on:** PR-013, PR-027
- **Allowed files:** `src/market_regime_engine/api/routes/evaluations.py`, `src/market_regime_engine/predictions/query.py`, `tests/unit/api/test_evaluations_route.py`, `tests/unit/predictions/test_query.py`, `tests/integration/test_oos_prediction_api.py`

### Acceptance criteria

- [ ] Retrieve metadata for explicit OOS build.
- [ ] Retrieve bounded date slices by explicit build ID.
- [ ] Cannot substitute fixed replay for OOS request.
- [ ] Response identifies profile/model/source/evaluation/engine version.
- [ ] Portfell can fetch leak-free probabilities without engine imports.
- [ ] Integration covers date filtering and mode mismatch.

## PR-031 — Add application CLI for train, evaluate, register, and serve

- **Status:** BLOCKED by PR-024, PR-026, PR-028, PR-029, PR-030
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/031-application-cli`
- **Depends on:** PR-024, PR-026, PR-028, PR-029, PR-030
- **Allowed files:** `src/market_regime_engine/cli.py`, `src/market_regime_engine/commands/*`, `tests/unit/test_cli.py`, `pyproject.toml`

### Acceptance criteria

- [ ] Console entry point `market-regime-engine`.
- [ ] CLI is thin adapter; no model math.
- [ ] `train` fits explicit profile/input.
- [ ] `evaluate` runs walk-forward candidate grid, logs MLflow aggregate scorecards, candidate fold metric histories, required fold/candidate diagnostic plots, and publishes OOS predictions.
- [ ] `evaluate` reports the parent evaluation run ID so the operator can open the candidate histories and plot artifacts directly in MLflow.
- [ ] `register` registers validated model and optional explicit alias.
- [ ] Production `register` uses `MLFLOW_TRACKING_URI`, documented value `http://10.10.1.3:5000`.
- [ ] `serve` starts FastAPI with environment host/port.
- [ ] Every command has help, deterministic exit codes, actionable errors.
- [ ] Unit tests mock services and use no external MLflow.

---

# Wave 5 — Deployment, shared MLflow verification, compatibility, end-to-end proof

PR-032 through PR-035 run in parallel after their prerequisites. PR-036 consolidates final operator/consumer/evaluation documentation.

## PR-032 — Add Docker image and NAS-friendly Compose configuration

- **Status:** BLOCKED by PR-029, PR-031
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/032-container-deployment`
- **Depends on:** PR-029, PR-031
- **Allowed files:** `Dockerfile`, `.dockerignore`, `compose.example.yaml`, `docs/deployment.md`, `tests/integration/test_container_config.py`

### Acceptance criteria

- [ ] Docker API runs non-root on Python-3.14-compatible base.
- [ ] Image excludes `.venv`, Git metadata, test cache, local MLflow state, and secrets.
- [ ] Compose exposes feature root, prediction root, API bind/port, model alias, and `MLFLOW_TRACKING_URI`.
- [ ] Compose default is exactly `MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://10.10.1.3:5000}`.
- [ ] Deployment doc identifies `http://10.10.1.3:5000` as existing shared NAS MLflow Tracking Server / Model Registry.
- [ ] Deployment doc states MLflow/PostgreSQL lifecycle is not managed by this compose stack.
- [ ] Healthcheck uses `/health/ready`.
- [ ] No credentials/tokens are embedded.
- [ ] Integration test asserts compose default and environment override behavior.

## PR-033 — Add regime-loader Gold compatibility integration test

- **Status:** BLOCKED by PR-008, PR-021
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/033-loader-contract-integration`
- **Depends on:** PR-008, PR-021
- **Allowed files:** `tests/fixtures/loader_gold/*`, `tests/integration/test_loader_gold_contract.py`, `docs/integrations/regime_loader.md`

### Acceptance criteria

- [ ] Fixture mirrors documented immutable Gold Parquet/manifest contract without importing loader code.
- [ ] Xetra profile resolves all required features.
- [ ] Build ID and feature version preserved.
- [ ] Missing/incompatible/duplicate/non-finite input fails closed.
- [ ] Cross-repo handoff documented; package dependency on loader forbidden.

## PR-034 — Verify the real shared MLflow Tracking Server / Model Registry

- **Status:** BLOCKED by PR-023, PR-026
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/034-external-mlflow-smoke`
- **Depends on:** PR-023, PR-026
- **Allowed files:** `tests/external/test_mlflow_external.py`, `docs/integrations/mlflow.md`, `scripts/verify_shared_mlflow.py`

### Acceptance criteria

- [ ] External test is marked `external_service` and excluded from required push/merge gates.
- [ ] Documented target is exactly `http://10.10.1.3:5000`.
- [ ] Manual command sets `MLFLOW_TRACKING_URI=http://10.10.1.3:5000` and explicitly opts into external testing.
- [ ] Verification fails if configured production URI differs unless explicit migration/override flag is supplied.
- [ ] Smoke test verifies Tracking Server reachability and server metadata before writes.
- [ ] Smoke test creates uniquely named disposable experiment/run, logs parameters/metrics/artifact, reads them back, and verifies registry create/read/version/alias behavior.
- [ ] Smoke test writes at least three ordered points for one representative `fold_*` metric using explicit steps, reads metric history back, and verifies step/value ordering.
- [ ] Smoke test uploads and reads back a representative deterministic PNG trend artifact plus `plots/manifest.json` under nested artifact paths.
- [ ] Smoke test verifies representative evaluation metric names, covariance=`full`, candidate scorecard artifact, fold timeline artifact, and plot artifact paths can round-trip through the real server.
- [ ] Disposable resources are uniquely namespaced; cleanup is limited strictly to resources created by the smoke test.
- [ ] No credential/token is committed.

## PR-035 — Add complete hermetic engine end-to-end integration proof

- **Status:** BLOCKED by PR-021, PR-024, PR-026, PR-027, PR-028, PR-029, PR-030
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/035-engine-e2e-proof`
- **Depends on:** PR-021, PR-024, PR-026, PR-027, PR-028, PR-029, PR-030
- **Allowed files:** `tests/integration/test_engine_e2e.py`, `tests/fixtures/e2e/*`, `EVALUATION.md`

### Acceptance criteria

- [ ] Uses deterministic fixture Parquet and local-file MLflow only.
- [ ] Candidate grid includes exactly Gaussian K=2 full, K=3 full, and K=4 full.
- [ ] E2E setup asserts that a diagonal/reduced-covariance candidate is rejected before fitting.
- [ ] Runs leak-free walk-forward OOS evaluation.
- [ ] Produces fold-level normalized OOS predictive log likelihood for every valid fold.
- [ ] For every required scalar fold-history metric, each candidate MLflow run contains exactly the expected valid-fold history points with deterministic steps matching `fold_index` and values matching `fold_metrics.parquet`.
- [ ] `fold_timeline.parquet` contains every planned fold, including invalid folds, with stable IDs and exact train/test bounds.
- [ ] Missing/invalid fold metrics are not interpolated; plot source data and manifest preserve the gap/invalid status.
- [ ] Every candidate run contains all required trend PNGs and `plots/manifest.json`; manifest metric keys and x-axis field match `EVALUATION.md`.
- [ ] Every valid fold contains transition-matrix heatmap and full-covariance heatmap for each persistent state, with exact feature order preserved.
- [ ] Parent evaluation run contains all required cross-candidate fold-trend plots and each plot covers the aligned K=2/K=3/K=4 candidate timeline.
- [ ] Produces candidate scorecards and candidate comparison table.
- [ ] Applies all hard gates and deterministic ranking exactly as documented in `EVALUATION.md`.
- [ ] Selects engine champion and records complete reason/ranking chain.
- [ ] Registers winner locally and assigns `engine-champion`.
- [ ] Registered winner round-trips complete per-state full covariance matrices including off-diagonal terms.
- [ ] Publishes immutable OOS predictions.
- [ ] Exercises fixed-model batch API and confirms `fixed_model_replay`.
- [ ] Exercises latest API and validates `RegimePredictionV1`.
- [ ] Exercises OOS retrieval.
- [ ] Same seeds/input reproduce deterministic values/lineage/history steps/plot manifest where required.
- [ ] Test never requires `10.10.1.3`; real shared MLflow is exclusively PR-034 external verification.
- [ ] Test asserts `EVALUATION.md`-documented required scorecard fields, fold history keys, and plot paths are present.

## PR-036 — Final operator, consumer, and evaluation documentation consistency

- **Status:** BLOCKED by PR-032, PR-033, PR-034, PR-035
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/036-operator-consumer-docs`
- **Depends on:** PR-032, PR-033, PR-034, PR-035
- **Allowed files:** `README.md`, `API.md`, `OPERATIONS.md`, `EVALUATION.md`, `docs/consumer_contract.md`, `docs/integrations/portfell.md`, `docs/integrations/mlflow.md`

### Acceptance criteria

- [ ] README shows loader -> engine -> MLflow/API -> consumer architecture.
- [ ] README gives clean-checkout `.venv` bootstrap for Python 3.14.7.
- [ ] Shared production MLflow endpoint is documented exactly as `http://10.10.1.3:5000`.
- [ ] Operations doc gives exact `MLFLOW_TRACKING_URI=http://10.10.1.3:5000` setup and train -> evaluate -> select -> register -> alias -> serve lifecycle.
- [ ] Operations/MLflow docs explain exactly where to find the parent run, candidate runs, native `fold_*` metric histories, auditable fold runs, `fold_timeline.parquet`, and plot artifacts in the MLflow UI.
- [ ] Documentation lists the required candidate trend plot paths, parent cross-candidate plot paths, transition heatmaps, and per-state full-covariance heatmaps.
- [ ] Documentation explains that MLflow history `step` is the deterministic fold index while the human-facing plot x-axis is the actual TEST-end date.
- [ ] Documentation explains invalid-fold gaps and explicitly forbids interpreting an interpolated line as measured evaluation evidence.
- [ ] API doc covers health, latest, fixed replay batch, and OOS retrieval.
- [ ] API doc warns `fixed_model_replay` is not leak-free OOS evidence.
- [ ] Consumer contract documents `RegimePrediction.v1` and immutable OOS fields.
- [ ] Portfell doc states Portfell owns ETF universe, regime-conditioned returns/covariances, portfolio optimization/backtesting/costs, and application-level model choice.
- [ ] `EVALUATION.md` lists exactly the full-covariance candidate models, walk-forward process, all implemented metrics, fold history keys, required diagnostic plots, MLflow hierarchy/artifacts, hard gates, ranking/tie-break order, and consumer-vs-engine boundary.
- [ ] Documentation explicitly states that reduced covariance modes are unsupported throughout the engine.
- [ ] Documentation cross-links `EVALUATION.md` from README/OPERATIONS/model-lifecycle docs where relevant.
- [ ] No documented metric, model candidate, covariance mode, history key, plot path, or selection rule disagrees with implementation/profile configuration.
- [ ] Future consumers can reuse registered models/API without HMM implementation imports or engine fork.

---

# Wave 6 — Optional model challengers after MVP

These adapters are isolated behind the model protocol and are not required for the first Gaussian-HMM platform. Because they change the candidate universe, each challenger PR must update `EVALUATION.md` in the same PR. The full-covariance-only architecture rule remains mandatory for all covariance-bearing challengers.

## PR-037 — Add Student-t HMM challenger adapter

- **Status:** OPTIONAL / BLOCKED by PR-010, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/037-student-t-hmm-challenger`
- **Depends on:** PR-010, PR-022
- **Allowed files:** `src/market_regime_engine/models/student_t_hmm.py`, `tests/unit/models/test_student_t_hmm.py`, `docs/models/student_t_hmm.md`, `pyproject.toml`, `EVALUATION.md`

### Acceptance criteria

- [ ] Satisfies common model/artifact/predictive-score protocol.
- [ ] Additional dependency supports Python 3.14.
- [ ] Degrees-of-freedom configuration explicit; covariance mode is fixed to `full` with a separate full covariance matrix per state.
- [ ] `diag`, `spherical`, `tied`, and any non-`full` covariance configuration fail closed.
- [ ] Filtered inference causal and alignment-compatible.
- [ ] Participates in existing walk-forward/grid/selection with no consumer special case.
- [ ] Emits the same required evaluation metrics/scorecard fields where semantically applicable.
- [ ] Deterministic heavy-tail multivariate synthetic test proves full-covariance fit/inference/evaluation and off-diagonal preservation.
- [ ] `EVALUATION.md` candidate table and any model-specific metric caveats are updated.

## PR-038 — Add duration-aware HSMM challenger adapter

- **Status:** OPTIONAL / BLOCKED by PR-010, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/038-hsmm-challenger`
- **Depends on:** PR-010, PR-022
- **Allowed files:** `src/market_regime_engine/models/hsmm.py`, `tests/unit/models/test_hsmm.py`, `docs/models/hsmm.md`, `pyproject.toml`, `EVALUATION.md`

### Acceptance criteria

- [ ] Satisfies common model protocol or documents minimal explicit extension for durations.
- [ ] New dependency supports Python 3.14.
- [ ] Duration parameters explicit, validated, serialized, versioned.
- [ ] Any multivariate covariance-bearing emission model is fixed to a separate full covariance matrix per hidden state; reduced covariance modes fail closed.
- [ ] Inference causal for OOS/production.
- [ ] Participates in same walk-forward/grid/selection interface with no consumer change.
- [ ] Emits common scorecard metrics plus explicit duration-model diagnostics.
- [ ] Persistent-regime synthetic test proves duration metadata, full covariance matrices, and prediction artifact round-trip.
- [ ] `EVALUATION.md` candidate table and HSMM-specific evaluation semantics are updated.

---

# Parallel execution plan

The orchestrator starts only branches whose prerequisites are merged to `main`.

```text
Wave 0A:
  PR-001

Wave 0B parallel:
  PR-002  PR-003

Wave 0C:
  PR-004, then execute and verify governance

Wave 1A parallel:
  PR-005  PR-006

Wave 1B parallel:
  PR-007  PR-008  PR-009  PR-010  PR-011  PR-012  PR-013

Wave 2A where dependencies allow:
  PR-014  PR-020  PR-021

Wave 2B parallel after PR-014:
  PR-015  PR-016  PR-018  PR-019

Wave 2C:
  PR-017

Wave 3A:
  PR-022

Wave 3B parallel:
  PR-023  PR-027

Wave 3C:
  PR-024

Wave 3D:
  PR-025

Wave 3E:
  PR-026

Wave 4A parallel:
  PR-028  PR-029  PR-030

Wave 4B:
  PR-031

Wave 5A parallel:
  PR-032  PR-033  PR-034  PR-035

Wave 5B:
  PR-036

Optional:
  PR-037  PR-038 in parallel once common protocol/evaluation are stable
```

## Weak-agent execution rules

1. An agent receives exactly one PR section.
2. It must not broaden scope or refactor unrelated code.
3. It may edit only listed allowed files.
4. If a prerequisite file is absent, it stops rather than recreating another PR's work.
5. Every acceptance checkbox is mandatory.
6. Tests ship in the same PR as behavior.
7. Dependency versions change only where `pyproject.toml` is explicitly allowed.
8. Agents never edit `BACKLOG.md`.
9. Every PR targets `main`; after governance is active, auto-merge is enabled immediately.
10. Normal implementation PRs are not manually merged; GitHub auto-merge completes the squash merge only after `merge-gate` succeeds.
11. Required tests never depend on the shared MLflow server; only PR-034's explicitly invoked `external_service` test may touch `http://10.10.1.3:5000`.
12. An agent must never delete or mutate MLflow experiments/models that it did not create itself.
13. Any PR changing evaluation behavior listed in architecture rule 13 must include `EVALUATION.md` in its allowed files and update it in the same PR.
14. Agents must treat metric names as public observability contracts once introduced; renames require updating implementation, tests, MLflow logging, and `EVALUATION.md` atomically.
15. Agents must never introduce a diagonal, spherical, tied, or other reduced covariance mode for a covariance-bearing HMM; only per-state full covariance is allowed.
16. Agents must treat `fold_*` MLflow history keys, deterministic fold IDs/indices, required diagnostic plot paths, and `plots/manifest.json` entries as public observability contracts; they may not rename, omit, reorder, interpolate, or silently substitute them.

## Definition of complete MVP

The MVP is complete after PR-036 when:

- a clean checkout reproducibly creates a Python 3.14.7 `.venv`;
- `main` is protected and direct/force pushes are blocked;
- push/merge gates run lint/type/unit/integration in parallel;
- PRs auto-complete only after `merge-gate`;
- versioned model profiles can be loaded;
- Parquet features are consumed without upstream loader imports;
- Gaussian HMM K=2 full, K=3 full, and K=4 full candidates can be compared, with all reduced covariance modes rejected;
- every fitted Gaussian state retains a complete full covariance matrix including off-diagonal cross-feature covariance terms;
- multi-start fitting, persistent alignment, causal filtering, and walk-forward OOS evaluation are implemented;
- each planned fold has stable ID/index and exact train/test UTC bounds;
- each valid fold produces normalized OOS predictive log likelihood per observation;
- candidate scorecards expose generalisation, fit, stability, state-quality, persistence, and uncertainty metrics;
- every required scalar fold metric is visible in MLflow as ordered candidate-run `fold_*` metric history with values traceable back to `fold_metrics.parquet`;
- each candidate run contains deterministic TEST-end-date trend plots for OOS likelihood, fit/complexity, multistart stability, occupancy, persistence, state drift, and uncertainty;
- each valid fold contains transition and per-state full-covariance heatmaps backed by machine-readable matrix artifacts;
- parent evaluation run contains aligned cross-candidate fold-trend plots for the required comparison metrics;
- invalid/missing fold metrics are shown as gaps/invalid evidence and are never silently interpolated;
- `fold_timeline.parquet` and `plots/manifest.json` make all fold ordering and plot sources auditable;
- AIC/BIC complexity accounting uses the correct full-covariance free-parameter count;
- hard gates prevent degenerate/unstable models or invalid covariance matrices from promotion;
- `xetra_cross_asset_v1` ranks valid candidates primarily by mean OOS predictive log likelihood per observation, then fold stability, then BIC/AIC/fewer-states deterministic tie breaks;
- MLflow tracks parent, candidate, fold evidence, metric histories, scorecard/comparison artifacts, and diagnostic plots;
- MLflow packages promoted models including complete full covariance matrices;
- production configuration points to the shared MLflow Tracking Server / Model Registry at `http://10.10.1.3:5000`;
- the real shared MLflow instance has an explicit successful opt-in smoke-test path;
- `engine-champion`, `challenger`, and consumer-specific aliases are supported;
- immutable `walk_forward_oos` prediction builds exist;
- fixed-model batch inference is clearly separated from historical OOS predictions;
- latest/realtime inference returns `RegimePrediction.v1`;
- `EVALUATION.md` is complete, cross-linked, and consistent with implementation/configuration;
- Portfell and future consumers use the API/registered models without HMM implementation imports;
- the complete hermetic local end-to-end proof passes the required merge gate.

---

# Wave 7 — Statistical feature selection and frozen HMM input

This section is authoritative for PR-045 through PR-050 and for its explicit feature-selection addenda to PR-021, PR-022, PR-024, PR-035, and PR-036. Where these feature-selection requirements conflict with older wording in earlier PR sections, this section wins. All feature-selection work remains subject to `DATA_SOURCE.md`, `EVALUATION.md`, `PLOT_STYLE.md`, and `CONTRIBUTING.md` in their existing areas of authority.

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
