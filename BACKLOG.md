# Market Regime Engine — Implementation Backlog

Status date: 2026-08-22

This backlog defines the complete implementation plan for `market-regime-engine` as a reusable market-regime model platform.

The repository owns model training, causal HMM inference, walk-forward validation, MLflow tracking/registry integration, immutable prediction artifacts, and batch/realtime inference APIs. It does **not** own market-data acquisition or portfolio optimization.

```text
market-regime-loader
    -> immutable causal feature data
    -> market-regime-engine
    -> MLflow Model Registry + RegimePrediction.v1 + OOS prediction artifacts + API
    -> portfell and future consumers
```

`market-regime-loader` remains the reusable data product. `portfell` remains the Xetra ETF portfolio application. `market-regime-engine` must stay consumer-agnostic so registered HMM models can later be reused by BTC, equity, rates, covered-call, or other projects.

## Repository bootstrap facts

- Default branch: `main`.
- Stable Python feature release selected for this repository: **Python 3.14.7**.
- Local development environment: repository-local `.venv`; `.venv/` is never committed.
- Python compatibility target for MVP: `>=3.14,<3.15`.
- Shared production MLflow Tracking Server / Model Registry: **`http://10.10.1.3:5000`**.
- Production/NAS examples use `MLFLOW_TRACKING_URI=http://10.10.1.3:5000`.
- Library/domain code still obtains the URI through configuration/environment so unit and integration tests can use isolated local stores.
- The shared MLflow infrastructure itself, including its PostgreSQL backend and deployment lifecycle, is external to this repository.
- All implementation PRs start from a clean, up-to-date `main` after declared dependencies are merged.
- Agents must not edit `BACKLOG.md`; backlog status is maintained only by the orchestrator.

## Non-negotiable architecture rules

1. No provider HTTP clients, CBOE/FRED/ECB/STOXX acquisition logic, or EODHD portfolio-data acquisition in this repository.
2. No portfolio optimizer, ETF weighting, Sharpe/Sortino application selection, transaction-cost model, or trading logic in this repository.
3. Model training may use only observations available inside the declared training window.
4. Backtest-safe inference uses filtered probabilities only. Retrospective smoothed probabilities are diagnostic-only.
5. All preprocessing parameters are fit on training data only.
6. Raw HMM state numbers are not stable consumer semantics; persistent state identity requires state alignment.
7. Every registered model carries exact feature order, preprocessing state, model parameters, state mapping, source/build lineage, package version, and Git commit.
8. Historical `walk_forward_oos` predictions and `fixed_model_replay` are different products and must never be conflated.
9. The shared production MLflow endpoint is `http://10.10.1.3:5000`; deployment/configuration examples must use it. Application source code reads the URI from `MLFLOW_TRACKING_URI` so hermetic tests remain local and future infrastructure migration does not require model-code changes.
10. MLflow credentials/tokens, if ever needed, are never committed.
11. Public consumer contracts are versioned. Initial prediction contract is `RegimePrediction.v1`.
12. MLflow Model Registry is the authoritative repository for promoted trained models; Git is authoritative for source/configuration, not fitted binary model artifacts.

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
- [ ] Runtime dependencies include Pydantic, NumPy, SciPy, scikit-learn, Polars, PyArrow, FastAPI, Uvicorn, HTTPX, MLflow, and a Python-3.14-compatible Gaussian-HMM implementation.
- [ ] Dev dependencies include pytest, pytest-cov, pytest-xdist, Ruff, mypy, and build tooling.
- [ ] Ruff targets Python 3.14; mypy is strict.
- [ ] pytest markers include `unit`, `integration`, and `external_service`.
- [ ] `.gitignore` excludes `.venv/`, caches, coverage/build output, local MLflow state, local prediction artifacts, and IDE files.
- [ ] Shell and PowerShell bootstrap scripts create `.venv` with Python 3.14, upgrade packaging tools, and install editable dev dependencies.
- [ ] Bootstrap scripts fail on the wrong interpreter and print the activation command without activating implicitly.
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

## PR-005 — Document durable architecture and MLflow ownership boundary

- **Status:** BLOCKED by PR-001
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/005-architecture-contract`
- **Depends on:** PR-001
- **Allowed files:** `ARCHITECTURE.md`, `docs/model_lifecycle.md`, `README.md`

### Acceptance criteria

- [ ] Architecture defines the engine as reusable model platform, not data loader or portfolio optimizer.
- [ ] Documents ports/adapters around feature input, model adapters, MLflow, prediction persistence, and API.
- [ ] Documents shared production MLflow Tracking Server / Registry as `http://10.10.1.3:5000`.
- [ ] States that MLflow/PostgreSQL deployment is external infrastructure and not owned by this repo.
- [ ] States that source code resolves MLflow through `MLFLOW_TRACKING_URI`; production/NAS value is the shared endpoint above.
- [ ] Separates `fixed_model_replay` from `walk_forward_oos`.
- [ ] Requires filtered probabilities for causal inference; smoothing is diagnostic-only.
- [ ] Requires persistent state alignment for consumer-facing predictions.
- [ ] Defines `market-regime-loader -> engine -> consumers` dependency direction without package coupling.
- [ ] Model lifecycle defines `candidate`, `validated`, `engine-champion`, `challenger`, and consumer-specific aliases such as `portfell-production`.
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
- [ ] Lineage includes engine version, Git SHA, training interval, source build, preprocessing version, and profile hash.
- [ ] `RegimePredictionV1` includes as-of, profile, model/version, persistent state IDs, probabilities, dominant state, entropy, confidence, lineage, and data-quality status.
- [ ] Validation rejects invalid/non-finite probabilities, duplicates, and non-normalized vectors.
- [ ] Raw library state labels cannot become consumer semantics without alignment ID.
- [ ] Serialization round-trip tests cover all public contracts.
- [ ] Contract layer has no model-library, MLflow, filesystem, FastAPI, or provider dependency.

## PR-007 — Add model-profile configuration schema and loader

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/007-model-profile-config`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/profiles/__init__.py`, `src/market_regime_engine/profiles/schema.py`, `src/market_regime_engine/profiles/loader.py`, `tests/unit/profiles/*`

### Acceptance criteria

- [ ] Schema declares profile ID/version, frequency, exact ordered features, candidate specs, training policy, inference policy, quality thresholds, and selection policy.
- [ ] YAML loading is deterministic and validated before model work.
- [ ] Unknown keys and duplicate features fail closed.
- [ ] Unsupported family/covariance/state count fails clearly.
- [ ] Profile has deterministic content hash.
- [ ] Unit tests cover valid, malformed, unknown, duplicate, unsupported, and hash cases.

## PR-008 — Add generic Parquet feature-source port and adapter

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/008-parquet-feature-source`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/features/__init__.py`, `src/market_regime_engine/features/ports.py`, `src/market_regime_engine/features/parquet_source.py`, `tests/unit/features/*`, `tests/integration/test_parquet_feature_source.py`

### Acceptance criteria

- [ ] Narrow `FeatureSource` protocol has no `market-regime-loader` import.
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
- **Allowed files:** `src/market_regime_engine/preprocessing/__init__.py`, `src/market_regime_engine/preprocessing/scaler.py`, `src/market_regime_engine/preprocessing/contracts.py`, `tests/unit/preprocessing/*`

### Acceptance criteria

- [ ] Serializable preprocessing contract.
- [ ] Standard scaling fit only on provided training slice.
- [ ] Exact feature order is preserved.
- [ ] Zero variance and non-finite values fail explicitly.
- [ ] Deterministic serialize/deserialize.
- [ ] Test proves future rows cannot alter training parameters.

## PR-010 — Define model-adapter and fitted-model artifact protocols

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/010-model-adapter-protocol`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/models/__init__.py`, `src/market_regime_engine/models/protocols.py`, `src/market_regime_engine/models/artifacts.py`, `tests/unit/models/test_protocols.py`, `tests/unit/models/test_artifacts.py`

### Acceptance criteria

- [ ] Adapter protocol defines fit, score, parameter extraction, reconstruction, and filtered-inference capability boundary.
- [ ] Fitted artifact contains initial probabilities, transition matrix, emissions, feature order, K, family, preprocessing reference, and convergence metadata.
- [ ] Artifact validation checks shapes, finite values, normalized rows, and feature/model consistency.
- [ ] Protocol has no MLflow, HTTP, filesystem, or portfolio dependency.
- [ ] Deterministic dummy adapter proves serialization/reconstruction contract.

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
- [ ] Settings expose the shared endpoint as the documented production/NAS default while allowing explicit local-file configuration for tests/development.
- [ ] Production validation rejects missing/blank tracking URI with actionable error.
- [ ] Narrow tracking and registry ports are defined for application code.
- [ ] No experiment ID, registered-model version, token, username, or password is hard-coded.
- [ ] Unit/local-file mode never contacts `10.10.1.3`.
- [ ] Unit tests cover production URI parsing, local-file URI, invalid URI, and no-network construction.

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

## PR-014 — Implement configurable Gaussian HMM adapter

- **Status:** BLOCKED by PR-009, PR-010
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/014-gaussian-hmm-adapter`
- **Depends on:** PR-009, PR-010
- **Allowed files:** `src/market_regime_engine/models/gaussian_hmm.py`, `tests/unit/models/test_gaussian_hmm.py`, `tests/fixtures/hmm/*`

### Acceptance criteria

- [ ] Implements model protocol.
- [ ] K configurable; tests K=2/3/4.
- [ ] Covariance type configurable; tests diagonal/full where supported.
- [ ] Seed, max iterations, and tolerance explicit.
- [ ] Fit returns convergence, iterations, log likelihood, initial probabilities, transition matrix, means, covariances.
- [ ] Invalid/non-converged fits are explicit and non-promotable.
- [ ] Fitted artifact reconstructs equivalent model.
- [ ] Deterministic synthetic tests verify normalization/shapes/reproducibility.

## PR-015 — Add deterministic multi-start HMM fitting

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/015-hmm-multistart`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/training/__init__.py`, `src/market_regime_engine/training/multistart.py`, `tests/unit/training/test_multistart.py`

### Acceptance criteria

- [ ] Explicit ordered seed list/count.
- [ ] Every start records convergence and final likelihood.
- [ ] Failed/non-finite fits excluded but retained diagnostically.
- [ ] Highest-likelihood valid converged fit wins with deterministic tie break.
- [ ] Minimum valid converged starts configurable.
- [ ] Insufficient stable starts fail closed.
- [ ] Tests cover mixed failures, ties, and insufficient valid starts.

## PR-016 — Implement causal forward filtering

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/016-causal-forward-filter`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/inference/__init__.py`, `src/market_regime_engine/inference/filtering.py`, `tests/unit/inference/test_filtering.py`

### Acceptance criteria

- [ ] Forward-only filtered state probabilities from fitted parameters.
- [ ] Output at `t` depends only on observations `<=t`.
- [ ] Probabilities finite/normalized.
- [ ] Numerical stabilization tested on long series.
- [ ] Appending future observations leaves historical filtered probabilities unchanged.
- [ ] Tests distinguish filtering from smoothing.
- [ ] Output can construct `RegimePredictionV1` after state alignment.

## PR-017 — Add transition-horizon probability forecasts

- **Status:** BLOCKED by PR-016
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/017-transition-forecasts`
- **Depends on:** PR-016
- **Allowed files:** `src/market_regime_engine/inference/forecasting.py`, `tests/unit/inference/test_forecasting.py`

### Acceptance criteria

- [ ] Forecasts state distribution for integer horizons from transition matrix.
- [ ] Horizon zero equals current filtered distribution.
- [ ] Outputs finite/normalized.
- [ ] Invalid matrices/horizons fail.
- [ ] Tests compare 1-step/multi-step against direct matrix powers.

## PR-018 — Add persistent state alignment

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/018-state-alignment`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/states/__init__.py`, `src/market_regime_engine/states/signatures.py`, `src/market_regime_engine/states/alignment.py`, `tests/unit/states/*`

### Acceptance criteria

- [ ] Normalized state signatures from emissions in exact feature order.
- [ ] Deterministic one-to-one mapping to reference states.
- [ ] Pure label permutations map correctly.
- [ ] Persistent IDs are independent of raw labels.
- [ ] Maximum signature drift configurable and fail-closed.
- [ ] Alignment artifact has deterministic hash/version.
- [ ] Tests cover permutation, small/excessive drift, ambiguity.

## PR-019 — Add model-quality diagnostics

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/019-model-diagnostics`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/evaluation/__init__.py`, `src/market_regime_engine/evaluation/diagnostics.py`, `tests/unit/evaluation/test_diagnostics.py`

### Acceptance criteria

- [ ] Train log likelihood, AIC, BIC.
- [ ] Hard occupancy and soft/effective occupancy.
- [ ] Transition/self-transition and empirical hard-state duration statistics.
- [ ] Empty/near-empty state detection with configurable thresholds.
- [ ] Non-finite parameter/invalid covariance detection.
- [ ] Metric definitions documented and unit tested.

## PR-020 — Add expanding walk-forward split planner

- **Status:** BLOCKED by PR-007, PR-008, PR-009
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/020-walk-forward-splits`
- **Depends on:** PR-007, PR-008, PR-009
- **Allowed files:** `src/market_regime_engine/evaluation/walk_forward_splits.py`, `tests/unit/evaluation/test_walk_forward_splits.py`

### Acceptance criteria

- [ ] Expanding windows with explicit minimum train observations/test size.
- [ ] Every test row is strictly after its training interval.
- [ ] No test row leaks into same fold's train data.
- [ ] Calendar gaps create no synthetic rows.
- [ ] Split plan deterministic/serializable.
- [ ] Tests cover normal, short, gaps, boundaries, partial final window.

## PR-021 — Add reusable Xetra cross-asset model profile

- **Status:** BLOCKED by PR-007, PR-008
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/021-xetra-cross-asset-profile`
- **Depends on:** PR-007, PR-008
- **Allowed files:** `configs/xetra_cross_asset_v1.yaml`, `docs/profiles/xetra_cross_asset_v1.md`, `tests/unit/profiles/test_xetra_profile.py`

### Acceptance criteria

- [ ] Profile ID/version is `xetra_cross_asset_v1`.
- [ ] Features come only from reusable `market-regime-loader` Gold contract; no ETF-return/portfolio feature is embedded.
- [ ] Candidate grid includes Gaussian HMM K=2/3/4 diagonal and K=3 full.
- [ ] Multi-start, convergence, minimum occupancy, drift, and walk-forward settings are explicit.
- [ ] Inference mode is filtered.
- [ ] Documentation explains Xetra is downstream application universe while profile models reusable cross-asset market state.
- [ ] Profile validation/hash test passes.

---

# Wave 3 — Walk-forward evaluation, MLflow, champion selection

PR-022 is the convergence point. PR-023 and PR-027 may run in parallel after it. PR-024 -> PR-025 -> PR-026 then complete the promotion path.

## PR-022 — Implement leak-free walk-forward evaluation runner

- **Status:** BLOCKED by PR-015, PR-016, PR-018, PR-019, PR-020
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/022-walk-forward-runner`
- **Depends on:** PR-015, PR-016, PR-018, PR-019, PR-020
- **Allowed files:** `src/market_regime_engine/evaluation/walk_forward.py`, `tests/unit/evaluation/test_walk_forward.py`, `tests/integration/test_walk_forward_runner.py`

### Acceptance criteria

- [ ] Each fold fits preprocessing only on training rows.
- [ ] Each fold fits HMM only on training rows.
- [ ] Alignment uses only current/prior training information.
- [ ] Test rows use causal filtering only.
- [ ] One OOS prediction row per eligible test timestamp with fold ID/lineage.
- [ ] Duplicate OOS timestamps fail unless explicitly disallowed before execution.
- [ ] Fold diagnostics include fit, occupancy, model, alignment, and date boundaries.
- [ ] Mutating future rows cannot change earlier OOS predictions.

## PR-023 — Add MLflow experiment-tracking adapter for shared registry architecture

- **Status:** BLOCKED by PR-012, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/023-mlflow-experiment-tracking`
- **Depends on:** PR-012, PR-022
- **Allowed files:** `src/market_regime_engine/mlflow_support/tracking.py`, `tests/unit/mlflow_support/test_tracking.py`, `tests/integration/test_mlflow_file_tracking.py`

### Acceptance criteria

- [ ] Adapter uses the configured `MLFLOW_TRACKING_URI`; production configuration from PR-012 points to `http://10.10.1.3:5000`.
- [ ] Experiment is created/reused from explicit profile/experiment configuration.
- [ ] Parent run records profile hash, engine version, Git SHA, feature version, source build, and evaluation plan.
- [ ] Candidate/fold runs log family, K, covariance, seeds, bounds, convergence, likelihood/AIC/BIC, occupancy, duration, and alignment diagnostics.
- [ ] OOS prediction artifact reference is logged.
- [ ] Unit tests use fake port; required integration uses local file-backed MLflow and no external service.
- [ ] No test in required gates attempts the shared NAS endpoint.

## PR-024 — Add candidate-grid orchestration

- **Status:** BLOCKED by PR-007, PR-015, PR-022, PR-023
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/024-candidate-grid-orchestrator`
- **Depends on:** PR-007, PR-015, PR-022, PR-023
- **Allowed files:** `src/market_regime_engine/training/candidate_grid.py`, `tests/unit/training/test_candidate_grid.py`, `tests/integration/test_candidate_grid.py`

### Acceptance criteria

- [ ] Expands only validated profile candidates.
- [ ] Deterministic candidate IDs.
- [ ] All candidates use same declared walk-forward plan.
- [ ] One candidate failure does not corrupt completed candidates.
- [ ] Output includes aggregate diagnostics and OOS reference.
- [ ] Deterministic for fixed input/profile/seeds.
- [ ] Integration covers K=2 and K=3 end-to-end.

## PR-025 — Add model validation gates and engine-champion selection

- **Status:** BLOCKED by PR-019, PR-024
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/025-engine-champion-selection`
- **Depends on:** PR-019, PR-024
- **Allowed files:** `src/market_regime_engine/evaluation/selection.py`, `tests/unit/evaluation/test_selection.py`

### Acceptance criteria

- [ ] Selection uses explicit profile policy; no hidden composite score.
- [ ] Hard gates include stable convergence, finite parameters, minimum effective occupancy, successful alignment, and drift limit.
- [ ] Failed candidates cannot win.
- [ ] Valid candidates rank by declared OOS model-centric criterion with deterministic ties.
- [ ] BIC/AIC are secondary/tie-break diagnostics, not sole champion criterion.
- [ ] Output records rejected candidates and reasons.
- [ ] Tests cover zero-valid, gate rejection, winner, and tie cases.

## PR-026 — Package fitted model in shared MLflow Model Registry and manage aliases

- **Status:** BLOCKED by PR-012, PR-016, PR-018, PR-025
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/026-mlflow-model-registry`
- **Depends on:** PR-012, PR-016, PR-018, PR-025
- **Allowed files:** `src/market_regime_engine/mlflow_support/model_package.py`, `src/market_regime_engine/mlflow_support/registry.py`, `tests/unit/mlflow_support/test_model_package.py`, `tests/unit/mlflow_support/test_registry.py`, `tests/integration/test_mlflow_registry_local.py`

### Acceptance criteria

- [ ] MLflow package contains preprocessing, fitted HMM, feature order, persistent state mapping/signature, profile hash, lineage, and inference-contract version.
- [ ] Artifact round-trip yields identical filtered prediction on deterministic fixture.
- [ ] Registry uses explicit registered model versions.
- [ ] Registry can set/move `engine-champion`, `challenger`, and arbitrary consumer aliases such as `portfell-production`.
- [ ] Alias movement is logged with source/destination version and reason.
- [ ] `engine-champion` can only come from validated selection result.
- [ ] Production registry operations use configured shared MLflow endpoint `http://10.10.1.3:5000`.
- [ ] Required integration tests remain local-file/no-network; shared-server behavior is verified by PR-034.

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
- [ ] Input is explicit feature-source/build reference and contract is validated.
- [ ] Output uses filtered inference and persistent state mapping.
- [ ] Metadata mode is `fixed_model_replay`.
- [ ] Never labels replay as walk-forward OOS.
- [ ] Date bounds deterministic/validated.
- [ ] Large results return immutable prediction-build reference rather than unbounded JSON.
- [ ] Local integration covers registered model + Parquet input without network.

## PR-029 — Add realtime/latest inference API

- **Status:** BLOCKED by PR-008, PR-013, PR-016, PR-018, PR-026
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/029-realtime-latest-api`
- **Depends on:** PR-008, PR-013, PR-016, PR-018, PR-026
- **Allowed files:** `src/market_regime_engine/inference/realtime.py`, `src/market_regime_engine/api/routes/latest.py`, `tests/unit/inference/test_realtime.py`, `tests/unit/api/test_latest_route.py`, `tests/integration/test_latest_api.py`

### Acceptance criteria

- [ ] Resolves explicit profile + model alias/version from configured MLflow registry.
- [ ] Production default configuration resolves registry through `http://10.10.1.3:5000`.
- [ ] Loads only feature data available up to as-of.
- [ ] Feature contract matches registered model exactly.
- [ ] Response validates `RegimePredictionV1` and includes resolved model version/lineage.
- [ ] MLflow unavailable, missing data, incompatible features, or quality failure returns explicit non-200; no stale/invented prediction.
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
- [ ] `evaluate` runs walk-forward candidate grid and publishes OOS predictions.
- [ ] `register` registers validated model and optional explicit alias.
- [ ] Production `register` uses `MLFLOW_TRACKING_URI`, documented value `http://10.10.1.3:5000`.
- [ ] `serve` starts FastAPI with environment host/port.
- [ ] Every command has help, deterministic exit codes, actionable errors.
- [ ] Unit tests mock services and use no external MLflow.

---

# Wave 5 — Deployment, shared MLflow verification, compatibility, end-to-end proof

PR-032 through PR-035 run in parallel after their prerequisites. PR-036 consolidates final operator/consumer documentation.

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
- [ ] Deployment doc identifies `http://10.10.1.3:5000` as the existing shared NAS MLflow Tracking Server / Model Registry.
- [ ] Deployment doc states MLflow/PostgreSQL lifecycle is not managed by this compose stack.
- [ ] Healthcheck uses `/health/ready`.
- [ ] No credentials/tokens are embedded.
- [ ] Integration test asserts the compose default and environment override behavior.

## PR-033 — Add market-regime-loader Gold compatibility integration test

- **Status:** BLOCKED by PR-008, PR-021
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/033-loader-contract-integration`
- **Depends on:** PR-008, PR-021
- **Allowed files:** `tests/fixtures/loader_gold/*`, `tests/integration/test_loader_gold_contract.py`, `docs/integrations/market_regime_loader.md`

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
- [ ] Verification fails if configured production URI differs from expected shared endpoint unless an explicit migration/override flag is supplied.
- [ ] Smoke test verifies Tracking Server reachability and server metadata before writes.
- [ ] Smoke test creates a uniquely named disposable experiment/run, logs parameters/metrics/artifact, reads them back, and verifies registry create/read/version/alias behavior.
- [ ] Disposable registered model/experiment names are namespaced with a unique test identifier; cleanup/delete is attempted and cleanup result is reported.
- [ ] No credential/token is committed.
- [ ] Documentation explains that the server is shared with other projects and destructive cleanup must be limited strictly to resources created by the smoke test.

## PR-035 — Add complete hermetic engine end-to-end integration proof

- **Status:** BLOCKED by PR-021, PR-024, PR-026, PR-027, PR-028, PR-029, PR-030
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/035-engine-e2e-proof`
- **Depends on:** PR-021, PR-024, PR-026, PR-027, PR-028, PR-029, PR-030
- **Allowed files:** `tests/integration/test_engine_e2e.py`, `tests/fixtures/e2e/*`

### Acceptance criteria

- [ ] Uses deterministic fixture Parquet and local-file MLflow only.
- [ ] Candidate grid includes at least Gaussian K=2/K=3.
- [ ] Runs walk-forward OOS evaluation.
- [ ] Applies hard gates and selects engine champion.
- [ ] Registers winner locally and assigns `engine-champion`.
- [ ] Publishes immutable OOS predictions.
- [ ] Exercises fixed-model batch API and confirms `fixed_model_replay`.
- [ ] Exercises latest API and validates `RegimePredictionV1`.
- [ ] Exercises OOS retrieval.
- [ ] Same seeds/input reproduce deterministic values/lineage where required.
- [ ] Test never requires `10.10.1.3`; real shared MLflow is exclusively PR-034 external verification.

## PR-036 — Final operator and consumer documentation

- **Status:** BLOCKED by PR-032, PR-033, PR-034, PR-035
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/036-operator-consumer-docs`
- **Depends on:** PR-032, PR-033, PR-034, PR-035
- **Allowed files:** `README.md`, `API.md`, `OPERATIONS.md`, `docs/consumer_contract.md`, `docs/integrations/portfell.md`, `docs/integrations/mlflow.md`

### Acceptance criteria

- [ ] README shows loader -> engine -> MLflow/API -> consumer architecture.
- [ ] README gives clean-checkout `.venv` bootstrap for Python 3.14.7.
- [ ] Shared production MLflow endpoint is documented exactly as `http://10.10.1.3:5000`.
- [ ] Operations doc gives exact `MLFLOW_TRACKING_URI=http://10.10.1.3:5000` setup and train -> evaluate -> select -> register -> alias -> serve lifecycle.
- [ ] Operations doc distinguishes shared MLflow infrastructure ownership from engine ownership.
- [ ] API doc covers health, latest, fixed replay batch, and OOS retrieval.
- [ ] API doc warns `fixed_model_replay` is not leak-free OOS evidence.
- [ ] Consumer contract documents `RegimePrediction.v1` and immutable OOS fields.
- [ ] Portfell doc states Portfell owns ETF universe, regime-conditioned returns/covariances, portfolio optimization/backtesting/costs, and application-level model choice.
- [ ] Future consumers can reuse registered models/API without HMM implementation imports or engine fork.

---

# Wave 6 — Optional model challengers after MVP

These adapters are isolated behind the model protocol and are not required for the first Gaussian-HMM platform.

## PR-037 — Add Student-t HMM challenger adapter

- **Status:** OPTIONAL / BLOCKED by PR-010, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/037-student-t-hmm-challenger`
- **Depends on:** PR-010, PR-022
- **Allowed files:** `src/market_regime_engine/models/student_t_hmm.py`, `tests/unit/models/test_student_t_hmm.py`, `docs/models/student_t_hmm.md`, `pyproject.toml`

### Acceptance criteria

- [ ] Satisfies common model/artifact protocol.
- [ ] Additional dependency supports Python 3.14.
- [ ] Degrees-of-freedom/covariance configuration explicit.
- [ ] Filtered inference causal and alignment-compatible.
- [ ] Participates in existing walk-forward/grid/selection with no consumer special case.
- [ ] Deterministic heavy-tail synthetic test proves fit/inference.

## PR-038 — Add duration-aware HSMM challenger adapter

- **Status:** OPTIONAL / BLOCKED by PR-010, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/038-hsmm-challenger`
- **Depends on:** PR-010, PR-022
- **Allowed files:** `src/market_regime_engine/models/hsmm.py`, `tests/unit/models/test_hsmm.py`, `docs/models/hsmm.md`, `pyproject.toml`

### Acceptance criteria

- [ ] Satisfies common model protocol or documents minimal explicit extension for durations.
- [ ] New dependency supports Python 3.14.
- [ ] Duration parameters explicit, validated, serialized, versioned.
- [ ] Inference causal for OOS/production.
- [ ] Participates in same walk-forward/grid/selection interface with no consumer change.
- [ ] Persistent-regime synthetic test proves duration metadata/prediction artifact round-trip.

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

## Definition of complete MVP

The MVP is complete after PR-036 when:

- a clean checkout reproducibly creates a Python 3.14.7 `.venv`;
- `main` is protected and direct/force pushes are blocked;
- push/merge gates run lint/type/unit/integration in parallel;
- PRs auto-complete only after `merge-gate`;
- versioned model profiles can be loaded;
- Parquet features are consumed without upstream loader imports;
- Gaussian HMM K=2/K=3/K=4 and diagonal/full candidates can be compared;
- multi-start fitting, persistent alignment, causal filtering, and walk-forward OOS evaluation are implemented;
- MLflow tracks candidates and packages promoted models;
- production configuration points to the shared MLflow Tracking Server / Model Registry at `http://10.10.1.3:5000`;
- the real shared MLflow instance has an explicit successful opt-in smoke-test path;
- `engine-champion`, `challenger`, and consumer-specific aliases are supported;
- immutable `walk_forward_oos` prediction builds exist;
- fixed-model batch inference is clearly separated from historical OOS predictions;
- latest/realtime inference returns `RegimePrediction.v1`;
- Portfell and future consumers use the API/registered models without HMM implementation imports;
- the complete hermetic local end-to-end proof passes the required merge gate.
