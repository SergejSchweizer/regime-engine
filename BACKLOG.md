# Market Regime Engine — Implementation Backlog

Status date: 2026-08-22

This backlog defines the complete implementation plan for `market-regime-engine` as a reusable market-regime model platform.

The repository owns model training, causal HMM inference, walk-forward validation, MLflow tracking/registry integration, immutable prediction artifacts, and batch/realtime inference APIs. It does **not** own market-data acquisition or portfolio optimization.

Upstream/downstream boundary:

```text
market-regime-loader
    -> immutable causal feature data
    -> market-regime-engine
    -> RegimePrediction.v1 / MLflow models / OOS prediction artifacts / API
    -> portfell and future consumers
```

`market-regime-loader` remains the reusable data product. `portfell` remains the Xetra ETF portfolio application. `market-regime-engine` must stay consumer-agnostic so the same registered HMM models and API can be reused by future BTC, equity, rates, covered-call, or other projects.

## Repository bootstrap facts

- Default branch: `main`.
- The repository started empty before this backlog was created.
- Stable Python feature release selected for this repository: **Python 3.14.7**.
- Local development environment: repository-local `.venv`; `.venv/` is never committed.
- Python compatibility target for MVP: `>=3.14,<3.15`.
- All implementation PRs start from a clean, up-to-date `main` after their declared dependencies are merged.
- Agents must not edit `BACKLOG.md` from implementation PRs. Backlog status is maintained only by the orchestrator to prevent merge conflicts.

## Non-negotiable architecture rules

1. No provider HTTP clients, CBOE/FRED/ECB/STOXX acquisition logic, or EODHD portfolio data acquisition in this repository.
2. No portfolio optimizer, ETF weighting, Sharpe/Sortino selection, transaction-cost model, or trading logic in this repository.
3. Model training may use only data available inside the declared training window.
4. Backtest-safe inference uses filtered probabilities only. Retrospective smoothed probabilities must never be exposed as causal OOS predictions.
5. All preprocessing parameters are fit on training data only.
6. Raw HMM state numbers are not stable consumer semantics. Persistent state identity must be produced by state alignment.
7. Every registered model carries exact feature order, preprocessing state, model parameters, state mapping, source/build lineage, package version, and Git commit.
8. Historical OOS predictions and fixed-model replay are different products and must never be conflated.
9. MLflow URI, credentials, storage locations, and API bind settings come from configuration/environment; no infrastructure URL or secret is hard-coded.
10. Public consumer contracts are versioned. Initial prediction contract is `RegimePrediction.v1`.

## Git discipline for every PR

Every PR entry below has an explicit branch and Git status. The following rules apply to every PR without exception:

```text
Before work:
  git switch main
  git pull --ff-only
  git status --short

Required result before branch creation:
  <empty output>

Create the exact branch named in the PR entry.

Before final push:
  git status --short

Required result after all intended files are committed:
  <empty output>
```

An agent must stop if `git status --short` is non-empty before branch creation, if a declared prerequisite is not merged, or if work requires files outside the PR's allowed-file scope.

## CI and merge policy target

Two independent GitHub Actions workflows are required.

### Push gate

Trigger: every branch push.

Parallel required jobs:

```text
lint
 type
 unit
 integration
```

The four jobs run independently and in parallel. A final `push-gate` job depends on all four and succeeds only when all four succeed.

### Merge gate

Trigger: pull requests targeting `main`.

Parallel required jobs:

```text
lint
 type
 unit
 integration
```

The four jobs run independently and in parallel. A final `merge-gate` job depends on all four and succeeds only when all four succeed.

### Protected `main`

After the merge workflow exists on `main`, repository governance must enforce:

- changes to `main` only through pull requests;
- required status check `merge-gate` with strict/up-to-date branch requirement;
- force pushes disabled;
- branch deletion disabled;
- conversation resolution required;
- administrators included in protection;
- repository auto-merge enabled;
- squash merge is the automated merge method;
- feature branch deleted after merge;
- auto-merge/auto-complete is enabled on implementation PRs after creation;
- a PR reaches `main` only after `merge-gate` succeeds.

The bootstrap/governance PRs are the only temporary exception while the repository is establishing the checks that protection itself requires.

---

# Wave 0 — Bootstrap and governance

Only PR-001 starts immediately. PR-002 and PR-003 may start in parallel after PR-001. PR-004 starts after PR-003 is merged because branch protection must reference an existing required check.

## PR-001 — Bootstrap Python 3.14.7 project and local `.venv`

- **Status:** TODO
- **Git status:** PLANNED — must start and finish with `git status --short` empty.
- **Branch:** `pr/001-bootstrap-python314`
- **Depends on:** none
- **Allowed files:** `.python-version`, `.gitignore`, `pyproject.toml`, `README.md`, `src/market_regime_engine/__init__.py`, `tests/unit/test_package_smoke.py`, `tests/conftest.py`, `scripts/bootstrap_venv.sh`, `scripts/bootstrap_venv.ps1`

### Scope

Create the smallest installable Python package and deterministic local environment bootstrap. Establish tool configuration used by every later PR.

### Acceptance criteria

- [ ] `.python-version` contains `3.14.7`.
- [ ] `pyproject.toml` defines package name `market-regime-engine` and `requires-python = ">=3.14,<3.15"`.
- [ ] Source layout is `src/market_regime_engine`.
- [ ] Runtime dependency set includes the libraries required by the planned engine layers: Pydantic, NumPy, SciPy, scikit-learn, Polars, PyArrow, FastAPI, Uvicorn, HTTPX, MLflow, and a Gaussian-HMM implementation dependency that successfully installs on Python 3.14.7.
- [ ] Dev dependency set includes pytest, pytest-cov, pytest-xdist, Ruff, mypy, and build tooling.
- [ ] Ruff is configured for Python 3.14 and 100-character lines.
- [ ] mypy is strict for `src/market_regime_engine`.
- [ ] pytest defines `unit`, `integration`, and `external_service` markers.
- [ ] `.gitignore` ignores `.venv/`, Python caches, test caches, coverage output, build output, local MLflow state, local prediction artifacts, and IDE-local files.
- [ ] `scripts/bootstrap_venv.sh` creates `.venv` with Python 3.14.7, upgrades packaging tools, installs the project editable with dev dependencies, and fails if the interpreter is not Python 3.14.x.
- [ ] `scripts/bootstrap_venv.ps1` provides equivalent Windows behavior using `py -3.14`/Python 3.14 and `.venv`.
- [ ] Neither bootstrap script activates the venv implicitly; both print the correct activation command on success.
- [ ] `README.md` documents the exact bootstrap command for Linux/macOS and Windows.
- [ ] `tests/unit/test_package_smoke.py` verifies the package imports and exposes `__version__`.
- [ ] From a clean checkout, `.venv` can be created and `python -m pytest tests/unit/test_package_smoke.py` passes.
- [ ] `.venv/` is not tracked by Git.

## PR-002 — Add parallel push quality gate

- **Status:** BLOCKED by PR-001
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/002-push-quality-gate`
- **Depends on:** PR-001
- **Allowed files:** `.github/workflows/push-gate.yml`

### Scope

Add the push workflow only. Do not add branch protection or PR merge policy here.

### Acceptance criteria

- [ ] Workflow triggers on pushes to repository branches.
- [ ] `lint`, `type`, `unit`, and `integration` are four separate jobs with no dependency on each other.
- [ ] All jobs use Python 3.14.7.
- [ ] `lint` runs Ruff check and Ruff format-check.
- [ ] `type` runs strict mypy.
- [ ] `unit` runs tests excluding the `integration` and `external_service` markers.
- [ ] `integration` runs tests marked `integration` and excludes `external_service`.
- [ ] External-service tests are never required for the push gate.
- [ ] A final job named exactly `push-gate` has `needs: [lint, type, unit, integration]` and fails if any required job fails.
- [ ] Workflow uses per-ref concurrency with cancellation of superseded runs.
- [ ] No secrets are required for this workflow.

## PR-003 — Add parallel merge quality gate

- **Status:** BLOCKED by PR-001
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/003-merge-quality-gate`
- **Depends on:** PR-001
- **Allowed files:** `.github/workflows/merge-gate.yml`

### Scope

Add the pull-request workflow that will become the protected-branch required check.

### Acceptance criteria

- [ ] Workflow triggers only for pull requests targeting `main`.
- [ ] `lint`, `type`, `unit`, and `integration` are separate parallel jobs.
- [ ] All jobs use Python 3.14.7.
- [ ] Commands and marker policy are equivalent to the push gate.
- [ ] A final job named exactly `merge-gate` depends on all four required jobs.
- [ ] `merge-gate` cannot succeed when any required job is failed, cancelled, or skipped unexpectedly.
- [ ] Workflow uses PR concurrency with cancellation when a newer commit is pushed to the same PR.
- [ ] No deployment, MLflow, network provider, or repository-admin secret is required.

## PR-004 — Configure protected `main` and repository auto-merge

- **Status:** BLOCKED by PR-003
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/004-repository-governance`
- **Depends on:** PR-003
- **Allowed files:** `scripts/configure_github_governance.sh`, `docs/repository_governance.md`

### Scope

Provide a deterministic, auditable one-time repository-admin command using the GitHub CLI/API to apply and verify repository settings. The script must not contain a token.

### Acceptance criteria

- [ ] Script requires authenticated `gh` CLI and fails clearly if authentication/admin permission is missing.
- [ ] Script targets exactly `SergejSchweizer/market-regime-engine` and branch `main`.
- [ ] Script enables repository auto-merge.
- [ ] Script enables squash merge as the automated merge method and configures deletion of merged feature branches.
- [ ] Script protects `main` and requires pull requests.
- [ ] Required status check is exactly `merge-gate` with strict/up-to-date branch enforcement.
- [ ] Force pushes to `main` are disabled.
- [ ] Deletion of `main` is disabled.
- [ ] Conversation resolution is required.
- [ ] Protection applies to administrators.
- [ ] Documentation states that the governance script is run **after this PR is merged**, because the required `merge-gate` check must already exist on `main`.
- [ ] Documentation gives an exact verification command for repository auto-merge state and branch protection state.
- [ ] Post-merge verification confirms the protected `main` policy matches every target above.
- [ ] All later PRs are created with auto-merge enabled so GitHub completes the squash merge automatically after `merge-gate` succeeds.

---

# Wave 1 — Public contracts and boundaries

After PR-001 is merged, PR-005 and PR-006 may run in parallel. After PR-006, PR-007 through PR-013 may run in parallel because they own disjoint modules/files.

## PR-005 — Document durable architecture and repository boundaries

- **Status:** BLOCKED by PR-001
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/005-architecture-contract`
- **Depends on:** PR-001
- **Allowed files:** `ARCHITECTURE.md`, `docs/model_lifecycle.md`, `README.md`

### Acceptance criteria

- [ ] `ARCHITECTURE.md` defines the engine as a reusable model platform, not a data loader or portfolio optimizer.
- [ ] Architecture documents ports/adapters around feature input, model adapters, MLflow, prediction persistence, and API.
- [ ] Architecture explicitly separates `fixed_model_replay` from `walk_forward_oos` predictions.
- [ ] Architecture states filtered probabilities are required for causal inference and smoothed probabilities are diagnostic-only.
- [ ] Architecture defines persistent state alignment as mandatory for consumer-facing predictions.
- [ ] Architecture defines `market-regime-loader -> engine -> consumers` dependency direction without Python package imports between repositories.
- [ ] `docs/model_lifecycle.md` defines candidate, validated, engine-champion, and consumer-specific production aliases.
- [ ] README contains a short system diagram and links to architecture/lifecycle docs.

## PR-006 — Define versioned core domain contracts

- **Status:** BLOCKED by PR-001
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/006-core-domain-contracts`
- **Depends on:** PR-001
- **Allowed files:** `src/market_regime_engine/contracts/__init__.py`, `src/market_regime_engine/contracts/features.py`, `src/market_regime_engine/contracts/models.py`, `src/market_regime_engine/contracts/predictions.py`, `src/market_regime_engine/contracts/lineage.py`, `tests/unit/contracts/*`

### Acceptance criteria

- [ ] Define immutable `FeatureFrameRef`/feature metadata contract with source dataset, source build ID, as-of range, feature version, and ordered feature names.
- [ ] Define immutable `ModelSpec` with model family, state count, covariance mode, feature profile/version, random/multi-start policy, and training-window policy.
- [ ] Define immutable model lineage with engine version, Git SHA, training interval, source build ID, and preprocessing version.
- [ ] Define `RegimePredictionV1` with as-of timestamp, profile ID, model name/version, persistent state IDs, probability vector, dominant state, entropy, confidence, lineage, and data-quality status.
- [ ] Contract validation rejects probabilities outside `[0,1]`, non-finite values, duplicate state IDs, and probability vectors that do not sum to one within declared tolerance.
- [ ] Raw library state labels are not accepted as persistent consumer semantics without an alignment identifier.
- [ ] Serialization round-trip tests exist for every public contract.
- [ ] Contract tests contain no model-library, MLflow, filesystem, FastAPI, or provider dependency.

## PR-007 — Add model-profile configuration schema and loader

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/007-model-profile-config`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/profiles/__init__.py`, `src/market_regime_engine/profiles/schema.py`, `src/market_regime_engine/profiles/loader.py`, `tests/unit/profiles/*`

### Acceptance criteria

- [ ] Profile schema declares profile ID/version, frequency, exact ordered feature list, candidate model specs, training-window policy, inference policy, state-quality thresholds, and selection policy.
- [ ] YAML loading is deterministic and validated before any model work starts.
- [ ] Unknown keys fail closed.
- [ ] Duplicate feature names fail validation.
- [ ] Unsupported model family/covariance/state-count values fail validation with actionable messages.
- [ ] Profile content has a deterministic hash used for lineage.
- [ ] Unit tests cover valid config, malformed YAML, unknown field, duplicate feature, unsupported model, and stable hashing.

## PR-008 — Add generic Parquet feature-source port and adapter

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/008-parquet-feature-source`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/features/__init__.py`, `src/market_regime_engine/features/ports.py`, `src/market_regime_engine/features/parquet_source.py`, `tests/unit/features/*`, `tests/integration/test_parquet_feature_source.py`

### Acceptance criteria

- [ ] Define a narrow `FeatureSource` protocol independent of `market-regime-loader` Python code.
- [ ] Initial adapter reads a Parquet dataset plus explicit lineage metadata/manifest input.
- [ ] Adapter selects profile features in exact declared order.
- [ ] Duplicate timestamps, non-monotonic timestamps, missing required features, duplicate columns, and non-finite values fail validation.
- [ ] Adapter never forward-fills, backward-fills, interpolates, or silently imputes values.
- [ ] Source build ID and feature version are preserved into the returned feature reference.
- [ ] Integration test proves a loader-shaped immutable Gold Parquet fixture can be consumed without importing `market-regime-loader`.

## PR-009 — Add train-only preprocessing pipeline

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/009-preprocessing-pipeline`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/preprocessing/__init__.py`, `src/market_regime_engine/preprocessing/scaler.py`, `src/market_regime_engine/preprocessing/contracts.py`, `tests/unit/preprocessing/*`

### Acceptance criteria

- [ ] Define a serializable preprocessing contract.
- [ ] Implement standard scaling with means/scales fit only from the provided training slice.
- [ ] Transform preserves exact feature order.
- [ ] Zero-variance and non-finite feature handling is explicit and fail-closed.
- [ ] Fitted preprocessing parameters serialize/deserialize deterministically.
- [ ] Unit test proves test/future rows cannot change fitted training parameters.

## PR-010 — Define model-adapter and fitted-model artifact protocols

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/010-model-adapter-protocol`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/models/__init__.py`, `src/market_regime_engine/models/protocols.py`, `src/market_regime_engine/models/artifacts.py`, `tests/unit/models/test_protocols.py`, `tests/unit/models/test_artifacts.py`

### Acceptance criteria

- [ ] Define model adapter methods for fit, score, fitted parameter extraction, and artifact reconstruction.
- [ ] Fitted artifact contains initial-state probabilities, transition matrix, emission parameters, feature order, state count, model family, preprocessing reference, and convergence metadata.
- [ ] Artifact validation checks matrix shapes, finite values, normalized probability rows, and feature/model consistency.
- [ ] Protocol has no MLflow, HTTP, filesystem, or portfolio dependency.
- [ ] Serialization/reconstruction contract tests use a deterministic dummy adapter.

## PR-011 — Add immutable prediction-store port and local Parquet adapter

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/011-prediction-store`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/predictions/__init__.py`, `src/market_regime_engine/predictions/ports.py`, `src/market_regime_engine/predictions/parquet_store.py`, `tests/unit/predictions/*`, `tests/integration/test_prediction_store.py`

### Acceptance criteria

- [ ] Define immutable prediction-set metadata with prediction mode (`walk_forward_oos` or `fixed_model_replay`), profile/version, model/version, date range, lineage, and build ID.
- [ ] Parquet adapter writes predictions atomically to a versioned build directory.
- [ ] Existing immutable builds cannot be overwritten.
- [ ] Manifest and data agree on row count, date range, profile, model, and prediction mode.
- [ ] Reader can resolve an explicit build ID; no silent `latest` behavior in research APIs.
- [ ] Integration test covers write, reload, checksum/metadata validation, and attempted overwrite failure.

## PR-012 — Add MLflow configuration and client boundary

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/012-mlflow-client-boundary`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/mlflow_support/__init__.py`, `src/market_regime_engine/mlflow_support/settings.py`, `src/market_regime_engine/mlflow_support/ports.py`, `tests/unit/mlflow_support/*`

### Acceptance criteria

- [ ] MLflow tracking URI is configuration/environment only.
- [ ] No NAS URL, credential, token, experiment ID, or model version is hard-coded.
- [ ] Define narrow tracking/registry ports used by application code.
- [ ] Settings distinguish unit/local-file MLflow from externally configured MLflow.
- [ ] Missing required production MLflow configuration fails with an actionable error.
- [ ] Unit tests use no network.

## PR-013 — Add FastAPI skeleton and health routes

- **Status:** BLOCKED by PR-006
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/013-api-skeleton`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/api/__init__.py`, `src/market_regime_engine/api/app.py`, `src/market_regime_engine/api/dependencies.py`, `src/market_regime_engine/api/routes/health.py`, `src/market_regime_engine/api/routes/latest.py`, `src/market_regime_engine/api/routes/batch.py`, `src/market_regime_engine/api/routes/evaluations.py`, `tests/unit/api/test_health.py`

### Acceptance criteria

- [ ] App factory creates FastAPI app without contacting MLflow, filesystem, or network at import time.
- [ ] `/health/live` returns process liveness.
- [ ] `/health/ready` delegates to injected readiness dependencies.
- [ ] Placeholder route modules for latest, batch, and evaluations are created now so later parallel PRs edit different files.
- [ ] No business logic exists in route modules.
- [ ] OpenAPI generation works in a unit test.

---

# Wave 2 — HMM core and causal inference

PR-014 starts after PR-009 and PR-010. After PR-014, PR-015, PR-016, PR-018, and PR-019 can run in parallel. PR-017 follows PR-016. PR-020 can run in parallel with the HMM work once its own dependencies are satisfied. PR-021 can run after PR-007 and PR-008 without waiting for the model implementation.

## PR-014 — Implement configurable Gaussian HMM adapter

- **Status:** BLOCKED by PR-009, PR-010
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/014-gaussian-hmm-adapter`
- **Depends on:** PR-009, PR-010
- **Allowed files:** `src/market_regime_engine/models/gaussian_hmm.py`, `tests/unit/models/test_gaussian_hmm.py`, `tests/fixtures/hmm/*`

### Acceptance criteria

- [ ] Adapter implements the model protocol for Gaussian HMM.
- [ ] State count is configurable; tests cover K=2, K=3, and K=4.
- [ ] Covariance type is configurable; tests cover diagonal and full covariance where supported.
- [ ] Random seed and iteration/tolerance settings are explicit.
- [ ] Fit returns convergence status, iteration count, log likelihood, initial probabilities, transition matrix, means, and covariance parameters.
- [ ] Invalid/non-converged fits are represented explicitly and are never silently promoted.
- [ ] Fitted artifact reconstructs an equivalent model.
- [ ] Deterministic synthetic-data tests confirm shape/normalization invariants and reproducible fits for a fixed seed.

## PR-015 — Add deterministic multi-start HMM fitting

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/015-hmm-multistart`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/training/__init__.py`, `src/market_regime_engine/training/multistart.py`, `tests/unit/training/test_multistart.py`

### Acceptance criteria

- [ ] Multi-start receives an explicit ordered seed list/count.
- [ ] Every start records convergence and final log likelihood.
- [ ] Failed/non-finite fits are excluded from winner selection and retained in diagnostics.
- [ ] Winner is the highest-likelihood valid converged fit with deterministic tie-breaking.
- [ ] Minimum required valid converged starts is configurable.
- [ ] Run fails closed if the minimum stable-fit requirement is not met.
- [ ] Tests cover mixed success/failure starts, tie breaking, and insufficient-valid-fit failure.

## PR-016 — Implement causal forward filtering

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/016-causal-forward-filter`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/inference/__init__.py`, `src/market_regime_engine/inference/filtering.py`, `tests/unit/inference/test_filtering.py`

### Acceptance criteria

- [ ] Implement forward-only filtered state probabilities from fitted Gaussian-HMM parameters.
- [ ] At timestamp `t`, output depends only on observations `<= t`.
- [ ] Probabilities are normalized and finite for every row.
- [ ] Numerical stabilization is implemented and tested on long synthetic series.
- [ ] Test appending future observations does not change previously produced filtered probabilities.
- [ ] Tests explicitly distinguish the implementation from smoothed/full-sample posterior probabilities.
- [ ] `RegimePredictionV1` can be created from a filtered probability row once persistent state mapping is supplied.

## PR-017 — Add transition-horizon probability forecasts

- **Status:** BLOCKED by PR-016
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/017-transition-forecasts`
- **Depends on:** PR-016
- **Allowed files:** `src/market_regime_engine/inference/forecasting.py`, `tests/unit/inference/test_forecasting.py`

### Acceptance criteria

- [ ] Forecast state distribution for configurable integer horizons using the fitted transition matrix.
- [ ] Horizon zero returns current filtered distribution.
- [ ] Every horizon output is normalized and finite.
- [ ] Invalid transition matrices/horizons fail validation.
- [ ] Tests compare 1-step and multi-step forecasts against direct matrix-power calculations.

## PR-018 — Add persistent state alignment

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/018-state-alignment`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/states/__init__.py`, `src/market_regime_engine/states/signatures.py`, `src/market_regime_engine/states/alignment.py`, `tests/unit/states/*`

### Acceptance criteria

- [ ] Define normalized state signatures from emission parameters in exact feature order.
- [ ] Implement deterministic one-to-one alignment of new fitted states to a reference state set.
- [ ] Alignment handles pure label permutations correctly.
- [ ] Alignment produces persistent state IDs independent of raw library state labels.
- [ ] Maximum allowed signature drift is configurable; excessive drift fails validation rather than silently relabeling.
- [ ] Alignment artifact has deterministic hash/version.
- [ ] Tests cover exact permutation, small drift, excessive drift, and ambiguous mapping.

## PR-019 — Add model-quality diagnostics

- **Status:** BLOCKED by PR-014
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/019-model-diagnostics`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/evaluation/__init__.py`, `src/market_regime_engine/evaluation/diagnostics.py`, `tests/unit/evaluation/test_diagnostics.py`

### Acceptance criteria

- [ ] Compute train log likelihood, AIC, and BIC from explicit model dimensions.
- [ ] Compute hard state occupancy from dominant filtered state.
- [ ] Compute soft/effective occupancy from filtered probabilities.
- [ ] Compute transition/self-transition statistics and empirical hard-state duration summary.
- [ ] Detect empty/near-empty states using configurable thresholds.
- [ ] Detect non-finite parameters and invalid covariance structures.
- [ ] All metric definitions are documented in docstrings and unit-tested on deterministic examples.

## PR-020 — Add expanding walk-forward split planner

- **Status:** BLOCKED by PR-007, PR-008, PR-009
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/020-walk-forward-splits`
- **Depends on:** PR-007, PR-008, PR-009
- **Allowed files:** `src/market_regime_engine/evaluation/walk_forward_splits.py`, `tests/unit/evaluation/test_walk_forward_splits.py`

### Acceptance criteria

- [ ] Planner supports expanding training windows with explicit minimum training observations and test-window size.
- [ ] Every test observation occurs strictly after the corresponding training interval.
- [ ] No overlap places a test row inside the same fold's training interval.
- [ ] Calendar gaps do not create synthetic observations.
- [ ] Split plan is deterministic and serializable.
- [ ] Tests cover normal series, short series rejection, gaps, exact boundary dates, and final partial-window policy.

## PR-021 — Add reusable Xetra cross-asset model profile

- **Status:** BLOCKED by PR-007, PR-008
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/021-xetra-cross-asset-profile`
- **Depends on:** PR-007, PR-008
- **Allowed files:** `configs/xetra_cross_asset_v1.yaml`, `docs/profiles/xetra_cross_asset_v1.md`, `tests/unit/profiles/test_xetra_profile.py`

### Acceptance criteria

- [ ] Profile is named/versioned `xetra_cross_asset_v1`.
- [ ] Feature list uses only columns available from the agreed reusable market-regime-loader Gold contract; no ETF return or portfolio feature is embedded in the model profile.
- [ ] Candidate grid includes Gaussian HMM K=2, K=3, K=4 with diagonal covariance and K=3 with full covariance.
- [ ] Multi-start, convergence, minimum occupancy, state-drift, and walk-forward settings are explicit rather than implicit defaults.
- [ ] Inference mode is filtered.
- [ ] Documentation explains that Xetra is the downstream application universe; the engine profile itself models reusable cross-asset market state.
- [ ] Profile schema/hash test passes.

---

# Wave 3 — Walk-forward evaluation, MLflow, champion selection

PR-022 is the main convergence point. After PR-022, PR-023 and PR-027 can run in parallel. PR-024 follows PR-023. PR-025 follows PR-024. PR-026 follows PR-025.

## PR-022 — Implement leak-free walk-forward evaluation runner

- **Status:** BLOCKED by PR-015, PR-016, PR-018, PR-019, PR-020
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/022-walk-forward-runner`
- **Depends on:** PR-015, PR-016, PR-018, PR-019, PR-020
- **Allowed files:** `src/market_regime_engine/evaluation/walk_forward.py`, `tests/unit/evaluation/test_walk_forward.py`, `tests/integration/test_walk_forward_runner.py`

### Acceptance criteria

- [ ] Each fold fits preprocessing only on that fold's training rows.
- [ ] Each fold fits HMM only on that fold's training rows.
- [ ] State alignment uses only current/prior training information and a declared reference mapping; no future test data influences alignment.
- [ ] Test rows use causal forward filtering only.
- [ ] Runner emits one OOS prediction row per eligible test timestamp with fold ID and complete lineage.
- [ ] No timestamp appears twice in final OOS output unless the configured overlap policy explicitly rejects the run.
- [ ] Fold diagnostics include fit/convergence, occupancy, model diagnostics, state-alignment diagnostics, and date boundaries.
- [ ] Integration test mutates future rows and proves earlier OOS predictions are unchanged.

## PR-023 — Add MLflow experiment-tracking adapter

- **Status:** BLOCKED by PR-012, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/023-mlflow-experiment-tracking`
- **Depends on:** PR-012, PR-022
- **Allowed files:** `src/market_regime_engine/mlflow_support/tracking.py`, `tests/unit/mlflow_support/test_tracking.py`, `tests/integration/test_mlflow_file_tracking.py`

### Acceptance criteria

- [ ] Tracking adapter creates/reuses an experiment from explicit profile/experiment configuration.
- [ ] Parent run records profile hash, engine version, Git SHA, feature version, source build ID, and evaluation plan.
- [ ] Candidate/fold runs record model family, K, covariance type, seeds, train/test bounds, convergence diagnostics, likelihood/AIC/BIC, occupancy, duration, and alignment diagnostics.
- [ ] OOS prediction artifact reference is logged, not silently embedded as untracked local output.
- [ ] Unit tests use a fake tracking port.
- [ ] Integration test uses a local file-backed MLflow store and requires no external service.

## PR-024 — Add candidate-grid orchestration

- **Status:** BLOCKED by PR-007, PR-015, PR-022, PR-023
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/024-candidate-grid-orchestrator`
- **Depends on:** PR-007, PR-015, PR-022, PR-023
- **Allowed files:** `src/market_regime_engine/training/candidate_grid.py`, `tests/unit/training/test_candidate_grid.py`, `tests/integration/test_candidate_grid.py`

### Acceptance criteria

- [ ] Orchestrator expands candidate model specs only from validated profile configuration.
- [ ] Every candidate gets a deterministic candidate ID.
- [ ] Every candidate runs the same declared walk-forward plan.
- [ ] Failure of one candidate is recorded and does not corrupt results of completed candidates.
- [ ] Candidate output includes aggregate diagnostics plus reference to OOS predictions.
- [ ] Grid results are deterministic for fixed input/profile/seeds.
- [ ] Integration test runs at least K=2 and K=3 Gaussian candidates end-to-end on synthetic data.

## PR-025 — Add model validation gates and engine-champion selection

- **Status:** BLOCKED by PR-019, PR-024
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/025-engine-champion-selection`
- **Depends on:** PR-019, PR-024
- **Allowed files:** `src/market_regime_engine/evaluation/selection.py`, `tests/unit/evaluation/test_selection.py`

### Acceptance criteria

- [ ] Selection is driven by explicit profile policy; no hidden weighting or magic composite score.
- [ ] Hard gates include convergence/stable-fit requirement, finite parameters, minimum effective occupancy, state alignment success, and declared state-drift limit.
- [ ] Candidates failing a hard gate cannot become engine champion regardless of likelihood.
- [ ] Among valid candidates, primary ranking uses declared OOS model-centric criterion; deterministic tie-break criteria are explicit.
- [ ] In-sample BIC/AIC may be used only as declared secondary/tie-break diagnostics, not as the sole champion criterion.
- [ ] Selection output records all rejected candidates and rejection reasons.
- [ ] Unit tests cover no-valid-candidate failure, gate rejection, deterministic winner, and deterministic tie break.

## PR-026 — Package fitted model in MLflow and manage registry aliases

- **Status:** BLOCKED by PR-012, PR-016, PR-018, PR-025
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/026-mlflow-model-registry`
- **Depends on:** PR-012, PR-016, PR-018, PR-025
- **Allowed files:** `src/market_regime_engine/mlflow_support/model_package.py`, `src/market_regime_engine/mlflow_support/registry.py`, `tests/unit/mlflow_support/test_model_package.py`, `tests/unit/mlflow_support/test_registry.py`, `tests/integration/test_mlflow_registry_local.py`

### Acceptance criteria

- [ ] MLflow model package contains preprocessing parameters, fitted HMM artifact, exact feature order, persistent state mapping/signature, profile hash, model lineage, and inference contract version.
- [ ] Loaded package produces the same filtered prediction as the pre-registration model on a deterministic fixture.
- [ ] Registry helper registers explicit versions; no consumer silently resolves an unversioned artifact.
- [ ] Registry helper can set/move aliases such as `engine-champion`, `challenger`, and arbitrary consumer-specific aliases such as `portfell-production`.
- [ ] Alias movement is logged with source/destination version and caller-supplied reason.
- [ ] `engine-champion` is set only from a validated selection result.
- [ ] Local integration test uses file/local MLflow and no external network.

## PR-027 — Publish immutable walk-forward OOS prediction builds

- **Status:** BLOCKED by PR-011, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/027-oos-prediction-publication`
- **Depends on:** PR-011, PR-022
- **Allowed files:** `src/market_regime_engine/predictions/oos_publication.py`, `tests/unit/predictions/test_oos_publication.py`, `tests/integration/test_oos_prediction_publication.py`

### Acceptance criteria

- [ ] Walk-forward output is published with mode exactly `walk_forward_oos`.
- [ ] Build manifest includes profile, candidate/model identity, fold plan hash, source build ID, feature version, engine Git SHA, and row/date counts.
- [ ] OOS output contains filtered probabilities, persistent state IDs, dominant state, entropy/confidence, fold ID, and lineage.
- [ ] Publishing the same content is deterministic/idempotent; publishing different content under an existing immutable build ID fails.
- [ ] Integration test reloads the build and validates all prediction rows against `RegimePredictionV1`.

---

# Wave 4 — Batch/realtime services and API

After PR-026, PR-028 and PR-029 can run in parallel. PR-030 depends only on PR-027 plus the API skeleton and can run in parallel with them. PR-031 integrates the already-complete application use cases into one CLI and runs after all three.

## PR-028 — Add fixed-model batch inference API

- **Status:** BLOCKED by PR-008, PR-013, PR-016, PR-018, PR-026
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/028-batch-inference-api`
- **Depends on:** PR-008, PR-013, PR-016, PR-018, PR-026
- **Allowed files:** `src/market_regime_engine/inference/batch.py`, `src/market_regime_engine/api/routes/batch.py`, `tests/unit/inference/test_batch.py`, `tests/unit/api/test_batch_route.py`, `tests/integration/test_batch_api.py`

### Acceptance criteria

- [ ] Batch service loads an explicit model version or explicit registry alias.
- [ ] Batch input is an explicit feature-source/build reference; feature order/contract is validated against the registered model.
- [ ] Batch output uses filtered inference and persistent state mapping.
- [ ] Response metadata clearly labels mode as `fixed_model_replay`.
- [ ] API never labels fixed-model historical replay as walk-forward OOS.
- [ ] Date-range bounds are validated and deterministic.
- [ ] Large result mode writes/returns an immutable prediction build reference rather than an unbounded JSON payload.
- [ ] Integration test covers a registered local model plus Parquet batch input.

## PR-029 — Add realtime/latest inference API

- **Status:** BLOCKED by PR-008, PR-013, PR-016, PR-018, PR-026
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/029-realtime-latest-api`
- **Depends on:** PR-008, PR-013, PR-016, PR-018, PR-026
- **Allowed files:** `src/market_regime_engine/inference/realtime.py`, `src/market_regime_engine/api/routes/latest.py`, `tests/unit/inference/test_realtime.py`, `tests/unit/api/test_latest_route.py`, `tests/integration/test_latest_api.py`

### Acceptance criteria

- [ ] Endpoint resolves explicit profile plus explicit model alias/version; default alias policy is documented and configurable.
- [ ] Service loads only feature data up to the requested/latest as-of time.
- [ ] Feature contract must exactly match registered model feature order/version policy.
- [ ] Response validates against `RegimePredictionV1`.
- [ ] Response includes resolved model version and complete lineage.
- [ ] MLflow unavailable, missing feature build, incompatible features, or data-quality failure returns explicit non-200 error; stale or invented predictions are never returned.
- [ ] Integration test proves deterministic latest prediction from a local registered model.

## PR-030 — Add walk-forward OOS prediction retrieval API

- **Status:** BLOCKED by PR-013, PR-027
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/030-oos-prediction-api`
- **Depends on:** PR-013, PR-027
- **Allowed files:** `src/market_regime_engine/api/routes/evaluations.py`, `src/market_regime_engine/predictions/query.py`, `tests/unit/api/test_evaluations_route.py`, `tests/unit/predictions/test_query.py`, `tests/integration/test_oos_prediction_api.py`

### Acceptance criteria

- [ ] Consumer can retrieve metadata for an explicit `walk_forward_oos` prediction build.
- [ ] Consumer can retrieve bounded date slices from an explicit build ID.
- [ ] Endpoint cannot substitute a fixed-model replay build when OOS mode is requested.
- [ ] Response identifies profile, candidate/model, source build, evaluation plan, and engine version.
- [ ] Portfell can obtain historical leak-free probabilities without importing engine Python modules.
- [ ] Integration test covers date filtering and mode-mismatch rejection.

## PR-031 — Add application CLI for train, evaluate, register, and serve

- **Status:** BLOCKED by PR-024, PR-026, PR-028, PR-029, PR-030
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/031-application-cli`
- **Depends on:** PR-024, PR-026, PR-028, PR-029, PR-030
- **Allowed files:** `src/market_regime_engine/cli.py`, `src/market_regime_engine/commands/*`, `tests/unit/test_cli.py`, `pyproject.toml`

### Acceptance criteria

- [ ] Package exposes one console entry point: `market-regime-engine`.
- [ ] CLI subcommands are thin adapters and contain no model math.
- [ ] `train` fits an explicit profile/input and produces a fitted candidate artifact/run.
- [ ] `evaluate` runs the declared walk-forward candidate grid and publishes OOS predictions.
- [ ] `register` registers a selected validated model and can assign an explicit alias.
- [ ] `serve` starts the FastAPI app through Uvicorn with environment-driven host/port.
- [ ] Every command supports `--help`, deterministic exit codes, and actionable validation errors.
- [ ] Tests mock application services; no external MLflow/network required.

---

# Wave 5 — Deployment, compatibility, and end-to-end proof

PR-032 through PR-035 can run in parallel after their prerequisites. PR-036 is the final documentation/release-readiness consolidation and starts only after the integration proof PRs are merged.

## PR-032 — Add Docker image and NAS-friendly Compose example

- **Status:** BLOCKED by PR-029, PR-031
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/032-container-deployment`
- **Depends on:** PR-029, PR-031
- **Allowed files:** `Dockerfile`, `.dockerignore`, `compose.example.yaml`, `docs/deployment.md`, `tests/integration/test_container_config.py`

### Acceptance criteria

- [ ] Docker image runs the API as a non-root user.
- [ ] Runtime uses Python 3.14-compatible base image.
- [ ] Image contains no `.venv`, Git metadata, test cache, local MLflow state, or secrets.
- [ ] Compose example accepts MLflow URI, feature input root, prediction root, API bind/port, and model alias through environment variables.
- [ ] Healthcheck uses `/health/ready`.
- [ ] No fixed NAS IP/address is embedded.
- [ ] Deployment doc shows local and NAS-style environment configuration without credentials.

## PR-033 — Add market-regime-loader Gold compatibility integration test

- **Status:** BLOCKED by PR-008, PR-021
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/033-loader-contract-integration`
- **Depends on:** PR-008, PR-021
- **Allowed files:** `tests/fixtures/loader_gold/*`, `tests/integration/test_loader_gold_contract.py`, `docs/integrations/market_regime_loader.md`

### Acceptance criteria

- [ ] Fixture mirrors the documented immutable `regime_features_daily` Parquet/manifest contract without importing loader code.
- [ ] `xetra_cross_asset_v1` resolves all required features from the fixture.
- [ ] Build ID and feature version are preserved.
- [ ] Missing required feature, incompatible feature version, duplicate timestamp, and non-finite input each fail closed.
- [ ] Documentation defines the cross-repository handoff contract and explicitly forbids package-level dependency on `market-regime-loader`.

## PR-034 — Add external MLflow smoke-test profile

- **Status:** BLOCKED by PR-023, PR-026
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/034-external-mlflow-smoke`
- **Depends on:** PR-023, PR-026
- **Allowed files:** `tests/external/test_mlflow_external.py`, `docs/integrations/mlflow.md`

### Acceptance criteria

- [ ] Test is marked `external_service` and therefore excluded from required push/merge gates.
- [ ] Test runs only when explicit MLflow environment configuration is present.
- [ ] Smoke test creates a disposable experiment/run, logs a minimal model artifact, reads it back, and verifies registry access where supported.
- [ ] No external MLflow URI or credential is committed.
- [ ] Documentation gives exact environment variables and manual command to run the smoke test.

## PR-035 — Add complete engine end-to-end integration proof

- **Status:** BLOCKED by PR-021, PR-024, PR-026, PR-027, PR-028, PR-029, PR-030
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/035-engine-e2e-proof`
- **Depends on:** PR-021, PR-024, PR-026, PR-027, PR-028, PR-029, PR-030
- **Allowed files:** `tests/integration/test_engine_e2e.py`, `tests/fixtures/e2e/*`

### Acceptance criteria

- [ ] Test uses deterministic synthetic/fixture Parquet feature data and local-file MLflow only.
- [ ] Executes candidate grid with at least Gaussian K=2 and K=3.
- [ ] Executes walk-forward OOS evaluation.
- [ ] Applies hard validation gates and selects an engine champion.
- [ ] Registers the winning model in local MLflow and assigns `engine-champion`.
- [ ] Publishes immutable `walk_forward_oos` predictions.
- [ ] Exercises fixed-model batch API and verifies its mode is `fixed_model_replay`.
- [ ] Exercises latest API and validates `RegimePredictionV1`.
- [ ] Exercises OOS prediction retrieval API and confirms it returns the stored walk-forward build.
- [ ] Re-running with the same seeds/input produces identical prediction values/lineage IDs where deterministic by contract.

## PR-036 — Final operator and consumer documentation

- **Status:** BLOCKED by PR-032, PR-033, PR-034, PR-035
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/036-operator-consumer-docs`
- **Depends on:** PR-032, PR-033, PR-034, PR-035
- **Allowed files:** `README.md`, `API.md`, `OPERATIONS.md`, `docs/consumer_contract.md`, `docs/integrations/portfell.md`

### Acceptance criteria

- [ ] README shows full `loader -> engine -> consumer` architecture.
- [ ] README gives clean-checkout `.venv` bootstrap instructions for Python 3.14.7.
- [ ] API doc covers health, latest, fixed-model batch, and OOS prediction retrieval endpoints.
- [ ] API doc clearly warns that `fixed_model_replay` is not leak-free historical OOS evidence.
- [ ] Operations doc covers train -> evaluate -> select -> register -> alias -> serve lifecycle.
- [ ] Consumer contract documents `RegimePrediction.v1` and immutable OOS prediction build fields.
- [ ] Portfell integration doc states Portfell owns ETF universe, regime-conditioned returns/covariances, portfolio optimization, backtesting, transaction costs, and application-level model choice.
- [ ] Documentation explains future consumers can reuse the engine by adding a new profile/feature producer and consuming the same API; no engine fork is required.

---

# Wave 6 — Optional model challengers after MVP

These PRs are intentionally isolated behind the model-adapter protocol. They are **not required to ship the first complete Gaussian-HMM platform** and may run in parallel after PR-010/PR-022. They must not modify the Gaussian-HMM implementation.

## PR-037 — Add Student-t HMM challenger adapter

- **Status:** OPTIONAL / BLOCKED by PR-010, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/037-student-t-hmm-challenger`
- **Depends on:** PR-010, PR-022
- **Allowed files:** `src/market_regime_engine/models/student_t_hmm.py`, `tests/unit/models/test_student_t_hmm.py`, `docs/models/student_t_hmm.md`, `pyproject.toml`

### Acceptance criteria

- [ ] Adapter satisfies the same model protocol and artifact contract as Gaussian HMM.
- [ ] Any additional dependency supports Python 3.14 before being added.
- [ ] Degrees-of-freedom and covariance configuration are explicit and validated.
- [ ] Filtered inference is causal and compatible with persistent state alignment.
- [ ] Candidate can participate in the existing walk-forward/grid/selection pipeline without special-case orchestration.
- [ ] Deterministic synthetic heavy-tail test demonstrates fit/inference end-to-end.

## PR-038 — Add duration-aware HSMM challenger adapter

- **Status:** OPTIONAL / BLOCKED by PR-010, PR-022
- **Git status:** PLANNED — clean before/after.
- **Branch:** `pr/038-hsmm-challenger`
- **Depends on:** PR-010, PR-022
- **Allowed files:** `src/market_regime_engine/models/hsmm.py`, `tests/unit/models/test_hsmm.py`, `docs/models/hsmm.md`, `pyproject.toml`

### Acceptance criteria

- [ ] Adapter satisfies the common model protocol or documents the smallest protocol extension required for explicit duration distributions.
- [ ] Any new dependency supports Python 3.14 before being added.
- [ ] Duration distribution parameters are explicit, validated, serialized, and versioned.
- [ ] Inference path is causal for production/OOS use.
- [ ] Candidate participates in the same walk-forward/grid/selection interface without consumer changes.
- [ ] Synthetic persistent-regime test demonstrates that duration metadata and predictions survive artifact round-trip.

---

# Parallel execution plan

The orchestrator should start only branches whose prerequisites are already merged into `main`. Within that restriction, maximize concurrency as follows:

```text
Wave 0A:
  PR-001

Wave 0B (parallel):
  PR-002  PR-003

Wave 0C:
  PR-004, then execute/verify governance script

Wave 1A (parallel):
  PR-005  PR-006

Wave 1B (parallel after PR-006):
  PR-007  PR-008  PR-009  PR-010  PR-011  PR-012  PR-013

Wave 2A (parallel where dependencies allow):
  PR-014  PR-020  PR-021

Wave 2B (parallel after PR-014):
  PR-015  PR-016  PR-018  PR-019

Wave 2C:
  PR-017

Wave 3A:
  PR-022

Wave 3B (parallel):
  PR-023  PR-027

Wave 3C:
  PR-024

Wave 3D:
  PR-025

Wave 3E:
  PR-026

Wave 4A (parallel):
  PR-028  PR-029  PR-030

Wave 4B:
  PR-031

Wave 5A (parallel):
  PR-032  PR-033  PR-034  PR-035

Wave 5B:
  PR-036

Optional challengers after common model protocol/evaluation are stable:
  PR-037  PR-038 in parallel
```

## Weak-agent execution rules

To keep each task safe for weak parallel agents:

1. An agent receives exactly one PR section from this backlog.
2. The agent must not broaden scope or refactor unrelated code.
3. The agent may edit only the files listed under `Allowed files`.
4. If an allowed file is missing because a dependency has not merged, the agent stops instead of recreating another PR's work.
5. Every acceptance checkbox is mandatory; there are no implied tasks outside the checklist.
6. Tests belong in the same PR as the behavior they verify.
7. No agent changes dependency versions unless its PR explicitly allows `pyproject.toml`.
8. No agent edits `BACKLOG.md`.
9. Every PR is opened against `main` and, after repository governance is active, auto-merge is enabled immediately.
10. The orchestrator never manually merges a normal implementation PR; `merge-gate` must succeed and GitHub auto-merge completes the squash merge.

## Definition of complete MVP

The MVP is complete after PR-036 when all of the following are true:

- a clean checkout creates a local Python 3.14.7 `.venv` reproducibly;
- `main` is protected and direct/force pushes are blocked;
- push and merge gates run lint/type/unit/integration jobs in parallel;
- PRs auto-complete only after `merge-gate` succeeds;
- a reusable versioned model profile can be loaded;
- Parquet feature data can be consumed without importing upstream loader code;
- Gaussian HMM K=2/K=3/K=4 and diagonal/full candidates can be compared;
- multi-start fitting, persistent state alignment, causal filtering, and walk-forward OOS evaluation are implemented;
- MLflow tracks candidate runs and stores registered packaged models;
- engine champion and consumer-specific aliases are supported;
- immutable `walk_forward_oos` prediction builds exist;
- fixed-model batch inference is separately labeled from historical OOS predictions;
- latest/realtime inference returns `RegimePrediction.v1`;
- Portfell and future projects can consume the API without importing HMM implementation code;
- the complete deterministic local end-to-end integration proof passes in the required merge gate.
