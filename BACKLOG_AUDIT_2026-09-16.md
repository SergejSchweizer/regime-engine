# Regime Engine — Repository Audit Backlog Supplement

Status date: 2026-09-16

Reference base: `origin/main` at `6291fd907cae87a5ceedf2c6b2ca2f06fdbcffb3`.

This supplement records defects and contract gaps found by a focused review of the
current `main` branch. It is intentionally planning-only: PR-448 changes no runtime
behavior. Every implementation item below is small and atomic and has a separate QA
follow-up that must validate the implementation independently.

## Audit findings

The review found five concrete gaps worth scheduling:

1. `pyproject.toml` pins coverage at 90%, but both GitHub gate workflows override it
   with `--fail-under=80`, and `tests/unit/test_ci_contract.py` currently locks that
   lower threshold in as expected behavior.
2. Merge and push gates run lint, type and unit lanes, but omit the repository's
   hermetic `integration` test lane even though those tests are explicitly marked and
   the local pre-commit contract already runs `integration and not slow and not external`.
3. Evaluation boundaries can downgrade broad Python exceptions into statistical
   invalidity. In particular, fold/prefix/K orchestration can turn implementation
   defects into invalid-fold or ineligible-model evidence instead of aborting the run.
4. Xetra v4 pins PCA variance threshold to `0.90` in the hashed model profile, while
   canonical evaluation/refit orchestration still exposes caller-supplied
   `pca_variance_threshold` overrides. A caller can therefore request computation that
   is numerically inconsistent with the profile identity unless every call site is
   disciplined manually.
5. Prefix candidate evidence currently sets `CandidateEvaluation.valid` from
   `valid_fold_count > 0`, while the actual statistical candidate gate requires a
   valid-fold rate of at least `0.80`. Diagnostic evidence can therefore label a
   candidate "valid" even when the ranking kernel rejects it.

The source/PCA completeness failure reported by the current live dataset is not treated
as a code bug here: the repository currently fails closed by design. Any change to that
statistical source contract requires a separate architecture decision and is outside this
audit tranche.

---

## PR-449 — Make 90% coverage the single CI authority

**Type:** implementation / CI correctness

**Depends on:** PR-448 only

**Scope:** `.github/workflows/merge-gate.yml`, `.github/workflows/push-gate.yml`, and
only the minimum existing CI-contract expectation needed to keep the branch testable.
Do not change test selection, dependency pins, artifact handling or branch protection.

### Acceptance

- [ ] `tool.coverage.report.fail_under = 90` remains the canonical threshold in
  `pyproject.toml`.
- [ ] Neither merge nor push workflow contains a lower command-line coverage override.
- [ ] The unit lane fails when measured coverage is below 90.0% and passes at or above
  90.0%, subject only to Coverage.py's documented precision behavior.
- [ ] Merge and push workflows use the same coverage command and the same data-file
  handling.
- [ ] Existing multiprocessing coverage combine behavior is preserved.
- [ ] No cross-job coverage artifact upload/download is introduced.
- [ ] No integration, slow or external test-selection behavior changes in this PR.
- [ ] Ruff, format, strict mypy and the existing unit suite pass.

### Non-goals

- Adding the missing integration lane is PR-451.
- Raising the threshold above 90% is out of scope.

---

## PR-450 — QA: enforce the 90% coverage contract against regression

**Type:** QA only

**Depends on:** PR-449

**Scope:** CI-contract tests only. Do not modify workflow behavior in this PR.

### Acceptance

- [ ] A test reads `pyproject.toml` and proves the configured threshold is exactly 90.
- [ ] A test inspects both gate workflows and rejects any explicit `--fail-under` value
  below 90.
- [ ] A mutation fixture/string mutation changing either workflow to 89 or 80 makes the
  QA assertion fail.
- [ ] The test proves merge and push workflows use equivalent unit-coverage commands.
- [ ] The test continues to reject cross-job coverage artifact plumbing.
- [ ] The QA does not depend on GitHub-hosted services or external credentials.

---

## PR-451 — Add hermetic integration lanes to merge and push gates

**Type:** implementation / CI completeness

**Depends on:** PR-449

**Scope:** `.github/workflows/merge-gate.yml` and `.github/workflows/push-gate.yml`.
Use the existing pinned environment/bootstrap pattern; do not add external-service tests.

### Acceptance

- [ ] Both workflows contain a dedicated `integration` job that runs in parallel with
  `lint`, `type` and `unit`.
- [ ] The integration selector is exactly the hermetic scope:
  `pytest -n auto tests -m "integration and not slow and not external"`.
- [ ] The integration lane uses Python 3.14.7 and the exact locked dependency install
  pattern used by the other lanes.
- [ ] The terminal `merge-gate` job requires `lint`, `type`, `unit` and `integration`.
- [ ] The terminal `push-gate` job requires `lint`, `type`, `unit` and `integration`.
- [ ] A failed or cancelled integration lane makes the terminal gate fail.
- [ ] `slow`, `external`, and unmarked E2E tests are not accidentally pulled into the
  GitHub integration lane.
- [ ] No NAS PostgreSQL, MLflow, network credential, production alias or external
  mutation is used.
- [ ] Existing lint/type/unit lane behavior remains unchanged apart from the PR-449
  coverage correction.

---

## PR-452 — QA: prove integration gating is mandatory and hermetic

**Type:** QA only

**Depends on:** PR-451

**Scope:** CI contract tests and, if necessary, one tiny hermetic fixture proving marker
selection. Do not change production code.

### Acceptance

- [ ] Static QA proves both workflows declare an `integration` job.
- [ ] Static QA proves both terminal gates list `integration` in `needs` and inspect its
  result before succeeding.
- [ ] Static QA proves the selector includes `integration`, excludes `slow`, and excludes
  `external`.
- [ ] Removing `integration` from either terminal gate dependency list makes the QA fail.
- [ ] Replacing the selector with `integration` alone makes the QA fail because slow or
  external tests would become eligible.
- [ ] The hermetic integration selector completes without requiring network access or
  secrets in the QA environment.

---

## PR-453 — Separate statistical invalidity from unexpected software failures

**Type:** implementation / correctness

**Depends on:** PR-448

**Primary files:** `src/market_regime_engine/evaluation/`,
`src/market_regime_engine/feature_discovery/prefix_search.py`, and
`src/market_regime_engine/evaluations/k_feature_selection.py` only as required for the
shared failure contract.

### Problem

Current orchestration has downgrade boundaries that catch broad exceptions and convert
them into `invalid` fold/prefix/K evidence. That can allow a programming defect to be
counted as ordinary statistical invalidity and, when enough other folds survive, still
participate in an 80% valid-fold decision.

### Acceptance

- [ ] Introduce one explicit recoverable evaluation-invalidity base exception with a
  stable, documented meaning: the computation is functioning correctly but a statistical
  or data eligibility gate makes this unit unusable.
- [ ] Only that recoverable exception family may be converted into invalid fold, invalid
  prefix, or ineligible K evidence at orchestration boundaries.
- [ ] Generic `Exception`, `RuntimeError`, `KeyError`, `AssertionError`, `TypeError`, and
  unrelated `ValueError` failures are not silently downgraded; they abort/re-raise.
- [ ] Existing expected gate failures (insufficient retained observations, explicit model
  eligibility/gate failure, alignment ambiguity when defined as statistical invalidity,
  and comparable documented domain failures) are converted to the typed recoverable
  exception at their owning boundary.
- [ ] Process-worker exceptions preserve their classification when surfaced in the
  parent process.
- [ ] Unexpected exceptions can never reduce `valid_fold_rate`; the evaluation terminates
  instead of recording an invalid fold.
- [ ] Failure messages stored as statistical evidence remain deterministic and do not
  contain traceback text, object addresses, credentials or raw vectors.
- [ ] Statistical thresholds, ranking order, seed policy and valid results are unchanged.
- [ ] No blanket `except Exception` remains at a boundary whose output is statistical
  eligibility evidence, except where it immediately re-raises without reclassification.

---

## PR-454 — QA: adversarial failure-classification matrix

**Type:** QA only

**Depends on:** PR-453

**Scope:** focused unit/integration tests for failure boundaries; no production behavior
changes.

### Acceptance

- [ ] Inject the explicit recoverable invalidity into a fold and prove it becomes one
  invalid fold with the expected stable reason.
- [ ] Inject the same recoverable invalidity into prefix selection and fixed-K selection
  and prove it becomes invalid/ineligible evidence, not a crash.
- [ ] Inject `RuntimeError`, `KeyError`, `AssertionError`, `TypeError` and an unrelated
  `ValueError`; each must escape the statistical evidence boundary.
- [ ] Prove an unexpected failure in one fold cannot be hidden by enough successful folds
  to satisfy the 0.80 gate.
- [ ] Prove serial and process-backed execution classify the same injected failure
  identically.
- [ ] Prove recoverable invalid evidence is canonical and completion-order independent.
- [ ] `KeyboardInterrupt` and `SystemExit` remain non-recoverable and propagate.

---

## PR-455 — Make the Xetra v4 profile the sole PCA-threshold authority

**Type:** implementation / identity correctness

**Depends on:** PR-448

**Scope:** canonical Xetra v4 orchestration only. Generic PCA primitives may retain a
threshold argument for isolated mathematical tests, but production/evaluation orchestration
must not permit a profile-inconsistent override.

### Problem

`PCAConfig` pins `variance_threshold == 0.90` and participates in `profile_hash`, while
walk-forward/candidate/prefix/fixed-K/final-refit orchestration still transports optional
caller-supplied PCA thresholds. This creates a path where numerical behavior can diverge
from the hashed profile identity.

### Acceptance

- [ ] Every canonical Xetra v4 evaluation path obtains the threshold exclusively from
  `profile.pca.variance_threshold`.
- [ ] Final production refit obtains the threshold exclusively from the same profile.
- [ ] Public canonical orchestration signatures remove redundant override arguments, or
  fail before computation if an override differs from the profile; silent override is
  forbidden.
- [ ] Process-task payloads do not carry a second independently mutable canonical PCA
  threshold when the profile is already present.
- [ ] A model/evaluation claiming the same `profile_hash` cannot be produced with PCA
  threshold 0.95, 0.80 or any value other than the profile's 0.90.
- [ ] Generic low-level PCA fitting utilities remain testable without weakening the
  canonical Xetra v4 contract.
- [ ] Component count remains pinned to eight; this PR does not redesign PCA.
- [ ] Existing valid 0.90 evaluation/refit outputs remain numerically and canonically
  unchanged apart from removal of redundant operational parameters.

---

## PR-456 — QA: profile/PCA identity invariance

**Type:** QA only

**Depends on:** PR-455

**Scope:** profile, walk-forward, prefix/fixed-K orchestration and final-refit tests.

### Acceptance

- [ ] Attempting to route 0.95 through every canonical entry point is impossible by
  signature or rejected before PCA fit starts.
- [ ] A spy around the generic PCA primitive proves canonical callers always pass
  `profile.pca.variance_threshold`.
- [ ] Evaluation, deployment selection and final refit use the same effective threshold.
- [ ] Serial and process-backed canonical paths use identical PCA configuration.
- [ ] The profile hash changes if the PCA profile were hypothetically changed, and no
  operational argument can bypass that identity.
- [ ] A regression that reintroduces an independently mutable threshold into a canonical
  task payload causes QA failure.

---

## PR-457 — Align prefix candidate `valid` evidence with the 0.80 hard gate

**Type:** implementation / evidence semantics

**Depends on:** PR-448

**Primary files:** `src/market_regime_engine/evaluation/selection.py`,
`src/market_regime_engine/feature_discovery/prefix_search.py`, and shared contracts only
if needed to avoid duplicate gate logic.

### Problem

Prefix candidate summaries currently use `valid_fold_count > 0` for
`CandidateEvaluation.valid`, while the statistical ranking kernel rejects candidates with
`valid_fold_rate < 0.80`. The same candidate can therefore be reported as valid in prefix
evidence and rejected by the selector.

### Acceptance

- [ ] `CandidateEvaluation.valid` means the candidate passes the same hard candidate
  eligibility gates used by same-feature ranking.
- [ ] Gate logic is single-sourced; prefix code must not reimplement a second 0.80 rule.
- [ ] A zero-valid-fold candidate is invalid with the precise zero-support reason.
- [ ] A candidate with finite metrics but valid-fold rate below 0.80 is invalid with the
  precise rate-gate reason.
- [ ] A candidate exactly at 0.80 is eligible when all other required metrics are finite.
- [ ] Finite diagnostic metrics may remain present on a rejected candidate; invalidity
  must not erase useful diagnostics.
- [ ] Missing/non-finite required ranking metrics produce an invalid evidence reason that
  matches the ranking kernel's rejection semantics.
- [ ] Prefix winner selection itself remains driven by the existing ranking kernel and
  soft-NMI rules; this PR corrects evidence semantics, not ranking policy.
- [ ] Canonical hashes are allowed to change only where the corrected `valid`/reason
  evidence changes.

---

## PR-458 — QA: candidate evidence/ranking gate equivalence

**Type:** QA only

**Depends on:** PR-457

**Scope:** unit tests around candidate aggregation, ranking and prefix evidence.

### Acceptance

- [ ] Cover `0/N`, `1/N`, `3/4`, `4/5`, and `N/N` valid-fold cases.
- [ ] Prove `3/4` is rejected and `4/5` is accepted under the 0.80 rule.
- [ ] For every candidate fixture, `CandidateEvaluation.valid` equals whether the ranking
  hard-gate evidence accepts that candidate before tie/rank ordering.
- [ ] Below-gate candidates retain finite diagnostic metrics when those metrics exist.
- [ ] Zero-support and non-finite-metric cases expose distinct deterministic reasons.
- [ ] Randomized candidate completion order produces identical candidate-evidence bytes.
- [ ] A mutation restoring `valid_fold_count > 0` causes the QA matrix to fail.

---

## Execution order

The intended order is:

```text
PR-449 -> PR-450
PR-449 -> PR-451 -> PR-452
PR-453 -> PR-454
PR-455 -> PR-456
PR-457 -> PR-458
```

The five implementation tracks are otherwise independent and should not be bundled into
one code PR. Each implementation branch must be created from current `origin/main`, and
must be rebased onto the then-current `origin/main` immediately before opening/updating
its GitHub PR. QA follow-ups must be rebased after their implementation dependency merges.

## Audit evidence inspected

- `.github/workflows/merge-gate.yml`
- `.github/workflows/push-gate.yml`
- `pyproject.toml`
- `tests/unit/test_ci_contract.py`
- `src/market_regime_engine/evaluation/walk_forward.py`
- `src/market_regime_engine/feature_discovery/prefix_search.py`
- `src/market_regime_engine/evaluations/k_feature_selection.py`
- `src/market_regime_engine/profiles/config.py`
- `src/market_regime_engine/training/final_refit.py`
- `src/market_regime_engine/feature_discovery/contracts.py`
- `src/market_regime_engine/evaluation/selection.py`

No production source, NAS MLflow state, registry alias or external service was mutated by
this planning audit.
