# Regime Engine — Canonical Backlog

Status date: 2026-09-16

This file is the **single authoritative backlog** for `regime-engine`.
Open and acceptance-pending work is kept at the top. Completed implementation and
historical acceptance work is condensed at the bottom. Separate backlog supplements
must not be created; new findings belong here.

Planning PR IDs such as `PR-449` are repository planning identities used in branch and
commit names. They do not need to equal the numeric GitHub pull-request number.

## Open / active work

### PR-448 — Consolidate repository audit planning into the canonical backlog

**Status:** ACTIVE — planning/documentation only

**GitHub:** #439

**Purpose:** make `BACKLOG.md` the only backlog, incorporate the focused repository
audit findings, keep active work first, and move completed history to the bottom.

#### Acceptance

- [ ] `BACKLOG.md` is the only backlog file in the repository.
- [ ] `BACKLOG_AUDIT_2026-09-16.md` is removed after its open items are incorporated.
- [ ] All acceptance-open work is listed before completed history.
- [ ] Finished work is condensed and kept at the bottom.
- [ ] No runtime, statistical, MLflow, PostgreSQL, registry, or serving behavior changes.

---

### PR-232 — Full current-Xetra computation and independent audit evidence

**Status:** ACCEPTANCE OPEN / EXTERNALLY BLOCKED

Implementation and local audit hardening are merged, including strict complete-dossier
checks and source/model identity cross-binding. Remaining work is external evidence from
a fresh current-Xetra computation.

#### Current blocker

The 2026-09-16 authorized full-evaluation attempt failed closed at source/model-clock
preflight before PCA/HMM/plotting/MLflow work:

- `potentially_valid_outer_folds=0/246`
- `first_inner_train_complete_observations=0`
- current source build: `20260915T212921Z`
- 169 raw features
- four `fed_*` columns have zero non-null observations across 16,768 rows
- five long-horizon `estr_*` columns have only 4.6–79.8% coverage in the latest
  1,260-row window

Canonical Xetra v4 requires the complete raw-plus-eight-component PCA source universe.
No fallback or imputation is permitted. The upstream `macro_loader` serving dataset must
be republished with a production-eligible raw universe before rerun.

#### Remaining acceptance

- [ ] Run the full current-Xetra computation on a production-eligible source build.
- [ ] Produce the complete independent math-audit dossier for every required fold.
- [ ] Prove source rows, feature identity, model identity, PCA artifacts and audit
  identities reconcile exactly.
- [ ] Record zero numerical audit errors under the current contract.
- [ ] Do not accept historical/raw-only runs as current evidence.

---

### PR-250 — External MLflow Model Metrics completeness proof

**Status:** ACCEPTANCE OPEN / EXTERNALLY BLOCKED

The verifier, exact metric-catalog checks, IEEE-754 value identity, artifact
hash/size/mtime/freshness checks, and local retry QA are merged.

#### Current blocker

The NAS MLflow service is healthy, but experiment `macro-regime-evaluation` contains
768 historical runs. It currently exposes zero visible LoggedModels and zero registered
model versions, while deleted-LoggedModel inventory is unavailable through the HTTP
RestStore. The strict namespace preflight therefore fails closed.

#### Remaining acceptance

- [ ] Resolve the historical 768-run namespace according to the explicit operator
  decision; do not silently delete production data.
- [ ] Run a fresh full evaluation after PR-232 source eligibility is satisfied.
- [ ] Verify complete Model Metrics projection against the current metric-catalog
  version and exact value identity.
- [ ] Verify all required plot families and artifact hashes/freshness against the same
  current evaluation identity.
- [ ] Verify the final registered model/version/alias readback only after authorized
  publication.

---

## Open K-slot production acceptance

The four-slot K=2..5 implementation is merged. The items below track the remaining
production/integration evidence and must remain independent rather than being collapsed
into one large PR.

### PR-423 — Four-slot outer validation production-callback integration

**Status:** IN PROGRESS — implementation merged in #418

#### Acceptance

- [ ] Exercise all K=2,3,4,5 slots through the production outer-validation callback
  path, not only isolated/hermetic adapters.
- [ ] Preserve per-K eligibility gates and deterministic process assembly.
- [ ] Prove serial/process parity for the production callback integration.
- [ ] Prove one ineligible K cannot contaminate another K slot.

### PR-424 — Per-K deployment/refit production-artifact integration

**Status:** IN PROGRESS — implementation merged in #418

#### Acceptance

- [ ] Exercise real per-K deployment selection through final refit and production
  package assembly.
- [ ] Bind source build, deployment cutoff, feature tuple, K and model family exactly.
- [ ] Prove deployment reruns selection on the deployment TRAIN clock rather than
  copying the last outer-fold configuration.
- [ ] Prove package round-trip identity for every eligible K slot.

### PR-425 — K-slot registry production matrix

**Status:** IN PROGRESS — implementation merged in #418

#### Acceptance

- [ ] Verify the complete production registry matrix for all K-slot aliases.
- [ ] Preserve immutable registration and audited compare-and-swap promotion behavior.
- [ ] Prove failed publication/promotion cannot mutate existing aliases.
- [ ] Preserve the default legacy `champion` alias contract where required.

### PR-426 — Full K-slot Model Metrics / plot artifact projection matrix

**Status:** IN PROGRESS — implementation merged in #418

#### Acceptance

- [ ] Project the full artifact/metric/plot matrix for every eligible K slot.
- [ ] Fail closed on incomplete lineage or missing required projection domains.
- [ ] Preserve serial/process completion-order canonical parity.
- [ ] Represent unavailable K slots explicitly and prove exact alias absence.

### PR-427 — Final independent mathematical acceptance closure

**Status:** IN PROGRESS — implementation merged in #418

#### Acceptance

- [ ] Close the independent stdlib math-oracle matrix for K=2..5.
- [ ] Cover prefixes, ties, state-label invariance, adversarial inputs and provenance
  mutations.
- [ ] Prove oracle results are independent of production implementation helpers.
- [ ] Record canonical serial/process parity for the final acceptance matrix.

### PR-428 — External K-slot registry/CAS durability

**Status:** IN PROGRESS — implementation merged in #418

#### Acceptance

- [ ] Verify external production durability across process kill/retry boundaries.
- [ ] Verify concurrent promotion races preserve compare-and-swap semantics.
- [ ] Prove no partial alias/registry state after interrupted publication.
- [ ] Keep the test independent from unrelated full-evaluation acceptance.

### PR-429 — Production-lineage four-slot E2E closure

**Status:** IN PROGRESS — implementation merged in #418

#### Acceptance

- [ ] Run the real Gaussian/GMM-HMM/Student-t K=2..5 four-slot path against current
  production lineage.
- [ ] Preserve process/serial and independent-process hash parity.
- [ ] Produce deployment packages, metrics and plots for eligible slots.
- [ ] Prove ineligible-slot fail-closed behavior and future-row invariance.
- [ ] Preserve the default `champion` alias contract.

### PR-430 — Authorized external publication/readback evidence

**Status:** IN PROGRESS / EXTERNALLY BLOCKED

The read-only verifier is implemented. Current NAS MLflow correctly reports that
Registered Model `regime-xetra` does not exist, so no publication or alias mutation was
attempted.

#### Acceptance

- [ ] Complete PR-232/PR-250 source/audit/namespace prerequisites first.
- [ ] Perform authorized publication only after prerequisites are green.
- [ ] Read back registered model, versions, aliases, lineage and package identity from
  the external NAS MLflow service.
- [ ] Prove publication is idempotent and failed readback/promotion leaves prior aliases
  unchanged.

---

## Open repository-audit correctness tranche

A focused review of current `main` found five correctness/contract gaps. Each
implementation PR is deliberately small and has a separate QA follow-up.

### PR-449 — Make 90% coverage the single CI authority

**Type:** implementation / CI correctness

**Depends on:** PR-448

**Scope:** `.github/workflows/merge-gate.yml`, `.github/workflows/push-gate.yml`, and the
minimum existing CI-contract expectation needed to keep the branch testable.

#### Acceptance

- [ ] `tool.coverage.report.fail_under = 90` remains the canonical threshold in
  `pyproject.toml`.
- [ ] Neither merge nor push workflow contains a lower command-line coverage override.
- [ ] The unit lane fails below 90.0% and passes at/above 90.0%, subject only to
  Coverage.py precision behavior.
- [ ] Merge and push use equivalent coverage commands and data-file handling.
- [ ] Existing multiprocessing coverage combine behavior is preserved.
- [ ] No cross-job coverage artifact upload/download is introduced.
- [ ] No test-selection behavior changes in this PR.
- [ ] Ruff, format, strict mypy and unit tests pass.

**Non-goal:** adding integration gating belongs to PR-451.

### PR-450 — QA: enforce the 90% coverage contract against regression

**Type:** QA only

**Depends on:** PR-449

#### Acceptance

- [ ] QA reads `pyproject.toml` and proves the configured threshold is exactly 90.
- [ ] QA rejects any explicit merge/push `--fail-under` below 90.
- [ ] Mutation to 89 or 80 fails the QA assertion.
- [ ] Merge/push unit-coverage commands are proven equivalent.
- [ ] QA continues to reject cross-job coverage artifact plumbing.
- [ ] QA is hermetic and requires no GitHub-hosted service or credential.

### PR-451 — Add hermetic integration lanes to merge and push gates

**Type:** implementation / CI completeness

**Depends on:** PR-449

#### Acceptance

- [ ] Both workflows contain a dedicated `integration` job running in parallel with
  `lint`, `type` and `unit`.
- [ ] Selector is exactly:
  `pytest -n auto tests -m "integration and not slow and not external"`.
- [ ] The lane uses Python 3.14.7 and the same locked dependency bootstrap pattern.
- [ ] Terminal merge/push gates require `lint`, `type`, `unit`, and `integration`.
- [ ] Failed/cancelled integration makes the terminal gate fail.
- [ ] Slow, external and unmarked E2E tests are not pulled into this lane.
- [ ] No NAS PostgreSQL/MLflow/network credential/alias mutation is used.

### PR-452 — QA: prove integration gating is mandatory and hermetic

**Type:** QA only

**Depends on:** PR-451

#### Acceptance

- [ ] Static QA proves both workflows declare `integration`.
- [ ] Static QA proves terminal gates depend on and inspect `integration`.
- [ ] Selector must include `integration` and exclude `slow` and `external`.
- [ ] Removing `integration` from either terminal gate makes QA fail.
- [ ] Replacing the selector with bare `integration` makes QA fail.
- [ ] The selected integration suite completes without network access or secrets.

### PR-453 — Separate statistical invalidity from unexpected software failures

**Type:** implementation / correctness

**Depends on:** PR-448

**Scope:** fold/prefix/K orchestration boundaries only as needed for one shared failure
contract.

#### Acceptance

- [ ] Introduce one explicit recoverable evaluation-invalidity base exception with a
  stable documented meaning: computation is functioning, but a statistical/data gate
  makes the unit unusable.
- [ ] Only that recoverable family may become invalid fold/prefix/K evidence.
- [ ] Generic `Exception`, `RuntimeError`, `KeyError`, `AssertionError`, `TypeError`, and
  unrelated `ValueError` failures must re-raise/abort.
- [ ] Expected gate failures are translated to the typed recoverable exception at their
  owning boundary.
- [ ] Process workers preserve failure classification in the parent.
- [ ] Unexpected failures can never merely reduce `valid_fold_rate`.
- [ ] Stored statistical failure messages remain deterministic and contain no traceback,
  object address, credential or raw vector.
- [ ] Statistical thresholds/ranking/seed policy remain unchanged.
- [ ] No blanket `except Exception` may produce statistical eligibility evidence.

### PR-454 — QA: adversarial failure-classification matrix

**Type:** QA only

**Depends on:** PR-453

#### Acceptance

- [ ] Typed recoverable invalidity becomes stable invalid evidence at fold, prefix and K
  boundaries.
- [ ] `RuntimeError`, `KeyError`, `AssertionError`, `TypeError`, and unrelated
  `ValueError` escape the evidence boundary.
- [ ] One unexpected fold failure cannot be hidden by enough successful folds to pass
  0.80 validity.
- [ ] Serial and process-backed execution classify injected failures identically.
- [ ] Recoverable invalid evidence is completion-order independent.
- [ ] `KeyboardInterrupt` and `SystemExit` propagate.

### PR-455 — Make the Xetra v4 profile the sole PCA-threshold authority

**Type:** implementation / identity correctness

**Depends on:** PR-448

#### Acceptance

- [ ] Every canonical Xetra v4 evaluation path reads the threshold exclusively from
  `profile.pca.variance_threshold`.
- [ ] Final production refit uses the same profile value exclusively.
- [ ] Canonical signatures remove redundant override arguments, or reject a differing
  override before computation; silent override is forbidden.
- [ ] Process payloads do not carry a second independently mutable canonical threshold
  when the profile is already present.
- [ ] The same `profile_hash` cannot produce PCA threshold 0.95, 0.80 or any value other
  than the pinned 0.90.
- [ ] Low-level generic PCA primitives may remain parameterized for isolated math tests.
- [ ] Component count remains pinned to eight.
- [ ] Valid 0.90 outputs remain numerically unchanged apart from redundant parameter
  removal.

### PR-456 — QA: profile/PCA identity invariance

**Type:** QA only

**Depends on:** PR-455

#### Acceptance

- [ ] Routing 0.95 through every canonical entry point is impossible or rejected before
  PCA fit starts.
- [ ] A spy proves canonical callers always pass `profile.pca.variance_threshold`.
- [ ] Evaluation, deployment selection and final refit use the same effective threshold.
- [ ] Serial and process paths use identical PCA configuration.
- [ ] Profile identity cannot be bypassed by an operational argument.
- [ ] Reintroducing an independently mutable threshold into a canonical task payload
  fails QA.

### PR-457 — Align prefix candidate `valid` evidence with the 0.80 hard gate

**Type:** implementation / evidence semantics

**Depends on:** PR-448

#### Acceptance

- [ ] `CandidateEvaluation.valid` means the candidate passes the same hard eligibility
  gates used by same-feature ranking.
- [ ] Gate logic is single-sourced; prefix code does not implement a second 0.80 rule.
- [ ] Zero-valid-fold candidates are invalid with the precise zero-support reason.
- [ ] Finite-metric candidates below 0.80 are invalid with the precise rate-gate reason.
- [ ] Exactly 0.80 is eligible when all other required metrics are finite.
- [ ] Rejected candidates retain finite diagnostics where available.
- [ ] Missing/non-finite required ranking metrics expose rejection reasons matching the
  ranking kernel.
- [ ] Prefix winner policy remains unchanged; this PR fixes evidence semantics only.

### PR-458 — QA: candidate evidence/ranking gate equivalence

**Type:** QA only

**Depends on:** PR-457

#### Acceptance

- [ ] Cover `0/N`, `1/N`, `3/4`, `4/5`, and `N/N` fold-validity cases.
- [ ] Prove `3/4` is rejected and `4/5` is accepted under the 0.80 rule.
- [ ] For every fixture, evidence `valid` equals hard-gate acceptance before ranking.
- [ ] Below-gate candidates retain finite diagnostics where present.
- [ ] Zero-support and non-finite-metric cases have distinct deterministic reasons.
- [ ] Randomized completion order produces identical candidate-evidence bytes.
- [ ] Restoring `valid_fold_count > 0` causes QA failure.

### Audit tranche execution order

```text
PR-449 -> PR-450
PR-449 -> PR-451 -> PR-452
PR-453 -> PR-454
PR-455 -> PR-456
PR-457 -> PR-458
```

The implementation tracks are otherwise independent. Every branch must start from the
then-current `origin/main` and be rebased immediately before opening/updating its GitHub
PR. QA follow-ups must be rebased after their implementation dependency merges.

---

## Current architectural decisions and non-goals

- Only **Xetra v4** is active. Legacy v1-v3 evaluation/package/serving compatibility is
  retired; Git history is the archive.
- PCA is mandatory for canonical v4. The feature universe is raw plus eight generated
  PCA components. There is no raw-only fallback.
- The current source/PCA completeness failure is a source-contract/architecture issue,
  not silently reclassified as a code defect. No imputation is permitted merely to make
  a run execute.
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

- GitHub #277, #284 and #317 were closed without merge and are superseded by later
  merged work.
- Draft planning IDs PR-186–PR-206 are superseded and must not be implemented.
- Historical requirements for v1-v3 compatibility or full-run computation-position
  resume are superseded by the architectural decisions above.
