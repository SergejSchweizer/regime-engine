# Regime Engine — Canonical Backlog

Status date: 2026-09-18

This file is the **single authoritative backlog** for `regime-engine`.
Open and acceptance-pending work is kept at the top. Completed implementation and
historical acceptance work is condensed at the bottom. Separate backlog supplements
must not be created; new findings belong here.

Planning PR IDs such as `PR-449` are repository planning identities used in branch and
commit names. They do not need to equal the numeric GitHub pull-request number.

## Open / active work

### Execution policy

Open work is intentionally ordered so that **runtime/statistical changes are implemented first**,
with small component-level QA PRs inserted immediately after risky changes, and only then is the
complete end-to-end acceptance run. Component QA may use deterministic fixtures and focused
integration tests; it must not be used as a substitute for the final complete-system proof.

Global rules for every planned PR:

- one PR owns one primary behavior or one QA responsibility;
- implementation and QA are separate whenever the behavior can be independently falsified;
- QA-only PRs must not introduce new production behavior;
- every PR starts from the then-current `origin/main` and is rebased immediately before opening
  or updating its GitHub PR;
- deterministic math must have exact fixtures, fixed seeds where randomness exists, and canonical
  result ordering;
- no outer-TEST information may enter TRAIN-side selection;
- no external PostgreSQL/MLflow mutation is allowed unless the PR is explicitly marked external;
- the full hermetic/system acceptance phase does not start until all implementation and focused QA
  PRs below it are green.

The intended order is:

```text
PR-448
  -> foundation corrections + focused QA
  -> medoid/L-elbow implementation + focused QA
  -> orchestration/observability/cutover + focused QA
  -> complete local/system acceptance
  -> current-source external acceptance
  -> external registry durability
  -> authorized publication/readback
```

---

### PR-448 — Consolidate and order the canonical backlog

**Status:** ACTIVE — planning/documentation only  
**GitHub:** #439

**Purpose:** keep `BACKLOG.md` as the single authoritative backlog, order all open work by
dependency, keep component QA close to the implementation it verifies, and move complete
end-to-end/external acceptance to the end.

#### Acceptance

- [ ] `BACKLOG.md` is the only backlog file in the repository.
- [ ] All active planning IDs appear exactly once in the execution order below.
- [ ] Every implementation PR has an explicit dependency and a bounded responsibility.
- [ ] Risky statistical/runtime changes have a separate focused QA PR.
- [ ] Complete-system and external acceptance PRs appear only after all implementation/cutover work.
- [ ] Superseded teacher-prefix work is kept only in historical traceability, not in active execution.
- [ ] No runtime, statistical, PostgreSQL, MLflow, registry or serving behavior changes in this PR.

---

## Phase 1 — Foundation corrections before the statistical redesign

These correctness changes are independent of the medoid/L-elbow algorithm but must be completed
before the new canonical path is integrated.

### PR-449 — Make 90% coverage the single CI authority

**Type:** implementation / CI correctness  
**Depends on:** PR-448

#### Acceptance

- [ ] `tool.coverage.report.fail_under = 90` remains the only canonical coverage threshold.
- [ ] Merge and push workflows contain no lower command-line override.
- [ ] Both workflows use equivalent coverage invocation and data-file handling.
- [ ] Existing multiprocessing coverage combine behavior is preserved.
- [ ] No cross-job coverage artifact plumbing is introduced.
- [ ] No test-selection behavior changes in this PR.
- [ ] A local threshold mutation below 90 fails before merge.
- [ ] Ruff, format, strict mypy and the existing unit lane pass.

### PR-450 — QA: regression-proof the 90% coverage contract

**Type:** QA only  
**Depends on:** PR-449

#### Acceptance

- [ ] Read `pyproject.toml` and prove the configured threshold is exactly 90.
- [ ] Reject any explicit merge/push `--fail-under` lower than 90.
- [ ] Mutations to 89 and 80 fail the QA test.
- [ ] Prove merge/push coverage commands are semantically equivalent.
- [ ] Prove no cross-job coverage artifact upload/download is required.
- [ ] QA is hermetic and requires no GitHub-hosted secret or network service.
- [ ] Production code is unchanged by this QA PR.

### PR-451 — Add hermetic integration lanes to merge and push gates

**Type:** implementation / CI completeness  
**Depends on:** PR-450

#### Acceptance

- [ ] Both workflows contain a dedicated `integration` job running in parallel with lint/type/unit.
- [ ] Selector is exactly `pytest -n auto tests -m "integration and not slow and not external"`.
- [ ] Python 3.14.7 and the repository-locked dependency bootstrap are used.
- [ ] Terminal gates require lint, type, unit and integration.
- [ ] Failed or cancelled integration makes the terminal gate fail.
- [ ] Slow, external and unrelated E2E tests are excluded.
- [ ] The lane uses no NAS PostgreSQL/MLflow access, credentials or alias mutation.
- [ ] Existing unit coverage behavior remains unchanged.

### PR-452 — QA: prove integration gating is mandatory and hermetic

**Type:** QA only  
**Depends on:** PR-451

#### Acceptance

- [ ] Static QA proves both workflows declare the integration job.
- [ ] Terminal gates must depend on and inspect integration status.
- [ ] The selector must include `integration` and exclude `slow` and `external`.
- [ ] Removing integration from either terminal gate fails QA.
- [ ] Replacing the selector with bare `integration` fails QA.
- [ ] The selected integration suite completes without network access or secrets.
- [ ] Production code is unchanged by this QA PR.

### PR-453 — Separate statistical invalidity from unexpected software failures

**Type:** implementation / correctness  
**Depends on:** PR-452

#### Acceptance

- [ ] Introduce one explicit recoverable evaluation-invalidity base exception.
- [ ] Only that typed family may become invalid fold/subset/K evidence.
- [ ] Generic `RuntimeError`, `KeyError`, `AssertionError`, `TypeError`, unrelated
  `ValueError`, `KeyboardInterrupt` and `SystemExit` cannot be converted into ordinary
  statistical invalidity.
- [ ] Expected statistical/data gates are translated at their owning boundary.
- [ ] Process workers preserve failure classification in the parent.
- [ ] Unexpected failures can never merely reduce a valid-fold rate.
- [ ] Stored recoverable-failure reasons are deterministic and contain no traceback, address,
  credential or raw vector.
- [ ] Thresholds, seeds and model-ranking semantics are otherwise unchanged.
- [ ] No blanket `except Exception` emits eligibility evidence.

### PR-454 — QA: adversarial failure-classification matrix

**Type:** QA only  
**Depends on:** PR-453

#### Acceptance

- [ ] Typed recoverable invalidity becomes stable invalid evidence at fold/subset/K boundaries.
- [ ] Injected `RuntimeError`, `KeyError`, `AssertionError`, `TypeError` and unrelated
  `ValueError` escape the evidence boundary.
- [ ] One unexpected fold failure cannot be hidden by enough successful folds to pass 0.80.
- [ ] Serial and process-backed execution classify the same injected failure identically.
- [ ] Recoverable-invalid evidence is completion-order independent.
- [ ] `KeyboardInterrupt` and `SystemExit` propagate.
- [ ] Production behavior is unchanged except for the intended failure classification.

### PR-455 — Make the Xetra profile the sole PCA-threshold authority

**Type:** implementation / identity correctness  
**Depends on:** PR-454

#### Acceptance

- [ ] Every canonical Xetra evaluation path reads the threshold exclusively from
  `profile.pca.variance_threshold`.
- [ ] Deployment selection and final refit use the identical profile value.
- [ ] Redundant override arguments are removed or rejected before computation.
- [ ] Process payloads do not carry a second independently mutable canonical threshold.
- [ ] The same `profile_hash` cannot produce a different effective threshold.
- [ ] Generic low-level PCA math may remain parameterized for isolated unit tests.
- [ ] Component count remains pinned to eight.
- [ ] Valid 0.90 outputs remain numerically unchanged apart from removal of redundant routing.

### PR-456 — QA: profile/PCA identity invariance

**Type:** QA only  
**Depends on:** PR-455

#### Acceptance

- [ ] Attempting to route 0.95 or 0.80 through every canonical entry point is impossible or fails
  before PCA fitting.
- [ ] A spy proves canonical callers use only the profile threshold.
- [ ] Evaluation, deployment selection and final refit resolve the same effective threshold.
- [ ] Serial and process paths use identical PCA configuration.
- [ ] Profile identity cannot be bypassed by an operational argument.
- [ ] Reintroducing an independently mutable threshold into a canonical task payload fails QA.
- [ ] Production code is otherwise unchanged by this QA PR.

---

## Phase 2 — Medoid-only feature universe and L-elbow algorithm

The new design removes the current
`medoid -> provisional teacher -> score all raw features -> cluster winner -> ranked prefix`
loop. After clustering, only medoids may enter HMM feature-count selection.

Canonical semantics:

```text
N     = quality-eligible feature count
M*    = silhouette-selected redundancy-cluster count
S_M   = exactly one deterministic medoid per cluster
L*    = medoid count selected by inner-OOS preservation elbow
S_L*  = one common final feature tuple used by K=2,3,4,5
K     = latent-state count, separate from L
```

### PR-459 — Define the medoid/L-elbow statistical contract

**Type:** architecture / contracts  
**Depends on:** PR-456

#### Acceptance

- [ ] Introduce a new explicit evaluation/profile version; do not silently change v4 meaning.
- [ ] Define `N`, `M*`, `S_M`, `L*`, `S_{L*}` and K separately.
- [ ] Preserve TRAIN-only quality filtering, absolute-Spearman distance, average-linkage hierarchy
  and silhouette-based `M*`.
- [ ] Define medoid = actual cluster member with minimum mean distance to the other members.
- [ ] Preserve the existing `1e-12` medoid tie tolerance and canonical ordinal tie-break.
- [ ] State that no non-medoid raw feature may re-enter selection after `S_M` is formed.
- [ ] Forbid provisional-teacher raw-feature scoring, `state_information_ratio`, eta-squared
  cluster-winner replacement and teacher-ranked-prefix selection in the new path.
- [ ] Require one common final feature tuple for K=2,3,4,5 under one selection identity.
- [ ] Preserve source/PCA/missing-value contracts and the current final model-family inventory.
- [ ] No production code path or external service is changed in this contract PR.

### PR-460 — QA: static contract consistency for the medoid/L-elbow profile

**Type:** QA only  
**Depends on:** PR-459

#### Acceptance

- [ ] Parse the new profile/evaluation contract and prove every required identity field is defined.
- [ ] Reject any contract in which non-medoids can re-enter after medoid selection.
- [ ] Reject any new-profile reference to raw-feature teacher scoring or teacher-ranked prefixes.
- [ ] Prove K and L are distinct quantities and the same `S_{L*}` is required for K2-K5.
- [ ] Prove the old v4 contract remains unchanged until cutover.
- [ ] Mutation of medoid tie policy, K-sharing rule or TRAIN-only boundary fails QA.
- [ ] QA introduces no production runtime behavior.

### PR-461 — Make medoids the sole post-clustering candidate universe

**Type:** implementation / feature selection  
**Depends on:** PR-460

#### Acceptance

- [ ] Consume the selected `M*` memberships and emit exactly one medoid per cluster.
- [ ] Persist cluster ID, complete member tuple, every member mean distance, selected medoid,
  canonical ordinal and deterministic medoid-set hash.
- [ ] Singleton cluster -> sole member.
- [ ] Non-singleton winner -> minimum mean distance to all other cluster members.
- [ ] Ties within `1e-12` -> smallest canonical ordinal.
- [ ] `S_M` order is canonical cluster order and immutable downstream.
- [ ] The new evaluation API exposes no replacement-member/raw-feature rescoring hook.
- [ ] Raw catalog and quality evidence remain auditable but cannot become HMM inputs after this
  boundary.
- [ ] Serial/process construction produces byte-identical medoid evidence.

### PR-462 — QA: independent medoid-selection oracle

**Type:** QA only  
**Depends on:** PR-461

#### Acceptance

- [ ] Independent test code recomputes every within-cluster mean distance without calling the
  production medoid selector.
- [ ] Cover singleton, two-member, multi-member and exact/near-`1e-12` tie clusters.
- [ ] Permuting input task completion order cannot change `S_M` or its hash.
- [ ] Mutating one distance entry changes the expected medoid when mathematically required.
- [ ] A spy proves no new-profile path calls raw-feature scoring/winner-selection modules.
- [ ] Duplicate/missing cluster membership fails closed.
- [ ] Serial and process-backed evidence bytes are identical.
- [ ] QA adds no runtime behavior.

### PR-463 — Build causal full-medoid HMM references for K=2..5

**Type:** implementation / inner walk-forward evidence  
**Depends on:** PR-462

#### Acceptance

- [ ] For each Outer-TRAIN fold, fit Gaussian full-covariance HMM K=2,3,4,5 on exactly `S_M`.
- [ ] All K values see the same feature order and exact same inner fold plan.
- [ ] Reuse deterministic multistart, covariance, occupancy and validity gates.
- [ ] Use only causal inner-TEST filtered probabilities as reference arrays.
- [ ] Smoothed probabilities and full-sample Viterbi labels are forbidden.
- [ ] Persist K, inner fold, timestamps, `S_M` hash, source identity, model identity and canonical
  posterior-evidence hash.
- [ ] Parameter-safety bounds are checked before fit; unsafe `M*` fails closed.
- [ ] No full-medoid HMM is allowed to rank or replace medoids.
- [ ] Serial/process execution yields the same canonical reference evidence.

### PR-464 — QA: full-medoid reference causality and identity

**Type:** QA only  
**Depends on:** PR-463

#### Acceptance

- [ ] Future inner-TEST rows cannot alter earlier filtered probabilities.
- [ ] A smoothed-posterior substitution fails QA.
- [ ] A Viterbi-label substitution fails QA.
- [ ] Feature-order mutation changes or invalidates the reference identity.
- [ ] K2, K3, K4 and K5 all bind the same `S_M` hash.
- [ ] Inner fold/timestamp mismatch fails closed.
- [ ] Real-HMM serial/process fixtures produce equivalent posterior evidence within the existing
  canonical numerical contract.
- [ ] QA adds no selection behavior.

### PR-465 — Implement greedy backward medoid elimination

**Type:** implementation / feature-count search  
**Depends on:** PR-464

For a current subset `S`, evaluate every one-medoid removal. A reduced subset is compared to
the **fixed full-medoid reference of the same K and inner fold**.

```text
nmi(K,f,S) =
    soft_NMI(full_medoid_posterior(K,f), reduced_posterior(K,f,S))

q_K(S) = median over valid inner folds f of nmi(K,f,S)
Q(S)   = min over K in {2,3,4,5} q_K(S)
```

#### Acceptance

- [ ] Start at `S_M` and produce one nested subset for every `L=M*,M*-1,...,2`.
- [ ] At each step test every possible single-medoid removal from the current subset.
- [ ] Unrestricted subset search and forward re-addition are forbidden.
- [ ] Every reduced candidate is fit for Gaussian K2-K5 on the same inner fold plan.
- [ ] Compare only same-K full vs reduced posteriors on exact shared timestamps.
- [ ] Reuse canonical soft NMI with finite/entropy/shared-support gates.
- [ ] Shared timestamp support must be at least 0.90.
- [ ] Every K must satisfy the canonical inner valid-fold-rate hard gate.
- [ ] `q_K` = median of valid fold-local NMI; `Q` = minimum across K2-K5.
- [ ] Define `Q(S_M)=1.0` by identity while retaining actual full-reference evidence separately.
- [ ] Remove the medoid whose resulting eligible subset has the largest `Q`.
- [ ] Tie within `1e-12` -> remove the largest canonical ordinal so lower ordinals remain stable.
- [ ] Persist every attempted removal, per-K/per-fold NMI, validity evidence, `q_K`, `Q`,
  chosen removal and resulting subset.
- [ ] Parallel task completion order cannot affect the selected nested path.

### PR-466 — QA: independent backward-elimination oracle

**Type:** QA only  
**Depends on:** PR-465

#### Acceptance

- [ ] Independent soft-NMI math recomputes all fold-local full-vs-reduced values in a golden path.
- [ ] Independent aggregation recomputes every `q_K` median and `Q=min_K(q_K)`.
- [ ] Mutation from minimum-K aggregation to mean/maximum fails QA.
- [ ] One K below the validity gate makes the candidate subset ineligible.
- [ ] State-label permutations leave NMI, `Q` and removal choice unchanged.
- [ ] Exact and near-`1e-12` removal ties exercise the canonical-ordinal rule.
- [ ] Randomized worker completion order yields identical path/evidence hashes.
- [ ] A moving-reference implementation fails QA; every reduced subset must compare to fixed
  `S_M` same-K references.
- [ ] QA adds no production behavior.

### PR-467 — Select L* with a deterministic preservation elbow

**Type:** implementation / model-dimension selection  
**Depends on:** PR-466

#### Acceptance

- [ ] Build raw `Q_L = Q(S_L)` for `L=2,...,M*`, with `Q_M=1.0`.
- [ ] Persist the raw curve unchanged for audit.
- [ ] Derive monotone decision curve `Qhat_L = max(Q_j for j <= L)`.
- [ ] If `M*=2`, select `L*=2`.
- [ ] If fewer than three distinct L candidates exist and no lower-L exact identity exists,
  conservatively select the full medoid count.
- [ ] If the smallest candidate has `Qhat_L=1.0` within `1e-12`, choose the smallest such L.
- [ ] Otherwise normalize endpoint coordinates to `x,y in [0,1]`.
- [ ] Compute interior elbow score `E_L=(y_L-x_L)/sqrt(2)`.
- [ ] Maximum `E_L` wins; ties within `1e-12` choose smaller L.
- [ ] Persist raw curve, monotone curve, normalized coordinates, elbow score, `L*` and exact
  `S_{L*}`.
- [ ] Outer TEST data cannot influence any elimination/elbow input.
- [ ] `S_{L*}` is one common tuple for K2-K5.

### PR-468 — QA: elbow boundary, noise and mutation matrix

**Type:** QA only  
**Depends on:** PR-467

#### Acceptance

- [ ] Golden fixtures cover `M*=2`, `M*=3`, clear elbow, flat curve and non-monotone raw curve.
- [ ] Independently recompute the monotone envelope and normalized elbow score.
- [ ] Exact lower-L identity selects the smallest equivalent L.
- [ ] Endpoint-only/insufficient-candidate cases follow the conservative rule exactly.
- [ ] Tie within `1e-12` chooses the smaller L.
- [ ] Mutation removing the monotone envelope fails QA.
- [ ] Mutation to a visual/manual or nondeterministic elbow fails QA.
- [ ] Future Outer-TEST mutation leaves `L*` byte-identical.
- [ ] QA adds no runtime behavior.

---

## Phase 3 — Integrate, expose evidence, then cut over

### PR-469 — Integrate medoid/L-elbow selection into validation, deployment and refit

**Type:** implementation / orchestration  
**Depends on:** PR-468

#### Acceptance

- [ ] New path is exactly quality -> Spearman -> clustering/`M*` -> medoids -> full-medoid
  K2-K5 references -> backward elimination -> elbow/`L*` -> common final tuple -> final grid ->
  Outer TEST.
- [ ] Old provisional teacher, raw-feature regime scoring, cluster-winner replacement and
  teacher-ranked-prefix selection are unreachable from the new path.
- [ ] Freeze `M*`, memberships, `S_M`, elimination path, `L*`, `S_{L*}` and final candidate
  configuration before Outer TEST.
- [ ] Final Gaussian/GMM-HMM/Student-t families all consume the same `S_{L*}`.
- [ ] No K or model family may rerun feature selection independently.
- [ ] Deployment selection reruns the complete TRAIN-only medoid/L-elbow procedure through the
  deployment cutoff; it never copies the last validation-fold tuple.
- [ ] Final refit/package identity binds source build, cutoffs, cluster/medoid evidence, `L*`,
  final tuple, K and family.
- [ ] Existing fold-local/model-version-local state identity semantics remain unchanged.
- [ ] Process and serial orchestration produce the same canonical selection identity.

### PR-470 — QA: orchestration boundary and future-data isolation

**Type:** QA only  
**Depends on:** PR-469

#### Acceptance

- [ ] Spy the complete new call graph and prove old teacher/scoring/winner/prefix modules are absent.
- [ ] Mutating Outer TEST rows cannot alter TRAIN-side `M*`, medoids, elimination path or `L*`.
- [ ] K2-K5 final slots bind one identical `S_{L*}` hash.
- [ ] Model-family changes cannot alter the selected feature tuple.
- [ ] Deployment rerun at a later cutoff is executed rather than last-fold reuse.
- [ ] Serial/process runs yield identical selection hashes.
- [ ] Injected recoverable vs unexpected failures follow PR-453 semantics.
- [ ] QA introduces no new production behavior.

### PR-471 — Project complete medoid/L-elbow evidence into MLflow and diagnostics

**Type:** implementation / observability  
**Depends on:** PR-470

#### Acceptance

- [ ] Log `N`, `M*`, silhouette curve, cluster memberships, medoids and mean-distance evidence.
- [ ] Log full-medoid K2-K5 reference identities without marking them as production champions.
- [ ] Log every backward-elimination candidate, selected removal, per-K `q_K`, aggregate `Q`
  and nested subset path.
- [ ] Add canonical L-preservation/elbow diagnostics with raw `Q_L`, monotone `Qhat_L`,
  `E_L` and selected `L*`.
- [ ] Log one common final feature tuple/hash and cross-check it across K2-K5.
- [ ] New-profile runs emit no state-information-ratio, eta-squared cluster-winner,
  provisional-teacher or teacher-prefix decision artifacts.
- [ ] Metric/artifact catalog versioning distinguishes the new profile from historical v4 evidence.
- [ ] Plot/artifact generation is deterministic and completion-order independent.
- [ ] Hermetic tests require no NAS MLflow endpoint.

### PR-472 — QA: MLflow/evidence completeness and old-artifact exclusion

**Type:** QA only  
**Depends on:** PR-471

#### Acceptance

- [ ] Independently enumerate every required medoid/elimination/elbow metric and artifact.
- [ ] Missing required evidence makes the run incomplete.
- [ ] K2-K5 final feature hashes must be identical for one selection identity.
- [ ] Historical teacher-only keys cannot satisfy new-profile completeness.
- [ ] New-profile runs containing teacher/raw-feature-scoring decision artifacts fail QA.
- [ ] Metric values and canonical hashes are independent of emission/completion order.
- [ ] Local FileStore/hermetic fixtures reproduce the complete evidence matrix without network.
- [ ] QA adds no new production selection behavior.

### PR-473 — Cut over canonical Xetra evaluation to the medoid/L-elbow profile

**Type:** implementation / controlled migration  
**Depends on:** PR-472

#### Acceptance

- [ ] Promote the new profile/evaluation version to the sole canonical Xetra selection path.
- [ ] Retire v4 teacher-based selection from canonical runtime entry points.
- [ ] No compatibility flag or silent fallback can reactivate the old algorithm.
- [ ] Historical v4 runs/packages remain historical evidence and are never relabelled.
- [ ] Update README, EVALUATION, source/lifecycle documentation and public identity constants.
- [ ] Static/runtime zero-legacy rules reject canonical calls to provisional teacher scoring,
  raw-feature winner selection and old teacher-prefix selection.
- [ ] Existing source/PCA quality rules, non-resumable full-run policy, external service boundaries
  and manual champion promotion remain unchanged.
- [ ] K-slot production semantics now require one common `S_{L*}` tuple across K2-K5.
- [ ] No production alias mutation occurs as part of code cutover.

### PR-474 — QA: cutover and zero-legacy proof

**Type:** QA only  
**Depends on:** PR-473

#### Acceptance

- [ ] Public Xetra entry points resolve only the new profile/evaluation identity.
- [ ] Attempts to invoke the old canonical teacher path fail explicitly.
- [ ] Static import/call scan finds no canonical dependency on old teacher/scoring/winner/prefix code.
- [ ] Historical v4 package/evidence readers remain distinguishable and read-only where required.
- [ ] All K2-K5 slots enforce one final feature hash per selection identity.
- [ ] Documentation and runtime constants agree on the new canonical version.
- [ ] No external service is mutated.
- [ ] This PR contains QA/documentation corrections only, not new selection behavior.

---

## Phase 4 — Complete local/system testing after all changes are implemented

No PR in this phase may begin as acceptance evidence before PR-474 is green.

### PR-475 — Full hermetic medoid/L-elbow end-to-end acceptance

**Type:** QA / complete-system acceptance  
**Depends on:** PR-474

#### Acceptance

- [ ] Run a complete hermetic evaluation with real HMM fits from immutable source snapshot through
  quality, clustering, medoids, full references, elimination, elbow, final grid and Outer TEST.
- [ ] Independently reproduce `M*`, every medoid, every elimination decision, raw/monotone L
  curves, elbow score and `L*`.
- [ ] Prove future Outer-TEST mutation cannot change any TRAIN-side decision.
- [ ] Prove every final K2-K5 slot uses the exact same feature tuple/hash.
- [ ] Prove deployment selection reruns the complete procedure at deployment cutoff.
- [ ] Prove deployment may legitimately choose a different `M*`, medoid set or `L*` than the
  last validation fold without identity collision.
- [ ] Verify serial/process canonical parity for selection, model and evidence hashes.
- [ ] Verify no legacy teacher/raw-feature-scoring decision artifact appears.
- [ ] Run lint, format, strict mypy, unit and hermetic integration gates.
- [ ] No external PostgreSQL/MLflow mutation.

### PR-423 — Four-slot outer-validation production-callback acceptance

**Status:** DEFERRED FINAL ACCEPTANCE — implementation previously merged  
**Depends on:** PR-475

#### Acceptance

- [ ] Exercise K2, K3, K4, K5 through the real production outer-validation callback.
- [ ] All four slots consume the exact same `S_{L*}` hash selected by the new canonical path.
- [ ] Preserve per-K eligibility gates and deterministic process assembly.
- [ ] Prove serial/process parity through the production callback.
- [ ] One ineligible K cannot contaminate another slot.
- [ ] No callback can rerun or mutate feature selection.
- [ ] No external service is required.

### PR-424 — Per-K deployment/refit production-artifact acceptance

**Status:** DEFERRED FINAL ACCEPTANCE — implementation previously merged  
**Depends on:** PR-423

#### Acceptance

- [ ] Exercise real deployment selection through final refit/package assembly for every eligible K.
- [ ] All K packages bind the same selected feature tuple/hash.
- [ ] Bind source build, validation cutoff, deployment cutoff, K and model family exactly.
- [ ] Prove deployment reruns selection rather than copying the last outer-fold tuple.
- [ ] Prove package round-trip identity for each eligible K.
- [ ] An ineligible K produces no package.
- [ ] No external publication occurs.

### PR-426 — Full K-slot Model Metrics and plot projection acceptance

**Status:** DEFERRED FINAL ACCEPTANCE — implementation previously merged  
**Depends on:** PR-424

#### Acceptance

- [ ] Project the complete metric/artifact/plot matrix for every eligible K slot.
- [ ] Include medoid/elimination/elbow lineage required by the new profile.
- [ ] Fail closed on incomplete lineage or missing projection domains.
- [ ] Preserve serial/process completion-order parity.
- [ ] Unavailable K slots are explicit and have no alias/package artifact.
- [ ] Old teacher-only evidence cannot satisfy new-profile completeness.
- [ ] No external publication occurs.

### PR-427 — Final independent mathematical acceptance closure

**Status:** DEFERRED FINAL ACCEPTANCE — implementation previously merged  
**Depends on:** PR-426

#### Acceptance

- [ ] Independent stdlib/math oracle covers K2-K5, medoids, same-K soft NMI, elimination and elbow.
- [ ] Cover ties, state-label permutations, adversarial inputs and provenance mutations.
- [ ] Oracle code must not call production selection helpers.
- [ ] Recompute every final feature tuple and K-slot identity for the golden fixture.
- [ ] Prove serial/process canonical parity.
- [ ] Mutation tests fail when K aggregation, tie rules, elbow rule or common-feature rule changes.
- [ ] No external service is required.

### PR-429 — Production-lineage four-slot E2E closure

**Status:** DEFERRED FINAL ACCEPTANCE — implementation previously merged  
**Depends on:** PR-427

#### Acceptance

- [ ] Run the full Gaussian/GMM-HMM/Student-t K2-K5 path against a production-shaped immutable
  lineage fixture.
- [ ] Preserve one common `S_{L*}` across K and families.
- [ ] Produce deployment packages, metrics and plots for eligible slots.
- [ ] Prove ineligible-slot fail-closed behavior.
- [ ] Prove future-row invariance.
- [ ] Prove process/serial and independent-process hash parity.
- [ ] Preserve default champion-alias semantics without mutating an external registry.

---

## Phase 5 — Current-source and external acceptance

These are the final proofs. They run only after all implementation and complete local/system
acceptance above is green.

### PR-232 — Full current-Xetra computation and independent audit evidence

**Status:** EXTERNALLY BLOCKED  
**Depends on:** PR-429 and a production-eligible upstream source build

#### Current source blocker

The 2026-09-16 attempt failed closed before model work:
`potentially_valid_outer_folds=0/246`,
`first_inner_train_complete_observations=0`, source build `20260915T212921Z`.
The upstream source must be republished with a production-eligible feature universe.

#### Acceptance

- [ ] Run the complete **new canonical medoid/L-elbow** evaluation on a production-eligible source.
- [ ] Record exact source/catalog/PCA/medoid/elimination/elbow/final-feature identities.
- [ ] Produce the independent mathematical audit dossier for every required fold.
- [ ] Independently reproduce selected `M*`, medoids, elimination path, `L*` and final tuple.
- [ ] Prove all K2-K5 slots share the same final feature hash.
- [ ] Reconcile source rows, model identities and all audit hashes exactly.
- [ ] Record zero numerical audit errors under the new canonical contract.
- [ ] Historical v4/raw-only runs are not accepted as evidence.

### PR-250 — External MLflow Model Metrics completeness proof

**Status:** EXTERNALLY BLOCKED  
**Depends on:** PR-232 and explicit namespace decision

#### Current blocker

Experiment `macro-regime-evaluation` contains 768 historical runs, zero visible LoggedModels and
zero registered versions; deleted-LoggedModel inventory is unavailable through HTTP RestStore.

#### Acceptance

- [ ] Resolve the historical namespace by explicit operator decision; no silent deletion.
- [ ] Project the fresh PR-232 run into the new-profile metric/artifact catalog.
- [ ] Verify every required medoid, elimination, elbow, K-slot and plot artifact is present.
- [ ] Verify exact dataset/model/feature lineage and metric value identity.
- [ ] Verify artifact hashes, sizes and freshness against the same evaluation identity.
- [ ] Prove historical teacher-only runs cannot satisfy new-profile completeness.
- [ ] This PR does not promote or mutate production aliases; publication/readback belongs to PR-430.

### PR-425 — K-slot registry matrix acceptance

**Status:** DEFERRED EXTERNAL ACCEPTANCE — implementation previously merged  
**Depends on:** PR-250

#### Acceptance

- [ ] Verify the complete registry naming/version matrix for all eligible K slots.
- [ ] Every K version binds the same new-profile final feature hash and its own K/family identity.
- [ ] Preserve immutable registration and audited compare-and-swap promotion semantics.
- [ ] Failed registration/promotion cannot mutate existing aliases.
- [ ] Ineligible K slots create no registered version or alias.
- [ ] Historical v4 versions cannot masquerade as new-profile versions.
- [ ] Use only explicitly authorized test/registry namespace operations.

### PR-428 — External K-slot registry/CAS durability

**Status:** DEFERRED EXTERNAL ACCEPTANCE — implementation previously merged  
**Depends on:** PR-425

#### Acceptance

- [ ] Verify external durability across process kill/retry boundaries.
- [ ] Verify concurrent promotion races preserve compare-and-swap semantics.
- [ ] Prove no partial alias/version state after interrupted publication attempt.
- [ ] Prove retry is idempotent for the same immutable package identity.
- [ ] Prove conflicting package identity is rejected rather than overwritten.
- [ ] Keep this test independent from a full reevaluation.
- [ ] Restore/retain the pre-test alias state unless the operator explicitly authorizes promotion.

### PR-430 — Authorized external publication and readback

**Status:** FINAL EXTERNAL GATE  
**Depends on:** PR-428

#### Acceptance

- [ ] PR-232, PR-250, PR-425 and PR-428 are all green first.
- [ ] Publish only explicitly authorized new-profile packages.
- [ ] Read back registered model names, versions, aliases, source lineage, `S_{L*}` hash, K,
  family and package digest from NAS MLflow.
- [ ] Verify every published K slot binds the same final feature tuple/hash.
- [ ] Verify publication is idempotent for the same package identity.
- [ ] Failed readback/promotion leaves the prior alias state unchanged.
- [ ] Champion promotion remains an explicit operator action.
- [ ] Archive the final external acceptance evidence and exact MLflow identities.

---

### Active dependency graph

```text
PR-448
  -> 449 -> 450 -> 451 -> 452 -> 453 -> 454 -> 455 -> 456
  -> 459 -> 460 -> 461 -> 462 -> 463 -> 464 -> 465 -> 466 -> 467 -> 468
  -> 469 -> 470 -> 471 -> 472 -> 473 -> 474
  -> 475 -> 423 -> 424 -> 426 -> 427 -> 429
  -> 232 -> 250 -> 425 -> 428 -> 430
```

PR-457 and PR-458 are superseded by the medoid/L-elbow design and are not active execution items.

---

## Current architectural decisions and non-goals

- **Target selection redesign:** PR-459–PR-468 replace the v4 teacher/raw-feature-scoring/prefix loop with medoid-only candidate selection plus causal inner-OOS backward elimination and a deterministic L elbow. After clustering, non-medoid raw features cannot re-enter selection. One common `S_{L*}` tuple is used by K=2,3,4,5. The current v4 production path remains active only until the controlled cutover in PR-467.
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
- PR-457/PR-458 are superseded historical planning for the retired teacher-prefix evidence path; they must not be implemented.
- Historical requirements for v1-v3 compatibility or full-run computation-position
  resume are superseded by the architectural decisions above.
