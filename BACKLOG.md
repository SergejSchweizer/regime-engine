# Regime Engine — Canonical Backlog

Status date: 2026-09-18

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

## Planned medoid-only L-elbow redesign

This tranche implements the user-directed replacement for the current
medoid -> provisional teacher -> raw-feature regime scoring -> ranked-prefix
selection loop. The replacement keeps the existing TRAIN-only quality filter,
absolute-Spearman redundancy structure, global hierarchical clustering and
cluster-count selection, but after clustering it permits **only the cluster
medoids** to enter HMM feature-count selection.

The target semantics are:

```text
eligible raw features
    -> absolute-Spearman distance
    -> hierarchical clustering
    -> M* clusters
    -> exactly one medoid per cluster
    -> full-medoid Gaussian HMM references for K=2,3,4,5
    -> greedy backward medoid elimination
    -> causal inner-OOS regime-preservation curve Q_L
    -> deterministic elbow/knee
    -> one common final L* medoid tuple
    -> final K=2,3,4,5/model-family evaluation
    -> strict outer OOS evaluation
```

The new path must not score non-medoid raw features against HMM states and must
not use a provisional teacher to replace a medoid with another member of the
same cluster. The HMM is used only to measure how much latent regime structure
is lost as medoids are removed.

For one outer TRAIN fold:

- `M*` = number of statistically distinct redundancy clusters selected from
  the TRAIN-only Spearman hierarchy.
- `S_M` = ordered tuple containing exactly one deterministic medoid from each
  of those `M*` clusters.
- `L*` = number of medoids retained after causal inner-OOS backward
  elimination and deterministic elbow selection.
- `K` remains separate from `L`; K=2,3,4,5 must all consume the same final
  `S_{L*}` feature tuple for a given selection identity.

The existing source, PCA, missing-value, quality, outer walk-forward, final
refit, MLflow, registry and state-identity contracts remain unchanged unless an
item below explicitly says otherwise.

### PR-459 — Define the medoid-only L-elbow evaluation contract

**Type:** architecture / contracts

**Depends on:** PR-448

**Purpose:** introduce the next evaluation/profile contract without silently
mutating the active v4 statistical meaning.

#### Acceptance

- [ ] Define a new canonical evaluation/profile version for the medoid-only
  L-elbow path; v4 remains unchanged until the explicit cutover PR.
- [ ] Define `N`, `M*`, `S_M`, `L*`, `S_{L*}`, and K=2,3,4,5 as
  separate quantities with non-overlapping meanings.
- [ ] Preserve the existing TRAIN-only quality filter, absolute-Spearman
  distance, hierarchical average-linkage clustering and silhouette-based
  `M*` selection.
- [ ] Define the cluster medoid as the actual cluster member with minimum mean
  distance to the other members; existing deterministic `1e-12` tie handling
  and canonical ordinal ordering remain authoritative.
- [ ] State explicitly that after medoid selection no non-medoid raw feature
  may re-enter the candidate universe.
- [ ] Forbid provisional-teacher feature scoring, `state_information_ratio`,
  eta-squared winner selection, raw-feature cluster-winner replacement and
  teacher-NMI ranked-prefix selection in the new path.
- [ ] Define one common final medoid tuple for K=2,3,4,5; K-specific feature
  tuples are forbidden for the same selection identity.
- [ ] Preserve mandatory PCA/source semantics and existing model-family
  availability; this tranche changes feature selection, not upstream feature
  generation or the final family inventory.
- [ ] No production alias, external PostgreSQL or MLflow mutation occurs in
  this contract-only PR.

### PR-460 — Make cluster medoids the sole post-clustering feature candidates

**Type:** implementation / feature-selection simplification

**Depends on:** PR-459

#### Acceptance

- [ ] Reuse the selected `M*` cluster memberships and emit exactly one medoid
  per cluster in canonical cluster order.
- [ ] Persist cluster ID, member tuple, selected medoid, every candidate mean
  distance, canonical ordinal and a deterministic medoid-set hash.
- [ ] Singleton clusters select their sole member.
- [ ] Ties within `1e-12` select the smallest canonical ordinal.
- [ ] Once `S_M` is emitted, the new evaluation path exposes no API that can
  substitute another raw feature from the cluster.
- [ ] `feature_discovery.scoring` and `feature_discovery.winners` are not
  called by the new path.
- [ ] Raw feature catalog/quality evidence remains available for audit, but raw
  non-medoids cannot become HMM inputs after this boundary.
- [ ] Serial and process-backed construction produce byte-identical canonical
  evidence.

### PR-461 — Build causal full-medoid K=2..5 reference evidence

**Type:** implementation / inner walk-forward HMM evidence

**Depends on:** PR-460

#### Acceptance

- [ ] For every outer TRAIN fold, evaluate Gaussian full-covariance HMM
  K=2,3,4,5 on the complete medoid tuple `S_M` using the canonical inner
  expanding walk-forward plan.
- [ ] Every K sees exactly the same medoid order and inner fold boundaries.
- [ ] Use existing deterministic multistart, occupancy, covariance and model
  validity gates.
- [ ] Persist only causal inner-TEST filtered state probabilities as
  regime-reference arrays; smoothed probabilities and full-sample Viterbi
  labels are forbidden.
- [ ] Persist exact K, fold, timestamps, feature tuple/hash, source identity,
  model identity and canonical evidence hash for every reference array.
- [ ] The reference HMMs do not rank raw features and do not choose replacement
  members inside clusters.
- [ ] Parameter-safety bounds are checked before fitting; an unsafe full-medoid
  dimension fails closed rather than silently dropping features.
- [ ] Serial and process execution produce the same canonical reference
  evidence.

### PR-462 — Implement greedy backward medoid elimination

**Type:** implementation / feature-count search

**Depends on:** PR-461

**Canonical elimination rule:** let `S_M` be the fixed full-medoid tuple. At
current size `L`, evaluate every one-medoid removal from the current nested
subset. Each reduced subset is compared against the **fixed full-medoid
reference of the same K and inner fold**, never against a moving teacher.

For an eligible subset `S`, define:

```text
nmi(K, f, S) =
    soft_NMI(
        full_medoid_filtered_probabilities(K, f),
        reduced_subset_filtered_probabilities(K, f, S)
    )

q_K(S) = median over valid inner folds f of nmi(K, f, S)
Q(S)   = min over K in {2,3,4,5} of q_K(S)
```

#### Acceptance

- [ ] Start from `S_M` and generate one nested subset for every
  `L=M*, M*-1, ..., 2`.
- [ ] At each step evaluate every possible single-medoid removal from the
  current subset; unrestricted subset search is forbidden.
- [ ] Each reduced candidate is fit for Gaussian K=2,3,4,5 on the same inner
  fold plan as the full-medoid references.
- [ ] Compare only same-K full vs reduced posterior arrays on exact shared
  timestamps.
- [ ] Reuse the existing soft-NMI primitive and require its canonical finite,
  entropy and shared-support gates; shared timestamp support must be at least
  0.90.
- [ ] Each K must satisfy the existing inner valid-fold-rate hard gate before
  the subset can be eligible.
- [ ] `q_K(S)` is the median of valid fold-local soft NMI values; `Q(S)` is
  the minimum across K=2,3,4,5 so a subset cannot hide degradation at one state
  resolution behind good results at another.
- [ ] Set `Q(S_M)=1.0` by identity and retain the separately computed
  full-reference evidence.
- [ ] Remove the medoid whose resulting subset has the largest eligible
  `Q(S)`.
- [ ] If removal candidates tie within `1e-12`, remove the candidate with the
  largest canonical ordinal so lower canonical ordinals remain stable.
- [ ] Persist every tested removal, per-K/per-fold NMI, validity evidence,
  aggregate `q_K`, aggregate `Q`, selected removal and resulting subset.
- [ ] Process parallelism may evaluate independent removals/K fits concurrently,
  but completion order cannot change the selected path or evidence bytes.

### PR-463 — Select L* with a deterministic medoid-preservation elbow

**Type:** implementation / deterministic model-dimension selection

**Depends on:** PR-462

The elbow operates only on the nested subsets emitted by PR-462. It does not
search new feature combinations.

#### Acceptance

- [ ] Build the raw preservation curve `Q_L = Q(S_L)` for
  `L=2,...,M*`, with `Q_M=1.0`.
- [ ] Preserve the raw curve for audit and derive the monotone decision curve
  `Qhat_L = max(Q_j for j <= L)` so finite-sample downward noise from adding
  a dimension cannot create a false reverse elbow.
- [ ] When `M*=2`, select `L*=2`.
- [ ] When fewer than three distinct L candidates exist and no exact lower-L
  identity exists, select the full medoid count conservatively.
- [ ] If the smallest candidate already has `Qhat_L=1.0` within `1e-12`,
  select the smallest such L.
- [ ] Otherwise normalize candidate coordinates to `x in [0,1]` and
  `y in [0,1]` using the endpoints of the monotone curve and compute the
  elbow score as the perpendicular/chord-equivalent distance
  `E_L = (y_L - x_L) / sqrt(2)`.
- [ ] Select the interior L with maximum `E_L`; ties within `1e-12` choose
  the smaller L.
- [ ] Persist raw `Q_L`, monotone `Qhat_L`, normalized coordinates,
  `E_L`, selected `L*` and the exact `S_{L*}` tuple.
- [ ] Outer TEST data cannot influence the elimination path, preservation curve
  or elbow.
- [ ] The selected `S_{L*}` is common to K=2,3,4,5.

### PR-464 — Integrate medoid-elbow selection into outer validation and deployment

**Type:** implementation / orchestration

**Depends on:** PR-463

#### Acceptance

- [ ] The new evaluation path is exactly:
  quality -> Spearman distance -> clustering/`M*` -> medoids -> full-medoid
  K2-K5 inner references -> backward elimination -> elbow/`L*` -> common
  final feature tuple -> final candidate grid -> outer OOS.
- [ ] The old provisional teacher, all-raw-feature regime scoring, cluster
  regime-winner replacement and ranked-prefix selection are unreachable from
  the new path.
- [ ] Freeze `M*`, cluster memberships, medoids, elimination path, `L*`,
  final tuple and final candidate configuration before touching Outer TEST.
- [ ] Final Gaussian/GMM-HMM/Student-t candidate families, where retained by the
  current production contract, all consume the same selected `S_{L*}` tuple;
  no model family or K may rerun feature selection independently.
- [ ] All K=2,3,4,5 production slots for one selection identity bind the same
  final feature tuple/hash; K-specific model parameters remain independent.
- [ ] Deployment selection reruns the complete medoid-elbow TRAIN-only
  procedure through the deployment cutoff and never copies the last outer-fold
  tuple.
- [ ] Final refit/package identities bind source build, cutoffs, cluster/medoid
  evidence, `L*`, final tuple, K and model family.
- [ ] Existing fold-local/model-version-local state identity semantics remain
  unchanged.

### PR-465 — QA: independent oracle for medoid elimination and elbow selection

**Type:** QA only

**Depends on:** PR-464

#### Acceptance

- [ ] Independent test code recomputes medoids from cluster memberships and the
  distance matrix without calling the production medoid selector.
- [ ] Independent soft-NMI math recomputes every full-vs-reduced K/fold value
  used by a deterministic synthetic elimination path.
- [ ] Golden fixtures cover M*=2, M*=3, a clear elbow, an exactly flat
  preservation curve, tied removal candidates, and a finite but non-monotone
  raw `Q_L` curve.
- [ ] State-label permutation leaves soft NMI, removal order, `Q_L`, elbow and
  `L*` unchanged.
- [ ] One K below the validity gate makes that subset ineligible even when the
  other three K values are perfect.
- [ ] Mutation of the K aggregation from minimum to mean/maximum fails QA.
- [ ] Mutation of the tie rule or monotone-envelope rule fails QA.
- [ ] Serial, randomized task completion and process-backed execution produce
  identical elimination/evidence hashes.
- [ ] A spy proves the new path never invokes raw-feature regime scoring,
  cluster-winner replacement or provisional-teacher selection.

### PR-466 — Project medoid-elbow evidence into MLflow and diagnostics

**Type:** implementation / observability

**Depends on:** PR-464

#### Acceptance

- [ ] Log `N`, `M*`, silhouette curve, cluster memberships, medoids and
  medoid mean-distance evidence.
- [ ] Log the full-medoid K2-K5 reference identities without treating them as
  production champions.
- [ ] Log every backward-elimination candidate, selected removal, per-K
  `q_K`, aggregate `Q`, and the nested subset path.
- [ ] Add a canonical L-preservation/elbow diagnostic exposing raw `Q_L`,
  monotone `Qhat_L`, elbow score and selected `L*`.
- [ ] Log the common final feature tuple/hash and prove it is identical across
  all K slots for one selection identity.
- [ ] New-path runs emit no `state_information_ratio`, eta-squared
  cluster-winner, provisional-teacher or teacher-prefix decision artifacts.
- [ ] Metric/artifact catalog versioning distinguishes medoid-elbow evidence
  from historical v4 teacher evidence.
- [ ] Plot/artifact generation is deterministic and independent of task
  completion order.
- [ ] Hermetic tests require no NAS MLflow endpoint; external publication
  remains separately authorized.

### PR-467 — Cut over canonical Xetra selection to the medoid-elbow profile

**Type:** implementation / controlled migration

**Depends on:** PR-465, PR-466

#### Acceptance

- [ ] Promote the new profile/evaluation version to the sole canonical Xetra
  discovery/selection path.
- [ ] Retire v4 teacher-based selection from canonical runtime entry points;
  there is no compatibility flag or silent fallback to the old algorithm.
- [ ] Historical v4 MLflow runs/packages remain historical evidence only and
  are never mislabelled as medoid-elbow runs.
- [ ] Update README, EVALUATION, source/lifecycle documentation and public
  identity constants to the new contract.
- [ ] Zero-legacy/static audits reject canonical imports/calls of provisional
  teacher scoring, raw-feature winner selection and old teacher-prefix
  selection.
- [ ] Existing source/PCA quality rules, non-resumable full-run policy,
  external PostgreSQL/MLflow boundaries and manual champion promotion remain
  unchanged.
- [ ] Open K-slot production acceptance PR-423..PR-430 is rebased conceptually
  onto the new selection identity: same `S_{L*}` feature tuple across K
  slots, independent K/model artifacts thereafter.
- [ ] No production alias mutation occurs as part of the code cutover.

### PR-468 — QA: full medoid-elbow end-to-end acceptance

**Type:** QA / acceptance closure

**Depends on:** PR-467

#### Acceptance

- [ ] Run a complete hermetic evaluation with real HMM fits from source
  snapshot through quality, clustering, medoids, elimination, elbow, final
  grid and Outer TEST evidence.
- [ ] Independently reproduce the selected `M*`, each medoid, the complete
  elimination path, raw/monotone L curves, elbow and `L*`.
- [ ] Prove future Outer-TEST row mutation cannot change any TRAIN-side
  clustering, medoid, elimination or elbow decision.
- [ ] Prove every final K=2,3,4,5 slot uses the exact same final feature
  tuple/hash.
- [ ] Prove deployment selection reruns the full procedure at the deployment
  cutoff and can legitimately choose a different `M*`, medoid set or `L*`
  from the last validation fold.
- [ ] Prove no teacher/raw-feature-scoring artifact or runtime call appears in
  the new canonical path.
- [ ] Verify complete deterministic MLflow/local evidence hashes in serial and
  process-backed execution.
- [ ] Run the zero-legacy audit and all merge/push quality gates before closure.

### Medoid-elbow execution order

```text
PR-459
  -> PR-460
  -> PR-461
  -> PR-462
  -> PR-463
  -> PR-464
       |-> PR-465
       |-> PR-466
PR-465 + PR-466
  -> PR-467
  -> PR-468
```

PR-459 through PR-468 supersede the unimplemented teacher-prefix evidence work
in PR-457/PR-458. The other repository-audit tracks remain independent unless
their implementation touches the new selection path.

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

**Status:** SUPERSEDED — do not implement; replaced by PR-459–PR-468

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

**Status:** SUPERSEDED — do not implement; replaced by PR-459–PR-468

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
PR-457 -> PR-458  # superseded; do not execute
```

The implementation tracks are otherwise independent. PR-457/PR-458 are retained only
as superseded traceability records and must not be implemented. Every active branch must
start from the then-current `origin/main` and be rebased immediately before
opening/updating its GitHub PR. QA follow-ups must be rebased after their implementation
dependency merges.

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
- Historical requirements for v1-v3 compatibility or full-run computation-position
  resume are superseded by the architectural decisions above.
