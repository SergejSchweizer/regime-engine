# Current-Xetra v4 full-compute audit

This is the operator contract for the intentionally external, unsampled
current-Xetra evaluation. It is not part of the GitHub push or merge gates.
The command reads the deployment `config.yaml`, uses the external MLflow
server configured there, captures one schema-wide PostgreSQL snapshot, and
then performs the complete v4 search and outer policy.

## Preconditions

- Install the exact repository environment with Python 3.14.7.
- Configure an absolute persistent `REGIME_ENGINE_STATE_ROOT`
  outside the repository through `config.yaml`.
- Configure absolute paths outside the repository for both
  `REGIME_EVALUATION_SUMMARY_PATH` and `REGIME_PERFORMANCE_REPORT_PATH`.
- Ensure the feature PostgreSQL password is available only through the
  configured password file.
- Verify the external MLflow health endpoint before starting.
- Do not run another v4 evaluation for the same deployment state root at the
  same time; the cron entrypoint takes an exclusive lock.

The run requires a production-eligible outer-policy result. If any valid
outer fold is missing its captured selection, or the policy is not eligible,
the command fails closed before audit tracking.

## Full command

From the repository root:

```text
scripts/run_xetra_v4_cron.sh
```

The cron wrapper performs no sampling, feature allowlisting, date truncation,
debug early stop, or model-family reduction. It calls the schema-wide source
API with `FeatureRequest.all_features()`, closes the read-only source
transaction before model work, and persists the immutable Arrow snapshot
before evaluation.

The evaluation uses the configured process-worker budget. Native BLAS/OpenMP
threads remain one per worker so independent HMM tasks can use all available
CPU cores without nested oversubscription.

## Required evidence

Set `REGIME_EVALUATION_SUMMARY_PATH` to a durable path outside the repository.
The JSON summary records the complete unbounded schema-wide source request,
source row/materialization bounds, source build/data/catalog/materialization/
snapshot/profile/plan/repository/lockfile hashes, the exact bounded M/prefix/K/
mixture/final-grid search envelope, every outer-fold selection (`M*`, `L*`,
candidate, `K`, OOS NMI and non-poolable OOS PLL), the first/middle/last valid
fold audit indices, production eligibility, MLflow parent run, evidence hash,
and paths to the independent math-audit expectations/report and performance
resource report. The independent verifier rejects a missing or altered audit
envelope before tracking begins.

The math-audit report must state the audited first, middle and last valid
outer-fold indices. It independently checks distance, silhouette,
state-information/eta scores, valid-prefix soft NMI, and selected TRAIN/OOS
Gaussian, GMM-HMM, or Student-t likelihoods. Any numerical discrepancy or
source identity change fails the command. The strict verifier requires the
source identity in the general math expectations to agree with the identity
inside the current-audit envelope, and it checks all source/materialized
timestamp bounds against the immutable Arrow snapshot. This prevents a stale
expectation from being paired with a different audit envelope.

The current contract version is `audit_contract.schema_version=2` inside the
expectation bundle (`schema_version=3`). The completed report must have
`status=verified`, `audit_contract_verified=true`, exactly the declared
first/middle/last audited folds, one outer-agreement record for every valid
fold, and zero (within `1e-10`) for every reported maximum absolute error.
The strict verifier also rejects any audited dossier that omits an eligible
feature score, bounded silhouette count, valid-prefix NMI, selected-model
TRAIN/OOS likelihood pair, or source row from the captured snapshot.

Record the command transcript, exit code, wall-clock duration, peak memory,
the summary JSON, the math-audit report, and the final evidence SHA-256 in the
release evidence directory. Do not copy passwords or populated deployment
configuration into that directory.

The local Arrow/JSON verifier is a hermetic identity and mathematical proof
only. It cannot prove that PostgreSQL was current when the snapshot was taken,
that the NAS MLflow historical namespace was empty, or that the complete
current-source evaluation was actually tracked. Those claims require the
opt-in external command and its read-only namespace/full-run evidence below;
they must not be inferred from local tests.

## Resume and rerun policy

This entrypoint is a single complete computation. An interrupted run must be
restarted from the beginning of the pinned snapshot computation. It must not
claim completion from a partial evaluation or attach a new result to a newer
live source build. Durable state and MLflow metric export may be inspected
after failure, but the external acceptance run is accepted only when a complete
fresh invocation finishes and all identities and audit outputs agree.
