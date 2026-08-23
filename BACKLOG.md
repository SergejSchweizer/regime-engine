# Regime Engine — Consolidated Implementation Backlog

Status date: 2026-08-23

This file is the single authoritative implementation backlog for `SergejSchweizer/regime-engine`.

It replaces all older duplicated Wave-7/Wave-8 addenda and conflicting legacy wording. Every PR below contains one effective scope, one dependency set, one allowed-file set, and one acceptance list. Weak agents must not infer requirements from superseded historical text.

---

# 1. Canonical identities

These names are fixed and must not be substituted:

| Concept | Canonical value |
|---|---|
| GitHub repository | `SergejSchweizer/regime-engine` |
| Repository short name | `regime-engine` |
| Python distribution | `market-regime-engine` |
| Python import package | `market_regime_engine` |
| Python runtime | `3.14.7` |
| MLflow version | `3.15.1` |
| Gaussian HMM backend | `hmmlearn==0.3.3` |
| MLflow custom app entry point | `regime-engine` |
| Production MLflow host | `10.10.1.3` |
| Production MLflow port | `5000` |
| External feature PostgreSQL host | `10.10.1.3` |
| External feature PostgreSQL port | `54321` |
| External feature reader | `regime-engine` |
| Initial public profile ID | `xetra` |
| Xetra profile configuration version | `1` |
| Xetra feature-selection policy | `xetra_semantic_medoid_v1` |
| Xetra MLflow registered model | `regime-xetra` |
| Production serving alias | `champion` |
| Non-production lifecycle alias | `challenger` |
| Initial prediction contract | `RegimePrediction.v1` |
| Invocation response contract | `RegimeInvocationResponse.v1` |
| Error contract | `RegimeError.v1` |

`xetra_cross_asset_v1` is not a public `profile_id`. The equivalent versioned model configuration is represented as `profile_id=xetra` plus `profile_config_version=1`.

`engine-champion` is not an MLflow serving alias. The phrase *statistical champion* means the candidate selected by the engine evaluation procedure; the production MLflow alias assigned after final refit is `champion`.

PR numbers `039`–`044` and `051`–`055` are historical planning/documentation IDs and are retired. They must never be reused as implementation PR numbers.

---

# 2. Ownership boundary

`regime-loader` owns:

- provider acquisition;
- Bronze/Silver/Gold processing;
- the 48 canonical causal source features;
- immutable Gold publication;
- replication of current Gold to PostgreSQL;
- PostgreSQL sync-state lineage.

`regime-engine` owns:

- feature-source consumption;
- statistical feature selection;
- preprocessing;
- HMM fitting;
- state alignment;
- leak-free-in-split walk-forward evaluation;
- statistical candidate comparison;
- final production refit;
- MLflow tracking/model registration;
- immutable OOS prediction artifacts;
- fixed-model latest/replay inference;
- the profile-routed inference API;
- serving/runtime guardrails.

`portfell` and other consumers own:

- ETF/asset returns;
- portfolio statistics;
- portfolio construction/optimization;
- transaction costs;
- Sharpe/Sortino/Calmar/drawdown/Expected Shortfall evaluation;
- application-specific economic model choice.

No portfolio/economic quantity may influence feature selection or the engine statistical champion.

---

# 3. Production topology

Production has exactly one externally published HTTP service:

```text
http://10.10.1.3:5000
    MLflow UI
    MLflow Tracking API
    MLflow Model Registry
    MLflow artifact serving
    regime-engine MLflow app
        POST /regime-engine/v1/profiles/{profile_id}/invocations
        GET  /regime-engine/v1/profiles/{profile_id}/oos-builds/{build_id}
        GET  /regime-engine/v1/health
```

The repository-owned Compose topology is exactly:

```text
docker-compose
├── mlflow
└── mlflow-postgres
```

Rules:

- `mlflow` is the only service publishing a host port.
- Host mapping is exactly `5000:5000`.
- `mlflow-postgres` is only the MLflow metadata backend.
- `mlflow-postgres` has no host `ports` mapping.
- The external feature PostgreSQL at `10.10.1.3:54321` is not a Compose service.
- No second `mlflow models serve` process exists.
- No standalone repository FastAPI/Uvicorn application exists.
- No nginx, Traefik, or other reverse proxy exists.
- No public `:5001` exists.
- MLflow Prometheus exposure is disabled; `--expose-prometheus` is forbidden.

MLflow `3.15.1` defaults to Uvicorn, but the `mlflow.app` extension used here is a Flask/WSGI application. Production therefore explicitly runs MLflow through Gunicorn; the default Uvicorn path is not permitted for this deployment.

Canonical server command shape:

```text
mlflow server
  --app-name regime-engine
  --host 0.0.0.0
  --port 5000
  --workers ${MLFLOW_WORKERS}
  --gunicorn-opts "--worker-class gthread --threads ${MLFLOW_THREADS_PER_WORKER} --timeout ${MLFLOW_HTTP_TIMEOUT_SECONDS} --graceful-timeout ${MLFLOW_GRACEFUL_TIMEOUT_SECONDS}"
```

Production defaults:

```text
MLFLOW_WORKERS=4
MLFLOW_THREADS_PER_WORKER=4
MLFLOW_HTTP_TIMEOUT_SECONDS=120
MLFLOW_GRACEFUL_TIMEOUT_SECONDS=30
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
```

No implementation may silently switch the server back to Uvicorn.

---

# 4. Security boundary

MVP security is explicitly **trusted-private-LAN only**.

- `http://10.10.1.3:5000` must not be exposed to the public Internet.
- NAS/firewall policy must restrict port 5000 to trusted private-network clients/operators.
- No wildcard Host or CORS policy is allowed.
- Default allowed hosts are `10.10.1.3`, `localhost`, and `127.0.0.1`; deployment may add explicit trusted hostnames.
- Browser CORS defaults to same-origin only; additional origins require explicit operator configuration.
- This MVP does not claim application-layer multi-tenant isolation between trusted LAN consumers.
- Registry write access is therefore part of the trusted-operator network boundary.
- If public/untrusted access is ever required, authentication/authorization is a separate versioned architecture change and must be composed into the single MLflow app; a reverse proxy must not be introduced implicitly.

Secrets are runtime-only. Passwords, credential-bearing DSNs, tokens, and secret values must never appear in Git, MLflow artifacts, API responses, plots, exceptions, or normal logs.

---

# 5. External feature PostgreSQL contract

Production source:

```text
host:              10.10.1.3
port:              54321
database:          mandatory runtime value; no default and never guessed
user:              regime-engine
dataset_id:        regime_features_daily
feature table:     regime_loader.regime_features_daily
sync-state table:  regime_loader_sync.gold_sync_state
temporal key:      timestamp_m1 TIMESTAMPTZ(6)
```

The dedicated PostgreSQL role is the quoted SQL identifier `"regime-engine"` and runtime username `regime-engine`.

It receives only:

- `CONNECT` on the explicitly supplied feature database;
- `USAGE` on `regime_loader`;
- `USAGE` on `regime_loader_sync`;
- `SELECT` on `regime_loader.regime_features_daily`;
- `SELECT` on `regime_loader_sync.gold_sync_state`.

It receives no writer/admin/ownership/CREATE privileges and no `SELECT` on `gold_row_hashes` in MVP.

Every production source read uses one transaction equivalent to:

```text
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
  read regime_loader_sync.gold_sync_state
  read bounded feature rows
  validate lineage and source bounds
COMMIT;
```

The transaction is kept only for data acquisition. Long model fitting/evaluation must operate on the already materialized immutable in-memory/source snapshot after the PostgreSQL transaction has closed.

Runtime names are separated from MLflow backend database configuration:

```text
REGIME_FEATURE_PGHOST=10.10.1.3
REGIME_FEATURE_PGPORT=54321
REGIME_FEATURE_PGDATABASE=<required>
REGIME_FEATURE_PGUSER=regime-engine
REGIME_FEATURE_PGPASSWORD_FILE=<preferred production secret file>
REGIME_FEATURE_PGPASSWORD=<optional local/test-only direct secret>
```

Generic `PGHOST`, `PGDATABASE`, `PGUSER`, and `PGPASSWORD` are not the regime feature-source production contract because the same container also uses an independent MLflow backend PostgreSQL.

---

# 6. Time semantics and scientific claim boundary

The current upstream `timestamp_m1` is observation-day identity. It is **not** provider release time and does not encode historical data vintages.

Therefore the engine makes exactly this claim:

> Walk-forward evaluation is split-leak-free and causal with respect to the current-vintage observation sequence supplied by `regime-loader`.

It does **not** claim:

- historical release-time correctness;
- vintage-correct macroeconomic data;
- tradability at `timestamp_m1`;
- a fully point-in-time trading backtest.

Canonical metadata value:

```text
data_time_semantics=current_vintage_observation_day
```

Consequences:

- historical `as_of`/`replay` cuts the current serving replica by observation timestamp;
- later upstream historical revisions may change a later replay even when `model_version` is pinned;
- `model_version` pins the model only, not an unavailable historical source vintage;
- immutable `walk_forward_oos` builds are the engine's reproducible evaluation outputs because they preserve their source-build lineage at publication time;
- downstream documentation must not describe current MVP OOS output as release-time/vintage-safe trading evidence;
- a future upstream `available_at_utc`/vintage contract requires a versioned source-contract change before the stronger claim may be made.

---

# 7. Missing-value and observation-clock semantics

Upstream Gold may contain SQL NULLs. NaN and infinity are invalid.

There are two distinct source modes.

## 7.1 Feature-selection source mode

- NULL values are allowed.
- Every non-null numeric value must be finite.
- Coverage and complete-case rules below determine eligibility.
- No fill, interpolation, carry, or synthetic calendar row is allowed.

## 7.2 Resolved-model mode

For a frozen final feature set, an HMM observation exists only at a timestamp where **all final features are non-null and finite**.

- Rows incomplete across final features are excluded from the model observation sequence.
- Excluded timestamps are retained as gap evidence/counts; they are never filled.
- HMM transition steps are per consecutive retained observation, not per calendar day.
- No `A^calendar_gap` extrapolation is performed.
- Walk-forward split bounds are calendar/source timestamps, while minimum usable TRAIN/TEST counts are evaluated after the resolved-model complete-case mask.
- `latest` predicts at the latest complete resolved-model timestamp at or before requested `as_of`.
- A replay returns predictions only for complete model observations inside `[start,end]` and reports skipped incomplete timestamps.

This rule is identical in training, evaluation, final refit, latest, and replay.

---

# 8. Xetra source universe and statistical feature selection

The source universe is exactly the 48 `regime-loader` feature-version-1 columns. `timestamp_m1` is never a candidate.

Canonical blocks:

1. `us_equity_volatility_spot`: `vix_level`, `vix_delta_5obs`, `vix_delta_20obs`, `vix_zscore_60obs`
2. `us_equity_volatility_term_structure`: `vix9d_level`, `vix9d_delta_5obs`, `vix9d_delta_20obs`, `vix9d_zscore_60obs`, `vix3m_level`, `vix3m_delta_5obs`, `vix3m_delta_20obs`, `vix3m_zscore_60obs`, `vix6m_level`, `vix6m_delta_5obs`, `vix6m_delta_20obs`, `vix6m_zscore_60obs`, `vix1y_level`, `vix1y_delta_5obs`, `vix1y_delta_20obs`, `vix1y_zscore_60obs`, `vix9d_vix_ratio`, `vix_vix3m_ratio`, `vix3m_minus_vix`, `vix6m_minus_vix`, `vix1y_minus_vix`
3. `europe_equity_volatility`: `vstoxx_level`, `vstoxx_delta_5obs`, `vstoxx_delta_20obs`, `vstoxx_zscore_60obs`
4. `rates_volatility`: `move_level`, `move_delta_5obs`, `move_delta_20obs`, `move_zscore_60obs`
5. `systemic_stress`: `ciss_level`, `ciss_delta_5obs`, `ciss_delta_20obs`
6. `credit_stress`: `euro_hy_oas_level`, `euro_hy_oas_delta_5obs`, `euro_hy_oas_delta_20obs`
7. `rates_yield_curve`: `us_2y_level`, `us_2y_delta_20obs`, `us_10y_level`, `us_10y_delta_20obs`, `estr_level`, `estr_delta_20obs`, `us_10y_minus_us_2y`
8. `usd_fx`: `usd_broad_level`, `usd_broad_delta_20obs`

Pinned policy:

```text
policy_id=xetra_semantic_medoid_v1
within_block_method=absolute_spearman_medoid
cross_block_method=absolute_spearman_prune
minimum_feature_coverage=0.90
minimum_nonzero_variance=1e-12
minimum_block_complete_observations=504
maximum_cross_block_abs_spearman=0.85
numeric_tie_abs_tolerance=1e-12
```

Selection uses only the TRAIN interval of the first walk-forward fold.

Spearman implementation is deterministic:

1. take complete rows for the relevant candidate set;
2. rank each column with average ranks for ties;
3. compute Pearson correlation of those rank columns;
4. all required correlations must be finite.

Stage 1:

- coverage = non-null count / first-fold TRAIN source-row count;
- eligible iff coverage >= 0.90, every non-null value finite, and population variance `ddof=0` > `1e-12`;
- complete-case rows are calculated across all eligible features in that block;
- require at least 504 block-complete observations;
- distance `d(i,j)=1-abs(rho(i,j))`;
- medoid score = arithmetic mean distance to all other eligible block candidates;
- one eligible feature has score 0;
- winner: lowest score, then higher coverage, then earlier configured candidate; values equal within absolute tolerance `1e-12` are treated as ties;
- exactly one medoid per block, therefore exactly eight preliminary medoids.

Stage 2:

- use one fixed complete-case Spearman matrix across the eight preliminary medoids;
- require at least 504 common observations;
- conflict iff `abs(rho)>0.85`; exactly `0.85` is allowed;
- repeatedly choose the remaining conflict with highest `abs(rho)`; ties use earlier first block then earlier second block in canonical block order;
- remove higher Stage-1 medoid score; tie -> lower coverage; tie -> later canonical block;
- do not recompute the matrix;
- do not search for replacements;
- final features are surviving medoids in canonical block order;
- legal final dimension is `1 <= d <= 8`.

The `d>=1` policy is deliberate. MVP does not impose a minimum retained block count beyond one. A future minimum-retained-block constraint is a new policy version.

The cross-block removal score compares medoid scores from different block sizes. This is a known pinned simplification of policy v1; it must be documented, not silently normalized.

No HMM fit/likelihood/AIC/BIC, ETF return, portfolio metric, or trading label may participate in feature selection.

## 8.1 Selection hashes

Two different hashes are mandatory:

`feature_selection_definition_hash` hashes only the policy and evidence determined by the first-fold TRAIN sample, including final features. It excludes whole-source `source_build_id`, full-build hash, later rows, and whole-evaluation-plan material unrelated to first-fold selection.

`feature_selection_execution_hash` hashes:

```text
feature_selection_definition_hash
+ source_build_id
+ data_sha256
+ evaluation_plan_hash
```

Appending or mutating rows strictly after first-fold `train_end` may change the source/execution hash but must not change the definition hash or Stage-1/Stage-2 selection evidence.

All candidates in one comparison must share the same final feature order, definition hash, execution hash, and source build.

---

# 9. Xetra evaluation profile — pinned numerical policy

The initial profile is:

```text
profile_id=xetra
profile_config_version=1
frequency=daily_observation_sequence
feature_selection_policy=xetra_semantic_medoid_v1
candidate_ids:
  gaussian_hmm_k2_full
  gaussian_hmm_k3_full
  gaussian_hmm_k4_full
```

Walk-forward policy:

```text
minimum_train_source_observations=1260
test_source_observations=63
step_source_observations=63
allow_partial_final_test=false
minimum_model_train_observations=504
minimum_model_test_observations=42
```

Multi-start/HMM policy:

```text
backend=hmmlearn==0.3.3
covariance_type=full
implementation=log
seeds=[11,23,37,53,71,89,107,131]
minimum_valid_starts=6
minimum_multistart_success_rate=0.75
n_iter=1000
tol=1e-4
min_covar=1e-6
startprob_prior=1.0
transmat_prior=1.0
means_prior=0.0
means_weight=0.0
covars_prior=0.01
covars_weight=1.0
params=stmc
init_params=stmc
```

Quality policy:

```text
minimum_train_hard_occupancy=0.03
minimum_train_soft_occupancy=0.05
minimum_valid_fold_rate=0.80
low_confidence_threshold=0.60
alignment_ambiguity_abs_tolerance=1e-10
```

State-signature drift is diagnostic in v1, not a hard threshold. No agent may invent a maximum drift gate. Alignment must still be finite, one-to-one, and unambiguous under the exact rule below.

---

# 10. Exact HMM inference and predictive-likelihood semantics

For each retained observation `x_t`, let `b_t(k)` be the state-k Gaussian emission density and `A` the transition matrix.

At the first observation of a sequence:

```text
u_0(k) = pi(k) * b_0(k)
c_0    = sum_k u_0(k)
alpha_0(k) = u_0(k) / c_0
loglik = log(c_0)
```

For each subsequent retained observation:

```text
prior_t = alpha_(t-1) @ A
u_t(k)  = prior_t(k) * b_t(k)
c_t     = sum_k u_t(k)
alpha_t(k) = u_t(k) / c_t
loglik += log(c_t)
```

Implement in log/stabilized form. Every alpha vector must be finite and normalized.

One transition occurs per retained complete observation; calendar gaps do not apply extra powers of `A`.

## 10.1 Walk-forward TEST predictive likelihood

TEST must continue from TRAIN, not restart from `startprob_`.

- filter the fitted model across retained TRAIN observations;
- retain terminal `alpha_train_end`;
- first retained TEST prior is `alpha_train_end @ A`;
- sequentially update over retained TEST observations;
- OOS predictive log likelihood is the sum of TEST-only `log(c_t)` terms;
- per-observation value divides by retained TEST observation count.

Calling a backend method that restarts the TEST sequence from `startprob_` is not accepted as OOS predictive likelihood.

Candidate aggregate `oos_predictive_loglik_mean` is the unweighted arithmetic mean of valid-fold `oos_predictive_loglik_per_obs` values. Standard deviation uses population `ddof=0`. A pooled observation-weighted value may be logged only as a separate diagnostic and never substituted for the ranking metric.

---

# 11. Exact state identity and alignment

Persistent consumer states are `state_0` through `state_{K-1}`. Raw library labels are never public semantics.

For fitted state `k` in standardized model feature space:

1. take mean vector `mu_k` in exact feature order;
2. take full covariance `Sigma_k`;
3. compute state standard deviations from covariance diagonal;
4. compute state correlation matrix from `Sigma_k`;
5. construct signature:

```text
signature_k = concat(
  mu_k,
  log(state_standard_deviations),
  upper_triangle_off_diagonal(state_correlation_matrix)
)
```

All signature components must be finite.

Distance between two signatures is root-mean-square difference:

```text
RMS(s1,s2) = sqrt(mean((s1-s2)^2))
```

For the first valid evaluation fold, raw states are assigned persistent IDs by lexicographic ascending order of the full signature vector; if signatures are equal within `1e-10` and cannot be ordered uniquely, the fold is invalid.

For every later fold with the same K:

- reference = previous valid fold's persistent-state signatures;
- enumerate all K! one-to-one permutations (`K<=4` in MVP);
- total assignment cost = sum of matched RMS distances;
- choose the unique minimum-cost permutation;
- if best and second-best total cost differ by <= `1e-10`, alignment is ambiguous and the fold is invalid;
- record per-state and maximum RMS drift as diagnostics.

The final production refit aligns to the last valid evaluation fold of the winning K using the same rule.

---

# 12. Evaluation metrics and gates

Per fold, record at least:

- TRAIN and TEST UTC bounds and source/usable observation counts;
- fit convergence/multi-start details;
- training log likelihood;
- TEST predictive log likelihood and per-observation value;
- AIC/BIC;
- TRAIN hard and soft occupancy per persistent state;
- OOS hard and soft occupancy diagnostics;
- transition/self-transition probabilities;
- dominant-state durations and switches;
- state-signature assignment/drift;
- OOS entropy and confidence.

Definitions:

```text
confidence_t = max_k alpha_t(k)
entropy_t = -sum_k alpha_t(k) * ln(alpha_t(k))
```

Natural logarithm is mandatory.

A Gaussian full-covariance HMM with K states and d features has parameter count:

```text
p=(K-1)+K(K-1)+Kd+K*d*(d+1)/2
```

AIC/BIC must use that count.

Covariance validity:

- exact shape `K x d x d`;
- finite;
- symmetry accepted only if maximum absolute asymmetry <= `1e-10`;
- symmetrize only after passing that tolerance by `(S+S.T)/2` for numerical validation;
- Cholesky must succeed without adding an unrecorded jitter;
- minimum diagonal variance must be >= `1e-12`;
- any backend reduced covariance mode fails closed.

Fold hard gates:

- at least 6 of 8 starts valid/converged;
- multistart success rate >= 0.75;
- finite/normalized model parameters;
- valid full covariance matrices;
- TRAIN hard occupancy >= 0.03 for every state;
- TRAIN soft occupancy >= 0.05 for every state;
- successful unambiguous persistent-state alignment;
- retained TRAIN observations >= 504;
- retained TEST observations >= 42.

Candidate hard gate: valid-fold rate >= 0.80.

Ranking after hard gates:

1. highest `oos_predictive_loglik_mean`;
2. lower `oos_predictive_loglik_std`;
3. higher `oos_predictive_loglik_worst_fold`;
4. lower `bic_mean`;
5. lower `aic_mean`;
6. fewer states K;
7. lexicographically earlier canonical candidate ID.

No weighted composite score exists.

---

# 13. Final production refit — mandatory

Walk-forward candidate models are evaluation evidence and are never directly registered as the production champion.

After the statistical winner K is selected:

1. keep the frozen feature-selection result unchanged;
2. use all source observations from the evaluation source snapshot up to the evaluation cutoff;
3. apply the same complete-case final-feature observation mask;
4. require at least 504 usable observations;
5. fit a new scaler on that full final-refit sample;
6. fit the winning K using the exact eight-seed multi-start policy;
7. apply all numerical, covariance, occupancy, and multistart gates again;
8. align final states to the last valid evaluation fold for the winning K;
9. run the causal filter over the complete refit observation sequence;
10. store initial probability, transition matrix, means, full covariances, scaler, mapping, signatures, inference origin, trained-through timestamp, and terminal filtered state distribution.

Required package temporal fields:

```text
inference_origin_timestamp
trained_through_timestamp
terminal_filtered_probabilities
```

The final refit does not alter the already computed OOS ranking/evaluation metrics.

Only this final-refit artifact may be registered as a production model version.

---

# 14. Production latest/replay initialization

Inference must be invariant to the client's requested replay start.

For requests wholly after `trained_through_timestamp`:

- initialize from stored `terminal_filtered_probabilities`;
- read every complete model observation after `trained_through_timestamp` through requested end;
- update causally and return only requested timestamps.

For a replay whose requested interval includes timestamps at or before `trained_through_timestamp`:

- read/filter from `inference_origin_timestamp` through requested end using the fitted model's initial probability vector;
- return only predictions within requested `[start,end]`.

A replay may never initialize from `startprob_` at the caller's arbitrary `start` unless that start is exactly the stored inference origin.

Required invariant test:

> Replays with different requested starts but overlapping returned timestamps must produce identical probabilities on the overlap for the same model version and source build.

---

# 15. Public API contract

## 15.1 Invocation route

```text
POST /regime-engine/v1/profiles/{profile_id}/invocations
```

`profile_id` is path-only. Any body `profile_id` is rejected. Unknown body fields are rejected.

`latest`:

```json
{
  "operation": "latest",
  "as_of": "optional RFC3339 UTC timestamp",
  "model_version": "optional exact immutable MLflow model version"
}
```

`replay`:

```json
{
  "operation": "replay",
  "start": "required RFC3339 UTC timestamp",
  "end": "required RFC3339 UTC timestamp",
  "model_version": "optional exact immutable MLflow model version"
}
```

Rules:

- timestamps must include UTC `Z` or explicit zero offset and are normalized to UTC;
- latest forbids `start`/`end`;
- replay forbids `as_of`;
- absent model version resolves `regime-{profile_id}@champion` and pins the exact resolved version before source access;
- explicit model version bypasses alias lookup;
- consumers never provide feature names/values, DB information, scaler/HMM parameters, source build, or selection evidence.

## 15.2 Response

Every successful latest/replay response is `RegimeInvocationResponse.v1` and contains:

```text
schema_version
request_id
profile_id
operation
prediction_mode
requested_as_of OR requested_start/requested_end
model.name
model.version
model.alias nullable
model.alias_resolved_at_utc nullable
model.trained_through_timestamp
source.dataset_id
source.source_build_id
source.data_sha256
source.schema_version
source.feature_version
source.synced_at_utc
source.data_time_semantics
feature_contract_hash
feature_selection_definition_hash nullable
feature_selection_execution_hash nullable
warmup_observation_count
skipped_incomplete_row_count
predictions[]
```

`latest` has exactly one prediction. `replay` requires at least one returned model observation or fails `422 no_complete_observations`.

Each prediction validates `RegimePrediction.v1` and contains timestamp, persistent-state probabilities, dominant state, confidence, entropy, model/source lineage and data-quality status.

`prediction_mode` values are exactly:

```text
fixed_model_latest
fixed_model_replay
```

Replay is never labelled `walk_forward_oos`.

## 15.3 OOS retrieval

```text
GET /regime-engine/v1/profiles/{profile_id}/oos-builds/{build_id}?start=<optional>&end=<optional>
```

- explicit immutable build ID is mandatory;
- no implicit/latest build resolution exists;
- bounded date filtering is allowed;
- response identifies mode exactly `walk_forward_oos`;
- it cannot substitute a fixed-model replay.

## 15.4 Error contract

All application errors use `RegimeError.v1`:

```json
{
  "schema_version": "RegimeError.v1",
  "request_id": "...",
  "error_code": "stable_machine_code",
  "message": "safe human message",
  "retryable": false,
  "details": {}
}
```

Stable HTTP mapping:

- `400`: malformed/forbidden fields or invalid timestamp syntax;
- `404`: unknown profile, build, or exact model version;
- `413`: replay range/row/internal-row/response-size limit;
- `422`: valid syntax but model/source/semantic/no-complete-observation contract failure;
- `503`: dependency/capacity/current-champion/source/model freshness failure;
- `504`: cooperative replay deadline exceeded.

No raw feature values or secrets appear in errors.

---

# 16. Latest freshness and model lifecycle health

Operational defaults:

```text
REGIME_SOURCE_STALE_WARN_DAYS=4
REGIME_SOURCE_STALE_FAIL_DAYS=7
REGIME_MODEL_STALE_WARN_DAYS=14
REGIME_MODEL_STALE_FAIL_DAYS=35
```

For default-alias latest calls:

- source staleness is calendar days from request time to returned prediction timestamp;
- model staleness is calendar days from latest usable source timestamp to champion `trained_through_timestamp`;
- warn threshold marks health `degraded` but still serves;
- fail threshold rejects default-champion latest with `503`;
- explicit-version research replay remains allowed when its requested historical source rows are valid, even if that model is old.

The only MVP concept/model-drift decision mechanism is periodic full walk-forward reevaluation on a new source build. No uncalibrated online drift detector may automatically change the champion.

---

# 17. High-load/replay controls

Production defaults:

```text
REGIME_MODEL_ALIAS_CACHE_TTL_SECONDS=30
REGIME_PG_POOL_MIN_SIZE=1
REGIME_PG_POOL_MAX_SIZE=4
REGIME_PG_ACQUIRE_TIMEOUT_SECONDS=5
REGIME_PG_STATEMENT_TIMEOUT_SECONDS=30
REGIME_REPLAY_MAX_ROWS=10000
REGIME_REPLAY_MAX_INTERNAL_ROWS=15000
REGIME_REPLAY_MAX_RANGE_DAYS=14610
REGIME_REPLAY_TIMEOUT_SECONDS=60
REGIME_REPLAY_MAX_RESPONSE_BYTES=26214400
REGIME_REPLAY_MAX_CONCURRENCY_PER_WORKER=1
```

Rules:

- each Gunicorn worker has its own process-local model cache and PostgreSQL pool;
- maximum configured feature-PG connections = `MLFLOW_WORKERS * REGIME_PG_POOL_MAX_SIZE`;
- maximum simultaneous admitted replays = `MLFLOW_WORKERS * REGIME_REPLAY_MAX_CONCURRENCY_PER_WORKER`;
- a worker's model load is single-flight per `(profile_id, exact_model_version)` so concurrent cache misses do not duplicate loads;
- cache has a bounded maximum of two loaded versions per profile per worker: current resolved version plus at most one previous version; inactive older versions are evicted LRU only when no active request holds them;
- alias target is loaded and validated before atomic cache replacement;
- if a newly resolved champion cannot load/validate, the request fails and the previous object is never falsely labelled current;
- replay admission uses a process-local semaphore and executes synchronously in the existing Gunicorn request thread; no extra unbounded thread/task executor is created;
- CPU filtering is chunked so cooperative deadline checks can stop work between chunks;
- a `504` is returned only after the underlying replay work has actually stopped; capacity is not released while hidden work continues;
- PG `statement_timeout` bounds database execution;
- response size is pre-estimated from row/state count and then verified against a bounded serialization buffer before HTTP response commit;
- no silent truncation, pagination, or opaque build substitution occurs.

PR-062 must prove that at configured replay capacity, standard MLflow health/tracking/registry reads and a `latest` request remain serviceable using the pinned Gunicorn `gthread` topology; it must not claim QoS after all OS process/thread capacity is externally exhausted beyond configured admission limits.

---

# 18. MLflow package and registry contract

Registered Xetra model name is exactly `regime-xetra`.

Every registered version comes only from the final production refit and includes:

- profile ID/config version;
- statistical-winner candidate ID and K;
- source/evaluation lineage;
- frozen feature order;
- `feature_contract_hash`;
- feature-selection definition/execution hashes;
- scaler;
- initial probabilities;
- transition matrix;
- means;
- complete full covariance matrix per state;
- persistent state map/signatures;
- inference origin;
- trained-through timestamp;
- terminal filtered probabilities;
- engine package version and Git SHA;
- `data_time_semantics`;
- inference-contract version.

No DB secret is embedded.

Alias operations:

- `challenger` may point to a validated newly registered version;
- `champion` may move only via explicit operator promotion;
- promotion requires `expected_current_version` plus `new_version` and a non-empty reason;
- if current alias differs from expected, operation fails without mutation;
- rollback uses the same compare-and-swap rule;
- every alias move logs previous version, new version, UTC time, reason, operator context when available, and source evaluation run.

---

# 19. Observability

Prometheus is not enabled.

Structured logs contain only safe operational metadata such as:

```text
request_id
profile_id
operation
model_version
source_build_id
duration_ms
returned_rows
skipped_rows
HTTP status
error_code
cache hit/miss
```

Raw feature vectors, passwords, credential-bearing DSNs, model binary payloads and secrets are forbidden.

MLflow evaluation observability remains mandatory:

- parent evaluation run;
- candidate runs;
- auditable fold records;
- deterministic fold timeline;
- candidate-run `fold_*` metric histories using `step=fold_index`;
- actual TEST-end date in plots;
- invalid folds as gaps/invalid markers, never interpolation;
- transition/full-covariance artifacts and heatmaps;
- candidate comparison plots;
- feature-selection evidence and visual audit;
- deterministic `plots/manifest.json`.

`PLOT_STYLE.md` is authoritative for plot rendering and does not change statistical semantics.

---

# 20. Dependency and image reproducibility

- Python is exactly `3.14.7` for development/CI/container runtime.
- `uv.lock` is committed and CI/bootstrap uses frozen lock resolution.
- MLflow is exactly `3.15.1` until changed by a dedicated compatibility PR.
- Gaussian HMM backend is exactly `hmmlearn==0.3.3` until changed by a dedicated model-backend PR.
- PR-001 must prove that `hmmlearn==0.3.3` can install and execute a full-covariance K=2 smoke fit under Python 3.14.7. If not, work stops; no agent changes Python/backend independently.
- MLflow backend Compose image is pinned to `postgres:18.6-alpine`.
- Python Docker base is pinned to `python:3.14.7-slim-bookworm`.
- `latest` floating dependency/image tags are forbidden.
- dependency/image version changes require lock update plus relevant integration tests.

Required CI coverage threshold is 90% measured across unit+integration test execution for repository source code. External-service tests are excluded from required coverage/gates.

---

# 21. Backup, restore and database migration

Normal MLflow container startup must not automatically run a backend schema migration.

Before changing the pinned MLflow version:

1. stop/quiesce the `mlflow` service;
2. create a consistent PostgreSQL `pg_dump` of the MLflow backend;
3. archive the MLflow artifact volume;
4. create a backup manifest containing UTC timestamp, MLflow version, PostgreSQL image version, dump/artifact SHA-256 hashes;
5. run an explicit one-shot `mlflow db upgrade` step;
6. start MLflow and execute health/registry/artifact smoke verification.

Restore requires MLflow stopped, restores backend DB and matching artifact archive from one manifest, then verifies MLflow metadata and a known artifact round-trip.

---

# 22. Git and weak-agent rules

Canonical PR name:

```text
PR-<three-digit-number>-<kebab-case-slug>
```

Branch:

```text
pr/<canonical-pr-name>
```

Commit:

```text
<type>(<canonical-pr-name>): <imperative description>
```

Every implementation agent must:

1. start from up-to-date `main`;
2. show `git status --short` and `git branch --show-current` before work;
3. stop if the tree is dirty or a dependency is unmerged;
4. edit only declared allowed files;
5. never edit `BACKLOG.md`;
6. never invent constants, thresholds, names, paths, database names, aliases, or fallback implementations;
7. stop rather than broaden scope;
8. include tests in the same PR;
9. finish with empty `git status --short` and report the current branch.

Shared normative files (`BACKLOG.md`, `CONTRIBUTING.md`, `DATA_SOURCE.md`, `EVALUATION.md`, `PLOT_STYLE.md`) are contract-owner files. Weak implementation agents do not modify them unless their PR explicitly lists the exact file and purpose. The implementation PRs below are designed so parallel lanes do not modify the same source files before a declared dependency merge.

---

# 23. CI and governance

Push and merge gates each run independent parallel jobs:

```text
lint
type
unit
integration
```

- Python `3.14.7`.
- `uv sync --frozen`.
- Ruff check + format-check.
- strict mypy.
- required tests have no NAS/network dependency.
- combined unit/integration coverage must be >=90%.
- final jobs are exactly `push-gate` and `merge-gate` respectively.

Protected `main` target:

- repository exactly `SergejSchweizer/regime-engine`;
- pull requests required;
- strict required check `merge-gate`;
- conversations resolved;
- admins included;
- force push/deletion disabled;
- squash merge;
- branch deletion after merge;
- repository auto-merge enabled after governance PR.

---

# 24. Atomic implementation PRs

## Wave 0 — bootstrap/governance

### PR-001 — Bootstrap exact Python/runtime dependency contract

- **Branch:** `pr/PR-001-bootstrap-python314`
- **Depends on:** none
- **Allowed files:** `.python-version`, `.gitignore`, `pyproject.toml`, `uv.lock`, `README.md`, `src/market_regime_engine/__init__.py`, `tests/unit/test_package_smoke.py`, `tests/integration/test_hmm_backend_smoke.py`, `tests/conftest.py`, `scripts/bootstrap_venv.sh`, `scripts/bootstrap_venv.ps1`

Acceptance:

- [ ] Python exactly 3.14.7; distribution/import names match identity table.
- [ ] MLflow exactly 3.15.1 and `hmmlearn==0.3.3` locked.
- [ ] Runtime includes Pydantic, NumPy, SciPy, scikit-learn, Polars, PyArrow, psycopg, psycopg-pool, Matplotlib and MLflow.
- [ ] FastAPI/Uvicorn are not direct repository application-server dependencies; MLflow transitive dependencies do not create a standalone app.
- [ ] `uv.lock` is deterministic and bootstrap uses `uv sync --frozen`.
- [ ] Wrong Python fails bootstrap.
- [ ] Full-covariance `hmmlearn` K=2 smoke fit works on Python 3.14.7 or PR fails with no fallback.
- [ ] `.venv/`, secrets, local MLflow state, artifacts and caches ignored.

### PR-002 — Push quality gate

- **Branch:** `pr/PR-002-push-quality-gate`
- **Depends on:** PR-001
- **Allowed files:** `.github/workflows/push-gate.yml`

Acceptance:

- [ ] Branch pushes trigger parallel lint/type/unit/integration jobs.
- [ ] Python 3.14.7 and frozen lock used.
- [ ] Required jobs are hermetic.
- [ ] Coverage >=90% is enforced across test jobs using combined coverage artifacts.
- [ ] Final job name exactly `push-gate`.
- [ ] Superseded runs cancel.

### PR-003 — Merge quality gate

- **Branch:** `pr/PR-003-merge-quality-gate`
- **Depends on:** PR-001
- **Allowed files:** `.github/workflows/merge-gate.yml`

Acceptance mirrors PR-002 for PRs targeting `main`; final job exactly `merge-gate` and required external/NAS access is forbidden.

### PR-004 — Repository governance

- **Branch:** `pr/PR-004-repository-governance`
- **Depends on:** PR-003
- **Allowed files:** `scripts/configure_github_governance.sh`, `docs/repository_governance.md`

Acceptance:

- [ ] Script targets exactly `SergejSchweizer/regime-engine` and `main`.
- [ ] Requires authenticated admin `gh`.
- [ ] Enables repository auto-merge, squash merge, branch deletion after merge.
- [ ] Protects main exactly as section 23.
- [ ] Verification commands are idempotent and documented.

## Wave 1 — contracts/source/framework boundaries

### PR-005 — Synchronize durable architecture contracts

- **Branch:** `pr/PR-005-architecture-contract`
- **Depends on:** PR-001
- **Allowed files:** `ARCHITECTURE.md`, `CONTRIBUTING.md`, `DATA_SOURCE.md`, `EVALUATION.md`, `README.md`, `docs/model_lifecycle.md`

Acceptance:

- [ ] Documents canonical identities without `market-regime-engine`/`regime-engine` confusion.
- [ ] Documents current-vintage observation-day limitation and exact claim boundary.
- [ ] Documents one-port MLflow/Flask-app/Gunicorn architecture and trusted-LAN security boundary.
- [ ] Documents external feature PG and dedicated `regime-engine` reader.
- [ ] Documents complete-case HMM observation clock.
- [ ] EVALUATION exactly matches sections 8–13 of this backlog, including pinned numeric settings and final refit.
- [ ] Removes `engine-champion` alias wording; statistical champion vs `champion` alias is explicit.
- [ ] CONTRIBUTING precedence points to this consolidated backlog; no Wave-7/Wave-8 override text remains.

### PR-006 — Versioned core domain contracts

- **Branch:** `pr/PR-006-core-domain-contracts`
- **Depends on:** PR-001
- **Allowed files:** `src/market_regime_engine/contracts/*`, `tests/unit/contracts/*`

Acceptance:

- [ ] Immutable feature/source lineage includes dataset/build/hash/schema/feature versions and `data_time_semantics`.
- [ ] Profile ID is separate from config version.
- [ ] ModelSpec enforces Gaussian covariance `full` only.
- [ ] RegimePredictionV1 exact probability/entropy/confidence validation.
- [ ] Invocation/Error response contracts match section 15.
- [ ] Feature-selection definition/execution hashes are separate fields.
- [ ] Final-refit temporal fields are represented.
- [ ] No MLflow/filesystem/HTTP/model-library coupling in contract layer.

### PR-007 — Model profile schema/loader

- **Branch:** `pr/PR-007-model-profile-config`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/profiles/*`, `tests/unit/profiles/*`

Acceptance:

- [ ] Schema encodes all pinned Xetra values from sections 8–12 with no hidden defaults.
- [ ] Unknown keys fail.
- [ ] `profile_id=xetra`, `profile_config_version=1` distinct.
- [ ] Static-feature vs selection-policy source is mutually exclusive.
- [ ] Reduced covariance rejected.
- [ ] Deterministic profile hash.

### PR-008 — FeatureSource port and PostgreSQL adapter

- **Branch:** `pr/PR-008-postgres-feature-source`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/features/__init__.py`, `src/market_regime_engine/features/ports.py`, `src/market_regime_engine/features/postgres_source.py`, `tests/unit/features/test_ports.py`, `tests/integration/test_postgres_feature_source.py`

Acceptance:

- [ ] Loader-independent FeatureSource port.
- [ ] PostgreSQL adapter binds sync state and rows in one repeatable-read read-only transaction.
- [ ] Transaction closes before caller model work.
- [ ] Selection mode permits NULL but rejects non-null nonfinite values.
- [ ] Resolved-model mode returns complete-case observation sequence plus excluded-gap metadata.
- [ ] Exact ordered features, monotonic timestamp and source bounds enforced.
- [ ] No fill/imputation/carry.
- [ ] Tests are local/fake only.

### PR-009 — Train-only preprocessing

- **Branch:** `pr/PR-009-preprocessing-pipeline`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/preprocessing/*`, `tests/unit/preprocessing/*`

Acceptance:

- [ ] StandardScaler-equivalent parameters fit only on supplied retained TRAIN observations.
- [ ] Exact feature order preserved.
- [ ] Population definitions pinned by implementation tests.
- [ ] Zero/near-zero variance <=1e-12 fails.
- [ ] Serialization deterministic.
- [ ] Future rows cannot alter fitted parameters.

### PR-010 — Model adapter/artifact protocols

- **Branch:** `pr/PR-010-model-adapter-protocol`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/models/protocols.py`, `src/market_regime_engine/models/artifacts.py`, `tests/unit/models/test_protocols.py`, `tests/unit/models/test_artifacts.py`

Acceptance:

- [ ] Protocol separates fitting, parameter extraction/reconstruction and causal forward inference.
- [ ] OOS predictive scoring accepts terminal TRAIN alpha and never implicitly resets TEST.
- [ ] Artifact retains full covariances/off-diagonals and all final-refit temporal fields where applicable.
- [ ] Shape/finite/normalization validation exact.

### PR-011 — Immutable prediction store

- **Branch:** `pr/PR-011-prediction-store`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/predictions/*`, `tests/unit/predictions/*`, `tests/integration/test_prediction_store.py`

Acceptance:

- [ ] Modes exactly distinguish `walk_forward_oos` and fixed-model outputs.
- [ ] Immutable explicit build IDs, atomic Parquet+manifest writes and checksums.
- [ ] No silent latest for research reader.
- [ ] Source/build/time-semantics lineage retained.

### PR-012 — MLflow client/registry ports

- **Branch:** `pr/PR-012-mlflow-client-boundary`
- **Depends on:** PR-006
- **Allowed files:** `src/market_regime_engine/mlflow_support/settings.py`, `src/market_regime_engine/mlflow_support/ports.py`, `tests/unit/mlflow_support/test_settings.py`, `tests/unit/mlflow_support/test_ports.py`

Acceptance:

- [ ] Production tracking URI exactly `http://10.10.1.3:5000` through runtime config.
- [ ] Tracking and registry boundaries injectable/no network at import.
- [ ] Alias resolution returns exact immutable version.
- [ ] Fold history logging supports explicit step/timestamp.
- [ ] No serving URI or secret hard-coded.

### PR-013 — MLflow custom Flask app skeleton

- **Branch:** `pr/PR-013-mlflow-app-skeleton`
- **Depends on:** PR-006
- **Allowed files:** `pyproject.toml`, `src/market_regime_engine/mlflow_app/__init__.py`, `src/market_regime_engine/mlflow_app/app.py`, `src/market_regime_engine/mlflow_app/contracts.py`, `src/market_regime_engine/mlflow_app/errors.py`, `tests/unit/mlflow_app/*`

Acceptance:

- [ ] Registers `[project.entry-points."mlflow.app"] regime-engine = market_regime_engine.mlflow_app.app:create_app`.
- [ ] Factory imports/extends `mlflow.server.app`; standard routes remain.
- [ ] No network/model/PG call at import/factory creation.
- [ ] Placeholder invocation/OOS/health route registration uses Flask only.
- [ ] Standard MLflow route plus custom route both pass Flask test client.
- [ ] No standalone ASGI app.

## Wave 2 — statistical feature selection

### PR-045 — Feature-selection contracts

- **Branch:** `pr/PR-045-feature-selection-contracts`
- **Depends on:** PR-007
- **Allowed files:** `src/market_regime_engine/feature_selection/contracts.py`, `src/market_regime_engine/feature_selection/__init__.py`, `tests/unit/feature_selection/test_contracts.py`

Acceptance:

- [ ] Immutable blocks/policy/evidence/result.
- [ ] Definition hash and execution hash are separate and validate required inclusion/exclusion rules.
- [ ] Exactly one preliminary medoid per block; final is order-preserving subset.
- [ ] Canonical JSON hashing uses UTF-8 `json.dumps(sort_keys=True,separators=(",",":"),allow_nan=False)` under pinned Python and SHA-256.
- [ ] No source/model/MLflow code.

### PR-046 — Exact Xetra 48-feature/eight-block policy

- **Branch:** `pr/PR-046-xetra-feature-blocks`
- **Depends on:** PR-045
- **Allowed files:** `configs/feature_selection/xetra_semantic_medoid_v1.yaml`, `docs/profiles/xetra_feature_selection_v1.md`, `tests/unit/feature_selection/test_xetra_feature_blocks.py`

Acceptance: exact block membership/order and all policy constants from section 8; each of 48 features occurs exactly once; no portfolio/HMM target fields.

### PR-047 — Pure Stage-1 selector

- **Branch:** `pr/PR-047-spearman-medoid-selector`
- **Depends on:** PR-045
- **Allowed files:** `src/market_regime_engine/feature_selection/selector.py`, `tests/unit/feature_selection/test_selector.py`, `tests/fixtures/feature_selection/*`

Acceptance: implements section-8 Stage 1 exactly, including average-rank Spearman, `ddof=0`, variance threshold, 504 rows and `1e-12` tie tolerance; no Stage 2/source/model logic.

### PR-020 — Expanding walk-forward planner

- **Branch:** `pr/PR-020-walk-forward-splits`
- **Depends on:** PR-007, PR-008
- **Allowed files:** `src/market_regime_engine/evaluation/walk_forward_splits.py`, `tests/unit/evaluation/test_walk_forward_splits.py`

Acceptance:

- [ ] Uses source timestamps and exact 1260/63/63/no-partial policy.
- [ ] Stable one-based fold index, fold ID and UTC bounds.
- [ ] No calendar synthesis.
- [ ] Usable-model row minima are validated later after resolved mask, not by fabricating rows.
- [ ] Deterministic plan hash.

### PR-048 — Stage-2 prune and first-TRAIN freeze

- **Branch:** `pr/PR-048-prune-freeze-first-train-features`
- **Depends on:** PR-020, PR-046, PR-047
- **Allowed files:** `src/market_regime_engine/feature_selection/freeze.py`, `tests/unit/feature_selection/test_freeze.py`, `tests/integration/test_feature_selection_freeze.py`

Acceptance:

- [ ] First-fold TRAIN only.
- [ ] Fixed eight-medoid matrix and exact >0.85 conflict/removal rules.
- [ ] No replacement/recompute.
- [ ] Future-row mutation cannot alter definition hash/evidence.
- [ ] Future-row/source-build mutation may alter execution hash and test asserts that distinction.
- [ ] Selection failure prevents HMM work.

### PR-021 — Xetra profile definition

- **Branch:** `pr/PR-021-xetra-profile`
- **Depends on:** PR-007, PR-046
- **Allowed files:** `configs/profiles/xetra_v1.yaml`, `docs/profiles/xetra_v1.md`, `tests/unit/profiles/test_xetra_profile.py`

Acceptance: all exact values from sections 8–12; public ID `xetra`; config version 1; three candidates only; no agent-chosen thresholds.

### PR-049 — Resolve frozen selected features into model profile

- **Branch:** `pr/PR-049-resolve-selected-feature-profile`
- **Depends on:** PR-021, PR-048
- **Allowed files:** `src/market_regime_engine/profiles/resolution.py`, `tests/unit/profiles/test_resolution.py`, `tests/integration/test_xetra_profile_resolution.py`

Acceptance:

- [ ] Exact final feature order shared by K2/K3/K4.
- [ ] Both selection hashes and source lineage checked.
- [ ] Original 48 universe/preliminary medoids retained separately.
- [ ] Mismatched execution/source build fails candidate comparison.

## Wave 3 — HMM/evaluation

### PR-014 — Full-covariance Gaussian HMM adapter

- **Branch:** `pr/PR-014-gaussian-hmm-adapter`
- **Depends on:** PR-009, PR-010
- **Allowed files:** `src/market_regime_engine/models/gaussian_hmm.py`, `tests/unit/models/test_gaussian_hmm.py`, `tests/fixtures/hmm/*`

Acceptance:

- [ ] Backend/config exactly section 9.
- [ ] Only full covariance; reduced modes fail.
- [ ] K=2/3/4 supported.
- [ ] Full covariance validation uses section-12 tolerances.
- [ ] Off-diagonals preserved round-trip.
- [ ] Exposes stabilized forward primitives needed by PR-016; backend TEST score reset cannot masquerade as OOS PLL.

### PR-015 — Deterministic multistart

- **Branch:** `pr/PR-015-hmm-multistart`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/training/multistart.py`, `tests/unit/training/test_multistart.py`

Acceptance: exact eight seeds; 6/8 and 0.75 gates; deterministic highest TRAIN-loglik winner with stable seed tie break; every failed start retained diagnostically.

### PR-016 — Causal forward filter and continued predictive likelihood

- **Branch:** `pr/PR-016-causal-forward-filter`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/inference/filtering.py`, `src/market_regime_engine/inference/predictive_likelihood.py`, `tests/unit/inference/test_filtering.py`, `tests/unit/inference/test_predictive_likelihood.py`

Acceptance:

- [ ] Implements section 10 exactly in stabilized form.
- [ ] Accepts explicit initial alpha continuation.
- [ ] TEST continuation from terminal TRAIN alpha verified against hand fixture.
- [ ] Future appended observations cannot change earlier alpha.
- [ ] Calendar gap does not create extra transitions.

### PR-017 — Transition-horizon forecasts

- **Branch:** `pr/PR-017-transition-forecasts`
- **Depends on:** PR-016
- **Allowed files:** `src/market_regime_engine/inference/forecasting.py`, `tests/unit/inference/test_forecasting.py`

Acceptance: horizon 0=current filtered distribution; integer observation-step horizons use matrix powers; normalized finite output; explicitly not calendar-day forecast.

### PR-018 — Persistent state alignment

- **Branch:** `pr/PR-018-state-alignment`
- **Depends on:** PR-014
- **Allowed files:** `src/market_regime_engine/states/signatures.py`, `src/market_regime_engine/states/alignment.py`, `tests/unit/states/*`

Acceptance: implements section 11 exactly, including means/log-std/correlation signature, RMS distance, first-fold lexicographic rule, K! enumeration and `1e-10` ambiguity rule; drift diagnostic only.

### PR-019 — Model diagnostics

- **Branch:** `pr/PR-019-model-diagnostics`
- **Depends on:** PR-014, PR-016
- **Allowed files:** `src/market_regime_engine/evaluation/diagnostics.py`, `tests/unit/evaluation/test_diagnostics.py`

Acceptance:

- [ ] Exact AIC/BIC parameter count.
- [ ] TRAIN hard/soft occupancy and gate metrics are separate from OOS occupancy diagnostics.
- [ ] Confidence=max alpha; entropy uses natural log.
- [ ] Duration is count of consecutive retained observations; switches/year uses actual UTC span: `switch_count / elapsed_days * 365.2425`, undefined for zero elapsed span.
- [ ] Covariance/numerical validation exact.

### PR-022 — Leak-free walk-forward runner

- **Branch:** `pr/PR-022-walk-forward-runner`
- **Depends on:** PR-015, PR-016, PR-018, PR-019, PR-020, PR-049
- **Allowed files:** `src/market_regime_engine/evaluation/walk_forward.py`, `tests/unit/evaluation/test_walk_forward.py`, `tests/integration/test_walk_forward_runner.py`

Acceptance:

- [ ] Frozen features/selection never rerun in folds.
- [ ] Scaler/HMM TRAIN only.
- [ ] TEST PLL continues from TRAIN terminal alpha.
- [ ] Retained model rows follow complete-case observation-clock contract.
- [ ] Fold usable minimums 504 TRAIN/42 TEST.
- [ ] State alignment uses only TRAIN/prior valid reference.
- [ ] OOS predictions filtered only.
- [ ] Fold result has explicit invalid reasons and all section-12 evidence.
- [ ] Future data mutation cannot alter earlier fold output.

### PR-023 — MLflow evaluation tracking/plots

- **Branch:** `pr/PR-023-mlflow-experiment-tracking`
- **Depends on:** PR-012, PR-022
- **Allowed files:** `src/market_regime_engine/mlflow_support/tracking.py`, `src/market_regime_engine/mlflow_support/plots.py`, `tests/unit/mlflow_support/test_tracking.py`, `tests/unit/mlflow_support/test_plots.py`, `tests/integration/test_mlflow_file_tracking.py`

Acceptance:

- [ ] Parent/candidate/fold hierarchy and all existing PLOT_STYLE requirements.
- [ ] Candidate run contains ordered `fold_*` histories with `step=fold_index`; timestamp is TEST end where supported.
- [ ] `fold_timeline.parquet`/`fold_metrics.parquet` include invalid folds with missing metrics, never interpolated.
- [ ] Required trend, transition and per-state covariance heatmaps/artifacts remain complete.
- [ ] Plot manifest deterministic and source-linked.
- [ ] Both selection hashes/source build included in candidate/evaluation lineage.

### PR-024 — Candidate-grid orchestration

- **Branch:** `pr/PR-024-candidate-grid-orchestrator`
- **Depends on:** PR-007, PR-015, PR-022, PR-023
- **Allowed files:** `src/market_regime_engine/training/candidate_grid.py`, `tests/unit/training/test_candidate_grid.py`, `tests/integration/test_candidate_grid.py`

Acceptance:

- [ ] Exactly K2/K3/K4 full candidates.
- [ ] Same source/build/folds/features/definition+execution hashes.
- [ ] Aggregate mean/std definitions exact (`ddof=0`).
- [ ] Invalid folds excluded from means but counted.
- [ ] Cross-candidate plots aligned by fold timeline.

### PR-025 — Deterministic statistical champion selection

- **Branch:** `pr/PR-025-statistical-champion-selection`
- **Depends on:** PR-019, PR-024
- **Allowed files:** `src/market_regime_engine/evaluation/selection.py`, `tests/unit/evaluation/test_selection.py`

Acceptance: exact hard gates and seven-step ranking from section 12; zero valid candidates fails; complete rejection/ranking chain retained; terminology is statistical champion, not MLflow alias.

### PR-050 — MLflow feature-selection evidence/visual audit

- **Branch:** `pr/PR-050-mlflow-feature-selection-evidence`
- **Depends on:** PR-023, PR-048, PR-049
- **Allowed files:** `src/market_regime_engine/mlflow_support/feature_selection_tracking.py`, `tests/unit/mlflow_support/test_feature_selection_tracking.py`, `tests/integration/test_mlflow_feature_selection_tracking.py`

Acceptance:

- [ ] Logs selection JSON, scores, within-block correlations, fixed 8x8 cross-block matrix, pruning evidence.
- [ ] Logs definition and execution hashes distinctly.
- [ ] Summary markdown includes eight winners and every removal reason.
- [ ] Stage-1 score plot, exactly eight block heatmaps, and fixed pre-pruning Stage-2 heatmap.
- [ ] Removed features remain visible in Stage-2 heatmap.
- [ ] All plots obey PLOT_STYLE and manifest lineage.
- [ ] No economic metric.

### PR-027 — Immutable walk-forward OOS publication

- **Branch:** `pr/PR-027-oos-prediction-publication`
- **Depends on:** PR-011, PR-022
- **Allowed files:** `src/market_regime_engine/predictions/oos_publication.py`, `tests/unit/predictions/test_oos_publication.py`, `tests/integration/test_oos_prediction_publication.py`

Acceptance: immutable mode exactly `walk_forward_oos`; source build/time semantics/fold plan/candidate/selection hashes retained; RegimePredictionV1 rows; deterministic/idempotent.

### PR-063 — Final statistical-winner production refit

- **Branch:** `pr/PR-063-final-production-refit`
- **Depends on:** PR-015, PR-016, PR-018, PR-019, PR-025, PR-049
- **Allowed files:** `src/market_regime_engine/training/final_refit.py`, `src/market_regime_engine/models/production_artifact.py`, `tests/unit/training/test_final_refit.py`, `tests/integration/test_final_refit.py`

Acceptance:

- [ ] Implements section 13 exactly.
- [ ] Uses winning K but does not rerun feature selection or candidate ranking.
- [ ] Full evaluation-cutoff sample refit with new scaler/multistart.
- [ ] Reapplies all hard numerical/occupancy/multistart gates.
- [ ] Aligns to final valid evaluation-fold reference.
- [ ] Stores origin/trained-through/terminal alpha.
- [ ] Evaluation metrics remain unchanged.

### PR-026 — MLflow model package/registry aliases

- **Branch:** `pr/PR-026-mlflow-model-registry`
- **Depends on:** PR-012, PR-063
- **Allowed files:** `src/market_regime_engine/mlflow_support/model_package.py`, `src/market_regime_engine/mlflow_support/registry.py`, `tests/unit/mlflow_support/test_model_package.py`, `tests/unit/mlflow_support/test_registry.py`, `tests/integration/test_mlflow_registry_local.py`

Acceptance:

- [ ] Only ProductionModelArtifact from PR-063 can register.
- [ ] Registered Xetra name exactly `regime-xetra`.
- [ ] Round-trip preserves scaler/full HMM/state mapping/origin/trained-through/terminal alpha.
- [ ] Supports `challenger` and `champion`; no `engine-champion` or consumer alias in default contract.
- [ ] Promotion/rollback primitives support compare-and-swap expected current version and audit reason.

## Wave 4 — production source/runtime/inference service

### PR-057 — Create dedicated read-only feature PostgreSQL role

- **Branch:** `pr/PR-057-regime-engine-postgres-reader`
- **Depends on:** PR-005
- **Allowed files:** `ops/postgres/regime_engine_reader.sql`, `scripts/bootstrap_regime_engine_reader.sh`, `scripts/verify_regime_engine_reader.sh`, `tests/unit/ops/test_regime_engine_reader_sql.py`, `docs/ops/feature_postgres_reader.md`

Acceptance:

- [ ] Exact quoted role `"regime-engine"`.
- [ ] Idempotent create/converge; LOGIN/NOSUPERUSER/NOCREATEDB/NOCREATEROLE/NOREPLICATION/NOBYPASSRLS.
- [ ] Default transaction read-only.
- [ ] DB name/password/admin credential mandatory runtime values; no defaults/secrets echoed.
- [ ] Exact least privileges from section 5 and no extras.
- [ ] Verification uses catalog privilege checks, not destructive writes.
- [ ] Never modifies loader writer/data/server lifecycle.

### PR-058 — Pooled production PostgreSQL runtime

- **Branch:** `pr/PR-058-postgres-serving-runtime`
- **Depends on:** PR-008
- **Allowed files:** `src/market_regime_engine/features/postgres_settings.py`, `src/market_regime_engine/features/postgres_pool.py`, `tests/unit/features/test_postgres_settings.py`, `tests/unit/features/test_postgres_pool.py`, `tests/integration/test_postgres_source_runtime.py`

Acceptance:

- [ ] Exact feature env names/defaults from sections 5/17.
- [ ] Password-file support preferred; secret never logged.
- [ ] One lazy process-local psycopg pool; import makes no connection.
- [ ] acquire and statement timeouts exact.
- [ ] Maximum connection formula tested/documented.
- [ ] Hermetic local/fake integration only.

### PR-056 — Profile/model resolver and bounded per-worker cache

- **Branch:** `pr/PR-056-profile-model-resolver-cache`
- **Depends on:** PR-007, PR-012, PR-026
- **Allowed files:** `src/market_regime_engine/serving/profile_registry.py`, `src/market_regime_engine/serving/model_resolver.py`, `src/market_regime_engine/serving/model_cache.py`, `tests/unit/serving/test_profile_registry.py`, `tests/unit/serving/test_model_resolver.py`, `tests/unit/serving/test_model_cache.py`

Acceptance:

- [ ] `xetra -> regime-xetra -> champion` data-driven mapping; future crypto requires mapping entry only.
- [ ] Explicit version bypasses alias.
- [ ] Alias TTL 30 seconds default.
- [ ] Single-flight load per profile/version under concurrent requests.
- [ ] Cache max two versions/profile/worker; LRU inactive eviction.
- [ ] Full package validation before atomic replacement.
- [ ] Invalid new champion causes explicit request failure, never stale mislabelling.

### PR-059 — Replay admission/deadline/size guardrails

- **Branch:** `pr/PR-059-replay-guardrails`
- **Depends on:** PR-013
- **Allowed files:** `src/market_regime_engine/serving/replay_limits.py`, `src/market_regime_engine/serving/replay_admission.py`, `tests/unit/serving/test_replay_limits.py`, `tests/unit/serving/test_replay_admission.py`

Acceptance:

- [ ] Exact defaults from section 17; no unlimited production value.
- [ ] RFC3339 UTC/range validation occurs before dependency work.
- [ ] Process-local semaphore max one replay by default.
- [ ] No extra worker/threadpool; execution stays in current request thread.
- [ ] Monotonic cooperative deadline contract.
- [ ] Row/internal-row/range/response errors -> 413; capacity ->503; stopped deadline ->504.
- [ ] Capacity released only after actual work ends.

### PR-029 — Profile-routed latest handler

- **Branch:** `pr/PR-029-latest-handler`
- **Depends on:** PR-016, PR-018, PR-056, PR-058
- **Allowed files:** `src/market_regime_engine/inference/latest.py`, `src/market_regime_engine/serving/latest_handler.py`, `tests/unit/inference/test_latest.py`, `tests/unit/serving/test_latest_handler.py`, `tests/integration/test_latest_handler.py`

Acceptance:

- [ ] Path profile + optional as_of/model_version only; no consumer features.
- [ ] Default champion exact version pinned before source access.
- [ ] Implements section-14 continuation and complete-case latest timestamp.
- [ ] Source/model stale warn/fail policy from section 16.
- [ ] Returns exact lineage and RegimePredictionV1.
- [ ] No stale/invented fallback.

### PR-028 — Profile-routed fixed-model replay handler

- **Branch:** `pr/PR-028-replay-handler`
- **Depends on:** PR-016, PR-018, PR-056, PR-058, PR-059
- **Allowed files:** `src/market_regime_engine/inference/replay.py`, `src/market_regime_engine/serving/replay_handler.py`, `tests/unit/inference/test_replay.py`, `tests/unit/serving/test_replay_handler.py`, `tests/integration/test_replay_handler.py`

Acceptance:

- [ ] Path profile + start/end + optional model version only.
- [ ] Model pinned before source access.
- [ ] Implements section-14 warmup/invariance exactly.
- [ ] Current-vintage source semantics explicit in response.
- [ ] All replay guardrails applied; no truncation/build substitution.
- [ ] Overlap-invariance test with different request starts.

### PR-030 — Profile-scoped immutable OOS retrieval handler

- **Branch:** `pr/PR-030-oos-prediction-handler`
- **Depends on:** PR-011, PR-027
- **Allowed files:** `src/market_regime_engine/predictions/query.py`, `src/market_regime_engine/serving/oos_handler.py`, `tests/unit/predictions/test_query.py`, `tests/unit/serving/test_oos_handler.py`, `tests/integration/test_oos_handler.py`

Acceptance:

- [ ] Explicit profile + build ID; optional UTC slice.
- [ ] Never resolves latest OOS implicitly.
- [ ] Rejects mode mismatch.
- [ ] No Flask route in this PR; PR-060 composes it.

### PR-060 — Compose final MLflow app service graph

- **Branch:** `pr/PR-060-compose-profile-service`
- **Depends on:** PR-013, PR-028, PR-029, PR-030, PR-056, PR-058, PR-059
- **Allowed files:** `src/market_regime_engine/mlflow_app/app.py`, `src/market_regime_engine/mlflow_app/dependencies.py`, `src/market_regime_engine/mlflow_app/dispatch.py`, `tests/unit/mlflow_app/test_dispatch.py`, `tests/integration/test_mlflow_app_invocations.py`

Acceptance:

- [ ] Exact routes from section 3/15.
- [ ] Body schemas reject unknown/forbidden fields and body profile ID.
- [ ] Error envelope/status mapping exact.
- [ ] Request IDs present.
- [ ] Standard MLflow UI/health/tracking/registry routes remain functional.
- [ ] Component health exposes no secrets and does not run full inference.
- [ ] No Prometheus route/exporter.

### PR-031 — Operator CLI

- **Branch:** `pr/PR-031-application-cli`
- **Depends on:** PR-024, PR-026, PR-027, PR-050, PR-063
- **Allowed files:** `src/market_regime_engine/cli.py`, `src/market_regime_engine/commands/*`, `tests/unit/test_cli.py`, `pyproject.toml`

Acceptance:

- [ ] Commands for evaluate, final-refit, register, publish-oos and status.
- [ ] No `serve` command for standalone API; serving belongs to MLflow container.
- [ ] Thin orchestration; no model math.
- [ ] OOS publication dependency is explicit.

### PR-064 — Model lifecycle, promotion/rollback and freshness workflow

- **Branch:** `pr/PR-064-model-lifecycle-operations`
- **Depends on:** PR-031, PR-056
- **Allowed files:** `src/market_regime_engine/commands/lifecycle.py`, `scripts/model_cycle.sh`, `tests/unit/commands/test_lifecycle.py`, `docs/operations/model_lifecycle.md`

Acceptance:

- [ ] `promote` and `rollback` require expected-current-version, target version and reason.
- [ ] Compare-and-swap mismatch produces no alias mutation.
- [ ] `model-cycle` detects new upstream source build; unchanged build is no-op.
- [ ] On changed build: evaluate -> select -> final refit -> register challenger; champion promotion remains explicit, not automatic.
- [ ] Recommended production cadence exactly once every 7 days after source synchronization; scheduling mechanism remains operator/NAS-owned.
- [ ] Warn/fail freshness values from section 16 documented/tested.
- [ ] No portfolio metric or uncalibrated online drift detector changes aliases.

## Wave 5 — image/compose/external verification/capacity/operations

### PR-032 — Unified MLflow/regime-engine image

- **Branch:** `pr/PR-032-container-image`
- **Depends on:** PR-031, PR-060
- **Allowed files:** `Dockerfile`, `.dockerignore`, `scripts/mlflow_entrypoint.sh`, `scripts/mlflow_db_upgrade.sh`, `tests/integration/test_container_image_contract.py`, `docs/container_image.md`

Acceptance:

- [ ] Base exactly `python:3.14.7-slim-bookworm`; package installed from frozen lock/build.
- [ ] MLflow exactly 3.15.1.
- [ ] Entrypoint starts one MLflow process with exact Gunicorn command/defaults.
- [ ] Does not run DB upgrade automatically.
- [ ] Separate one-shot upgrade script exists and never logs DSN secrets.
- [ ] No `mlflow models serve`, Uvicorn serving mode, reverse proxy or Prometheus.
- [ ] Non-root runtime.

### PR-033 — Real feature-PostgreSQL compatibility smoke

- **Branch:** `pr/PR-033-feature-postgres-smoke`
- **Depends on:** PR-021, PR-057, PR-058
- **Allowed files:** `tests/external/test_feature_postgres_external.py`, `scripts/verify_feature_postgres.py`, `tests/fixtures/loader_gold/*`, `tests/integration/test_loader_gold_contract.py`

Acceptance:

- [ ] Required integration is hermetic.
- [ ] Opt-in external target exactly `10.10.1.3:54321`; database/password runtime-required.
- [ ] Authenticates as `regime-engine`, proves exact SELECT/transaction path and privilege metadata.
- [ ] Never attempts production writes or uses loader writer.

### PR-061 — Exact two-service Compose deployment

- **Branch:** `pr/PR-061-two-service-mlflow-compose`
- **Depends on:** PR-032, PR-057, PR-058, PR-059, PR-060
- **Allowed files:** `compose.example.yaml`, `.env.example`, `tests/integration/test_compose_config.py`, `docs/deployment.md`

Acceptance:

- [ ] Exactly services `mlflow` and `mlflow-postgres`.
- [ ] MLflow backend image exactly `postgres:18.6-alpine`.
- [ ] Only host port 5000.
- [ ] Persistent backend and artifact volumes.
- [ ] Compose secrets used for MLflow backend password and feature-PG password; examples contain placeholders only.
- [ ] Feature PostgreSQL remains external.
- [ ] All exact worker/thread/cache/pool/replay/staleness/BLAS defaults exposed.
- [ ] Allowed-host/CORS policy follows trusted-LAN section; no wildcards.
- [ ] No Prometheus, proxy, model-server, port 5001.
- [ ] Normal startup does not automatically migrate MLflow DB.

### PR-034 — Unified real MLflow service smoke

- **Branch:** `pr/PR-034-external-mlflow-smoke`
- **Depends on:** PR-023, PR-026, PR-061
- **Allowed files:** `tests/external/test_mlflow_external.py`, `tests/external/test_regime_service_external.py`, `scripts/verify_shared_mlflow.py`

Acceptance:

- [ ] External-only and excluded from required CI.
- [ ] Exactly `http://10.10.1.3:5000` serves standard MLflow and custom health.
- [ ] Disposable tracking/registry/artifact/history tests round-trip.
- [ ] If `regime-xetra@champion` exists, opt-in latest call is read-only and verified.
- [ ] No :5001/proxy/Prometheus assumptions.

### PR-062 — Hermetic capacity/failure-isolation proof

- **Branch:** `pr/PR-062-serving-capacity-proof`
- **Depends on:** PR-060, PR-061
- **Allowed files:** `tests/integration/test_serving_capacity.py`, `tests/integration/test_serving_failure_isolation.py`, `tests/fixtures/serving/*`

Acceptance:

- [ ] No NAS access.
- [ ] Cache warm/single-flight/atomic version change/LRU behavior proven.
- [ ] PG acquire timeout/exhaustion bounded.
- [ ] Replay range/rows/internal rows/bytes ->413 with no partial output.
- [ ] Replay admission saturation ->503 and recovers after actual completion.
- [ ] Cooperative deadline ->504 only after work stops and slot/resources released.
- [ ] At admitted replay capacity, MLflow health/tracking/registry read and latest remain responsive under pinned 4x4 gthread test topology.
- [ ] No secret/raw-feature log leakage.
- [ ] No Prometheus configuration.

### PR-065 — MLflow backup/restore and controlled migration

- **Branch:** `pr/PR-065-mlflow-backup-restore`
- **Depends on:** PR-061
- **Allowed files:** `scripts/backup_mlflow.sh`, `scripts/restore_mlflow.sh`, `scripts/verify_mlflow_backup.sh`, `tests/integration/test_backup_manifest_contract.py`, `docs/operations/backup_restore.md`

Acceptance:

- [ ] Backup stops/quiesces `mlflow`, dumps backend DB, archives artifacts, hashes both, writes one manifest, restarts service.
- [ ] Manifest records UTC, MLflow version, Postgres image version and SHA-256 values.
- [ ] Restore requires MLflow stopped and matching dump/artifact manifest.
- [ ] Restore verification checks metadata plus known artifact.
- [ ] Version-upgrade documentation mandates successful backup before explicit `mlflow db upgrade`.
- [ ] No automatic migration on ordinary start.

### PR-035 — Complete hermetic engine E2E proof

- **Branch:** `pr/PR-035-engine-e2e-proof`
- **Depends on:** PR-024, PR-026, PR-027, PR-028, PR-029, PR-030, PR-050, PR-060, PR-062, PR-063
- **Allowed files:** `tests/integration/test_engine_e2e.py`, `tests/fixtures/e2e/*`

Acceptance:

- [ ] Full 48-feature source fixture through selection -> K2/K3/K4 evaluation -> champion -> final refit -> local MLflow register.
- [ ] Selection definition hash future-row invariant and execution hash lineage behavior proven.
- [ ] TEST PLL continuation from TRAIN alpha proven.
- [ ] Initial/later/final-refit state alignment exact.
- [ ] All MLflow fold/history/plot/feature-selection visual evidence present.
- [ ] Final registered model is final refit, not a fold model.
- [ ] Latest and replay exact profile route exercised; different-start replay overlap invariance proven.
- [ ] Current-vintage time-semantics metadata present.
- [ ] OOS retrieval stays distinct from replay.
- [ ] All limit/error contracts exercised.
- [ ] No NAS dependency.

### PR-036 — Final operator/consumer documentation consistency

- **Branch:** `pr/PR-036-final-documentation`
- **Depends on:** PR-033, PR-034, PR-035, PR-061, PR-064, PR-065
- **Allowed files:** `README.md`, `API.md`, `OPERATIONS.md`, `ARCHITECTURE.md`, `DATA_SOURCE.md`, `EVALUATION.md`, `CONTRIBUTING.md`, `docs/consumer_contract.md`, `docs/integrations/portfell.md`, `docs/integrations/mlflow.md`

Acceptance:

- [ ] No obsolete Addendum/Wave-override wording.
- [ ] Canonical identities exactly match section 1.
- [ ] Documents current-vintage vs point-in-time limitation prominently.
- [ ] Documents complete-case observation clock and predictive-likelihood continuation.
- [ ] Documents feature-selection definition/execution hash distinction.
- [ ] Documents final refit before registry.
- [ ] Documents one-port Gunicorn MLflow app, trusted-LAN boundary, exact Compose topology, role `regime-engine` and no Prometheus.
- [ ] API latest/replay/OOS/error schemas exact.
- [ ] Documents replay warmup/start invariance, operational limits and staleness.
- [ ] Documents lifecycle/promote/rollback/model-cycle and backup/restore/migration.
- [ ] Portfell boundary and non-economic engine selection explicit.
- [ ] No implementation/config/document contradiction remains.

## Optional challengers after MVP

### PR-037 — Student-t HMM challenger

- **Branch:** `pr/PR-037-student-t-hmm-challenger`
- **Depends on:** PR-010, PR-022, PR-036
- **Allowed files:** dedicated adapter/tests/profile extension docs only.
- Must preserve full covariance, causal continuation, common feature/fold/evaluation contracts and require a separate explicit dependency/backend decision; it cannot alter MVP silently.

### PR-038 — HSMM challenger

- **Branch:** `pr/PR-038-hsmm-challenger`
- **Depends on:** PR-010, PR-022, PR-036
- Same isolation rule; duration semantics and any protocol extension must be explicit before candidate inclusion.

---

# 25. Parallel execution plan

Only merged dependencies unlock a PR.

```text
A1:
  PR-001

A2 parallel:
  PR-002  PR-003  PR-005  PR-006

A3:
  PR-004 after PR-003

B1 parallel after PR-006:
  PR-007  PR-008  PR-009  PR-010  PR-011  PR-012  PR-013

B2:
  PR-045 after PR-007
  PR-014 after PR-009+010
  PR-020 after PR-007+008
  PR-057 after PR-005
  PR-058 after PR-008
  PR-059 after PR-013

B3 parallel:
  PR-046 + PR-047 after PR-045
  PR-015 + PR-016 + PR-018 after PR-014

B4:
  PR-019 after PR-014+016
  PR-048 after PR-020+046+047
  PR-021 after PR-007+046

B5:
  PR-049 after PR-021+048

C1:
  PR-022 after 015+016+018+019+020+049

C2 parallel:
  PR-023 after PR-012+022
  PR-027 after PR-011+022

C3 parallel:
  PR-024 after PR-007+015+022+023
  PR-050 after PR-023+048+049

C4:
  PR-025 after PR-019+024
  PR-063 after PR-015+016+018+019+025+049
  PR-026 after PR-012+063

D1 parallel after PR-026/source/runtime dependencies:
  PR-056
  PR-033 when 021+057+058 ready

D2 parallel:
  PR-029 after 016+018+056+058
  PR-028 after 016+018+056+058+059
  PR-030 after 011+027

D3:
  PR-060 after 013+028+029+030+056+058+059
  PR-031 after 024+026+027+050+063

D4 parallel:
  PR-032 after 031+060
  PR-064 after 031+056

D5:
  PR-061 after 032+057+058+059+060

D6 parallel:
  PR-034 after 023+026+061
  PR-062 after 060+061
  PR-065 after 061

D7:
  PR-035 after all declared deps

D8:
  PR-036 final
```

Parallel lanes intentionally have disjoint source ownership wherever they overlap in time. If Git reports a conflict in a contract-owner file, the later PR rebases after the dependency merge rather than resolving by broad rewriting.

---

# 26. Complete MVP definition

MVP is complete only when all required PRs through PR-036 plus PR-056–065 are merged and:

- exact Python/dependency/container pins are reproducible;
- CI/gate coverage is >=90%;
- main governance is enabled;
- production feature source is external PostgreSQL with exact read-only `regime-engine` role;
- time semantics are honestly labelled current-vintage observation-day and never claimed point-in-time tradable;
- missing-value/retained-observation semantics are identical across selection/evaluation/refit/serving;
- 48 -> eight medoids -> deterministic >0.85 pruning -> frozen d<=8 works exactly;
- definition hash is future-row invariant and execution hash preserves complete lineage;
- K2/K3/K4 full-covariance candidates use exact pinned backend/settings;
- OOS PLL continues from terminal TRAIN alpha;
- state signatures/alignment are fully deterministic;
- hard gates/ranking use exact numeric definitions;
- statistical winner is refit on the full evaluation-cutoff sample before registration;
- final model stores inference origin/trained-through/terminal alpha;
- latest/replay are causal and replay result overlap is independent of caller start;
- one MLflow/Gunicorn service at `10.10.1.3:5000` serves tracking/registry/UI/artifacts/profile API;
- Compose has exactly `mlflow` + private `mlflow-postgres` using pinned PostgreSQL 18.6;
- no reverse proxy, second model server, FastAPI app, port 5001 or Prometheus exists;
- replay load is bounded and does not silently starve standard admitted MLflow/latest traffic;
- aliases use compare-and-swap auditable promotion/rollback;
- model/source staleness and weekly reevaluation workflow are defined;
- MLflow backend/artifacts have tested backup/restore and controlled migration;
- real NAS PostgreSQL and MLflow have explicit opt-in smoke paths;
- final documentation has no stale names, aliases, framework assumptions or duplicated override sections.
