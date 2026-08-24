# Model lifecycle

The production alias `champion` is distinct from the statistical champion chosen during evaluation. A statistical champion identifies the winning Gaussian-HMM candidate family/K; an MLflow `champion` alias is assigned only to a separately fitted final-production artifact.

## Evaluation and final refit

1. Bind one current-vintage source build and its exact lineage.
2. Build the deterministic expanding walk-forward plan.
3. Select/freeze the Xetra feature subset using first-fold TRAIN rows only.
4. Evaluate K=2/K=3/K=4 full-covariance candidates with causal TEST likelihood continuation and persistent state alignment.
5. Apply hard gates and deterministic statistical ranking.
6. Refit the winning K from scratch on the complete-case source sequence through the exact final evaluation cutoff using a fresh full-sample scaler and the exact eight-seed multistart policy.
7. Align final-refit states to the last valid evaluation fold of the winning K and persist `inference_origin_timestamp`, `trained_through_timestamp`, and terminal filtered probabilities.
8. Register only that final-refit artifact as a version of `regime-xetra`.

Walk-forward OOS evidence is immutable and is not rewritten by final refit.

## Promotion and rollback

Promotion and rollback use compare-and-swap inputs: `expected_current_version`, target version, and a non-empty reason. A mismatch performs no registry mutation. Every alias move is auditable. Aliases are limited to `challenger` and `champion`.

The recommended model-cycle cadence is every seven days after upstream synchronization. Scheduling is operator/NAS owned; the engine performs one deterministic reevaluation/refit cycle when invoked.

## Freshness

Staleness is elapsed UTC seconds divided by exactly 86400. Default-champion latest uses prediction timestamp for source staleness and the gap from `trained_through_timestamp` to prediction timestamp for model staleness. Warn thresholds degrade health while serving continues; fail thresholds reject default-champion latest. Explicit-version historical replay remains available when its requested source/model contracts remain valid.

## Serving initialization

Inference after `trained_through_timestamp` continues from the stored terminal filtered probabilities. Requests that include an earlier timestamp filter from the stored inference origin and model initial probabilities. A caller-supplied replay start is never treated as a fresh HMM initial condition, so overlapping timestamps for the same exact model version and source build are invariant to the requested start.
