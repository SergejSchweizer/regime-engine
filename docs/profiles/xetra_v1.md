# Xetra model profile v1

The public profile ID is exactly `xetra` with `profile_config_version=1`. Its registered MLflow model is `regime-xetra`; production and non-production aliases are `champion` and `challenger`.

The profile resolves features through `xetra_semantic_medoid_v1`, not a static feature list. It pins the first-TRAIN feature-selection thresholds, expanding walk-forward plan, exact K=2/K=3/K=4 full-covariance Gaussian HMM candidate set, deterministic eight-seed multistart settings, occupancy gates, numerical tolerances, and candidate-ranking tolerance defined by `EVALUATION.md`.

The profile does not contain portfolio, ETF-return, trading-target, or economic-ranking settings. Those concerns are outside regime-engine statistical champion selection.
