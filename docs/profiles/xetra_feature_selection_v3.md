# Xetra feature selection v3

`xetra_semantic_medoid_v3` is a versioned extension of the historical v1/v2 policy. It does not mutate either older policy or their reproducibility hashes. The immutable feature-selection domain contracts explicitly recognize v3 as a supported policy/result identity.

The policy retains the same eight semantic blocks, Stage-1 absolute-Spearman medoid selection, Stage-2 absolute-Spearman pruning, first-fold TRAIN-only semantics, and all v2 thresholds. Its canonical universe contains 61 unique features: the 48 v2 features plus 13 existing one-observation delta columns from PostgreSQL.

The added columns are assigned as follows:

- US equity volatility spot: `vix_delta_1obs`.
- US equity volatility term structure: `vix9d_delta_1obs`, `vix3m_delta_1obs`, `vix6m_delta_1obs`, `vix1y_delta_1obs`.
- Europe equity volatility: `vstoxx_delta_1obs`.
- Rates volatility: `move_delta_1obs`.
- Systemic stress: `ciss_delta_1obs`.
- Credit stress: `euro_hy_oas_delta_1obs`.
- Rates / yield curve: `us_2y_delta_1obs`, `us_10y_delta_1obs`, `estr_delta_1obs`.
- USD FX: `usd_broad_delta_1obs`.

A delta feature is a normal Stage-1 candidate. If it becomes its block medoid and survives Stage 2, it is part of the frozen multivariate feature set. HMM performance and downstream economic metrics do not feed back into feature selection.

The normative statistical contract remains `EVALUATION.md`; this document is a profile-specific implementation reference.
