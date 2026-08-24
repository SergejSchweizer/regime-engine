# Xetra feature-selection policy v1

`xetra_semantic_medoid_v1` partitions the 48 `regime-loader` feature-version-1 inputs into eight ordered, exhaustive semantic blocks. The policy first selects one representative per block with the absolute-Spearman medoid rule and subsequently applies the fixed cross-block absolute-Spearman pruning rule defined by `EVALUATION.md`.

The canonical block order is: US equity volatility spot, US equity volatility term structure, European equity volatility, rates volatility, systemic stress, credit stress, rates/yield curve, and USD FX. The configured block sizes are respectively 4, 21, 4, 4, 3, 3, 7, and 2, for 48 unique source features total.

Semantic blocks prevent the large VIX-family feature family from dominating the full-covariance Gaussian HMM merely because it contributes many highly related levels, changes, z-scores, ratios, and term-structure spreads. Stage 1 therefore produces exactly eight preliminary representatives in canonical block order. Stage 2 may remove cross-block representatives whose fixed first-TRAIN absolute Spearman correlation is strictly greater than 0.85, so the final production/evaluation dimension is an order-preserving subset of those eight representatives rather than an unconditional eight-feature set.

The policy is statistical only. `timestamp_m1`, ETF returns, portfolio metrics, trading targets, regime profitability, Sharpe/Sortino/Calmar values, drawdowns, and labels are not candidate inputs. Feature selection uses only the first planned walk-forward TRAIN sample and is frozen for evaluation; downstream economic evaluation cannot alter it.
