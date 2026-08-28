# Evaluation metrics sidecar

This sidecar defines every numerical metric used by the Xetra evaluation
pipeline. It is descriptive only: source code and `EVALUATION.md` remain the
executable and selection-contract authorities.

## Notation

- (c): candidate; (f): planned walk-forward fold; (F_c): valid folds of
  candidate (c); (n_f): retained TEST observations in fold (f).
- (y_t): observation at time (t); (mathcal{D}_{f,\mathrm{train}}) and
  (mathcal{D}_{f,\mathrm{test}}): fold-local retained TRAIN and TEST data.
- (K): state count; (p): fitted free-parameter count; (gamma_t(k)):
  filtered probability of state (k) at time (t).

## Fold-level likelihood and information criteria

| Metric | Equation | Use |
|---|---|---|
| TRAIN log likelihood | (ell_{\mathrm{train},f}=\sum_{t\in\mathcal{D}_{f,\mathrm{train}}}\log p(y_t\mid y_{<t})) | Canonical causal-filter likelihood after fit/filter parity passes; input to AIC and BIC. |
| OOS predictive log likelihood | (ell_{\mathrm{OOS},f}=\sum_{t\in\mathcal{D}_{f,\mathrm{test}}}\log p(y_t\mid y_{<t},\mathcal{D}_{f,\mathrm{train}})) | Diagnostic fold total; TEST continuation is causal and starts from the terminal TRAIN filter. |
| OOS PLL per observation | (q_f=\ell_{\mathrm{OOS},f}/n_f) | Primary fold score. Higher is better. |
| AIC | (mathrm{AIC}_f=2p-2\ell_{\mathrm{train},f}) | Complexity-penalized TRAIN diagnostic and fifth candidate ranking stage; lower is better. |
| BIC | (mathrm{BIC}_f=p\log(n_{\mathrm{train},f})-2\ell_{\mathrm{train},f}) | Complexity-penalized TRAIN diagnostic and fourth candidate ranking stage; lower is better. |
| Fit/filter parity delta | (Delta_f=|\ell_{\mathrm{fit},f}-\ell_{\mathrm{train},f}|) | Hard fold validity check: (Delta_f\le10^{-10}\max(1,|\ell_{\mathrm{fit},f}|,|\ell_{\mathrm{train},f}|)). |

## Candidate aggregation and hard gates

All candidate comparison metrics use valid folds only. For final champion
selection, they are recomputed over the common valid-fold intersection of all
accepted candidates; invalid folds are never imputed.

| Metric | Equation | Use |
|---|---|---|
| Valid-fold rate | (r_c=|F_c|/|F_{\mathrm{planned}}|) | Hard gate: (r_c\ge0.80). |
| Common-valid-fold rate | (r_{\cap}=|\cap_c F_c|/|F_{\mathrm{planned}}|) | Comparability gate after candidate gates: (r_{\cap}\ge0.80). |
| Mean OOS PLL/obs | (ar q_c=|F|^{-1}\sum_{f\in F}q_f) | Ranking stage 1; higher is better. |
| OOS PLL/obs population std | (s_c=\sqrt{|F|^{-1}\sum_{f\in F}(q_f-\bar q_c)^2}) | Ranking stage 2; lower is better. Uses `ddof=0`. |
| Worst-fold OOS PLL/obs | (w_c=\min_{f\in F}q_f) | Ranking stage 3; higher is better. |
| Best-fold OOS PLL/obs | (b_c=\max_{f\in F}q_f) | Diagnostic only; not a ranking stage. |
| Mean BIC | (overline{\mathrm{BIC}}_c=|F|^{-1}\sum_{f\in F}\mathrm{BIC}_f) | Ranking stage 4; lower is better. |
| Mean AIC | (overline{\mathrm{AIC}}_c=|F|^{-1}\sum_{f\in F}\mathrm{AIC}_f) | Ranking stage 5; lower is better. |

The remaining exact ranking stages are state count (K) ascending, then
candidate ID lexicographically ascending. Every numeric stage uses an anchored
absolute tolerance of (10^{-12}): candidates within the anchored tie set
advance together to the next stage.

## State-occupancy diagnostics and gates

| Metric | Equation | Use |
|---|---|---|
| Soft occupancy | (o_k^{\mathrm{soft}}=T^{-1}\sum_{t=1}^{T}\gamma_t(k)) | TRAIN hard/soft occupancy gates and OOS diagnostics. |
| Hard occupancy | (o_k^{\mathrm{hard}}=T^{-1}\sum_{t=1}^{T}\mathbf{1}[k=\arg\max_j\gamma_t(j)]) | TRAIN hard/soft occupancy gates and OOS diagnostics. |
| TRAIN hard minimum | (min_k o_{k,\mathrm{train}}^{\mathrm{hard}}) | Hard gate: at least (0.03). |
| TRAIN soft minimum | (min_k o_{k,\mathrm{train}}^{\mathrm{soft}}) | Hard gate: at least (0.05). |

## Univariate agreement metrics

Medoid-univariate and delta-1-univariate candidate grids use the same
candidate metrics above per feature. Their *feature* champion is selected from
agreement with the multivariate statistical champion, not from an economic
metric.

| Metric | Equation | Use |
|---|---|---|
| Shared-fold rate | (r_{\mathrm{shared}}=|F_{\mathrm{uni}}\cap F_{\mathrm{multi}}|/|F_{\mathrm{planned}}|) | Feature eligibility gate: at least (0.80). |
| Shared timestamp count | (N_{\mathrm{shared}}=|\{t:\text{aligned uni and multi states exist}\}|) | Tie-break after NMI. |
| Dominant-state NMI | (mathrm{NMI}(U,M)=I(U;M)/\sqrt{H(U)H(M)}) | Feature ranking stage 1; higher is better. (U) and (M) are dominant (`argmax`) aligned state sequences. |

For NMI ties within (10^{-12}), the largest shared timestamp count wins; a
remaining tie is resolved by the canonical feature-name order.

## Diagnostic-only quantities

Observation-weighted pooled OOS likelihood,
(sum_f\ell_{\mathrm{OOS},f}/\sum_fn_f), may be shown in plots but is never
a candidate ranking, winner, registry, or promotion input. State persistence,
transition probabilities, feature-attribution plots, alignment RMS/drift and
forecast probabilities are model diagnostics, not candidate-selection metrics.
