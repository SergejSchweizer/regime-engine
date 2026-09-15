# Hermetic K-champion portfolio QA

`tests/e2e/test_k_champion_portfolio.py` is the local, no-network PR-429
proof. It uses the production K-slot outer-policy, deployment-selection,
Model-Metrics, plot-payload, and registry-boundary modules. Numerical models
are not mocked: every fold/K evaluates real Gaussian-HMM, two-mixture
GMM-HMM, and Student-t HMM adapters on deterministic fixture data.

The fixture uses three expanding outer folds, K=2..5, and a different frozen
feature tuple for each K. The three family fits within one K receive the same
ordered tuple. Candidate fits, selected refits, and teacher fits are submitted
to process pools; native numerical thread pools are capped by the existing
`cpu_process_pool` initializer. Outer evidence and plot payloads are assembled
in canonical order.

Run the focused proof without the repository-wide xdist fan-out (the proof
creates its own process pools):

```bash
.venv/bin/pytest -n 0 -q tests/e2e/test_k_champion_portfolio.py
```

The proof checks:

- real Gaussian/GMM/Student-t computation for every K and every outer fold;
- shared per-K feature identity and independent K-specific feature tuples;
- process/serial outer-result hash parity and independent-process canonical
  payload hash parity;
- one deployment package and one selected LoggedModel projection per eligible
  K, with exactly one local registry version and `champion-k2` through
  `champion-k5` alias target;
- ineligible K behavior: no selected model, package, plot, or alias;
- eight per-K plot payloads (two metrics × four K slots), each with three
  family series and `no_evaluation_recomputation=True`;
- rejection of cross-K raw likelihood, AIC, and BIC plotting;
- mutation of rows after the validation cutoff cannot change completed outer
  selections or promotion evidence.

The module is marked `integration` and `slow`, so it is excluded from the
normal pre-commit integration lane and run explicitly by the local proof
workflow. The independent-process test rebuilds the real numerical portfolio
and compares the complete outer-evidence hash with the fixture run.

This is intentionally hermetic. It does not contact NAS PostgreSQL, remote
MLflow, or the full evaluation. The registry and package boundaries are local
test doubles/filesystem fixtures, so external MLflow side-effect, LoggedModel
lineage, and remote alias durability remain PR-430/registry acceptance work.
