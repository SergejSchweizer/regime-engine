# PCA feature-generator rollout

PCA is a mandatory part of the canonical Xetra v4 profile. There is no
`pca.enabled` option and no raw-feature-only canonical evaluation; raw-only
input is not acceptance evidence for v4. The fixed component count and
variance threshold are part of the profile hash:

```yaml
pca:
  variance_threshold: 0.90
  component_count: 8
```

The rollout contract is:

1. Fit PCA only on complete rows in the frozen TRAIN clock. Components are
   sign-canonicalized and named `pca_pc_001`, `pca_pc_002`, and so on.
2. Preserve raw columns and append generated columns to the immutable catalog
   and snapshot. Both raw and generated columns then enter the same global
   feature universe and pass through the same quality, distance, clustering,
   scoring, prefix-search, and final-grid stages. The catalog, materialized
   data, PCA fit, and source lineage are hash-bound.
3. Route the generated snapshot through the same v4 discovery/teacher/prefix/
   final-grid policy. Every walk-forward fold refits PCA on that fold's raw
   TRAIN rows and transforms TEST only with that fold artifact.
4. Fit the final production artifact with a separate `production` PCA clock.
   The resulting `PCATwoStageScalerArtifact` is stored in the production
   package and validated against the HMM/scaler feature order.

The process-backed candidate, fold, and multistart schedulers retain the same
CPU budget on the mandatory PCA path. PCA is fitted on each outer/inner TRAIN
clock only, while the generated columns remain in the fixed catalog universe.
Results are reassembled in canonical order, so worker completion order cannot
change statistical hashes.

Raw-only data remains useful for explicitly named unit fixtures and matched
diagnostic comparisons, but it is not a canonical Xetra v4 deployment input.

The rollout proof is intentionally local and synthetic. Run the focused
checks without contacting PostgreSQL or MLflow:

```bash
PYTHONPATH=src .venv/bin/pytest -n auto \
  tests/unit/preprocessing \
  tests/unit/evaluation/test_walk_forward_pca.py \
  tests/unit/evaluations/test_pca_v4.py \
  tests/unit/training/test_final_refit.py::test_final_refit_reconstructs_and_persists_fold_independent_pca
```

The full current-source evaluation is not part of the profile/configuration
proof or of the merge/push gates.
