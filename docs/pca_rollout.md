# PCA feature-generator rollout

PCA is an explicit opt-in of the Xetra v4 profile. The checked-in profile
keeps `pca.enabled: false` so a raw-feature evaluation cannot silently change
its statistical universe. Enabling it changes the profile hash and requires a
`PCAGeneratedFeatureSet` from the complete raw catalog:

```yaml
pca:
  enabled: true
  variance_threshold: 0.90
```

The rollout contract is:

1. Fit PCA only on complete rows in the frozen TRAIN clock. Components are
   sign-canonicalized and named `pca_pc_001`, `pca_pc_002`, and so on.
2. Preserve raw columns and append generated columns to the immutable catalog
   and snapshot. The catalog, materialized data, PCA fit, and source lineage
   are hash-bound.
3. Route the generated snapshot through the same v4 discovery/teacher/prefix/
   final-grid policy. Every walk-forward fold refits PCA on that fold's raw
   TRAIN rows and transforms TEST only with that fold artifact.
4. Fit the final production artifact with a separate `production` PCA clock.
   The resulting `PCATwoStageScalerArtifact` is stored in the production
   package and validated against the HMM/scaler feature order.

The process-backed candidate, fold, and multistart schedulers retain the
same CPU budget when PCA is enabled. Results are reassembled in canonical
order, so worker completion order cannot change statistical hashes.

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
