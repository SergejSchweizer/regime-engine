from __future__ import annotations

import inspect

import market_regime_engine.commands.v4_backend as backend


def test_xetra_backend_call_graph_has_only_the_canonical_evaluation_boundary() -> None:
    source = inspect.getsource(backend)

    assert "run_canonical_xetra_evaluation" in source
    assert "FeatureSelectionMetadataStore" in source
    assert "fit_family_pca_stages" in source
    assert "CanonicalRefitValidation" in source
    assert "evaluate_global_regime_v4_from_source" not in source
    assert "select_deployment_configuration" not in source
    assert "run_walk_forward_candidate" not in source
