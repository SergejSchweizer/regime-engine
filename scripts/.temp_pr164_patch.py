from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected patch anchor missing in {path}: {old[:80]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/market_regime_engine/models/protocols.py",
    '''@dataclass(frozen=True, slots=True)\nclass FitResult:\n    artifact: GaussianHMMArtifact\n    train_log_likelihood: float\n    converged: bool\n    iterations: int\n    seed: int\n''',
    '''@dataclass(frozen=True, slots=True)\nclass FitResult:\n    artifact: GaussianHMMArtifact\n    train_log_likelihood: float\n    converged: bool\n    iterations: int\n    seed: int\n    em_log_likelihood_history: tuple[float, ...] = ()\n\n    def __post_init__(self) -> None:\n        history = self.em_log_likelihood_history\n        if history and len(history) != self.iterations:\n            raise ValueError("EM log-likelihood history length must equal completed iterations")\n        if history and not all(np.isfinite(value) for value in history):\n            raise ValueError("EM log-likelihood history must contain only finite values")\n''',
)

replace_once(
    "src/market_regime_engine/models/gaussian_hmm.py",
    '''def _has_material_likelihood_regression(history: Iterable[float]) -> bool:\n    values = tuple(float(value) for value in history)\n    return any(\n        current - previous < -_HMMLEARN_REGRESSION_TOLERANCE\n        for previous, current in pairwise(values)\n    )\n\n\n@dataclass(frozen=True, slots=True)\n''',
    '''def _has_material_likelihood_regression(history: Iterable[float]) -> bool:\n    values = tuple(float(value) for value in history)\n    return any(\n        current - previous < -_HMMLEARN_REGRESSION_TOLERANCE\n        for previous, current in pairwise(values)\n    )\n\n\ndef _validated_em_history(history: Iterable[float], iterations: int) -> tuple[float, ...]:\n    values = tuple(float(value) for value in history)\n    if not values:\n        raise ValueError("successful HMM fit requires a non-empty EM log-likelihood history")\n    if len(values) != iterations:\n        raise ValueError("EM log-likelihood history length must equal completed iterations")\n    if not all(isfinite(value) for value in values):\n        raise ValueError("EM log-likelihood history must contain only finite values")\n    return values\n\n\n@dataclass(frozen=True, slots=True)\n''',
)

replace_once(
    "src/market_regime_engine/models/gaussian_hmm.py",
    '''        train_log_likelihood = float(model.score(values))\n        if not isfinite(train_log_likelihood):\n            raise ValueError("TRAIN log likelihood must be finite")\n        return FitResult(\n            artifact=artifact,\n            train_log_likelihood=train_log_likelihood,\n            converged=bool(model.monitor_.converged)\n            and not _has_material_likelihood_regression(model.monitor_.history),\n            iterations=int(model.monitor_.iter),\n            seed=seed,\n        )\n''',
    '''        train_log_likelihood = float(model.score(values))\n        if not isfinite(train_log_likelihood):\n            raise ValueError("TRAIN log likelihood must be finite")\n        iterations = int(model.monitor_.iter)\n        history = _validated_em_history(model.monitor_.history, iterations)\n        return FitResult(\n            artifact=artifact,\n            train_log_likelihood=train_log_likelihood,\n            converged=bool(model.monitor_.converged)\n            and not _has_material_likelihood_regression(history),\n            iterations=iterations,\n            seed=seed,\n            em_log_likelihood_history=history,\n        )\n''',
)

replace_once(
    "src/market_regime_engine/models/student_t_hmm.py",
    '''from market_regime_engine.models.gaussian_hmm import forward_filter, gaussian_log_emissions\n''',
    '''from market_regime_engine.models.gaussian_hmm import (\n    _validated_em_history,\n    forward_filter,\n    gaussian_log_emissions,\n)\n''',
)

replace_once(
    "src/market_regime_engine/models/student_t_hmm.py",
    '''        dimension = values.shape[1]\n        identity = np.eye(dimension)\n        for iteration in range(1, self.settings.n_iter + 1):\n''',
    '''        dimension = values.shape[1]\n        identity = np.eye(dimension)\n        history: list[float] = []\n        for iteration in range(1, self.settings.n_iter + 1):\n''',
)

replace_once(
    "src/market_regime_engine/models/student_t_hmm.py",
    '''            gamma, xi_sum, likelihood, _ = _expectation(values, current)\n            start = np.maximum(gamma[0], 1e-12)\n''',
    '''            gamma, xi_sum, likelihood, _ = _expectation(values, current)\n            history.append(float(likelihood))\n            start = np.maximum(gamma[0], 1e-12)\n''',
)

replace_once(
    "src/market_regime_engine/models/student_t_hmm.py",
    '''        result = _artifact(self.feature_order, start, transition, means, scales, nu)\n        final_likelihood = forward_filter(values, result).log_likelihood\n        self._artifact = result\n        return FitResult(\n            artifact=result,\n            train_log_likelihood=final_likelihood,\n            converged=converged,\n            iterations=iterations,\n            seed=seed,\n        )\n''',
    '''        result = _artifact(self.feature_order, start, transition, means, scales, nu)\n        final_likelihood = forward_filter(values, result).log_likelihood\n        em_history = _validated_em_history(history, iterations)\n        self._artifact = result\n        return FitResult(\n            artifact=result,\n            train_log_likelihood=final_likelihood,\n            converged=converged,\n            iterations=iterations,\n            seed=seed,\n            em_log_likelihood_history=em_history,\n        )\n''',
)

replace_once(
    "tests/unit/models/test_gaussian_hmm.py",
    '''    HmmlearnGMMHMMAdapter,\n    _has_material_likelihood_regression,\n''',
    '''    HmmlearnGMMHMMAdapter,\n    _has_material_likelihood_regression,\n    _validated_em_history,\n''',
)

replace_once(
    "tests/unit/models/test_gaussian_hmm.py",
    '''    assert result.artifact.state_count == 2\n    assert result.artifact.feature_dimension == 2\n\n\ndef test_hmmlearn_likelihood_regression_detection_allows_numerical_noise() -> None:\n''',
    '''    assert result.artifact.state_count == 2\n    assert result.artifact.feature_dimension == 2\n    assert len(result.em_log_likelihood_history) == result.iterations\n    assert result.em_log_likelihood_history\n    assert np.all(np.isfinite(result.em_log_likelihood_history))\n\n\ndef test_em_history_validation_rejects_missing_nonfinite_and_wrong_lengths() -> None:\n    with pytest.raises(ValueError, match="non-empty"):\n        _validated_em_history((), 1)\n    with pytest.raises(ValueError, match="length"):\n        _validated_em_history((-10.0,), 2)\n    with pytest.raises(ValueError, match="finite"):\n        _validated_em_history((-10.0, np.nan), 2)\n\n\ndef test_hmmlearn_likelihood_regression_detection_allows_numerical_noise() -> None:\n''',
)

replace_once(
    "tests/unit/models/test_gaussian_hmm.py",
    '''    assert all(len(weights) == 2 for weights in result.artifact.mixture_weights)\n    assert np.isfinite(result.train_log_likelihood)\n    filtered = adapter.causal_filter(values[:5])\n''',
    '''    assert all(len(weights) == 2 for weights in result.artifact.mixture_weights)\n    assert np.isfinite(result.train_log_likelihood)\n    assert result.em_log_likelihood_history\n    assert len(result.em_log_likelihood_history) == result.iterations\n    assert np.all(np.isfinite(result.em_log_likelihood_history))\n    filtered = adapter.causal_filter(values[:5])\n''',
)

replace_once(
    "tests/unit/models/test_student_t_hmm.py",
    '''    assert np.isfinite(result.train_log_likelihood)\n    filtered = adapter.causal_filter(sample()[:8])\n''',
    '''    assert np.isfinite(result.train_log_likelihood)\n    assert result.em_log_likelihood_history\n    assert len(result.em_log_likelihood_history) == result.iterations\n    assert np.all(np.isfinite(result.em_log_likelihood_history))\n    filtered = adapter.causal_filter(sample()[:8])\n''',
)

replace_once(
    "tests/unit/training/test_multistart.py",
    '''        converged=converged,\n        iterations=17,\n        seed=seed,\n    )\n''',
    '''        converged=converged,\n        iterations=17,\n        seed=seed,\n        em_log_likelihood_history=tuple(float(index) for index in range(17)),\n    )\n''',
)

replace_once(
    "tests/unit/training/test_multistart.py",
    '''    assert result.winner.seed == 89\n    assert tuple(item.seed for item in result.diagnostics) == MULTISTART_SEEDS\n''',
    '''    assert result.winner.seed == 89\n    assert result.winner.em_log_likelihood_history == tuple(float(index) for index in range(17))\n    assert tuple(item.seed for item in result.diagnostics) == MULTISTART_SEEDS\n''',
)
