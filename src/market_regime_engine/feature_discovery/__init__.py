"""Lightweight public contracts for feature discovery.

Implementation modules stay explicit imports so multiprocessing ``spawn``
does not eagerly load preprocessing and re-enter ``features.ports``.
"""

from market_regime_engine.feature_discovery.contracts import *  # noqa: F403
from market_regime_engine.feature_discovery.feature_roles import *  # noqa: F403
