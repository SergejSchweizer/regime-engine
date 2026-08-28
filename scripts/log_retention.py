"""Run retention daily for persistent compose deployments."""

from __future__ import annotations

import time

from market_regime_engine.observability import apply_retention, configure_debug_logging


def main() -> None:
    logger = configure_debug_logging("retention")
    while True:
        try:
            apply_retention()
            logger.debug("log_retention_completed")
        except Exception:
            logger.exception("log_retention_failed")
        time.sleep(24 * 60 * 60)


if __name__ == "__main__":
    main()
