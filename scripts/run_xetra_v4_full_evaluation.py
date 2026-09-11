#!/usr/bin/env python3
"""Run the complete unsampled Xetra v4 evaluation and external tracking flow."""

from __future__ import annotations

# Keep one implementation of the source/evaluation orchestration.  This named
# entry point exists for release/audit commands; the cron wrapper adds locking,
# environment loading, and the MLflow health check around the same full run.
from run_xetra_v4_evaluation import main

if __name__ == "__main__":
    main()
