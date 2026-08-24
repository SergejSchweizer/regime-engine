#!/usr/bin/env bash
# Shell fragment for Compose commands.  Compose interpolates build arguments
# while parsing compose.yaml, including for --no-build and down operations.
export REGIME_ENGINE_GIT_SHA="$(git rev-parse HEAD)"
export REGIME_ENGINE_BUILD_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
