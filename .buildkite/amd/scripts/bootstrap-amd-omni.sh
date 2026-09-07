#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Compatibility entry for external AMD Buildkite callers. New pipelines invoke
# upload_pipeline.py directly; keep this wrapper for existing trigger settings.
set -euo pipefail

exec python3 .buildkite/amd/scripts/upload_pipeline.py --upload "$@"
