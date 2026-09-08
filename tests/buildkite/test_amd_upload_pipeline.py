# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UPLOADER_PATH = Path(".buildkite/amd/scripts/upload_pipeline.py")
SPEC = importlib.util.spec_from_file_location("amd_upload_pipeline", UPLOADER_PATH)
assert SPEC is not None and SPEC.loader is not None
UPLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPLOADER)


def test_amd_bootstrap_injects_amd_condition_only() -> None:
    path = Path(".buildkite/amd/bootstrap-upload-steps.yml")
    rendered = UPLOADER._render_bootstrap(
        path,
        SimpleNamespace(skip_all=False, skip_l2_l3=False),
    )
    document = yaml.safe_load(rendered)
    keys = {step["key"] for step in document["steps"]}
    assert keys == {"image-build", "upload-amd-pipeline"}
    assert "mirror_hardwares" not in rendered
    assert "if" not in document["steps"][0]


def test_amd_filter_preserves_native_hardware_and_filters_sources() -> None:
    steps = [
        {
            "label": "AMD native",
            "mirror_hardwares": ["amdproduction"],
            "source_file_dependencies": ["tests/e2e/qwen.py"],
            "commands": ["pytest tests/e2e/qwen.py"],
        },
        {
            "label": "Unrelated",
            "mirror_hardwares": ["amdproduction"],
            "source_file_dependencies": ["tests/e2e/other.py"],
            "commands": ["pytest tests/e2e/other.py"],
        },
    ]
    filtered = UPLOADER._filter_steps(steps, ["tests/e2e/qwen.py"])
    assert [step["label"] for step in filtered] == ["AMD native"]
    assert filtered[0]["mirror_hardwares"] == ["amdproduction"]
    assert "source_file_dependencies" not in filtered[0]


def test_docs_only_amd_bootstrap_is_schedule_only() -> None:
    decision = SimpleNamespace(skip_all=True, skip_l2_l3=False)
    assert UPLOADER._amd_upload_if(decision) == UPLOADER.NIGHTLY_MAIN_IF


def test_qwen3_runner_fails_closed_on_empty_collection() -> None:
    runner = Path(".buildkite/amd/scripts/run-qwen3-omni-ci-test.sh").read_text()
    assert "pytest --collect-only" in runner
    assert "collection_count" in runner
    assert "Qwen3 selector collected no tests" in runner
    assert "exit 5" in runner
