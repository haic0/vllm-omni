#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Upload AMD Buildkite bootstrap and native MI300 test pipelines."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

try:
    import yaml
except ModuleNotFoundError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pyyaml"],
        check=True,
    )
    import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
COMMON_SCRIPTS = ROOT / ".buildkite/common/scripts"
if str(COMMON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPTS))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from select_test_suites import (  # noqa: E402
    SUITE_SPECS,
    select_amd_test_suites,
)
from skip_ci import resolve_ci_context_from_git  # noqa: E402

BOOTSTRAP_STEPS = ROOT / ".buildkite/amd/bootstrap-upload-steps.yml"
TEMPLATE = ROOT / ".buildkite/amd/test-template-amd-omni.j2"
NIGHTLY_MAIN_IF = 'build.branch == "main" && build.env("NIGHTLY") == "1"'
NIGHTLY_LABEL_IF = (
    f"({NIGHTLY_MAIN_IF}) || "
    '(build.branch != "main" && build.pull_request.labels includes "nightly-test")'
)
BOOTSTRAP_ENABLED_IF = "true"


def _format_if(expression: str) -> str:
    return expression if expression in {"true", "false"} else f"({expression})"


def _amd_upload_if(decision: Any) -> str:
    """Return the condition for image construction and AMD suite upload."""
    if decision.skip_all:
        return NIGHTLY_MAIN_IF
    if decision.skip_l2_l3 and not (
        decision.is_run("amd", "l2") or decision.is_run("amd", "l3")
    ):
        return NIGHTLY_LABEL_IF
    return BOOTSTRAP_ENABLED_IF


def _render_bootstrap(path: Path, decision: Any) -> str:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("steps"), list):
        raise ValueError(f"invalid AMD bootstrap pipeline: {path}")

    expression = _format_if(_amd_upload_if(decision))
    steps = []
    for step in document["steps"]:
        if not isinstance(step, dict):
            steps.append(step)
            continue
        step = copy.deepcopy(step)
        if step.get("key") in {"image-build", "upload-amd-pipeline"}:
            step["if"] = expression
        if step.get("if") == "true":
            step.pop("if")
        steps.append(step)
    document["steps"] = steps
    return yaml.safe_dump(document, sort_keys=False)


def _pr_labels() -> tuple[str, ...]:
    pull_request = os.environ.get("BUILDKITE_PULL_REQUEST", "false")
    if pull_request == "false":
        return ()
    url = f"https://api.github.com/repos/vllm-project/vllm-omni/pulls/{pull_request}"
    with urlopen(url) as response:  # noqa: S310
        payload = yaml.safe_load(response.read())
    return tuple(label["name"] for label in payload.get("labels", []))


def _changed_files(context: Any) -> list[str] | None:
    return context.changed_files


def _source_matches(changed_files: list[str] | None, dependencies: list[str]) -> bool:
    if changed_files is None:
        return True
    return any(
        path == dependency.rstrip("/")
        or path.startswith(f"{dependency.rstrip('/')}/")
        for path in changed_files
        for dependency in dependencies
    )


def _filter_steps(steps: list[Any], changed_files: list[str] | None) -> list[Any]:
    filtered: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            filtered.append(step)
            continue
        dependencies = step.get("source_file_dependencies")
        if dependencies is not None and not isinstance(dependencies, list):
            raise ValueError("source_file_dependencies must be a list")
        if dependencies is not None and not _source_matches(changed_files, dependencies):
            continue
        nested = step.get("steps")
        if isinstance(nested, list):
            nested_step = copy.deepcopy(step)
            nested_step.pop("source_file_dependencies", None)
            nested_step["steps"] = _filter_steps(nested, changed_files)
            if nested_step["steps"]:
                filtered.append(nested_step)
            continue
        step = copy.deepcopy(step)
        step.pop("source_file_dependencies", None)
        filtered.append(step)
    return filtered


def _combine_suites(
    suites: tuple[str, ...],
    changed_files: list[str] | None,
) -> Path:
    combined: dict[str, Any] = {"env": {}, "steps": []}
    for suite_name in suites:
        suite_path = ROOT / ".buildkite/amd" / SUITE_SPECS[suite_name].split(":", 1)[1]
        suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        for name, value in (suite.get("env") or {}).items():
            previous = combined["env"].get(name, value)
            if previous != value:
                raise ValueError(f"conflicting AMD environment value for {name}")
            combined["env"][name] = value
        for entry in suite.get("steps") or []:
            if isinstance(entry, dict) and "group" in entry:
                children = _filter_steps(entry.get("steps") or [], changed_files)
                if children:
                    combined["steps"].append(
                        {"group": entry["group"], "steps": children},
                    )
            else:
                combined["steps"].extend(
                    _filter_steps([entry], changed_files),
                )

    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yml",
        prefix="amd-selected-",
        delete=False,
    )
    with temporary:
        yaml.safe_dump(combined, temporary, sort_keys=False)
    return Path(temporary.name)


def _render_tests(context: Any) -> str:
    labels = _pr_labels()
    nightly = os.environ.get("NIGHTLY", "0") == "1"
    suites = select_amd_test_suites(
        branch=os.environ.get("BUILDKITE_BRANCH", "main"),
        labels=labels,
        debug_test_yaml=os.environ.get("DEBUG_TEST_YAML", ""),
        nightly=nightly,
    )
    runnable = tuple(
        suite
        for suite in suites
        if suite == "nightly"
        or context.decision.is_run(
            "amd",
            "l2" if suite == "ready" else "l3",
        )
    )
    if not runnable:
        return ""

    selected = _combine_suites(runnable, _changed_files(context))
    minijinja = _ensure_minijinja()
    try:
        result = subprocess.run(
            [
                minijinja,
                str(TEMPLATE),
                str(selected),
                "-D",
                "mirror_hw=amdproduction",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        selected.unlink(missing_ok=True)
    return result.stdout


def _ensure_minijinja() -> str:
    executable = shutil.which("minijinja-cli")
    if executable:
        return executable
    subprocess.run(
        [
            "bash",
            "-o",
            "pipefail",
            "-c",
            "curl -sSfL "
            "https://github.com/mitsuhiko/minijinja/releases/download/2.3.1/"
            "minijinja-cli-installer.sh | sh",
        ],
        check=True,
    )
    for candidate in (
        "/var/lib/buildkite-agent/.cargo/bin/minijinja-cli",
        str(Path.home() / ".cargo/bin/minijinja-cli"),
    ):
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("minijinja-cli installer completed without an executable")


def _upload(content: str) -> None:
    subprocess.run(
        ["buildkite-agent", "pipeline", "upload"],
        input=content,
        text=True,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("pipeline", nargs="?", default=str(BOOTSTRAP_STEPS))
    args = parser.parse_args()

    context = resolve_ci_context_from_git()
    if args.bootstrap:
        content = _render_bootstrap(Path(args.pipeline), context.decision)
    else:
        content = _render_tests(context)
    if args.upload:
        if content:
            _upload(content)
    else:
        sys.stdout.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
