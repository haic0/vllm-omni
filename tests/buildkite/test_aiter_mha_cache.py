# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Behavior tests for the immutable BF16 AITER MHA cache."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SCRIPT = Path(".buildkite/amd/scripts/aiter_mha_cache.py")


def _load_cache():
    spec = importlib.util.spec_from_file_location("aiter_mha_cache", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache = _load_cache()


def _write_module(path: Path, payload: bytes) -> None:
    path.write_bytes(payload + cache.ARCH_MARKER + b"\n")


def _identity() -> dict[str, str]:
    return {
        "schema": cache.SCHEMA,
        "module": cache.MODULE_NAME,
        "arch": cache.ARCH,
        "aiter": "1.2.3",
        "torch": "2.8.0",
        "hip": "6.4.0",
    }


def test_cache_key_changes_when_the_runtime_identity_changes() -> None:
    original = cache.cache_key(_identity())
    changed = _identity()
    changed["hip"] = "6.5.0"

    assert cache.cache_key(_identity()) == original
    assert cache.cache_key(changed) != original
    with pytest.raises(ValueError, match="missing"):
        cache.cache_key({**_identity(), "aiter": ""})


def test_publish_is_immutable_and_restore_verifies_the_copy(tmp_path: Path) -> None:
    store = tmp_path / "store"
    first_job = tmp_path / "job-a"
    second_job = tmp_path / "job-b"
    restore_dir = tmp_path / "restore"
    for path in (first_job, second_job, restore_dir):
        path.mkdir()
    _write_module(first_job / cache.module_filename(), b"first-build")
    _write_module(second_job / cache.module_filename(), b"second-build")

    assert cache.publish_module(first_job, store, "job-a") == "published"
    published = cache.published_dir(store) / cache.module_filename()
    first_bytes = published.read_bytes()
    assert cache.publish_module(second_job, store, "job-b") == "present"
    assert published.read_bytes() == first_bytes

    assert cache.restore_module(cache.published_dir(store), restore_dir)
    assert (restore_dir / cache.module_filename()).read_bytes() == first_bytes

    published.chmod(0o644)
    published.write_bytes(first_bytes + b"tampered")
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    assert not cache.restore_module(cache.published_dir(store), rejected)
    assert not (rejected / cache.module_filename()).exists()
    assert list(rejected.glob(".*.partial")) == []


def test_invalid_module_is_not_published(tmp_path: Path) -> None:
    store = tmp_path / "store"
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / cache.module_filename()).write_bytes(b"host-only-object")

    assert cache.publish_module(job_dir, store, "job") == "skipped"
    assert not cache.published_dir(store).exists()


def test_corrupt_publication_is_not_replaced(tmp_path: Path) -> None:
    store = tmp_path / "store"
    published = cache.published_dir(store)
    published.mkdir(parents=True)
    module = published / cache.module_filename()
    _write_module(module, b"corrupt")
    (published / "module.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    (published / "READY").write_text("ok\n", encoding="utf-8")
    original = module.read_bytes()

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_module(job_dir / cache.module_filename(), b"replacement")

    assert cache.publish_module(job_dir, store, "job") == "skipped"
    assert module.read_bytes() == original


def test_seed_skips_the_cached_mha_module(tmp_path: Path) -> None:
    jit_dir = tmp_path / "jit"
    job_dir = tmp_path / "job"
    jit_dir.mkdir()
    job_dir.mkdir()
    (jit_dir / "module_aiter_core.so").write_bytes(b"core")
    _write_module(jit_dir / cache.module_filename(), b"image-copy")

    cache.seed_prebuilt_modules(job_dir, jit_dir)

    assert (job_dir / "module_aiter_core.so").read_bytes() == b"core"
    assert not (job_dir / cache.module_filename()).exists()


def test_prepare_exports_a_job_local_directory_without_a_shared_store() -> None:
    env = os.environ.copy()
    env["HF_HOME"] = ""
    env["BUILDKITE_JOB_ID"] = "unit.job/1"
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "prepare"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0
    assert "export AITER_JIT_DIR=/tmp/vllm-omni-aiter-mha-unit.job_1\n" in completed.stdout
    assert "export GPU_ARCHS=gfx942\n" in completed.stdout
    assert "unset VLLM_OMNI_AITER_MHA_STORE\n" in completed.stdout
    assert cache.CACHE_DIRNAME not in completed.stdout
