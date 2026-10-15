# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Sidecar allocation must stop when the device allocator stalls."""

from __future__ import annotations

import time

import pytest

from tests.dfx.reliability.helpers import _build_sidecar_cmd, allocate_device_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_allocate_device_chunk_returns_chunk_when_free_memory_drops() -> None:
    free = {"bytes": 100}

    def allocate(req_elems: int) -> str:
        free["bytes"] -= req_elems
        return f"chunk-{req_elems}"

    chunk = allocate_device_chunk(
        10,
        allocate=allocate,
        free_bytes=lambda: free["bytes"],
        timeout_sec=1.0,
    )

    assert chunk == "chunk-10"


def test_allocate_device_chunk_rejects_runtime_error() -> None:
    def allocate(req_elems: int) -> str:
        raise RuntimeError(f"oom {req_elems}")

    assert (
        allocate_device_chunk(
            4,
            allocate=allocate,
            free_bytes=lambda: 0,
            timeout_sec=1.0,
        )
        is None
    )


def test_allocate_device_chunk_rejects_when_free_memory_does_not_fall() -> None:
    def allocate(req_elems: int) -> str:
        return f"chunk-{req_elems}"

    assert (
        allocate_device_chunk(
            4,
            allocate=allocate,
            free_bytes=lambda: 50,
            timeout_sec=1.0,
        )
        is None
    )


def test_allocate_device_chunk_rejects_stalled_allocator() -> None:
    def allocate(req_elems: int) -> str:
        time.sleep(2.0)
        return f"chunk-{req_elems}"

    started = time.monotonic()
    chunk = allocate_device_chunk(
        4,
        allocate=allocate,
        free_bytes=lambda: 50,
        timeout_sec=0.05,
    )

    assert chunk is None
    assert time.monotonic() - started < 1.0


def test_sidecar_program_uses_rejection_helper_and_compiles() -> None:
    command = _build_sidecar_cmd(0, 0.95, 0, True, 256)
    program = command[3]

    assert "allocate_device_chunk" in program
    compile(program, "<oom-sidecar>", "exec")
