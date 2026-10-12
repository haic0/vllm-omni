#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Restore or publish the unmasked BF16 AITER MHA module.

Z-Image and the quarantined Stable Audio layerwise job both cold-compile
``mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale``. AITER loads an existing
``.so`` from ``AITER_JIT_DIR`` instead of compiling it. This script keeps that
module in an immutable directory keyed by the AITER, Torch, and ROCm versions
plus ``gfx942``. Each job receives its own writable copy.

The shared directory is never used as ``AITER_JIT_DIR``. AITER deletes a module
at the start of a rebuild, so a writable cross-job directory would let one pod
destroy another's cache. Publication is a single rename of a complete staging
directory. Readers accept the result only when the checksum matches and the
``.so`` contains ``gfx942`` device code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import shlex
import shutil
import sys
import sysconfig
from pathlib import Path

MODULE_NAME = "mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale"
ARCH = "gfx942"
ARCH_MARKER = b"amdhsa--gfx942"
CACHE_DIRNAME = "vllm-omni-aiter-jit"
SCHEMA = "1"
KEY_FIELDS = ("schema", "module", "arch", "aiter", "torch", "hip")
_READ_SIZE = 1024 * 1024


def log(message: str) -> None:
    print(f"aiter-mha-cache: {message}", file=sys.stderr, flush=True)


def module_filename() -> str:
    return f"{MODULE_NAME}.so"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_READ_SIZE):
            digest.update(block)
    return digest.hexdigest()


def contains_marker(path: Path, marker: bytes = ARCH_MARKER) -> bool:
    """Return whether ``marker`` occurs in ``path`` without reading it whole."""
    if not path.is_file() or path.stat().st_size < len(marker):
        return False
    overlap = len(marker) - 1
    previous = b""
    with path.open("rb") as handle:
        while block := handle.read(_READ_SIZE):
            window = previous + block
            if marker in window:
                return True
            previous = window[-overlap:] if overlap else b""
    return False


def cache_key(identity: dict[str, str]) -> str:
    missing = [field for field in KEY_FIELDS if not identity.get(field)]
    if missing:
        raise ValueError(f"AITER cache identity is missing {', '.join(missing)}")
    payload = "\n".join(f"{field}={identity[field]}" for field in KEY_FIELDS)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def published_dir(store: Path) -> Path:
    return store / "published"


def is_valid_publication(published: Path) -> bool:
    module = published / module_filename()
    checksum_path = published / "module.sha256"
    ready = published / "READY"
    if not (module.is_file() and checksum_path.is_file() and ready.is_file()):
        return False
    recorded = checksum_path.read_text(encoding="utf-8").strip()
    if recorded != sha256_file(module):
        log(f"ignoring cache with a checksum mismatch: {module}")
        return False
    if not contains_marker(module):
        log(f"ignoring cache without {ARCH} device code: {module}")
        return False
    return True


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def restore_module(published: Path, job_dir: Path) -> bool:
    """Copy a verified module into ``job_dir``. Leave ``job_dir`` unchanged on failure."""
    if not is_valid_publication(published):
        return False
    source = published / module_filename()
    expected = (published / "module.sha256").read_text(encoding="utf-8").strip()
    partial = job_dir / f".{module_filename()}.partial"
    shutil.copyfile(source, partial)
    if sha256_file(partial) != expected or not contains_marker(partial):
        partial.unlink(missing_ok=True)
        log(f"restored copy failed verification: {source}")
        return False
    os.replace(partial, job_dir / module_filename())
    return True


def publish_module(job_dir: Path, store: Path, job_id: str) -> str:
    """Publish ``job_dir``'s module when the immutable slot is empty.

    Returns ``present``, ``published``, or ``skipped``. An existing publication
    is never replaced.
    """
    published = published_dir(store)
    if is_valid_publication(published):
        return "present"
    if published.exists():
        log(f"leaving invalid immutable cache in place: {published}")
        return "skipped"

    local = job_dir / module_filename()
    if not local.is_file() or not contains_marker(local):
        log("job-local BF16 AITER module is absent; nothing to publish")
        return "skipped"
    try:
        local.resolve().relative_to(store.resolve())
    except ValueError:
        pass
    else:
        log("refusing to publish a module that already lives in the store")
        return "skipped"

    store.mkdir(parents=True, exist_ok=True)
    staging = store / f".staging-{_sanitize(job_id)}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        shutil.copyfile(local, staging / module_filename())
        digest = sha256_file(staging / module_filename())
        (staging / "module.sha256").write_text(f"{digest}\n", encoding="utf-8")
        (staging / "READY").write_text("ok\n", encoding="utf-8")
        _fsync_tree(staging)
        if not is_valid_publication(staging):
            log("staged module failed verification; not publishing")
            return "skipped"
        os.rename(staging, published)
    except OSError as exc:
        log(f"immutable publish did not complete: {exc}")
        if is_valid_publication(published):
            return "present"
        return "skipped"
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    try:
        for path in published.iterdir():
            path.chmod(0o444)
        published.chmod(0o555)
    except OSError as exc:
        log(f"published module remains readable but was not locked: {exc}")
    log(f"published {module_filename()} to {published}")
    return "published"


def seed_prebuilt_modules(job_dir: Path, jit_dir: Path | None) -> None:
    """Copy image-shipped AITER modules so ``AITER_JIT_DIR`` can still import them."""
    if jit_dir is None or not jit_dir.is_dir():
        return
    for source in sorted(jit_dir.glob("*.so")):
        if source.name == module_filename():
            continue
        destination = job_dir / source.name
        if destination.exists():
            continue
        shutil.copyfile(source, destination)


def installed_jit_dir() -> Path | None:
    candidate = Path(sysconfig.get_path("purelib")) / "aiter" / "jit"
    if candidate.is_dir():
        return candidate
    return None


def _sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)


def job_directory() -> Path:
    job_id = _sanitize(os.environ.get("BUILDKITE_JOB_ID") or "local")
    return Path(f"/tmp/vllm-omni-aiter-mha-{job_id}")


def read_hip_version(version_file: Path = Path("/opt/rocm/.info/version")) -> str:
    if not version_file.is_file():
        return ""
    text = version_file.read_text(encoding="utf-8", errors="replace").strip()
    return text.splitlines()[0].strip() if text else ""


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name).strip()
        except importlib.metadata.PackageNotFoundError:
            continue
    return ""


def collect_identity() -> dict[str, str]:
    hip = read_hip_version()
    if not hip:
        try:
            torch = __import__("torch")
            hip = str(getattr(torch.version, "hip", "") or "").strip()
        except ImportError:
            hip = ""
    return {
        "schema": SCHEMA,
        "module": MODULE_NAME,
        "arch": ARCH,
        "aiter": _distribution_version("amd-aiter", "aiter"),
        "torch": _distribution_version("torch"),
        "hip": hip,
    }


def store_for_identity(hf_home: Path, identity: dict[str, str]) -> Path | None:
    if not hf_home.is_absolute() or hf_home == Path("/") or not hf_home.is_dir():
        return None
    try:
        key = cache_key(identity)
    except ValueError as exc:
        log(str(exc))
        return None
    return hf_home / CACHE_DIRNAME / key


def emit_shell_exports(job_dir: Path, store: Path | None) -> None:
    print(f"export AITER_JIT_DIR={shlex.quote(str(job_dir))}")
    print(f"export GPU_ARCHS={shlex.quote(ARCH)}")
    print(f"export VLLM_OMNI_AITER_MHA_JOB_DIR={shlex.quote(str(job_dir))}")
    if store is None:
        print("unset VLLM_OMNI_AITER_MHA_STORE")
    else:
        print(f"export VLLM_OMNI_AITER_MHA_STORE={shlex.quote(str(store))}")


def prepare() -> int:
    job_dir = job_directory()
    job_dir.mkdir(parents=True, exist_ok=True)
    seed_prebuilt_modules(job_dir, installed_jit_dir())
    hf_home = Path(os.environ.get("HF_HOME", ""))
    store = store_for_identity(hf_home, collect_identity()) if hf_home else None
    if store is not None and restore_module(published_dir(store), job_dir):
        log(f"restored {module_filename()} from {published_dir(store)}")
    elif store is not None:
        log(f"no verified module at {published_dir(store)}; compiling into {job_dir}")
    else:
        log("shared AITER cache is unavailable; compiling into the job-local directory")
    emit_shell_exports(job_dir, store)
    return 0


def publish() -> int:
    store_value = os.environ.get("VLLM_OMNI_AITER_MHA_STORE", "")
    job_value = os.environ.get("VLLM_OMNI_AITER_MHA_JOB_DIR", "")
    if not store_value or not job_value:
        log("shared AITER cache was not selected; skipping publish")
        return 0
    outcome = publish_module(
        Path(job_value),
        Path(store_value),
        os.environ.get("BUILDKITE_JOB_ID") or "local",
    )
    log(f"publish result: {outcome}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish"))
    args = parser.parse_args(argv)
    if args.action == "prepare":
        return prepare()
    return publish()


if __name__ == "__main__":
    sys.exit(main())
