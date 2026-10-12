#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Restore or publish the unmasked BF16 AITER MHA module.

Z-Image and the quarantined Stable Audio layerwise job both cold-compile
``mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale``. AITER loads an existing
``.so`` from ``AITER_JIT_DIR`` instead of compiling it. This script keeps that
module in a directory keyed by the Python, pybind, compiler, libstdc++, image
dependency, AITER core, Torch, and ROCm identity. Each job receives its own
writable copy.

Only a non-PR main build of ``vllm-project/vllm-omni`` may publish. Pull
requests may restore that publication and cannot replace it. The shared
directory is never ``AITER_JIT_DIR``: AITER deletes a module at the start of a
rebuild. Readers accept a publication only when its checksum, ``gfx942`` device
code, trust namespace, and ABI identity all match. A main build may replace a
publication that fails those checks after pytest has succeeded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Mapping
from pathlib import Path

MODULE_NAME = "mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale"
ARCH = "gfx942"
ARCH_MARKER = b"amdhsa--gfx942"
CACHE_DIRNAME = "vllm-omni-aiter-jit"
SCHEMA = "2"
TRUST_NAMESPACE = "main"
TRUSTED_REPOSITORY = "vllm-project/vllm-omni"
KEY_FIELDS = (
    "schema",
    "module",
    "arch",
    "aiter",
    "torch",
    "hip",
    "python",
    "python_cache_tag",
    "abi_flags",
    "pybind11",
    "compiler",
    "glibcxx",
    "image_deps",
    "core",
)
_IMAGE_DEPENDENCY_PATHS = (
    "docker/Dockerfile.rocm",
    "docker/Dockerfile.rocm.dockerignore",
    "pyproject.toml",
    "setup.py",
    "tools/install_torchcodec_rocm.sh",
)
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


def cache_key(identity: Mapping[str, str]) -> str:
    missing = [field for field in KEY_FIELDS if not identity.get(field)]
    if missing:
        raise ValueError(f"AITER cache identity is missing {', '.join(missing)}")
    payload = "\n".join(f"{field}={identity[field]}" for field in KEY_FIELDS)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def published_dir(store: Path) -> Path:
    return store / "published"


def _read_identity(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    identity: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            return None
        field, value = line.split("=", 1)
        identity[field] = value
    if set(identity) != set(KEY_FIELDS):
        return None
    return {field: identity[field] for field in KEY_FIELDS}


def is_valid_publication(published: Path) -> bool:
    module = published / module_filename()
    checksum_path = published / "module.sha256"
    ready = published / "READY"
    trust = published / "TRUST"
    identity = _read_identity(published / "identity.txt")
    if not (module.is_file() and checksum_path.is_file() and ready.is_file() and trust.is_file()):
        return False
    if trust.read_text(encoding="utf-8").strip() != TRUST_NAMESPACE:
        log(f"ignoring cache outside the {TRUST_NAMESPACE} trust namespace: {published}")
        return False
    recorded = checksum_path.read_text(encoding="utf-8").strip()
    if recorded != sha256_file(module):
        log(f"ignoring cache with a checksum mismatch: {module}")
        return False
    if not contains_marker(module):
        log(f"ignoring cache without {ARCH} device code: {module}")
        return False
    if identity is None:
        log(f"ignoring cache without a complete ABI identity: {published}")
        return False
    try:
        key = cache_key(identity)
    except ValueError as exc:
        log(str(exc))
        return False
    if published.parent.name != key or published.parent.parent.name != TRUST_NAMESPACE:
        log(f"ignoring cache whose directory does not match its ABI identity: {published}")
        return False
    return True


def ldd_resolves(path: Path) -> bool:
    """Return whether the dynamic loader can resolve ``path``.

    A missing ``ldd`` does not reject the module. An unresolved dependency does,
    so a module built for another libstdc++ or Python is not restored.
    """
    try:
        completed = subprocess.run(
            ["ldd", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"could not run ldd on {path}: {exc}")
        return True
    if completed.returncode != 0 or "not found" in completed.stdout:
        log(f"cached module does not load on this runtime: {path}")
        return False
    return True


def publication_matches(
    published: Path,
    identity: Mapping[str, str],
    *,
    ldd_ok: Callable[[Path], bool] = ldd_resolves,
) -> bool:
    """Return whether ``published`` is the module for this exact runtime."""
    if not is_valid_publication(published):
        return False
    recorded = _read_identity(published / "identity.txt")
    if recorded != {field: identity[field] for field in KEY_FIELDS}:
        log(f"cached ABI identity does not match this runtime: {published}")
        return False
    return ldd_ok(published / module_filename())


def is_trusted_main_producer(env: Mapping[str, str] | None = None) -> bool:
    """Return whether this process may publish the shared main cache.

    Pull-request jobs restore that cache and do not update it. The repository
    check keeps another checkout that shares the Hugging Face volume from
    becoming the writer.
    """
    values = os.environ if env is None else env
    pull_request = values.get("BUILDKITE_PULL_REQUEST", "")
    repository = values.get("BUILDKITE_REPO", "")
    return (
        values.get("BUILDKITE_BRANCH") == TRUST_NAMESPACE
        and pull_request in ("", "false")
        and TRUSTED_REPOSITORY in repository
    )


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


def restore_module(
    published: Path,
    job_dir: Path,
    *,
    accept: Callable[[Path], bool] | None = None,
) -> bool:
    """Copy a verified module into ``job_dir``. Leave ``job_dir`` unchanged on failure."""
    if not is_valid_publication(published):
        return False
    if accept is not None and not accept(published):
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


def _write_identity(path: Path, identity: Mapping[str, str]) -> None:
    lines = [f"{field}={identity[field]}" for field in KEY_FIELDS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unlock_tree(path: Path) -> None:
    if path.is_dir():
        path.chmod(0o755)
        for child in path.iterdir():
            _unlock_tree(child)
    elif path.exists():
        path.chmod(0o644)


def _quarantine(published: Path, job_id: str) -> None:
    rejected = published.parent / f".rejected-{_sanitize(job_id)}"
    if rejected.exists():
        _unlock_tree(rejected)
        shutil.rmtree(rejected)
    _unlock_tree(published)
    os.rename(published, rejected)


def publish_module(
    job_dir: Path,
    store: Path,
    job_id: str,
    identity: Mapping[str, str],
    *,
    replace_incompatible: bool = False,
    accept: Callable[[Path], bool] | None = None,
) -> str:
    """Publish ``job_dir``'s module into the trusted main slot.

    Returns ``present``, ``published``, or ``skipped``. A compatible
    publication is kept. An incompatible one is replaced only when
    ``replace_incompatible`` is set, which the caller limits to main builds.
    """
    if cache_key(identity) != store.name or store.parent.name != TRUST_NAMESPACE:
        log("refusing to publish outside the trusted main ABI directory")
        return "skipped"
    published = published_dir(store)

    def accepted(path: Path) -> bool:
        if not is_valid_publication(path):
            return False
        return True if accept is None else accept(path)

    if published.exists() and accepted(published):
        return "present"
    if published.exists() and not replace_incompatible:
        log(f"leaving incompatible cache in place: {published}")
        return "skipped"
    if published.exists():
        try:
            _quarantine(published, job_id)
        except OSError as exc:
            log(f"could not replace incompatible cache: {exc}")
            if published.exists() and accepted(published):
                return "present"
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
        (staging / "TRUST").write_text(f"{TRUST_NAMESPACE}\n", encoding="utf-8")
        _write_identity(staging / "identity.txt", identity)
        _fsync_tree(staging)
        if not is_valid_publication(staging):
            log("staged module failed verification; not publishing")
            return "skipped"
        os.rename(staging, published)
    except OSError as exc:
        log(f"immutable publish did not complete: {exc}")
        if published.exists() and accepted(published):
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


def dependency_identity(root: Path) -> str:
    """Hash the same ROCm image inputs used by ``build-ci-image.sh``."""
    requirement_dir = root / "requirements"
    if not requirement_dir.is_dir():
        return ""
    requirement_files = sorted(path for path in requirement_dir.glob("*.txt") if path.is_file())
    files = [root / relative for relative in _IMAGE_DEPENDENCY_PATHS]
    files.extend(requirement_files)
    if any(not path.is_file() for path in files):
        return ""
    digest = hashlib.sha256()
    digest.update(f"rocm-arch={ARCH}\n".encode())
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"path={relative}\n".encode())
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _max_glibcxx(path: Path) -> str:
    versions: list[str] = []
    data = path.read_bytes()
    for match in re.finditer(rb"GLIBCXX_([0-9.]+)", data):
        versions.append(match.group(1).decode())
    if not versions:
        return ""

    def sort_key(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split(".") if part.isdigit())

    return max(versions, key=sort_key)


def glibcxx_version() -> str:
    candidates = (
        Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6"),
        Path("/lib/x86_64-linux-gnu/libstdc++.so.6"),
        Path("/usr/lib64/libstdc++.so.6"),
    )
    for candidate in candidates:
        if candidate.is_file() or candidate.is_symlink():
            version = _max_glibcxx(candidate)
            if version:
                return version
    return ""


def compiler_id() -> str:
    for binary in ("hipcc", "clang"):
        try:
            completed = subprocess.run(
                [binary, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = (completed.stdout or completed.stderr).splitlines()
        if lines and lines[0].strip():
            return f"{binary} {lines[0].strip()}"
    return platform.python_compiler().strip()


def collect_identity(root: Path | None = None, jit_dir: Path | None = None) -> dict[str, str] | None:
    """Return the ABI identity, or None when a required field is unavailable."""
    hip = read_hip_version()
    if not hip:
        try:
            torch = __import__("torch")
            hip = str(getattr(torch.version, "hip", "") or "").strip()
        except ImportError:
            hip = ""
    checkout = Path(__file__).resolve().parents[3] if root is None else root
    core_path = (installed_jit_dir() if jit_dir is None else jit_dir) or Path()
    core_file = core_path / "module_aiter_core.so"
    identity = {
        "schema": SCHEMA,
        "module": MODULE_NAME,
        "arch": ARCH,
        "aiter": _distribution_version("amd-aiter", "aiter"),
        "torch": _distribution_version("torch"),
        "hip": hip,
        "python": sys.version.split()[0],
        "python_cache_tag": sys.implementation.cache_tag or "",
        "abi_flags": getattr(sys, "abiflags", "") or "none",
        "pybind11": _distribution_version("pybind11") or "absent",
        "compiler": compiler_id(),
        "glibcxx": glibcxx_version(),
        "image_deps": dependency_identity(checkout),
        "core": sha256_file(core_file) if core_file.is_file() else "",
    }
    try:
        cache_key(identity)
    except ValueError as exc:
        log(str(exc))
        return None
    return identity


def store_for_identity(hf_home: Path, identity: Mapping[str, str]) -> Path | None:
    if not hf_home.is_absolute() or hf_home == Path("/") or not hf_home.is_dir():
        return None
    try:
        key = cache_key(identity)
    except ValueError as exc:
        log(str(exc))
        return None
    return hf_home / CACHE_DIRNAME / TRUST_NAMESPACE / key


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
    identity = collect_identity() if hf_home else None
    store = store_for_identity(hf_home, identity) if hf_home and identity is not None else None
    published = published_dir(store) if store is not None else None
    restored = False
    if store is not None and identity is not None and published is not None:
        runtime_identity = identity
        restored = restore_module(
            published,
            job_dir,
            accept=lambda path: publication_matches(path, runtime_identity),
        )
    if restored and published is not None:
        log(f"restored {module_filename()} from {published}")
    elif store is not None:
        log(f"no verified module at {published}; compiling into {job_dir}")
    else:
        log("shared AITER cache is unavailable; compiling into the job-local directory")
    emit_shell_exports(job_dir, store)
    return 0


def publish() -> int:
    if not is_trusted_main_producer():
        log(f"skipping publish; shared cache updates are limited to non-PR main builds of {TRUSTED_REPOSITORY}")
        return 0
    store_value = os.environ.get("VLLM_OMNI_AITER_MHA_STORE", "")
    job_value = os.environ.get("VLLM_OMNI_AITER_MHA_JOB_DIR", "")
    identity = collect_identity()
    hf_home = Path(os.environ.get("HF_HOME", ""))
    expected = store_for_identity(hf_home, identity) if identity is not None else None
    if not store_value or not job_value or expected is None:
        log("shared AITER cache was not selected; skipping publish")
        return 0
    if identity is None or Path(store_value).resolve() != expected.resolve():
        log("skipping publish; store path does not match the trusted main ABI key")
        return 0
    runtime_identity = identity
    outcome = publish_module(
        Path(job_value),
        Path(store_value),
        os.environ.get("BUILDKITE_JOB_ID") or "local",
        runtime_identity,
        replace_incompatible=True,
        accept=lambda path: publication_matches(path, runtime_identity),
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
