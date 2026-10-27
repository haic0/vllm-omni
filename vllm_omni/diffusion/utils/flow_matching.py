# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import numpy as np
import torch
from torch import nn

# ROCm wheels raise this from torch.linalg on CPU when LAPACK was not compiled in.
_CPU_LAPACK_ERROR = "requires compiling PyTorch with LAPACK"


def _cpu_lapack_missing(matrix: torch.Tensor, exc: RuntimeError) -> bool:
    return matrix.device.type == "cpu" and _CPU_LAPACK_ERROR in str(exc)


def safe_linalg_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(matrix, rhs)
    except RuntimeError as exc:
        if not _cpu_lapack_missing(matrix, exc):
            raise
        # Some ROCm wheels still fail here at runtime even when the LAPACK probe reports support.
        solved = np.linalg.solve(matrix.detach().cpu().numpy(), rhs.detach().cpu().numpy())
        return torch.from_numpy(solved).to(device=matrix.device, dtype=matrix.dtype)


def safe_linalg_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """Return a float32 Cholesky factor on ``matrix.device``.

    CPU inputs fall back to NumPy when the wheel has no LAPACK. A non-CPU
    failure, and any other ``RuntimeError``, propagate.
    """
    try:
        factor = torch.linalg.cholesky(matrix)
    except RuntimeError as exc:
        if not _cpu_lapack_missing(matrix, exc):
            raise
        factor = torch.from_numpy(np.linalg.cholesky(matrix.detach().cpu().numpy()))
    return factor.to(device=matrix.device, dtype=torch.float32)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (math.log(10000.0) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # (B, T, embedding_dim)
        return enc
