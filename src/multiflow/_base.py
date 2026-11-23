"""Base-distribution helpers shared by training and sampling."""

from __future__ import annotations

import torch


def sample_joint_standard_normal(
    n: int,
    rna_dim: int,
    atac_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one full-rank joint Gaussian state and return its modality blocks.

    The sampled random variable has shape ``[n, rna_dim + atac_dim]`` and
    distribution ``N(0, I)``.  Its RNA and ATAC coordinate blocks are
    marginally independent, but together constitute one joint source state.
    """

    if n < 1 or rna_dim < 1 or atac_dim < 1:
        raise ValueError("n, rna_dim, and atac_dim must be positive")
    joint = torch.randn(
        (n, rna_dim + atac_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return joint[:, :rna_dim], joint[:, rna_dim:]
