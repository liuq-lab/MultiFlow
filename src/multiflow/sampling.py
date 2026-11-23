"""ODE sampling for trained MultiFlow vector fields."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ._base import sample_joint_standard_normal
from .normalization import LatentStandardizer


def _condition_vector(
    value: int | torch.Tensor | None,
    n: int,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.long, device=device)
    if tensor.ndim == 0:
        return tensor.repeat(n)
    tensor = tensor.reshape(-1)
    if tensor.numel() != n:
        raise ValueError(f"{name} must be a scalar or contain n values")
    return tensor


def _validate_condition_range(
    tensor: torch.Tensor | None,
    count: int | None,
    name: str,
    *,
    required: bool,
) -> None:
    if tensor is None:
        if required:
            raise ValueError(f"{name} is required by this conditioned model")
        return
    if count is None:
        raise ValueError(f"{name} was provided to a model that does not use it")
    if tensor.numel() and (int(tensor.min()) < 0 or int(tensor.max()) >= int(count)):
        raise ValueError(f"{name} must be in [0, {int(count) - 1}]")


@torch.no_grad()
def sample_paired_latents(
    model: nn.Module,
    n: int,
    *,
    label: int | torch.Tensor | None = None,
    perturbation: int | torch.Tensor | None = None,
    steps: int = 100,
    batch_size: int = 512,
    device: str | torch.device | None = None,
    seed: int = 0,
    standardizer: LatentStandardizer | None = None,
    rng_mode: Literal["joint", "legacy_interleaved", "batch_invariant"] = "joint",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate paired latent states with midpoint ODE integration.

    ``joint`` draws one full-rank Gaussian random variable in the concatenated
    RNA-ATAC latent space per sampling batch and splits it into modality blocks.
    ``legacy_interleaved`` reproduces the historical two-call random-draw order.
    ``batch_invariant`` draws the complete joint initial state first; it gives
    identical samples across batch sizes but uses O(n) device memory.
    """
    if n < 1 or steps < 1 or batch_size < 1:
        raise ValueError("n, steps, and batch_size must be positive")
    if rng_mode not in {"joint", "legacy_interleaved", "batch_invariant"}:
        raise ValueError(
            "rng_mode must be joint, legacy_interleaved, or batch_invariant"
        )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model = model.to(device).eval()
    rna_dim, atac_dim = int(model.rna_dim), int(model.atac_dim)
    shared_noise = bool(getattr(model, "requires_shared_base", False))
    if shared_noise and rna_dim != atac_dim:
        raise ValueError("shared source noise requires equal latent widths")
    all_rna: list[torch.Tensor] = []
    all_atac: list[torch.Tensor] = []
    labels = _condition_vector(label, n, device, "label")
    perturbations = _condition_vector(perturbation, n, device, "perturbation")
    label_count = getattr(model, "num_contexts", getattr(model, "num_classes", None))
    _validate_condition_range(
        labels,
        label_count,
        "label",
        required=label_count is not None,
    )
    perturbation_count = getattr(model, "num_perturbations", None)
    _validate_condition_range(
        perturbations,
        perturbation_count,
        "perturbation",
        required=False,
    )
    time_grid = torch.linspace(0, 1, steps + 1, device=device)
    if rng_mode == "batch_invariant":
        if shared_noise:
            base_rna = torch.randn((n, rna_dim), device=device, generator=generator)
            base_atac = base_rna
        else:
            base_rna, base_atac = sample_joint_standard_normal(
                n,
                rna_dim,
                atac_dim,
                device=device,
                generator=generator,
            )
    else:
        base_rna = base_atac = None

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        count = end - start
        if rng_mode == "batch_invariant":
            rna = base_rna[start:end].clone()
            atac = base_atac[start:end].clone()
        elif rng_mode == "legacy_interleaved":
            rna = torch.randn((count, rna_dim), device=device, generator=generator)
            atac = (
                rna.clone()
                if shared_noise
                else torch.randn((count, atac_dim), device=device, generator=generator)
            )
        elif shared_noise:
            rna = torch.randn((count, rna_dim), device=device, generator=generator)
            atac = rna.clone()
        else:
            rna, atac = sample_joint_standard_normal(
                count,
                rna_dim,
                atac_dim,
                device=device,
                generator=generator,
            )
        label_batch = None if labels is None else labels[start:end]
        pert_batch = None if perturbations is None else perturbations[start:end]
        for index in range(steps):
            t0, t1 = time_grid[index], time_grid[index + 1]
            dt = t1 - t0
            time = t0.expand(count)
            velocity_rna, velocity_atac = model(
                rna, atac, time, label=label_batch, pert_label=pert_batch
            )
            midpoint_time = ((t0 + t1) / 2).expand(count)
            midpoint_rna = rna + 0.5 * dt * velocity_rna
            midpoint_atac = atac + 0.5 * dt * velocity_atac
            midpoint_velocity_rna, midpoint_velocity_atac = model(
                midpoint_rna,
                midpoint_atac,
                midpoint_time,
                label=label_batch,
                pert_label=pert_batch,
            )
            rna = rna + dt * midpoint_velocity_rna
            atac = atac + dt * midpoint_velocity_atac
        all_rna.append(rna.cpu())
        all_atac.append(atac.cpu())
    rna_output, atac_output = torch.cat(all_rna), torch.cat(all_atac)
    if standardizer is not None:
        rna_output, atac_output = standardizer.inverse_transform(rna_output, atac_output)
    return rna_output, atac_output
