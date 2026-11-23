"""Latent-space normalization used by MultiFlow."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LatentStandardizer:
    """Per-feature mean and standard deviation for paired latent states."""

    rna_mean: torch.Tensor
    rna_std: torch.Tensor
    atac_mean: torch.Tensor
    atac_std: torch.Tensor

    @classmethod
    def fit(
        cls,
        rna: torch.Tensor,
        atac: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> LatentStandardizer:
        rna, atac = torch.as_tensor(rna).float(), torch.as_tensor(atac).float()
        if rna.ndim != 2 or atac.ndim != 2 or rna.shape[0] != atac.shape[0]:
            raise ValueError("RNA and ATAC must be paired two-dimensional tensors")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if rna.shape[0] < 2:
            raise ValueError("at least two cells are required to estimate standard deviations")
        if not torch.isfinite(rna).all() or not torch.isfinite(atac).all():
            raise ValueError("RNA and ATAC latents must contain only finite values")
        return cls(
            rna.mean(0, keepdim=True).cpu(),
            (rna.std(0, keepdim=True) + epsilon).cpu(),
            atac.mean(0, keepdim=True).cpu(),
            (atac.std(0, keepdim=True) + epsilon).cpu(),
        )

    def transform(
        self,
        rna: torch.Tensor,
        atac: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rna, atac = torch.as_tensor(rna).float(), torch.as_tensor(atac).float()
        return (
            (rna - self.rna_mean.to(rna.device)) / self.rna_std.to(rna.device),
            (atac - self.atac_mean.to(atac.device)) / self.atac_std.to(atac.device),
        )

    def inverse_transform(
        self,
        rna: torch.Tensor,
        atac: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rna, atac = torch.as_tensor(rna).float(), torch.as_tensor(atac).float()
        return (
            rna * self.rna_std.to(rna.device) + self.rna_mean.to(rna.device),
            atac * self.atac_std.to(atac.device) + self.atac_mean.to(atac.device),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "rna_mean": self.rna_mean.cpu(),
            "rna_std": self.rna_std.cpu(),
            "atac_mean": self.atac_mean.cpu(),
            "atac_std": self.atac_std.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> LatentStandardizer:
        required = {"rna_mean", "rna_std", "atac_mean", "atac_std"}
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"standardizer state is missing: {missing}")
        return cls(
            rna_mean=torch.as_tensor(state["rna_mean"]).float().cpu(),
            rna_std=torch.as_tensor(state["rna_std"]).float().cpu(),
            atac_mean=torch.as_tensor(state["atac_mean"]).float().cpu(),
            atac_std=torch.as_tensor(state["atac_std"]).float().cpu(),
        )
