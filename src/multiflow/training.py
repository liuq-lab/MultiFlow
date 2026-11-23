"""Latent flow-matching training utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ._base import sample_joint_standard_normal
from .normalization import LatentStandardizer


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 600
    batch_size: int = 512
    learning_rate: float = 1e-4
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    standardize: bool = True
    noise_mode: str = "auto"
    reseed: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        if self.noise_mode not in {"auto", "joint", "independent", "shared"}:
            raise ValueError(
                "noise_mode must be auto, joint, independent, or shared"
            )


@dataclass
class TrainingResult:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    history: list[dict[str, float]] = field(default_factory=list)
    standardizer: LatentStandardizer | None = None


def seed_everything(seed: int) -> None:
    """Seed NumPy and PyTorch before constructing a model.

    Call this before model construction when using the Python API. By default,
    ``fit`` seeds its training stream; the command-line interface seeds once
    before model construction and disables the second seed explicitly.
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _as_tensor(values: np.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=dtype, device="cpu")
    if not torch.isfinite(tensor).all():
        raise ValueError("training arrays contain non-finite values")
    return tensor


def _as_index_tensor(
    values: np.ndarray | torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    original = torch.as_tensor(values, device="cpu")
    if original.is_floating_point():
        if not torch.isfinite(original).all() or not torch.equal(original, original.round()):
            raise ValueError(f"{name} must contain finite integer indices")
    tensor = original.to(dtype=torch.long).reshape(-1)
    if tensor.numel() and int(tensor.min()) < 0:
        raise ValueError(f"{name} cannot contain negative indices")
    return tensor


def _validate_conditions(
    model: nn.Module,
    labels: torch.Tensor | None,
    perturbations: torch.Tensor | None,
) -> None:
    label_count = getattr(model, "num_contexts", getattr(model, "num_classes", None))
    if label_count is not None:
        if labels is None:
            raise ValueError("labels are required by this conditioned model")
        if labels.numel() and int(labels.max()) >= int(label_count):
            raise ValueError(f"labels must be in [0, {int(label_count) - 1}]")
    elif labels is not None:
        raise ValueError("labels were provided to an unconditional model")

    perturbation_count = getattr(model, "num_perturbations", None)
    if perturbation_count is None:
        if perturbations is not None:
            raise ValueError("perturbations are only accepted by PerturbationFlow")
    elif perturbations is not None and perturbations.numel():
        if int(perturbations.max()) >= int(perturbation_count):
            raise ValueError(
                f"perturbations must be in [0, {int(perturbation_count) - 1}]"
            )


def fit(
    model: nn.Module,
    rna: np.ndarray | torch.Tensor,
    atac: np.ndarray | torch.Tensor,
    *,
    labels: np.ndarray | torch.Tensor | None = None,
    perturbations: np.ndarray | torch.Tensor | None = None,
    config: TrainingConfig | None = None,
    verbose: bool = False,
) -> TrainingResult:
    """Fit a MultiFlow vector field to paired latent states.

    ``verbose=True`` prints the two losses about ten times during the run.

    Inputs are target latent states.  For the canonical models, one joint
    Gaussian source in the concatenated RNA-ATAC latent space and a uniform
    flow time are sampled for each optimization step, and the model regresses
    the straight-line conditional vector field.  The historical ``independent``
    spelling is retained as an alias for the same full-rank joint Gaussian
    distribution; ``shared`` denotes only the rank-deficient legacy concat
    ablation.
    """
    config = config or TrainingConfig()
    if config.reseed:
        seed_everything(config.seed)

    rna_tensor = _as_tensor(rna, torch.float32)
    atac_tensor = _as_tensor(atac, torch.float32)
    if rna_tensor.ndim != 2 or atac_tensor.ndim != 2:
        raise ValueError("RNA and ATAC latents must be two-dimensional")
    if rna_tensor.shape[0] != atac_tensor.shape[0]:
        raise ValueError("RNA and ATAC latents must contain the same paired cells")
    if rna_tensor.shape[0] < 1:
        raise ValueError("at least one paired cell is required")
    if rna_tensor.shape[1] != getattr(model, "rna_dim", None):
        raise ValueError("RNA latent width does not match model.rna_dim")
    if atac_tensor.shape[1] != getattr(model, "atac_dim", None):
        raise ValueError("ATAC latent width does not match model.atac_dim")

    n_cells = rna_tensor.shape[0]
    label_tensor = None if labels is None else _as_index_tensor(labels, name="labels")
    perturbation_tensor = (
        None
        if perturbations is None
        else _as_index_tensor(perturbations, name="perturbations")
    )
    for name, tensor in (("labels", label_tensor), ("perturbations", perturbation_tensor)):
        if tensor is not None and tensor.shape[0] != n_cells:
            raise ValueError(f"{name} length does not match paired latent rows")
    _validate_conditions(model, label_tensor, perturbation_tensor)

    standardizer = LatentStandardizer.fit(rna_tensor, atac_tensor) if config.standardize else None
    if standardizer is not None:
        rna_tensor, atac_tensor = standardizer.transform(rna_tensor, atac_tensor)

    device = torch.device(config.device)
    rna_tensor = rna_tensor.to(device)
    atac_tensor = atac_tensor.to(device)
    if label_tensor is not None:
        label_tensor = label_tensor.to(device)
    if perturbation_tensor is not None:
        perturbation_tensor = perturbation_tensor.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mse = nn.MSELoss()
    history: list[dict[str, float]] = []
    model_requires_shared = bool(getattr(model, "requires_shared_base", False))
    if config.noise_mode != "auto":
        requested_shared = config.noise_mode == "shared"
        if requested_shared != model_requires_shared:
            expected = "shared" if model_requires_shared else "joint"
            raise ValueError(
                f"{type(model).__name__} requires noise_mode={expected!r}; "
                f"received {config.noise_mode!r}"
            )
    shared_noise = model_requires_shared
    if shared_noise and rna_tensor.shape[1] != atac_tensor.shape[1]:
        raise ValueError("shared source noise requires equal RNA and ATAC latent widths")

    for epoch in range(config.epochs):
        model.train()
        permutation = torch.randperm(n_cells, device=device)
        rna_loss_sum = 0.0
        atac_loss_sum = 0.0
        batches = 0
        for start in range(0, n_cells, config.batch_size):
            index = permutation[start : start + config.batch_size]
            target_rna = rna_tensor[index]
            target_atac = atac_tensor[index]
            label_batch = None if label_tensor is None else label_tensor[index]
            pert_batch = (
                None if perturbation_tensor is None else perturbation_tensor[index]
            )
            if shared_noise:
                base_rna = torch.randn_like(target_rna)
                base_atac = base_rna.clone()
            else:
                base_rna, base_atac = sample_joint_standard_normal(
                    int(index.shape[0]),
                    int(target_rna.shape[1]),
                    int(target_atac.shape[1]),
                    device=device,
                    dtype=target_rna.dtype,
                )
            time = torch.rand(index.shape[0], 1, device=device)
            state_rna = (1 - time) * base_rna + time * target_rna
            state_atac = (1 - time) * base_atac + time * target_atac
            velocity_rna = target_rna - base_rna
            velocity_atac = target_atac - base_atac
            predicted_rna, predicted_atac = model(
                state_rna,
                state_atac,
                time[:, 0],
                label=label_batch,
                pert_label=pert_batch,
            )
            loss_rna = mse(predicted_rna, velocity_rna)
            loss_atac = mse(predicted_atac, velocity_atac)
            loss = loss_rna + loss_atac
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rna_loss_sum += float(loss_rna.detach())
            atac_loss_sum += float(loss_atac.detach())
            batches += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss_rna": rna_loss_sum / batches,
                "loss_atac": atac_loss_sum / batches,
                "loss_total": (rna_loss_sum + atac_loss_sum) / batches,
            }
        )
        every = max(1, config.epochs // 10)
        if verbose and (epoch == 0 or (epoch + 1) % every == 0 or epoch + 1 == config.epochs):
            print(
                f"multiflow epoch {epoch + 1}/{config.epochs}  "
                f"rna {history[-1]['loss_rna']:.4f}  atac {history[-1]['loss_atac']:.4f}",
                flush=True,
            )
    return TrainingResult(model, optimizer, history, standardizer)
