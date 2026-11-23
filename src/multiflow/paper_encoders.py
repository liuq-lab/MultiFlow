"""RNA VAE used by the MultiFlow cell-state generation workflow.

The architecture and checkpoint-loading rules follow the scDiffusion VAE used
by the original MultiFlow generation notebook.  The code is kept in a small,
explicit module so the public workflow does not depend on an unversioned local
checkout of scDiffusion.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class RNAEncoder(nn.Module):
    """Three-layer scDiffusion/SCimilarity RNA encoder."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (1024, 1024, 1024),
    ) -> None:
        super().__init__()
        dims = [int(n_genes), *map(int, hidden_dims)]
        self.network = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=0.0),
                    nn.Linear(dims[index], dims[index + 1]),
                    nn.BatchNorm1d(dims[index + 1]),
                    nn.PReLU(),
                )
                for index in range(len(hidden_dims))
            ]
        )
        self.network.append(nn.Linear(int(hidden_dims[-1]), int(latent_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.network:
            x = layer(x)
        return F.normalize(x, p=2, dim=1)


class RNADecoder(nn.Module):
    """Symmetric scDiffusion/SCimilarity RNA decoder."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (1024, 1024, 1024),
    ) -> None:
        super().__init__()
        dims = [int(latent_dim), *map(int, hidden_dims)]
        blocks: list[nn.Module] = []
        for index in range(len(hidden_dims)):
            block: list[nn.Module] = [
                nn.Linear(dims[index], dims[index + 1]),
                nn.BatchNorm1d(dims[index + 1]),
                nn.PReLU(),
            ]
            if index > 0:
                block.insert(0, nn.Dropout(p=0.0))
            blocks.append(nn.Sequential(*block))
        self.network = nn.ModuleList(blocks)
        self.network.append(nn.Linear(int(hidden_dims[-1]), int(n_genes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.network:
            x = layer(x)
        return x


class RNAExpressionVAE(nn.Module):
    """The deterministic 128-dimensional RNA autoencoder used in the paper."""

    def __init__(self, n_genes: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.encoder = RNAEncoder(self.n_genes, self.latent_dim)
        self.decoder = RNADecoder(self.n_genes, self.latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_latent: bool = False,
        return_decoded: bool = False,
    ) -> torch.Tensor:
        if return_decoded:
            return torch.relu(self.decoder(x))
        latent = self.encoder(x)
        if return_latent:
            return latent
        return self.decoder(latent)


def _trusted_torch_load(path: str | Path, *, map_location: str | torch.device) -> object:
    """Load a user-supplied local checkpoint with explicit trust semantics."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=map_location)


def load_rna_vae_checkpoint(
    checkpoint: str | Path,
    *,
    n_genes: int,
    device: str | torch.device,
) -> RNAExpressionVAE:
    model = RNAExpressionVAE(n_genes=n_genes, latent_dim=128)
    state = _trusted_torch_load(checkpoint, map_location=device)
    if not isinstance(state, dict):
        raise ValueError("RNA VAE checkpoint must contain a state dictionary")
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_scimilarity_initialization(
    model: RNAExpressionVAE,
    model_dir: str | Path,
    *,
    device: str | torch.device,
) -> None:
    """Load transferable SCimilarity layers while replacing gene I/O layers."""
    root = Path(model_dir).expanduser().resolve()
    encoder_path = root / "encoder.ckpt"
    decoder_path = root / "decoder.ckpt"
    gene_order = root / "gene_order.tsv"
    for path in (encoder_path, decoder_path, gene_order):
        if not path.is_file():
            raise FileNotFoundError(path)

    encoder_payload = _trusted_torch_load(encoder_path, map_location=device)
    decoder_payload = _trusted_torch_load(decoder_path, map_location=device)
    encoder_state = dict(encoder_payload["state_dict"])
    decoder_state = dict(decoder_payload["state_dict"])

    # These dimensions depend on the SCimilarity gene space and are replaced
    # by the dataset-specific 13,431-gene input/output layers in the notebook.
    for key in list(encoder_state):
        if key.startswith("network.0.1.") or key.startswith("network.0.2."):
            encoder_state.pop(key)
    for key in ("network.3.weight", "network.3.bias"):
        decoder_state.pop(key, None)

    model.encoder.load_state_dict(encoder_state, strict=False)
    model.decoder.load_state_dict(decoder_state, strict=False)

