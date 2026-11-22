"""Paired RNA/ATAC autoencoder used to build MultiFlow latent states.

The network, its likelihoods and its training recipe are ported from the
multimodal autoencoder of scDiffusion-X (MIT license; commit
``d60d928b635ab7a52ace030a641efd02137c193f``, see ``THIRD_PARTY_NOTICES.md``).
Parameter names are unchanged, so checkpoints written by scDiffusion-X load
into :class:`MultimodalAutoencoder` directly.

RNA is encoded from raw counts and decoded as a negative-binomial mean (a
softmax over genes times the library size). ATAC is encoded from binary
accessibility and decoded as Bernoulli probabilities.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.distributions import Bernoulli

CHECKPOINT_FORMAT = "multiflow-autoencoder"
DEFAULT_RNA_DIMS = (512, 300, 128)
DEFAULT_ATAC_DIMS = (1024, 512, 128)


class MLP(nn.Module):
    """Linear-norm-ELU blocks followed by one linear output layer."""

    def __init__(
        self,
        dims: Sequence[int],
        *,
        norm: bool = True,
        norm_type: str = "batchnorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("an MLP needs an input and an output size")
        if norm_type not in {"batchnorm", "layernorm"}:
            raise ValueError("norm_type must be batchnorm or layernorm")
        self.dims = [int(value) for value in dims]
        layers: list[nn.Module] = []
        for index in range(len(self.dims) - 2):
            block: list[nn.Module] = [nn.Linear(self.dims[index], self.dims[index + 1])]
            if norm:
                block.append(
                    nn.BatchNorm1d(self.dims[index + 1])
                    if norm_type == "batchnorm"
                    else nn.LayerNorm(self.dims[index + 1])
                )
            block.append(nn.ELU())
            if dropout > 0:
                block.append(nn.Dropout(float(dropout)))
            layers.append(nn.Sequential(*block))
        layers.append(nn.Linear(self.dims[-2], self.dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def negative_binomial_log_prob(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-probability of counts under NB(mean ``mu``, inverse dispersion ``theta``)."""
    log_theta_mu = torch.log(theta + mu + eps)
    return (
        theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )


class MultimodalAutoencoder(nn.Module):
    """Modality-specific encoders and decoders for paired RNA and ATAC."""

    def __init__(
        self,
        n_genes: int,
        n_peaks: int,
        *,
        rna_dims: Sequence[int] = DEFAULT_RNA_DIMS,
        atac_dims: Sequence[int] = DEFAULT_ATAC_DIMS,
        norm: bool = True,
        norm_type: str = "batchnorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_peaks = int(n_peaks)
        self.rna_dims = [int(value) for value in rna_dims]
        self.atac_dims = [int(value) for value in atac_dims]
        self.norm = bool(norm)
        self.norm_type = str(norm_type)
        self.dropout = float(dropout)
        options = {"norm": self.norm, "norm_type": self.norm_type, "dropout": self.dropout}
        # Creation order (theta, RNA encoder/decoder, ATAC encoder/decoder)
        # follows scDiffusion-X, so a fixed seed gives the same initial weights.
        self.theta = nn.Parameter(torch.randn(self.n_genes))
        rna = [self.n_genes, *self.rna_dims]
        atac = [self.n_peaks, *self.atac_dims]
        encoder: dict[str, nn.Module] = {}
        decoder: dict[str, nn.Module] = {}
        encoder["rna"] = MLP(rna, **options)
        decoder["rna"] = MLP(rna[::-1], **options)
        encoder["atac"] = MLP(atac, **options)
        decoder["atac"] = MLP(atac[::-1], **options)
        self.encoder = nn.ModuleDict(encoder)
        self.decoder = nn.ModuleDict(decoder)

    @property
    def rna_latent_dim(self) -> int:
        return self.rna_dims[-1]

    @property
    def atac_latent_dim(self) -> int:
        return self.atac_dims[-1]

    def get_config(self) -> dict[str, Any]:
        return {
            "n_genes": self.n_genes,
            "n_peaks": self.n_peaks,
            "rna_dims": list(self.rna_dims),
            "atac_dims": list(self.atac_dims),
            "norm": self.norm,
            "norm_type": self.norm_type,
            "dropout": self.dropout,
        }

    def encode(self, rna: torch.Tensor, atac: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw RNA counts and binary ATAC into the two latent states."""
        return self.encoder["rna"](rna), self.encoder["atac"](atac)

    def decode_rna(self, z_rna: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        """Expected RNA counts: gene proportions times the library size."""
        return torch.softmax(self.decoder["rna"](z_rna), dim=1) * library_size

    def sample_rna(self, z_rna: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        """RNA counts drawn from the negative binomial of the decoder.

        The mean is :meth:`decode_rna` and the inverse dispersion ``exp(theta)``, the
        distribution the autoencoder is trained with. Sampled as a gamma-Poisson mixture.
        """
        mean = self.decode_rna(z_rna, library_size)
        theta = torch.exp(self.theta)
        rate = torch.distributions.Gamma(
            concentration=theta, rate=theta / mean.clamp_min(1e-8)
        ).sample()
        return torch.poisson(rate.clamp(max=1e8))

    def decode_atac(self, z_atac: torch.Tensor) -> torch.Tensor:
        """Accessibility probability of every peak."""
        return torch.sigmoid(self.decoder["atac"](z_atac))

    def loss(self, rna: torch.Tensor, atac: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood summed over features and averaged over cells."""
        z_rna, z_atac = self.encode(rna, atac)
        mu = self.decode_rna(z_rna, rna.sum(dim=1, keepdim=True))
        rna_term = negative_binomial_log_prob(rna, mu, torch.exp(self.theta)).sum(dim=1).mean()
        atac_term = Bernoulli(probs=self.decode_atac(z_atac)).log_prob(atac).sum(dim=1).mean()
        return -(rna_term + atac_term)


# ----------------------------------------------------------------------------- checkpoints
class _Placeholder(dict):
    """Stands in for classes of packages that are not installed (e.g. omegaconf)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def __setstate__(self, state: Any) -> None:
        pass

    def append(self, item: Any) -> None:
        pass

    def extend(self, items: Any) -> None:
        pass

    def add(self, item: Any) -> None:
        pass


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Placeholder


class _TolerantPickle:
    """``pickle_module`` for ``torch.load`` that skips classes it cannot import.

    Lightning checkpoints written by scDiffusion-X carry training metadata
    (Hydra/omegaconf objects) next to the weights. Only the weights are used
    here, so the metadata classes need not be importable.
    """

    __name__ = "pickle"
    Unpickler = _TolerantUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def _read_payload(path: Path, *, device: str | torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception:  # noqa: BLE001 - foreign (Lightning) checkpoint; the file is user-supplied
        payload = torch.load(
            path, map_location=device, weights_only=False, pickle_module=_TolerantPickle
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"{path} is not an autoencoder checkpoint (no state_dict)")
    return payload


def _mlp_dims(state: dict[str, torch.Tensor], prefix: str) -> list[int]:
    weights = [
        (key, value)
        for key, value in state.items()
        if key.startswith(prefix) and key.endswith("weight") and value.ndim == 2
    ]
    if not weights:
        raise ValueError(f"checkpoint has no layers under {prefix!r}")
    weights.sort(key=lambda item: int(item[0][len(prefix):].split(".")[0]))
    return [int(weights[0][1].shape[1]), *[int(value.shape[0]) for _, value in weights]]


def config_from_state_dict(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Recover the architecture from the parameter shapes of a checkpoint."""
    rna = _mlp_dims(state, "encoder.rna.net.")
    atac = _mlp_dims(state, "encoder.atac.net.")
    norm_keys = [key for key in state if key.startswith("encoder.rna.net.0.1.")]
    return {
        "n_genes": rna[0],
        "n_peaks": atac[0],
        "rna_dims": rna[1:],
        "atac_dims": atac[1:],
        "norm": bool(norm_keys),
        "norm_type": (
            "batchnorm" if any(key.endswith("running_mean") for key in norm_keys) else "layernorm"
        ),
        "dropout": 0.0,
    }


def load_autoencoder(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> MultimodalAutoencoder:
    """Load a MultiFlow or scDiffusion-X autoencoder checkpoint in eval mode."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _read_payload(path, device=device)
    state = payload["state_dict"]
    config = payload.get("config") if payload.get("format") == CHECKPOINT_FORMAT else None
    model = MultimodalAutoencoder(**(config or config_from_state_dict(state)))
    model.load_state_dict(state)
    return model.to(device).eval()


def save_autoencoder(
    path: str | Path,
    model: MultimodalAutoencoder,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write weights, architecture and metadata as plain tensors and JSON."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "schema_version": 1,
        "config": model.get_config(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": json.loads(json.dumps(metadata or {}, default=str)),
    }
    torch.save(payload, destination)
    return destination


# ----------------------------------------------------------------------------- data access
def _rows(matrix: Any, index: np.ndarray | slice) -> np.ndarray:
    block = matrix[index]
    if sparse.issparse(block):
        block = block.toarray()
    return np.asarray(block, dtype=np.float32)


def _check_inputs(rna: Any, atac: Any) -> None:
    if rna.shape[0] != atac.shape[0]:
        raise ValueError("RNA and ATAC must contain the same cells in the same order")
    if rna.shape[0] < 2:
        raise ValueError("at least two paired cells are required")


# ----------------------------------------------------------------------------- training
@dataclass(frozen=True)
class AutoencoderConfig:
    """Training settings; the defaults are those of the paper."""

    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    validation_fraction: float = 0.1
    rna_dims: tuple[int, ...] = DEFAULT_RNA_DIMS
    atac_dims: tuple[int, ...] = DEFAULT_ATAC_DIMS
    seed: int = 20260717
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AutoencoderResult:
    model: MultimodalAutoencoder
    history: list[dict[str, float]] = field(default_factory=list)
    config: AutoencoderConfig | None = None


def fit_autoencoder(
    rna: Any,
    atac: Any,
    *,
    config: AutoencoderConfig | None = None,
    verbose: bool = True,
) -> AutoencoderResult:
    """Train the paired autoencoder on raw RNA counts and binary ATAC.

    ``rna`` and ``atac`` are cell-by-feature matrices (SciPy sparse or NumPy)
    with identical row order. Batches are densified one at a time, so the full
    matrices stay sparse in memory.
    """
    cfg = config or AutoencoderConfig()
    _check_inputs(rna, atac)
    if not 0 <= cfg.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    n_cells = int(rna.shape[0])
    order = torch.randperm(n_cells, generator=generator).numpy()
    n_valid = int(round(cfg.validation_fraction * n_cells))
    valid_index, train_index = np.sort(order[:n_valid]), order[n_valid:]
    batch_size = min(int(cfg.batch_size), len(train_index))
    if batch_size < 2:
        raise ValueError("the training split needs at least two cells")

    model = MultimodalAutoencoder(
        rna.shape[1], atac.shape[1], rna_dims=cfg.rna_dims, atac_dims=cfg.atac_dims
    ).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    def tensors(index: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        index = np.sort(index)
        return (
            torch.from_numpy(_rows(rna, index)).to(cfg.device),
            torch.from_numpy(_rows(atac, index)).to(cfg.device),
        )

    history: list[dict[str, float]] = []
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        shuffled = train_index[torch.randperm(len(train_index), generator=generator).numpy()]
        total, batches = 0.0, 0
        for start in range(0, len(shuffled) - batch_size + 1, batch_size):
            loss = model.loss(*tensors(shuffled[start : start + batch_size]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        record = {"epoch": float(epoch), "train_loss": total / max(batches, 1)}
        if len(valid_index):
            model.eval()
            weighted, seen = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(valid_index), batch_size):
                    block = valid_index[start : start + batch_size]
                    weighted += float(model.loss(*tensors(block))) * len(block)
                    seen += len(block)
            record["valid_loss"] = weighted / seen
        history.append(record)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs):
            message = f"autoencoder epoch {epoch}/{cfg.epochs}  train {record['train_loss']:.2f}"
            if "valid_loss" in record:
                message += f"  valid {record['valid_loss']:.2f}"
            print(message, flush=True)
    return AutoencoderResult(model=model.eval(), history=history, config=cfg)


# ----------------------------------------------------------------------------- encoding
@torch.no_grad()
def encode_latents(
    model: MultimodalAutoencoder,
    rna: Any,
    atac: Any,
    *,
    batch_size: int = 1000,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encoder outputs for every cell, before the latent rescaling."""
    _check_inputs(rna, atac)
    target = next(model.parameters()).device if device is None else torch.device(device)
    model = model.to(target).eval()
    z_rna: list[np.ndarray] = []
    z_atac: list[np.ndarray] = []
    for start in range(0, int(rna.shape[0]), int(batch_size)):
        block = slice(start, start + int(batch_size))
        a, b = model.encode(
            torch.from_numpy(_rows(rna, block)).to(target),
            torch.from_numpy(_rows(atac, block)).to(target),
        )
        z_rna.append(a.cpu().numpy())
        z_atac.append(b.cpu().numpy())
    return np.concatenate(z_rna), np.concatenate(z_atac)


def latent_scale(latent: np.ndarray) -> float:
    """Ten times the mean per-dimension standard deviation (scDiffusion-X ``std10``)."""
    values = torch.as_tensor(np.asarray(latent, dtype=np.float32))
    return float(values.std(dim=0).mean() * 10)


def training_summary(result: AutoencoderResult) -> dict[str, Any]:
    """JSON-ready description of a finished training run."""
    summary: dict[str, Any] = {"history": result.history}
    if result.config is not None:
        summary["training"] = asdict(result.config)
    return summary
