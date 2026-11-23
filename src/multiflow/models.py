"""Neural vector fields used by MultiFlow.

The cross-attention implementation follows the active computation used in the
MultiFlow generation and perturbation experiments: RNA queries attend to ATAC
keys/values, while ATAC queries attend to RNA keys/values.

Some low-level residual, time-embedding, and cross-attention patterns were
adapted from scDiffusion-X.  See ``THIRD_PARTY_NOTICES.md`` for attribution and
license terms.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Create sinusoidal embeddings for continuous flow time."""
    if dim < 2:
        raise ValueError("time embedding dimension must be at least 2")
    if timesteps.ndim == 2:
        if timesteps.shape[1] != 1:
            raise ValueError("two-dimensional time tensors must have shape [batch, 1]")
        timesteps = timesteps[:, 0]
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape [batch] or [batch, 1]")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class ResidualBlock(nn.Module):
    """Feature-wise residual block with time/condition modulation."""

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        out_channels: int | None = None,
        use_scale_shift_norm: bool = True,
    ) -> None:
        super().__init__()
        out_channels = int(out_channels or channels)
        self.use_scale_shift_norm = bool(use_scale_shift_norm)
        self.norm1 = nn.LayerNorm(channels)
        self.fc = nn.Linear(channels, out_channels)
        self.condition_projection = nn.Linear(
            condition_dim,
            2 * out_channels if self.use_scale_shift_norm else out_channels,
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.fc_out = nn.Linear(out_channels, out_channels)
        self.skip = nn.Linear(channels, out_channels) if channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("residual block input must have shape [batch, tokens, channels]")
        hidden = self.fc(F.silu(self.norm1(x)))
        projected = self.condition_projection(F.silu(condition))
        if self.use_scale_shift_norm:
            scale, shift = projected.chunk(2, dim=-1)
            hidden = self.norm2(hidden) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            hidden = hidden + projected.unsqueeze(1)
        hidden = self.fc_out(F.silu(hidden))
        output = self.skip(x) + hidden
        return output.squeeze(1) if squeeze else output


class BidirectionalCrossAttention(nn.Module):
    """Exchange information between RNA and ATAC feature streams.

    The research architecture represents each modality as one token whose
    channels are the hidden features, so inputs have shape ``[batch, 1,
    channels]``.  This is feature-level cross-attention, not sequence-level
    attention over multiple tokens.
    """

    def __init__(self, channels: int, attention_dim: int = 64) -> None:
        super().__init__()
        if channels < 1 or attention_dim < 1:
            raise ValueError("channels and attention_dim must be positive")
        self.channels = int(channels)
        self.attention_dim = int(attention_dim)
        self.q_rna = nn.Linear(1, attention_dim, bias=False)
        self.k_rna = nn.Linear(1, attention_dim, bias=False)
        self.v_rna = nn.Linear(1, attention_dim, bias=False)
        self.q_atac = nn.Linear(1, attention_dim, bias=False)
        self.k_atac = nn.Linear(1, attention_dim, bias=False)
        self.v_atac = nn.Linear(1, attention_dim, bias=False)
        self.to_rna = nn.Linear(attention_dim, 1, bias=False)
        self.to_atac = nn.Linear(attention_dim, 1, bias=False)
        self.scale = attention_dim**-0.5

    def forward(self, rna: torch.Tensor, atac: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rna.shape != atac.shape or rna.ndim != 3:
            raise ValueError("cross-attention inputs must share shape [batch, 1, channels]")
        if rna.shape[1] != 1:
            raise ValueError("cross-attention requires a singleton token dimension")
        if rna.shape[-1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, observed {rna.shape[-1]}")
        rna_features = rna.transpose(-1, -2)
        atac_features = atac.transpose(-1, -2)

        q_rna, k_rna, v_rna = (
            self.q_rna(rna_features),
            self.k_rna(rna_features),
            self.v_rna(rna_features),
        )
        q_atac, k_atac, v_atac = (
            self.q_atac(atac_features),
            self.k_atac(atac_features),
            self.v_atac(atac_features),
        )
        rna_weights = F.softmax(q_rna @ k_atac.transpose(-1, -2) * self.scale, dim=-1)
        atac_weights = F.softmax(q_atac @ k_rna.transpose(-1, -2) * self.scale, dim=-1)
        rna_features = rna_features + self.to_rna(rna_weights @ v_atac)
        atac_features = atac_features + self.to_atac(atac_weights @ v_rna)
        return rna_features.transpose(-1, -2), atac_features.transpose(-1, -2)


class CellStateFlow(nn.Module):
    """Cell-type-conditioned paired RNA/ATAC vector field."""

    model_type = "cell_state"
    requires_shared_base = False
    source_distribution = "joint_standard_normal"

    def __init__(
        self,
        rna_dim: int,
        atac_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        num_classes: int | None = None,
        cross_attention_dim: int = 64,
    ) -> None:
        super().__init__()
        if rna_dim < 1 or atac_dim < 1 or hidden_dim < 2 or num_blocks < 1:
            raise ValueError("latent dimensions, hidden_dim, and num_blocks must be positive")
        if num_classes is not None and num_classes < 1:
            raise ValueError("num_classes must be positive when provided")
        if cross_attention_dim < 1:
            raise ValueError("cross_attention_dim must be positive")
        self.rna_dim = int(rna_dim)
        self.atac_dim = int(atac_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.cross_attention_dim = int(cross_attention_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.label_embedding = (
            nn.Embedding(self.num_classes, hidden_dim) if self.num_classes is not None else None
        )
        self.rna_in = nn.Linear(rna_dim, hidden_dim)
        self.atac_in = nn.Linear(atac_dim, hidden_dim)
        self.down_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.cross1 = BidirectionalCrossAttention(hidden_dim, cross_attention_dim)
        self.middle_cross = BidirectionalCrossAttention(hidden_dim, cross_attention_dim)
        self.middle_residual = ResidualBlock(hidden_dim, hidden_dim)
        self.cross3 = BidirectionalCrossAttention(hidden_dim, cross_attention_dim)
        self.up_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.rna_out = nn.Linear(hidden_dim, rna_dim)
        self.atac_out = nn.Linear(hidden_dim, atac_dim)

    def forward(
        self,
        x_rna: torch.Tensor,
        x_atac: torch.Tensor,
        time: torch.Tensor,
        label: torch.Tensor | None = None,
        pert_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del pert_label
        squeeze = x_rna.ndim == 2
        if squeeze:
            x_rna, x_atac = x_rna.unsqueeze(1), x_atac.unsqueeze(1)
        if x_rna.ndim != 3 or x_atac.ndim != 3:
            raise ValueError("latent inputs must have shape [batch, latent_dim]")
        if x_rna.shape[-1] != self.rna_dim or x_atac.shape[-1] != self.atac_dim:
            raise ValueError("input latent dimensions do not match the model")

        condition = self.time_mlp(timestep_embedding(time, self.hidden_dim))
        if self.label_embedding is not None:
            if label is None:
                raise ValueError("label is required for a class-conditioned model")
            condition = condition + self.label_embedding(label.long())

        rna, atac = self.rna_in(x_rna), self.atac_in(x_atac)
        rna_skips: list[torch.Tensor] = []
        atac_skips: list[torch.Tensor] = []
        for block in self.down_blocks:
            rna, atac = block(rna, condition), block(atac, condition)
            rna_skips.append(rna)
            atac_skips.append(atac)
        rna, atac = self.cross1(rna, atac)
        rna, atac = self.middle_cross(rna, atac)
        rna = self.middle_residual(rna, condition)
        atac = self.middle_residual(atac, condition)
        for index, block in enumerate(self.up_blocks):
            rna, atac = rna + rna_skips.pop(), atac + atac_skips.pop()
            if index == 0:
                rna, atac = self.cross3(rna, atac)
            rna, atac = block(rna, condition), block(atac, condition)
        rna_velocity, atac_velocity = self.rna_out(rna), self.atac_out(atac)
        if squeeze:
            rna_velocity, atac_velocity = rna_velocity.squeeze(1), atac_velocity.squeeze(1)
        return rna_velocity, atac_velocity

    def get_config(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "rna_dim": self.rna_dim,
            "atac_dim": self.atac_dim,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "num_classes": self.num_classes,
            "cross_attention_dim": self.cross_attention_dim,
        }


class PerturbationFlow(CellStateFlow):
    """Context- and perturbation-conditioned paired vector field."""

    model_type = "perturbation"

    def __init__(
        self,
        rna_dim: int,
        atac_dim: int,
        context_matrix: torch.Tensor | None = None,
        *,
        num_contexts: int | None = None,
        context_dim: int | None = None,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        cross_attention_dim: int = 64,
        freeze_context: bool = True,
        context_scale: float = 1.0,
        num_perturbations: int = 1,
        perturbation_scale: float = 1.0,
    ) -> None:
        if context_matrix is None:
            if num_contexts is None:
                raise ValueError("provide context_matrix or num_contexts")
            context_dim = int(context_dim or hidden_dim)
            context_matrix = torch.zeros(int(num_contexts), context_dim)
        context_matrix = torch.as_tensor(context_matrix, dtype=torch.float32)
        if context_matrix.ndim != 2 or context_matrix.shape[0] < 1:
            raise ValueError("context_matrix must have shape [num_contexts, context_dim]")
        if not torch.isfinite(context_matrix).all():
            raise ValueError("context_matrix must contain only finite values")
        context_width = int(context_matrix.shape[1])
        if context_width != hidden_dim and 2 * context_width != hidden_dim:
            raise ValueError("context width must equal hidden_dim or hidden_dim / 2")
        if num_perturbations < 1:
            raise ValueError("num_perturbations must be at least one (control)")

        super().__init__(
            rna_dim=rna_dim,
            atac_dim=atac_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_classes=None,
            cross_attention_dim=cross_attention_dim,
        )
        self.num_contexts = int(context_matrix.shape[0])
        self.context_dim = int(context_matrix.shape[1])
        self.num_perturbations = int(num_perturbations)
        self.freeze_context = bool(freeze_context)
        self.context_scale = float(context_scale)
        self.perturbation_scale = float(perturbation_scale)
        self.context_embedding = nn.Embedding.from_pretrained(
            context_matrix,
            freeze=freeze_context,
        )
        self.perturbation_embedding = nn.Embedding(num_perturbations, hidden_dim)
        with torch.no_grad():
            self.perturbation_embedding.weight[0].zero_()
        self.perturbation_embedding.weight.register_hook(self._zero_control_gradient)
        self.time_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.context_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.perturbation_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    @staticmethod
    def _zero_control_gradient(gradient: torch.Tensor | None) -> torch.Tensor | None:
        if gradient is None:
            return None
        gradient = gradient.clone()
        gradient[0].zero_()
        return gradient

    def _condition(
        self,
        time: torch.Tensor,
        label: torch.Tensor,
        pert_label: torch.Tensor | None,
    ) -> torch.Tensor:
        time_emb = self.time_mlp(timestep_embedding(time, self.hidden_dim))
        context_emb = self.context_embedding(label.long())
        if context_emb.shape[-1] * 2 == self.hidden_dim:
            context_emb = torch.cat([context_emb, context_emb], dim=-1)
        if pert_label is None:
            pert_label = torch.zeros_like(label)
        pert_label = pert_label.long()
        perturbation_emb = self.perturbation_embedding(pert_label)
        perturbation_emb = torch.where(
            (pert_label == 0).unsqueeze(-1),
            torch.zeros_like(perturbation_emb),
            perturbation_emb,
        )
        return (
            self.time_norm(time_emb)
            + self.context_scale * self.context_norm(context_emb)
            + self.perturbation_scale * self.perturbation_norm(perturbation_emb)
        )

    def forward(
        self,
        x_rna: torch.Tensor,
        x_atac: torch.Tensor,
        time: torch.Tensor,
        label: torch.Tensor | None = None,
        pert_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if label is None:
            raise ValueError("label is required by PerturbationFlow")
        squeeze = x_rna.ndim == 2
        if squeeze:
            x_rna, x_atac = x_rna.unsqueeze(1), x_atac.unsqueeze(1)
        if x_rna.shape[-1] != self.rna_dim or x_atac.shape[-1] != self.atac_dim:
            raise ValueError("input latent dimensions do not match the model")
        condition = self._condition(time, label, pert_label)
        rna, atac = self.rna_in(x_rna), self.atac_in(x_atac)
        rna_skips: list[torch.Tensor] = []
        atac_skips: list[torch.Tensor] = []
        for block in self.down_blocks:
            rna, atac = block(rna, condition), block(atac, condition)
            rna_skips.append(rna)
            atac_skips.append(atac)
        rna, atac = self.cross1(rna, atac)
        rna, atac = self.middle_cross(rna, atac)
        rna = self.middle_residual(rna, condition)
        atac = self.middle_residual(atac, condition)
        for index, block in enumerate(self.up_blocks):
            rna, atac = rna + rna_skips.pop(), atac + atac_skips.pop()
            if index == 0:
                rna, atac = self.cross3(rna, atac)
            rna, atac = block(rna, condition), block(atac, condition)
        rna_velocity, atac_velocity = self.rna_out(rna), self.atac_out(atac)
        if squeeze:
            rna_velocity, atac_velocity = rna_velocity.squeeze(1), atac_velocity.squeeze(1)
        return rna_velocity, atac_velocity

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "model_type": self.model_type,
                "num_contexts": self.num_contexts,
                "context_dim": self.context_dim,
                "freeze_context": self.freeze_context,
                "context_scale": self.context_scale,
                "num_perturbations": self.num_perturbations,
                "perturbation_scale": self.perturbation_scale,
            }
        )
        config.pop("num_classes", None)
        return config


class _FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mlp(x))


class _ConcatInputResidualBlock(nn.Module):
    """Input residual block used by the executed concat notebook graph."""

    def __init__(self, in_features: int, out_features: int, condition_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.condition_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, out_features),
        )
        self.activation = nn.SiLU()
        self.skip = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.activation(self.norm1(self.fc1(x)))
        hidden = hidden + self.condition_projection(condition)
        hidden = self.activation(self.norm2(self.fc2(hidden)))
        return residual + hidden


class ConditionalConcatFlow(nn.Module):
    """Historical concat ablation with a replicated, rank-deficient base."""

    model_type = "concat"
    requires_shared_base = True
    source_distribution = "replicated_shared_normal_legacy"

    def __init__(
        self,
        rna_dim: int,
        atac_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_classes: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if rna_dim < 1 or atac_dim < 1 or hidden_dim < 1:
            raise ValueError("latent dimensions and hidden_dim must be positive")
        if num_layers < 2:
            raise ValueError("num_layers must be at least two")
        if rna_dim != atac_dim:
            raise ValueError("ConditionalConcatFlow requires equal RNA and ATAC latent widths")
        if num_classes is not None and num_classes < 1:
            raise ValueError("num_classes must be positive when provided")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.rna_dim, self.atac_dim = int(rna_dim), int(atac_dim)
        self.hidden_dim, self.num_layers = int(hidden_dim), int(num_layers)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.dropout = float(dropout)
        self.fusion = nn.Linear(rna_dim + atac_dim, rna_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.label_embedding = (
            nn.Embedding(self.num_classes, hidden_dim) if self.num_classes is not None else None
        )
        self.input_block = _ConcatInputResidualBlock(rna_dim, hidden_dim, hidden_dim)
        self.down = nn.ModuleList(
            [_FeedForwardBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.up = nn.ModuleList(
            [_FeedForwardBlock(hidden_dim, dropout) for _ in range(num_layers - 1)]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, rna_dim),
            nn.Linear(rna_dim, rna_dim + atac_dim),
        )

    def forward(
        self,
        x_rna: torch.Tensor,
        x_atac: torch.Tensor,
        time: torch.Tensor,
        label: torch.Tensor | None = None,
        pert_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del pert_label
        if x_rna.ndim != 2 or x_atac.ndim != 2:
            raise ValueError("concat inputs must have shape [batch, latent_dim]")
        if x_rna.shape[0] != x_atac.shape[0]:
            raise ValueError("RNA and ATAC batches must contain the same paired cells")
        if x_rna.shape[1] != self.rna_dim or x_atac.shape[1] != self.atac_dim:
            raise ValueError("input latent dimensions do not match the model")
        condition = self.time_mlp(timestep_embedding(time, self.hidden_dim))
        if self.label_embedding is not None:
            if label is None:
                raise ValueError("label is required for a class-conditioned model")
            condition = condition + self.label_embedding(label.long())
        hidden = self.fusion(torch.cat([x_rna, x_atac], dim=-1))
        hidden = self.input_block(hidden, condition)
        skips = []
        for layer in self.down:
            hidden = layer(hidden)
            skips.append(hidden)
        skips.pop()
        for layer in self.up:
            hidden = layer(hidden) + skips.pop()
        joint = self.output(hidden)
        return torch.split(joint, [self.rna_dim, self.atac_dim], dim=-1)

    def get_config(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "rna_dim": self.rna_dim,
            "atac_dim": self.atac_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }


def build_model(config: Mapping[str, object] | None = None, /, **kwargs: object) -> nn.Module:
    """Build a MultiFlow model from a serialized configuration."""
    values = dict(config or {})
    values.update(kwargs)
    model_type = str(values.pop("model_type", "cell_state"))
    if model_type == "cell_state":
        return CellStateFlow(**values)
    if model_type == "perturbation":
        return PerturbationFlow(**values)
    if model_type == "concat":
        return ConditionalConcatFlow(**values)
    raise ValueError(f"unknown model_type: {model_type!r}")
