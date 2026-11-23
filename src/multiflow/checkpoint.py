"""Portable, weights-only-safe MultiFlow checkpoints."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ._version import __version__
from .models import build_model
from .normalization import LatentStandardizer

CHECKPOINT_FORMAT = 1


def _weights_only_safe(value: Any, *, field: str) -> Any:
    """Return a serialization-safe copy or reject arbitrary Python objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} mapping keys must be strings")
            output[key] = _weights_only_safe(item, field=f"{field}.{key}")
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _weights_only_safe(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field} contains unsupported {type(value).__name__}; use tensors, "
        "NumPy scalars, or JSON-compatible primitive values"
    )


def save_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    *,
    standardizer: LatentStandardizer | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    history: list[dict[str, float]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save model and optional training state."""
    if not hasattr(model, "get_config"):
        raise TypeError("model must implement get_config()")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT,
        "package_version": __version__,
        "model_config": model.get_config(),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "standardizer": None if standardizer is None else standardizer.state_dict(),
        "epoch": None if epoch is None else int(epoch),
        "history": _weights_only_safe(list(history or []), field="history"),
        "metadata": _weights_only_safe(dict(metadata or {}), field="metadata"),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _safe_load(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=True)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, LatentStandardizer | None, dict[str, Any]]:
    """Load a portable checkpoint and reconstruct its model."""
    source = Path(path).expanduser().resolve()
    payload = _safe_load(source, map_location)
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format_version')!r}")
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=strict)
    model.to(map_location)
    standardizer_state = payload.get("standardizer")
    standardizer = (
        None
        if standardizer_state is None
        else LatentStandardizer.from_state_dict(standardizer_state)
    )
    return model, standardizer, payload
