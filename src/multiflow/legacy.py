"""Migration helpers for trusted historical MultiFlow checkpoints.

Historical research scripts used several checkpoint envelopes and internal
module names.  This module converts those files into the portable checkpoint
format implemented by :mod:`multiflow.checkpoint`.  Loading is restricted
to PyTorch's ``weights_only`` mode unless the caller explicitly confirms that
the source is trusted.
"""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .checkpoint import save_checkpoint
from .models import build_model
from .normalization import LatentStandardizer


class LegacyCheckpointError(ValueError):
    """Raised when a historical checkpoint cannot be migrated safely."""


_CELL_KEY_REPLACEMENTS = (
    ("label_emb.", "label_embedding."),
    ("mid_cross.", "middle_cross."),
    ("mid_res.", "middle_residual."),
    (".emb_proj.", ".condition_projection."),
    (".trans_rna.", ".to_rna."),
    (".trans_atac.", ".to_atac."),
)

_PERTURBATION_KEY_REPLACEMENTS = (
    ("context_emb.", "context_embedding."),
    ("pert_emb.", "perturbation_embedding."),
    ("time_ln.", "time_norm."),
    ("context_ln.", "context_norm."),
    ("pert_ln.", "perturbation_norm."),
) + _CELL_KEY_REPLACEMENTS

_CONCAT_KEY_REPLACEMENTS = (
    ("fusion_linear.", "fusion."),
    ("time_embedding.time_embed.", "time_mlp."),
    ("input_block.emb_layer.", "input_block.condition_projection."),
    ("down_layers.", "down."),
    ("up_layers.", "up."),
    ("out1.", "output.0."),
    ("out_norm.", "output.1."),
    ("out2.", "output.3."),
    ("output_linear.", "output.4."),
)


def _load_payload(
    path: Path,
    *,
    map_location: str | torch.device,
    trust_source: bool,
) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError) as safe_error:
        if not trust_source:
            raise LegacyCheckpointError(
                "the historical checkpoint is not compatible with PyTorch's safe "
                "weights-only loader; inspect the source and pass trust_source=True "
                "only when you trust the file and the environment that produced it"
            ) from safe_error
        warnings.warn(
            "Loading a trusted historical checkpoint with weights_only=False. "
            "Arbitrary pickle code may execute; the converted checkpoint will be "
            "written in MultiFlow's safe format.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def _is_state_dict(value: Any) -> bool:
    return bool(value) and isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(item, torch.Tensor)
        for key, item in value.items()
    )


def _strip_parallel_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state)
    strip_module = bool(keys) and all(key.startswith("module.") for key in keys)
    output: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        normalized = key[7:] if strip_module else key
        if normalized in output:
            raise LegacyCheckpointError(f"duplicate state key after prefix removal: {normalized}")
        output[normalized] = value
    return output


def _unpack_payload(
    payload: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str, Mapping[str, Any]]:
    if _is_state_dict(payload):
        return _strip_parallel_prefix(payload), {}, "raw_state_dict", {}
    if not isinstance(payload, Mapping):
        raise LegacyCheckpointError("historical checkpoint must contain a mapping")
    if int(payload.get("format_version", -1)) == 1 and "model_state" in payload:
        raise LegacyCheckpointError(
            "this is already a current MultiFlow checkpoint; use load_checkpoint()"
        )

    if "model_state" in payload:
        state = payload["model_state"]
        container = f"format_v{payload.get('format_version', 'unknown')}"
    elif "model_state_dict" in payload:
        state = payload["model_state_dict"]
        container = "model_state_dict"
    else:
        raise LegacyCheckpointError(
            "checkpoint contains neither a raw state_dict, model_state, nor model_state_dict"
        )
    if not _is_state_dict(state):
        raise LegacyCheckpointError("historical model state is not a tensor state_dict")

    config = payload.get("model_config", payload.get("model_kwargs", {}))
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise LegacyCheckpointError("historical model configuration must be a mapping")
    return _strip_parallel_prefix(state), dict(config), container, payload


def _replace_key(key: str, replacements: tuple[tuple[str, str], ...]) -> str:
    output = key
    for old, new in replacements:
        if old.startswith("."):
            output = output.replace(old, new)
        elif output.startswith(old):
            output = new + output[len(old) :]
    return output


def _remap_state(
    state: Mapping[str, torch.Tensor],
    model_type: str,
) -> dict[str, torch.Tensor]:
    replacements = {
        "cell_state": _CELL_KEY_REPLACEMENTS,
        "perturbation": _PERTURBATION_KEY_REPLACEMENTS,
        "concat": _CONCAT_KEY_REPLACEMENTS,
    }[model_type]
    output: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        mapped = _replace_key(key, replacements)
        if mapped in output:
            raise LegacyCheckpointError(f"legacy state keys collide at {mapped!r}")
        output[mapped] = value
    return output


def _normalize_model_type(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "cell": "cell_state",
        "cell_state": "cell_state",
        "cellstateflow": "cell_state",
        "perturbation": "perturbation",
        "perturbationflow": "perturbation",
        "concat": "concat",
        "conditionalconcatflow": "concat",
    }
    if normalized not in aliases:
        raise LegacyCheckpointError(f"unknown historical model type: {value!r}")
    return aliases[normalized]


def _infer_model_type(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    explicit: str | None,
) -> str:
    if explicit is not None:
        return _normalize_model_type(explicit)
    configured = config.get("model_type")
    if configured is not None:
        return _normalize_model_type(str(configured))
    variant = str(config.get("model_variant", "")).lower()
    if "concat" in variant:
        return "concat"
    if "notebook_cross_attention" in variant:
        return "cell_state"
    keys = set(state)
    if any(key.startswith(("context_emb.", "context_embedding.", "pert_emb.")) for key in keys):
        return "perturbation"
    if any(key.startswith(("fusion_linear.", "fusion.")) for key in keys):
        return "concat"
    if "rna_in.weight" in keys and "atac_in.weight" in keys:
        return "cell_state"
    raise LegacyCheckpointError(
        "could not infer the model architecture; pass model_type explicitly"
    )


def _first(config: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return default


def _weight(
    state: Mapping[str, torch.Tensor],
    *names: str,
) -> torch.Tensor | None:
    for name in names:
        if name in state:
            return state[name]
    return None


def _require_weight(
    state: Mapping[str, torch.Tensor],
    *names: str,
) -> torch.Tensor:
    value = _weight(state, *names)
    if value is None:
        raise LegacyCheckpointError(f"state_dict is missing one of {names!r}")
    return value


def _module_count(state: Mapping[str, torch.Tensor], *prefixes: str) -> int:
    indices: set[int] = set()
    for key in state:
        for prefix in prefixes:
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                first = suffix.split(".", 1)[0]
                if first.isdigit():
                    indices.add(int(first))
    if not indices:
        raise LegacyCheckpointError(f"could not infer module count for {prefixes!r}")
    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise LegacyCheckpointError(f"non-contiguous module indices for {prefixes!r}: {indices}")
    return len(indices)


def _optional_num_classes(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> int | None:
    configured = _first(config, "num_classes")
    if configured is not None:
        return int(configured)
    embedding = _weight(state, "label_emb.weight", "label_embedding.weight")
    return None if embedding is None else int(embedding.shape[0])


def _cell_config(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rna_in = _require_weight(state, "rna_in.weight")
    atac_in = _require_weight(state, "atac_in.weight")
    attention = _require_weight(state, "cross1.q_rna.weight")
    return {
        "model_type": "cell_state",
        "rna_dim": int(_first(config, "rna_dim", default=rna_in.shape[1])),
        "atac_dim": int(_first(config, "atac_dim", default=atac_in.shape[1])),
        "hidden_dim": int(_first(config, "hidden_dim", default=rna_in.shape[0])),
        "num_blocks": int(
            _first(
                config,
                "num_blocks",
                default=_module_count(state, "down_blocks."),
            )
        ),
        "num_classes": _optional_num_classes(state, config),
        "cross_attention_dim": int(
            _first(
                config,
                "cross_attention_dim",
                "cross_attn_feature_dim",
                default=attention.shape[0],
            )
        ),
    }


def _perturbation_config(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    base = _cell_config(state, config)
    context_from_state = _weight(
        state,
        "context_emb.weight",
        "context_embedding.weight",
    )
    context_from_payload = envelope.get("context_matrix")
    if context_from_state is not None:
        context = torch.as_tensor(context_from_state).float()
    elif context_from_payload is not None:
        context = torch.as_tensor(context_from_payload).float()
    else:
        raise LegacyCheckpointError("perturbation checkpoint is missing its context matrix")
    if context.ndim != 2 or context.shape[0] < 1 or not torch.isfinite(context).all():
        raise LegacyCheckpointError("legacy context matrix must be a finite two-dimensional tensor")
    if context_from_state is not None and context_from_payload is not None:
        payload_context = torch.as_tensor(context_from_payload)
        if payload_context.shape != context.shape:
            raise LegacyCheckpointError("state and payload context matrices have different shapes")

    perturbation = _weight(
        state,
        "pert_emb.weight",
        "perturbation_embedding.weight",
    )
    num_perturbations = int(
        _first(
            config,
            "num_perturbations",
            default=1 if perturbation is None else perturbation.shape[0],
        )
    )
    base.pop("num_classes", None)
    base.update(
        {
            "model_type": "perturbation",
            "context_matrix": context,
            "freeze_context": bool(_first(config, "freeze_context", default=True)),
            "context_scale": float(_first(config, "context_scale", default=1.0)),
            "num_perturbations": num_perturbations,
            "perturbation_scale": float(
                _first(config, "perturbation_scale", default=1.0)
            ),
        }
    )
    return base


def _concat_config(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fusion = _require_weight(state, "fusion_linear.weight", "fusion.weight")
    inferred_rna = int(fusion.shape[0])
    inferred_atac = int(fusion.shape[1]) - inferred_rna
    hidden = _require_weight(
        state,
        "time_embedding.time_embed.0.weight",
        "time_mlp.0.weight",
    )
    return {
        "model_type": "concat",
        "rna_dim": int(_first(config, "rna_dim", default=inferred_rna)),
        "atac_dim": int(_first(config, "atac_dim", default=inferred_atac)),
        "hidden_dim": int(_first(config, "hidden_dim", default=hidden.shape[0])),
        "num_layers": int(
            _first(
                config,
                "num_layers",
                default=_module_count(state, "down_layers.", "down."),
            )
        ),
        "num_classes": _optional_num_classes(state, config),
        "dropout": float(_first(config, "dropout", default=0.1)),
    }


def _build_legacy_model(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    envelope: Mapping[str, Any],
    model_type: str,
    map_location: str | torch.device,
) -> nn.Module:
    if model_type == "cell_state":
        normalized = _cell_config(state, config)
    elif model_type == "perturbation":
        normalized = _perturbation_config(state, config, envelope)
    else:
        normalized = _concat_config(state, config)
    model = build_model(normalized)
    remapped = _remap_state(state, model_type)
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    allowed_missing: set[str] = set()
    if model_type == "perturbation":
        if "context_embedding.weight" not in remapped and "context_matrix" in envelope:
            allowed_missing.add("context_embedding.weight")
        if (
            "perturbation_embedding.weight" not in remapped
            and int(model.num_perturbations) == 1
        ):
            allowed_missing.add("perturbation_embedding.weight")
    remaining_missing = set(missing).difference(allowed_missing)
    if remaining_missing or unexpected:
        raise LegacyCheckpointError(
            "legacy model state is incompatible after key migration: "
            f"missing={sorted(remaining_missing)}, unexpected={sorted(unexpected)}"
        )
    return model.to(map_location)


def _coerce_standardizer(
    value: LatentStandardizer | Mapping[str, torch.Tensor] | None,
) -> LatentStandardizer | None:
    if value is None or isinstance(value, LatentStandardizer):
        return value
    if isinstance(value, Mapping):
        return LatentStandardizer.from_state_dict(dict(value))
    raise TypeError("standardizer must be LatentStandardizer, a state mapping, or None")


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    return str(value)


def _migration_info(
    source: Path,
    container: str,
    model_type: str,
    envelope: Mapping[str, Any],
    standardizer: LatentStandardizer | None,
) -> dict[str, Any]:
    epoch = envelope.get("completed_epochs", envelope.get("epoch"))
    labels = envelope.get("label_classes")
    perturbations = envelope.get("perturbation_classes")
    info: dict[str, Any] = {
        "source_name": source.name,
        "container": container,
        "legacy_format_version": _safe_scalar(envelope.get("format_version")),
        "model_type": model_type,
        "epoch": _safe_scalar(epoch),
        "standardizer_recovered": standardizer is not None,
        "optimizer_state_omitted": bool(
            "optimizer_state" in envelope or "optimizer_state_dict" in envelope
        ),
    }
    if isinstance(labels, (list, tuple)):
        info["label_classes"] = [str(item) for item in labels]
    if isinstance(perturbations, (list, tuple)):
        info["perturbation_classes"] = [str(item) for item in perturbations]
    if "heldout_cell_type" in envelope:
        info["heldout_cell_type"] = str(envelope["heldout_cell_type"])
    return info


def load_legacy_checkpoint(
    path: str | Path,
    *,
    model_type: str | None = None,
    standardizer: LatentStandardizer | Mapping[str, torch.Tensor] | None = None,
    map_location: str | torch.device = "cpu",
    trust_source: bool = False,
) -> tuple[nn.Module, LatentStandardizer | None, dict[str, Any]]:
    """Load a historical checkpoint without writing a converted file.

    ``trust_source`` is deliberately false by default.  Set it only for a file
    whose provenance has been independently verified when PyTorch's safe
    weights-only loader cannot decode its legacy NumPy/Python objects.
    """
    source = Path(path).expanduser().resolve()
    payload = _load_payload(
        source,
        map_location=map_location,
        trust_source=trust_source,
    )
    state, config, container, envelope = _unpack_payload(payload)
    inferred_type = _infer_model_type(state, config, model_type)
    model = _build_legacy_model(
        state,
        config,
        envelope,
        inferred_type,
        map_location,
    )
    recovered = _coerce_standardizer(standardizer)
    if recovered is None and isinstance(envelope.get("latent_stats"), Mapping):
        recovered = LatentStandardizer.from_state_dict(dict(envelope["latent_stats"]))
    info = _migration_info(source, container, inferred_type, envelope, recovered)
    return model, recovered, info


def migrate_legacy_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    model_type: str | None = None,
    standardizer: LatentStandardizer | Mapping[str, torch.Tensor] | None = None,
    map_location: str | torch.device = "cpu",
    trust_source: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Convert one historical checkpoint into the current safe format."""
    model, recovered, info = load_legacy_checkpoint(
        source,
        model_type=model_type,
        standardizer=standardizer,
        map_location=map_location,
        trust_source=trust_source,
    )
    output_metadata: dict[str, Any] = {"legacy_migration": info}
    for key in ("label_classes", "perturbation_classes"):
        if key in info:
            output_metadata[key] = list(info[key])
    if metadata is not None:
        output_metadata.update(dict(metadata))
    config = model.get_config()
    expected_mappings = {
        "label_classes": (
            config.get("num_contexts")
            if config.get("model_type") == "perturbation"
            else config.get("num_classes")
        ),
        "perturbation_classes": (
            config.get("num_perturbations")
            if config.get("model_type") == "perturbation"
            else None
        ),
    }
    for key, expected in expected_mappings.items():
        if expected is None:
            continue
        classes = output_metadata.get(key)
        if classes is None:
            warnings.warn(
                f"legacy checkpoint requires {int(expected)} {key}, but their names "
                "were not stored; provide the mapping during migration before using "
                "the generate CLI",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if not isinstance(classes, (list, tuple)) or len(classes) != int(expected):
            observed = len(classes) if isinstance(classes, (list, tuple)) else "not a list"
            raise LegacyCheckpointError(
                f"{key} must contain exactly {int(expected)} names; observed {observed}"
            )
        output_metadata[key] = [str(item) for item in classes]
    epoch = info.get("epoch")
    return save_checkpoint(
        destination,
        model,
        standardizer=recovered,
        epoch=None if epoch is None else int(epoch),
        metadata=output_metadata,
    )
