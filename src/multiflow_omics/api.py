"""High-level Python API: one function per step of a MultiFlow analysis.

Every function reads and writes files, returns the path it wrote, and records
its settings next to the result, so a notebook and the ``multiflow`` command
line produce the same artefacts::

    import multiflow_omics as mf

    ae = mf.train_autoencoder("train.h5mu", "runs/autoencoder")
    mf.encode("train.h5mu", "train_encoded.h5mu", task="perturbation", autoencoder=ae)
    run = mf.train("train_encoded.h5mu", "runs/multiflow")
    mf.generate(run, "generated.h5mu", n=1000, cell_type="CD14+ Mono")
"""

from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import paper as _paper
from ._version import __version__
from .checkpoint import load_checkpoint, save_checkpoint
from .h5mu import (
    DEFAULT_ATAC_REPRESENTATION,
    DEFAULT_RNA_REPRESENTATION,
    read_paired_latents,
    sha256_file,
    write_generated_h5mu,
)
from .models import CellStateFlow, ConditionalConcatFlow, PerturbationFlow
from .sampling import sample_paired_latents
from .training import TrainingConfig, fit, seed_everything

PathLike = str | Path


def _device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _empty_run_directory(path: PathLike, *, create: bool) -> Path:
    run = Path(path).expanduser().resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run}")
    if create:
        run.mkdir(parents=True, exist_ok=True)
    return run


def default_sampling_steps(model: str) -> int:
    """Midpoint ODE steps used in the paper: 50 for perturbation, 100 otherwise."""
    return 50 if model == "perturbation" else 100


# ----------------------------------------------------------------------------- data and encoders
def split_h5mu(
    input: PathLike,
    output_dir: PathLike,
    *,
    condition_key: str = "cell_type",
    train_fraction: float = 0.8,
    seed: int = 20260717,
) -> tuple[Path, Path]:
    """Split raw paired profiles into ``train.h5mu`` and ``test.h5mu``, stratified by condition."""
    return _paper.split_h5mu(
        input_path=input,
        output_dir=output_dir,
        condition_key=condition_key,
        train_fraction=train_fraction,
        seed=seed,
    )


def train_autoencoder(
    input: PathLike,
    output_dir: PathLike,
    *,
    condition_key: str = "cell_type",
    epochs: int = 300,
    batch_size: int = 256,
    rna_dims: tuple[int, ...] = (512, 300, 128),
    atac_dims: tuple[int, ...] = (1024, 512, 128),
    seed: int = 20260717,
    device: str | None = None,
    verbose: bool = True,
) -> Path:
    """Train the paired RNA/ATAC autoencoder on raw profiles; returns ``autoencoder.pt``.

    ``rna_dims`` and ``atac_dims`` list the hidden widths of each encoder and end
    with the latent width; the decoders mirror them. The defaults are the paper's.
    """
    return _paper.train_autoencoder_h5mu(
        input_path=input,
        output_dir=output_dir,
        condition_key=condition_key,
        epochs=epochs,
        batch_size=batch_size,
        rna_dims=tuple(rna_dims),
        atac_dims=tuple(atac_dims),
        seed=seed,
        device=device,
        verbose=verbose,
    )


def train_rna_vae(
    input: PathLike,
    output_dir: PathLike,
    *,
    scimilarity_model_dir: PathLike | None = None,
    max_steps: int = 200_000,
    checkpoint_every: int = 50_000,
    batch_size: int = 128,
    seed: int = 1234,
    device: str | None = None,
    verbose: bool = True,
) -> Path:
    """Train the RNA VAE used for cell-state generation; returns ``rna_vae.pt``."""
    return _paper.train_rna_vae_h5mu(
        input_path=input,
        output_dir=output_dir,
        scimilarity_model_dir=scimilarity_model_dir,
        max_steps=max_steps,
        checkpoint_every=checkpoint_every,
        batch_size=batch_size,
        seed=seed,
        device=device,
        verbose=verbose,
    )


def encode(
    input: PathLike,
    output: PathLike,
    *,
    task: str,
    autoencoder: PathLike,
    rna_vae: PathLike | None = None,
    condition_key: str = "cell_type",
    batch_size: int = 2048,
    device: str | None = None,
) -> Path:
    """Encode raw RNA/ATAC profiles into the latent states MultiFlow is trained on."""
    return _paper.encode_paper_h5mu(
        input_path=input,
        output_path=output,
        task=task,
        autoencoder_checkpoint=autoencoder,
        rna_vae_checkpoint=rna_vae,
        condition_key=condition_key,
        device=_device(device),
        batch_size=batch_size,
    )


def decode(
    input: PathLike,
    output: PathLike,
    *,
    task: str,
    reference: PathLike,
    autoencoder: PathLike,
    rna_vae: PathLike | None = None,
    condition_key: str = "cell_type",
    batch_size: int = 2048,
    seed: int = 0,
    device: str | None = None,
) -> Path:
    """Decode generated latent states into RNA and ATAC profiles."""
    return _paper.decode_paper_h5mu(
        generated_path=input,
        reference_path=reference,
        output_path=output,
        task=task,
        autoencoder_checkpoint=autoencoder,
        rna_vae_checkpoint=rna_vae,
        condition_key=condition_key,
        device=_device(device),
        batch_size=batch_size,
        seed=seed,
    )


def prepare_perturbation_fold(
    input: PathLike,
    output: PathLike,
    *,
    held_out_cell_type: str,
    condition_key: str = "cell_type",
    perturbation_key: str = "perturbation",
    control_perturbation: str = "control",
) -> Path:
    """Remove the perturbed cells of one cell type and build the context matrix."""
    return _paper.prepare_perturbation_fold(
        input_path=input,
        output_path=output,
        held_out_cell_type=held_out_cell_type,
        condition_key=condition_key,
        perturbation_key=perturbation_key,
        control_perturbation=control_perturbation,
    )


def debias_perturbation(
    input: PathLike,
    output: PathLike,
    *,
    training_fold: PathLike,
    held_out_cell_type: str,
    perturbation: str,
    condition_key: str = "cell_type",
    perturbation_key: str = "perturbation",
    control_perturbation: str = "control",
    alpha: float = 1.0,
    min_cells: int = 5,
) -> Path:
    """Apply the training-only latent mean shift to generated perturbed cells."""
    return _paper.debias_perturbation_h5mu(
        generated_path=input,
        training_fold_path=training_fold,
        output_path=output,
        held_out_cell_type=held_out_cell_type,
        perturbation=perturbation,
        condition_key=condition_key,
        perturbation_key=perturbation_key,
        control_perturbation=control_perturbation,
        alpha=alpha,
        min_cells=min_cells,
    )


def predict_perturbations(
    run: PathLike,
    output: PathLike,
    *,
    training_fold: PathLike,
    held_out_cell_type: str,
    autoencoder: PathLike,
    n_cells: int | Mapping[str, int],
    perturbations: Sequence[str] | None = None,
    condition_key: str = "cell_type",
    perturbation_key: str = "perturbation",
    control_perturbation: str = "control",
    debias: bool = True,
    alpha: float = 1.0,
    min_cells: int = 5,
    steps: int | None = None,
    batch_size: int = 256,
    seed: int = 0,
    device: str | None = None,
) -> Path:
    """Predict the profiles of a held-out cell type under several perturbations.

    Samples latent states from a trained perturbation run, applies the
    training-only mean shift (``debias=True``, the paper's protocol) and
    decodes RNA and ATAC, all in one pass. ``n_cells`` is one number for every
    perturbation or a mapping ``{perturbation: number of cells}``; with a
    mapping and no ``perturbations`` argument, its keys are predicted. The
    result is one H5MU whose ``obs`` has ``cell_type`` and ``perturbation``.
    """
    return _paper.predict_perturbations_h5mu(
        run=run,
        training_fold_path=training_fold,
        output_path=output,
        held_out_cell_type=held_out_cell_type,
        autoencoder_checkpoint=autoencoder,
        n_cells=dict(n_cells) if isinstance(n_cells, Mapping) else int(n_cells),
        perturbations=None if perturbations is None else list(perturbations),
        condition_key=condition_key,
        perturbation_key=perturbation_key,
        control_perturbation=control_perturbation,
        debias=debias,
        alpha=alpha,
        min_cells=min_cells,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        device=_device(device),
    )


# ----------------------------------------------------------------------------- MultiFlow
def train(
    input: PathLike,
    output: PathLike,
    *,
    model: str = "cell-state",
    condition_key: str | None = "cell_type",
    perturbation_key: str | None = None,
    control_perturbation: str = "control",
    epochs: int = 600,
    batch_size: int = 512,
    learning_rate: float = 1e-4,
    hidden_dim: int = 512,
    num_blocks: int = 2,
    num_layers: int = 8,
    cross_attention_dim: int = 64,
    sampling_steps: int | None = None,
    standardize: bool = True,
    seed: int = 0,
    device: str | None = None,
    hash_input: bool = False,
    rna_representation: str = DEFAULT_RNA_REPRESENTATION,
    atac_representation: str = DEFAULT_ATAC_REPRESENTATION,
    context_matrix_key: str = "multiflow_context_matrix",
    context_classes_key: str = "multiflow_context_classes",
    verbose: bool = True,
) -> Path:
    """Train MultiFlow on an encoded H5MU and save a run directory.

    ``model`` is ``"cell-state"`` (generation, conditioned on ``condition_key``),
    ``"perturbation"`` (needs ``perturbation_key`` and a fold written by
    :func:`prepare_perturbation_fold`) or ``"concat"`` (the ablation without
    cross-attention). The run directory holds ``model.pt``, ``history.csv`` and
    ``run.json``.
    """
    input_path = Path(input).expanduser().resolve()
    if input_path.suffix.lower() != ".h5mu":
        raise ValueError("MultiFlow training input must be a paired .h5mu file")
    if model not in {"cell-state", "perturbation", "concat"}:
        raise ValueError("model must be cell-state, perturbation or concat")
    _empty_run_directory(output, create=False)
    if model == "perturbation" and perturbation_key is None:
        raise ValueError("model='perturbation' requires perturbation_key")
    if model != "perturbation" and perturbation_key is not None:
        raise ValueError("perturbation_key is accepted only by model='perturbation'")
    data = read_paired_latents(
        input_path,
        rna_representation=rna_representation,
        atac_representation=atac_representation,
        condition_key=condition_key,
        perturbation_key=perturbation_key,
        context_matrix_key=context_matrix_key,
        context_classes_key=context_classes_key,
        control_perturbation=control_perturbation,
    )
    target = _device(device)
    seed_everything(seed)
    num_classes = None if data.labels is None else len(data.label_classes)
    network: torch.nn.Module
    if model == "cell-state":
        network = CellStateFlow(
            data.rna.shape[1],
            data.atac.shape[1],
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_classes=num_classes,
            cross_attention_dim=cross_attention_dim,
        )
    elif model == "perturbation":
        if data.context_matrix is None:
            raise ValueError("perturbation training requires a context matrix")
        network = PerturbationFlow(
            data.rna.shape[1],
            data.atac.shape[1],
            torch.as_tensor(data.context_matrix),
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            cross_attention_dim=cross_attention_dim,
            num_perturbations=len(data.perturbation_classes),
        )
    else:
        network = ConditionalConcatFlow(
            data.rna.shape[1],
            data.atac.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=target,
        seed=seed,
        standardize=standardize,
        reseed=False,
    )
    result = fit(
        network,
        data.rna,
        data.atac,
        labels=data.labels,
        perturbations=data.perturbations,
        config=config,
        verbose=verbose,
    )
    run = _empty_run_directory(output, create=True)
    input_metadata: dict[str, object] = {
        "name": input_path.name,
        "size_bytes": input_path.stat().st_size,
    }
    if hash_input:
        input_metadata["sha256"] = sha256_file(input_path)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "input": input_metadata,
        "h5mu_contract": {
            "rna_representation": rna_representation,
            "atac_representation": atac_representation,
            "condition_key": condition_key,
            "perturbation_key": perturbation_key,
            "control_perturbation": (
                control_perturbation if perturbation_key is not None else None
            ),
            "paired_cells": int(data.rna.shape[0]),
        },
        "label_classes": data.label_classes,
        "perturbation_classes": data.perturbation_classes,
        "training": asdict(config),
        "sampling": {
            "default_steps": (
                int(sampling_steps) if sampling_steps is not None else default_sampling_steps(model)
            )
        },
    }
    checkpoint = save_checkpoint(
        run / "model.pt",
        result.model,
        standardizer=result.standardizer,
        optimizer=result.optimizer,
        epoch=epochs,
        history=result.history,
        metadata=metadata,
    )
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.history[0].keys())
        writer.writeheader()
        writer.writerows(result.history)
    manifest = {
        "package_version": __version__,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_config": result.model.get_config(),
        **metadata,
    }
    (run / "run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run


def _named_condition(
    value: str | None, classes: list[str], *, name: str
) -> tuple[int | None, str | None]:
    if not classes:
        if value is not None:
            raise ValueError(f"{name} was provided to a run trained without it")
        return None, None
    if value is None:
        raise ValueError(f"{name} is required; choose from {classes}")
    if value not in classes:
        raise ValueError(f"unknown {name} {value!r}; choose from {classes}")
    return classes.index(value), value


def generate(
    run: PathLike,
    output: PathLike,
    *,
    n: int | Mapping[str, int],
    cell_type: str | None = None,
    perturbation: str | None = None,
    steps: int | None = None,
    batch_size: int = 512,
    seed: int = 0,
    device: str | None = None,
    keep_standardized: bool = False,
    rng_mode: str = "joint",
) -> Path:
    """Sample paired latent states from a trained run and write them as H5MU.

    ``n`` is the number of cells for ``cell_type``, or a mapping
    ``{cell type: number of cells}`` to sample several cell types into one
    file (cell type ``i`` of the mapping is sampled with ``seed + i``).
    """
    source = Path(run).expanduser().resolve()
    checkpoint = source / "model.pt" if source.is_dir() else source
    target = _device(device)
    model, standardizer, payload = load_checkpoint(checkpoint, map_location=target)
    metadata: dict[str, Any] = payload.get("metadata", {})
    label_classes = list(metadata.get("label_classes", []))
    perturbation_classes = list(metadata.get("perturbation_classes", []))
    model_config = model.get_config()
    is_perturbation = model_config.get("model_type") == "perturbation"
    expected_labels = model_config.get("num_contexts" if is_perturbation else "num_classes")
    expected_perturbations = model_config.get("num_perturbations") if is_perturbation else None
    for mapping_name, classes, expected in (
        ("label_classes", label_classes, expected_labels),
        ("perturbation_classes", perturbation_classes, expected_perturbations),
    ):
        if expected is None or (mapping_name == "perturbation_classes" and int(expected) == 1):
            continue
        if len(classes) != int(expected):
            raise ValueError(
                f"checkpoint requires {int(expected)} {mapping_name}, but contains "
                f"{len(classes)}; remigrate the checkpoint with --{mapping_name.replace('_', '-')}"
            )
    if isinstance(n, Mapping):
        if cell_type is not None:
            raise ValueError("give either one cell_type with a number, or a mapping as n")
        requests = [(str(name), int(count)) for name, count in n.items()]
        if not requests:
            raise ValueError("the mapping n is empty")
    else:
        requests = [(cell_type, int(n))]
    perturbation_code, perturbation_name = _named_condition(
        perturbation, perturbation_classes, name="perturbation"
    )
    fallback = default_sampling_steps(str(model_config.get("model_type")))
    ode_steps = (
        int(metadata.get("sampling", {}).get("default_steps", fallback))
        if steps is None
        else int(steps)
    )
    standardizer_present = standardizer is not None
    invert = standardizer_present and not keep_standardized
    if not standardizer_present and "legacy_migration" in metadata:
        warnings.warn(
            "legacy checkpoint has no latent standardizer; generated values remain "
            "in the model's recorded output space",
            RuntimeWarning,
            stacklevel=2,
        )
    rna_blocks, atac_blocks = [], []
    label_names: list[str | None] = []
    label_codes: list[int | None] = []
    for index, (name, count) in enumerate(requests):
        label_code, label_name = _named_condition(name, label_classes, name="cell_type")
        rna, atac = sample_paired_latents(
            model,
            count,
            label=label_code,
            perturbation=perturbation_code,
            steps=ode_steps,
            batch_size=batch_size,
            device=target,
            seed=seed + index,
            standardizer=standardizer if invert else None,
            rng_mode=rng_mode,  # type: ignore[arg-type]
        )
        rna_blocks.append(rna.numpy())
        atac_blocks.append(atac.numpy())
        label_names.extend([label_name] * count)
        label_codes.extend([label_code] * count)
    total = len(label_names)
    labelled = label_names[0] is not None
    return write_generated_h5mu(
        output,
        np.concatenate(rna_blocks),
        np.concatenate(atac_blocks),
        cell_type=label_names if labelled else None,
        cell_type_code=label_codes if labelled else None,
        perturbation=None if perturbation_name is None else [perturbation_name] * total,
        perturbation_code=None if perturbation_code is None else [perturbation_code] * total,
        metadata={
            "schema_version": 1,
            "package_version": __version__,
            "output_space": "latent",
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_config": model_config,
            "seed": int(seed),
            "ode_steps": ode_steps,
            "batch_size": int(batch_size),
            "rng_mode": rng_mode,
            "standardizer_present": standardizer_present,
            "standardization_inverted": invert,
            "label_classes": label_classes,
            "perturbation_classes": perturbation_classes,
        },
    )
