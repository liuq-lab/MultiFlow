"""File-level workflow between raw paired H5MU profiles and MultiFlow latents.

MultiFlow itself is trained on latent states. This module provides the steps
around it, exactly as used in the paper: train the paired RNA/ATAC autoencoder
(and, for cell-state generation, the RNA VAE), encode raw profiles, build a
leave-one-cell-type-out fold, apply the training-only mean shift, and decode
generated latents back to RNA and ATAC profiles.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse

from .autoencoder import (
    DEFAULT_ATAC_DIMS,
    DEFAULT_RNA_DIMS,
    AutoencoderConfig,
    encode_latents,
    fit_autoencoder,
    latent_scale,
    load_autoencoder,
    save_autoencoder,
    training_summary,
)
from .checkpoint import load_checkpoint
from .h5mu import prepare_h5mu_for_write, sha256_file, validate_h5mu
from .metrics import log1p_cp10k
from .paper_encoders import (
    RNAExpressionVAE,
    load_rna_vae_checkpoint,
    load_scimilarity_initialization,
)
from .sampling import sample_paired_latents
from .training import seed_everything


def _paper_dependencies() -> tuple[Any, Any]:
    try:
        import anndata as ad
        import mudata as mu
    except ImportError as exc:
        raise RuntimeError(
            "Reading and writing H5MU files needs AnnData and MuData: "
            "`python -m pip install anndata mudata`."
        ) from exc
    return ad, mu


def _dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    result = np.asarray(matrix, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("expected a finite two-dimensional matrix")
    return result


@torch.no_grad()
def _encode_rna_vae(
    adata: Any,
    checkpoint: str | Path,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model = load_rna_vae_checkpoint(
        checkpoint,
        n_genes=adata.n_vars,
        device=device,
    )
    chunks: list[np.ndarray] = []
    for start in range(0, adata.n_obs, batch_size):
        stop = min(start + batch_size, adata.n_obs)
        x = torch.as_tensor(log1p_cp10k(adata.X[start:stop]), device=device)
        chunks.append(model(x, return_latent=True).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def split_h5mu(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    condition_key: str = "cell_type",
    train_fraction: float = 0.8,
    seed: int = 20260717,
) -> tuple[Path, Path]:
    """Write a condition-stratified ``train.h5mu`` / ``test.h5mu`` split."""
    _, mu = _paper_dependencies()
    import pandas as pd

    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    report = validate_h5mu(source, condition_key=condition_key)
    if not report["paired_raw_contract_ok"]:
        raise ValueError("input RNA must be raw counts and input ATAC must be binary")
    mdata = mu.read_h5mu(source)
    labels = mdata["rna"].obs[condition_key].astype(str).to_numpy()
    names = np.asarray(mdata["rna"].obs_names.astype(str))
    rng = np.random.default_rng(seed)
    records: list[tuple[str, str, str]] = []
    for name in sorted(set(labels.tolist())):
        cells = names[labels == name]
        if len(cells) < 2:
            raise ValueError(f"condition {name!r} needs at least two cells")
        cells = cells[rng.permutation(len(cells))]
        n_train = min(max(int(np.floor(len(cells) * train_fraction)), 1), len(cells) - 1)
        records.extend((cell, name, "train") for cell in cells[:n_train])
        records.extend((cell, name, "test") for cell in cells[n_train:])
    manifest = pd.DataFrame(records, columns=["obs_name", condition_key, "split"])
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "split_manifest.csv", index=False)
    paths: dict[str, Path] = {}
    for split_name in ("train", "test"):
        selected = manifest.loc[manifest["split"] == split_name, "obs_name"].tolist()
        subset = _subset_cells(mdata, selected)
        paths[split_name] = output / f"{split_name}.h5mu"
        prepare_h5mu_for_write(subset).write_h5mu(paths[split_name])
    metadata = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "train_fraction": float(train_fraction),
        "seed": int(seed),
        "condition_key": condition_key,
        "n_total": int(mdata.n_obs),
        "n_train": int((manifest["split"] == "train").sum()),
        "n_test": int((manifest["split"] == "test").sum()),
    }
    (output / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths["train"], paths["test"]


def train_autoencoder_h5mu(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    condition_key: str = "cell_type",
    epochs: int = 300,
    batch_size: int = 256,
    seed: int = 20260717,
    device: str | None = None,
    rna_dims: tuple[int, ...] = DEFAULT_RNA_DIMS,
    atac_dims: tuple[int, ...] = DEFAULT_ATAC_DIMS,
    verbose: bool = True,
) -> Path:
    """Train the paired RNA/ATAC autoencoder and save ``autoencoder.pt``."""
    _, mu = _paper_dependencies()
    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    checkpoint = output / "autoencoder.pt"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite {checkpoint}")
    report = validate_h5mu(source, condition_key=condition_key)
    if not report["paired_raw_contract_ok"]:
        raise ValueError("the autoencoder needs raw-count RNA and binary ATAC")
    mdata = mu.read_h5mu(source)
    config = AutoencoderConfig(
        epochs=int(epochs),
        batch_size=int(batch_size),
        seed=int(seed),
        rna_dims=tuple(int(value) for value in rna_dims),
        atac_dims=tuple(int(value) for value in atac_dims),
        **({} if device is None else {"device": device}),
    )
    result = fit_autoencoder(mdata["rna"].X, mdata["atac"].X, config=config, verbose=verbose)
    summary = {
        "schema_version": 1,
        "train_h5mu": str(source),
        "train_h5mu_sha256": sha256_file(source),
        "n_cells": int(mdata.n_obs),
        "n_genes": int(mdata["rna"].n_vars),
        "n_peaks": int(mdata["atac"].n_vars),
        **training_summary(result),
    }
    save_autoencoder(
        checkpoint,
        result.model,
        metadata={key: value for key, value in summary.items() if key != "history"},
    )
    summary["checkpoint"] = str(checkpoint)
    summary["checkpoint_sha256"] = sha256_file(checkpoint)
    (output / "autoencoder_training.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint


def train_rna_vae_h5mu(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    scimilarity_model_dir: str | Path | None = None,
    max_steps: int = 200_000,
    checkpoint_every: int = 50_000,
    batch_size: int = 128,
    seed: int = 1234,
    device: str | None = None,
    verbose: bool = True,
) -> Path:
    """Train the generation RNA VAE on ``log1p`` CP10k expression and save ``rna_vae.pt``.

    The paper starts the hidden layers from the SCimilarity annotation model
    (``scimilarity_model_dir``). Without it the network starts from random
    weights, which is a different setting from the paper.
    """
    _, mu = _paper_dependencies()
    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    final = output / "rna_vae.pt"
    if final.exists():
        raise FileExistsError(f"refusing to overwrite {final}")
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    matrix = mu.read_h5mu(source)["rna"].X
    n_cells, n_genes = (int(value) for value in matrix.shape)
    batch = min(int(batch_size), n_cells)
    if batch < 2:
        raise ValueError("the RNA VAE needs at least two cells")
    model = RNAExpressionVAE(n_genes=n_genes, latent_dim=128)
    if scimilarity_model_dir is not None:
        load_scimilarity_initialization(model, scimilarity_model_dir, device=target)
    model.to(target).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    order = np.empty(0, dtype=np.int64)
    position = 0
    for step in range(int(max_steps)):
        if position + batch > len(order):
            order, position = torch.randperm(n_cells, generator=generator).numpy(), 0
        index = np.sort(order[position : position + batch])
        position += batch
        genes = torch.as_tensor(log1p_cp10k(matrix[index]), device=target)
        loss = torch.nn.functional.mse_loss(model(genes), genes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if verbose and step % 1_000 == 0:
            print(f"rna vae step {step}/{max_steps}  mse {loss.item():.6f}", flush=True)
        if checkpoint_every > 0 and step > 0 and step % int(checkpoint_every) == 0:
            torch.save(model.state_dict(), output / f"rna_vae_step_{step:06d}.pt")
    torch.save(model.state_dict(), final)
    metadata = {
        "schema_version": 1,
        "train_h5mu": str(source),
        "train_h5mu_sha256": sha256_file(source),
        "n_cells": n_cells,
        "n_genes": n_genes,
        "rna_input_scale": "normalize_total(1e4)+log1p",
        "latent_dim": 128,
        "hidden_dims": [1024, 1024, 1024],
        "scimilarity_initialization": scimilarity_model_dir is not None,
        "optimizer": "AdamW",
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "max_steps": int(max_steps),
        "batch_size": batch,
        "seed": int(seed),
        "checkpoint": str(final),
        "checkpoint_sha256": sha256_file(final),
    }
    (output / "rna_vae_training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return final


def encode_paper_h5mu(
    *,
    input_path: str | Path,
    output_path: str | Path,
    task: str,
    autoencoder_checkpoint: str | Path,
    rna_vae_checkpoint: str | Path | None = None,
    condition_key: str = "cell_type",
    device: str = "cpu",
    batch_size: int = 2048,
) -> Path:
    """Encode raw paired profiles into ``X_multiflow`` latent states.

    ``task="generation"`` takes RNA from the RNA VAE and ATAC from the paired
    autoencoder; ``task="perturbation"`` takes both from the paired autoencoder.
    """
    _, mu = _paper_dependencies()
    if task not in {"generation", "perturbation"}:
        raise ValueError("task must be generation or perturbation")
    if task == "generation" and rna_vae_checkpoint is None:
        raise ValueError("generation encoding requires an RNA VAE checkpoint")

    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    ae_checkpoint = Path(autoencoder_checkpoint).expanduser().resolve()
    for path in (source, ae_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    mdata = mu.read_h5mu(source)
    if set(("rna", "atac")).difference(mdata.mod):
        raise ValueError("input H5MU must contain rna and atac modalities")
    if not np.array_equal(
        np.asarray(mdata["rna"].obs_names.astype(str)),
        np.asarray(mdata["atac"].obs_names.astype(str)),
    ):
        raise ValueError("RNA and ATAC cell order must match")
    if condition_key not in mdata["rna"].obs:
        raise KeyError(f"rna.obs[{condition_key!r}] is missing")

    model = load_autoencoder(ae_checkpoint, device=device)
    if (mdata["rna"].n_vars, mdata["atac"].n_vars) != (model.n_genes, model.n_peaks):
        raise ValueError(
            f"the autoencoder expects {model.n_genes} genes and {model.n_peaks} peaks; "
            f"the input has {mdata['rna'].n_vars} and {mdata['atac'].n_vars}"
        )
    z_rna, z_atac = encode_latents(
        model, mdata["rna"].X, mdata["atac"].X, batch_size=min(batch_size, 1000), device=device
    )
    # Latents are divided by ten times their mean per-dimension standard deviation.
    rna_std10, atac_std10 = latent_scale(z_rna), latent_scale(z_atac)
    mm_rna = (z_rna / rna_std10).astype(np.float32)
    atac = (z_atac / atac_std10).astype(np.float32)
    if task == "generation":
        rna = _encode_rna_vae(
            mdata["rna"],
            Path(rna_vae_checkpoint),
            device=device,
            batch_size=batch_size,
        )
        rna_encoder = "RNA VAE; normalize_total(1e4)+log1p"
    else:
        rna = mm_rna
        rna_encoder = "paired autoencoder, RNA branch; raw counts"
    if rna.shape[0] != mdata.n_obs or atac.shape[0] != mdata.n_obs:
        raise ValueError("encoder output rows do not match the paired input cells")

    mdata["rna"].obsm["X_multiflow"] = rna
    mdata["atac"].obsm["X_multiflow"] = atac
    metadata = {
        "schema_version": 1,
        "task": task,
        "rna_encoder": rna_encoder,
        "atac_encoder": "paired autoencoder, ATAC branch; binary input",
        "rna_latent_dim": int(rna.shape[1]),
        "atac_latent_dim": int(atac.shape[1]),
        "rna_std10": rna_std10,
        "atac_std10": atac_std10,
        "autoencoder_sha256": sha256_file(ae_checkpoint),
        "rna_vae_sha256": (
            sha256_file(Path(rna_vae_checkpoint)) if rna_vae_checkpoint else None
        ),
        "condition_key": condition_key,
        "source_h5mu": str(source),
    }
    mdata.uns["multiflow_encoder"] = json.dumps(metadata, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_h5mu_for_write(mdata).write_h5mu(output)
    return output


def _subset_cells(mdata: Any, names: Any) -> Any:
    """Copy of the given cells of a MuData, keeping the global ``uns``.

    mudata before 0.3 does not carry ``uns`` into a view, so ``mdata[names, :].copy()``
    loses it there.
    """
    subset = mdata[names, :].copy()
    for key, value in mdata.uns.items():
        if key not in subset.uns:
            subset.uns[key] = copy.deepcopy(value)
    return subset


def prepare_perturbation_fold(
    *,
    input_path: str | Path,
    output_path: str | Path,
    held_out_cell_type: str,
    condition_key: str,
    perturbation_key: str,
    control_perturbation: str,
) -> Path:
    """Create the exact flow-level LOCO split used by the perturbation notebook."""
    _, mu = _paper_dependencies()
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    mdata = mu.read_h5mu(source)
    obs = mdata["rna"].obs
    for key in (condition_key, perturbation_key):
        if key not in obs:
            raise KeyError(f"rna.obs[{key!r}] is missing")
    for modality in ("rna", "atac"):
        if "X_multiflow" not in mdata[modality].obsm:
            raise KeyError(f"{modality}.obsm['X_multiflow'] is missing")

    cell_type = obs[condition_key].astype(str).to_numpy()
    perturbation = obs[perturbation_key].astype(str).to_numpy()
    if held_out_cell_type not in set(cell_type):
        raise ValueError(f"unknown held-out cell type: {held_out_cell_type!r}")
    holdout = (cell_type == held_out_cell_type) & (
        perturbation != control_perturbation
    )
    if not holdout.any():
        raise ValueError("the requested fold contains no held-out perturbed cells")
    train_names = np.asarray(mdata["rna"].obs_names.astype(str))[~holdout]
    train = _subset_cells(mdata, train_names)

    rna = _dense_float32(train["rna"].obsm["X_multiflow"])
    atac = _dense_float32(train["atac"].obsm["X_multiflow"])
    labels = train["rna"].obs[condition_key].astype(str).to_numpy()
    classes = sorted(set(labels.tolist()))
    context = np.stack(
        [np.concatenate((rna[labels == name], atac[labels == name]), axis=1).mean(0)
         for name in classes],
        axis=0,
    ).astype(np.float32)
    train.uns["multiflow_context_matrix"] = context
    train.uns["multiflow_context_classes"] = np.asarray(classes, dtype=object)
    train.uns["multiflow_fold"] = json.dumps(
        {
            "schema_version": 1,
            "protocol": "leave-one-cell-type-out_noncontrol",
            "held_out_cell_type": held_out_cell_type,
            "control_perturbation": control_perturbation,
            "n_input": int(mdata.n_obs),
            "n_train": int(train.n_obs),
            "n_held_out_perturbed": int(holdout.sum()),
            "context_source": "unstandardized paired encoder latents from flow training rows",
        },
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_h5mu_for_write(train).write_h5mu(output)
    return output


class _FoldLatents:
    """Latent states and labels of one training fold, read once."""

    def __init__(
        self,
        fold: Any,
        *,
        condition_key: str,
        perturbation_key: str,
        control_perturbation: str,
    ) -> None:
        obs = fold["rna"].obs
        for key in (condition_key, perturbation_key):
            if key not in obs:
                raise KeyError(f"rna.obs[{key!r}] is missing in the training fold")
        self.cell_types = obs[condition_key].astype(str).to_numpy()
        self.perturbations = obs[perturbation_key].astype(str).to_numpy()
        self.control = control_perturbation
        self.control_mask = self.perturbations == control_perturbation
        self.latent = {
            modality: _dense_float32(fold[modality].obsm["X_multiflow"])
            for modality in ("rna", "atac")
        }
        self.control_mean = {
            modality: (
                values[self.control_mask].mean(axis=0) if self.control_mask.any() else None
            )
            for modality, values in self.latent.items()
        }

    def masks(
        self, held_out_cell_type: str, perturbation: str, min_cells: int
    ) -> tuple[np.ndarray, np.ndarray]:
        target_mask = self.perturbations == perturbation
        heldout_control_mask = self.control_mask & (self.cell_types == held_out_cell_type)
        for name, mask in (
            ("training controls", self.control_mask),
            (f"training perturbation {perturbation!r}", target_mask),
            (f"held-out controls for {held_out_cell_type!r}", heldout_control_mask),
        ):
            if int(mask.sum()) < min_cells:
                raise ValueError(
                    f"{name} has {int(mask.sum())} cells; need at least {min_cells}"
                )
        return target_mask, heldout_control_mask

    def shift(
        self,
        generated: np.ndarray,
        *,
        modality: str,
        target_mask: np.ndarray,
        heldout_control_mask: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Move the generated group mean to control mean + alpha x training effect."""
        latent = self.latent[modality]
        training_delta = latent[target_mask].mean(axis=0) - self.control_mean[modality]
        target_mean = latent[heldout_control_mask].mean(axis=0) + float(alpha) * training_delta
        corrected = generated + (target_mean - generated.mean(axis=0))
        return corrected.astype(np.float32, copy=False)


def debias_perturbation_h5mu(
    *,
    generated_path: str | Path,
    training_fold_path: str | Path,
    output_path: str | Path,
    held_out_cell_type: str,
    perturbation: str,
    condition_key: str,
    perturbation_key: str,
    control_perturbation: str,
    alpha: float = 1.0,
    min_cells: int = 5,
) -> Path:
    """Apply the executed training-only perturbation mean-shift correction."""
    _, mu = _paper_dependencies()
    generated_file = Path(generated_path).expanduser().resolve()
    fold_file = Path(training_fold_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    generated = mu.read_h5mu(generated_file)
    fold = _FoldLatents(
        mu.read_h5mu(fold_file),
        condition_key=condition_key,
        perturbation_key=perturbation_key,
        control_perturbation=control_perturbation,
    )
    target_mask, heldout_control_mask = fold.masks(held_out_cell_type, perturbation, min_cells)
    for modality in ("rna", "atac"):
        generated[modality].X = fold.shift(
            _dense_float32(generated[modality].X),
            modality=modality,
            target_mask=target_mask,
            heldout_control_mask=heldout_control_mask,
            alpha=alpha,
        )

    generated.uns["multiflow_debias"] = json.dumps(
        {
            "schema_version": 1,
            "method": "training_only_perturbation_mean_shift",
            "held_out_cell_type": held_out_cell_type,
            "perturbation": perturbation,
            "control_perturbation": control_perturbation,
            "alpha": float(alpha),
            "training_fold": str(fold_file),
            "n_training_controls": int(fold.control_mask.sum()),
            "n_training_perturbation": int(target_mask.sum()),
            "n_heldout_controls": int(heldout_control_mask.sum()),
        },
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_h5mu_for_write(generated).write_h5mu(output)
    return output


def _encoder_metadata(mdata: Any) -> dict[str, Any]:
    raw = mdata.uns.get("multiflow_encoder")
    if raw is None:
        raise ValueError("reference H5MU has no multiflow_encoder metadata")
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _log_library_stats_by_class(
    rna: Any,
    *,
    condition_key: str,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD of the log library size per condition, without densifying profiles."""
    labels = rna.obs[condition_key].astype(str).to_numpy()
    matrix = rna.X
    row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    if row_sums.shape[0] != rna.n_obs or not np.isfinite(row_sums).all():
        raise ValueError("reference RNA library sizes are invalid")
    if np.any(row_sums <= 0):
        raise ValueError("reference RNA contains cells with zero library size")
    log_sizes = np.log(row_sums)
    means: list[float] = []
    standard_deviations: list[float] = []
    for name in classes:
        values = log_sizes[labels == name]
        if values.size == 0:
            raise ValueError(f"reference has no RNA cells for condition {name!r}")
        means.append(float(values.mean()))
        # Sample standard deviation (Bessel's correction), as in the original workflow.
        standard_deviations.append(
            float(values.std(ddof=1)) if values.size > 1 else float("nan")
        )
    return (
        np.asarray(means, dtype=np.float32),
        np.asarray(standard_deviations, dtype=np.float32),
    )


@torch.no_grad()
def _decode_latents(
    model: Any,
    rna_latent: np.ndarray,
    atac_latent: np.ndarray,
    labels: np.ndarray,
    *,
    task: str,
    rna_std10: float,
    atac_std10: float,
    rna_vae: Any = None,
    library_mean: np.ndarray | None = None,
    library_sd: np.ndarray | None = None,
    device: str = "cpu",
    batch_size: int = 2048,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode latent states to RNA and binary ATAC.

    Generation returns log-normalised expression. Perturbation returns counts drawn
    from the negative binomial of the autoencoder, so that predicted cells carry the
    same count noise as observed cells when the two are normalised and compared.
    """
    from torch.distributions import Bernoulli, Normal

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rna_blocks: list[np.ndarray] = []
    atac_blocks: list[np.ndarray] = []
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        z_atac = torch.as_tensor(
            atac_latent[start:stop] * atac_std10,
            dtype=torch.float32,
            device=device,
        )
        atac_probability = model.decode_atac(z_atac)
        atac_blocks.append(Bernoulli(probs=atac_probability).sample().cpu().numpy())

        z_rna = torch.as_tensor(rna_latent[start:stop], dtype=torch.float32, device=device)
        if task == "generation":
            rna_out = rna_vae(z_rna, return_decoded=True)
        else:
            z_rna = z_rna * rna_std10
            batch_labels = labels[start:stop]
            assert library_mean is not None and library_sd is not None
            mean = torch.as_tensor(library_mean)[batch_labels].to(device)
            sd = torch.nan_to_num(
                torch.as_tensor(library_sd)[batch_labels].to(device),
                nan=1e-4,
                posinf=1e-4,
                neginf=1e-4,
            ).clamp_min(1e-4)
            library_size = torch.exp(Normal(mean, sd).sample()).view(-1, 1)
            rna_out = model.sample_rna(z_rna, library_size)
        rna_blocks.append(rna_out.cpu().numpy())
    return (
        np.concatenate(rna_blocks).astype(np.float32, copy=False),
        np.concatenate(atac_blocks).astype(np.float32, copy=False),
    )


def _profile_mudata(
    rna_matrix: np.ndarray,
    atac_matrix: np.ndarray,
    obs: Any,
    reference: Any,
    *,
    task: str,
) -> Any:
    ad, mu = _paper_dependencies()
    rna = ad.AnnData(X=rna_matrix, obs=obs.copy(), var=reference["rna"].var.copy())
    # binary accessibility is mostly zeros: store it sparse
    atac = ad.AnnData(
        X=sparse.csr_matrix(atac_matrix), obs=obs.copy(), var=reference["atac"].var.copy()
    )
    rna.uns["matrix_scale"] = (
        "log1p_normalized_1e4_reconstruction"
        if task == "generation"
        else "raw_count_sample"
    )
    atac.uns["matrix_scale"] = "binary"
    return mu.MuData({"rna": rna, "atac": atac})


def decode_paper_h5mu(
    *,
    generated_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    task: str,
    autoencoder_checkpoint: str | Path,
    rna_vae_checkpoint: str | Path | None = None,
    condition_key: str = "cell_type",
    device: str = "cpu",
    batch_size: int = 2048,
    seed: int = 0,
) -> Path:
    """Decode generated latents with the decoders that match their encoders."""
    _, mu = _paper_dependencies()
    if task not in {"generation", "perturbation"}:
        raise ValueError("task must be generation or perturbation")
    generated_file = Path(generated_path).expanduser().resolve()
    reference_file = Path(reference_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    checkpoint = Path(autoencoder_checkpoint).expanduser().resolve()
    generated = mu.read_h5mu(generated_file)
    rna_latent = _dense_float32(generated["rna"].X)
    atac_latent = _dense_float32(generated["atac"].X)
    model = load_autoencoder(checkpoint, device=device)
    reference = mu.read_h5mu(reference_file)
    if (reference["rna"].n_vars, reference["atac"].n_vars) != (model.n_genes, model.n_peaks):
        raise ValueError("the reference features do not match the autoencoder checkpoint")
    metadata = _encoder_metadata(reference)
    classes = sorted(reference["rna"].obs[condition_key].astype(str).unique())
    generated_labels = generated["rna"].obs[condition_key].astype(str).to_numpy()
    mapping = {name: index for index, name in enumerate(classes)}
    try:
        labels = np.asarray([mapping[name] for name in generated_labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"generated cell type is absent from reference: {exc.args[0]}") from exc

    rna_vae = None
    library_mean: np.ndarray | None = None
    library_sd: np.ndarray | None = None
    if task == "generation":
        if rna_vae_checkpoint is None:
            raise ValueError("generation decoding requires an RNA VAE checkpoint")
        rna_vae = load_rna_vae_checkpoint(
            rna_vae_checkpoint,
            n_genes=reference["rna"].n_vars,
            device=device,
        )
    else:
        library_mean, library_sd = _log_library_stats_by_class(
            reference["rna"],
            condition_key=condition_key,
            classes=classes,
        )
    rna_matrix, atac_matrix = _decode_latents(
        model,
        rna_latent,
        atac_latent,
        labels,
        task=task,
        rna_std10=float(metadata["rna_std10"]),
        atac_std10=float(metadata["atac_std10"]),
        rna_vae=rna_vae,
        library_mean=library_mean,
        library_sd=library_sd,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    decoded = _profile_mudata(
        rna_matrix, atac_matrix, generated["rna"].obs, reference, task=task
    )
    decoded.uns["multiflow_decode"] = json.dumps(
        {
            "schema_version": 1,
            "task": task,
            "source_generated_h5mu": str(generated_file),
            "reference_h5mu": str(reference_file),
            "autoencoder_sha256": sha256_file(checkpoint),
            "rna_vae_sha256": (
                sha256_file(Path(rna_vae_checkpoint)) if rna_vae_checkpoint else None
            ),
            "seed": int(seed),
        },
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_h5mu_for_write(decoded).write_h5mu(output)
    return output


def predict_perturbations_h5mu(
    *,
    run: str | Path,
    training_fold_path: str | Path,
    output_path: str | Path,
    held_out_cell_type: str,
    autoencoder_checkpoint: str | Path,
    n_cells: int | dict[str, int],
    perturbations: list[str] | None = None,
    condition_key: str = "cell_type",
    perturbation_key: str = "perturbation",
    control_perturbation: str = "control",
    debias: bool = True,
    alpha: float = 1.0,
    min_cells: int = 5,
    steps: int | None = None,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
) -> Path:
    """Predict RNA and ATAC profiles of a held-out cell type under several perturbations.

    One call replaces ``generate`` + ``debias_perturbation`` + ``decode`` per
    perturbation: the trained flow and the training fold are read once, and all
    predicted cells are written to one H5MU with ``cell_type`` and
    ``perturbation`` columns. Perturbation ``i`` is sampled with ``seed + i``.
    """
    _, mu = _paper_dependencies()
    import pandas as pd

    source = Path(run).expanduser().resolve()
    flow_checkpoint = source / "model.pt" if source.is_dir() else source
    fold_file = Path(training_fold_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    ae_checkpoint = Path(autoencoder_checkpoint).expanduser().resolve()

    flow, standardizer, payload = load_checkpoint(flow_checkpoint, map_location=device)
    flow_metadata: dict[str, Any] = payload.get("metadata", {})
    label_classes = list(flow_metadata.get("label_classes", []))
    perturbation_classes = list(flow_metadata.get("perturbation_classes", []))
    if flow.get_config().get("model_type") != "perturbation":
        raise ValueError("predict_perturbations needs a run trained with model='perturbation'")
    if held_out_cell_type not in label_classes:
        raise ValueError(f"unknown cell_type {held_out_cell_type!r}; choose from {label_classes}")
    if isinstance(n_cells, dict):
        counts = {str(name): int(value) for name, value in n_cells.items()}
        names = list(counts) if perturbations is None else [str(name) for name in perturbations]
        missing = [name for name in names if name not in counts]
        if missing:
            raise ValueError(f"n_cells has no entry for {missing}")
    else:
        names = (
            [name for name in perturbation_classes if name != control_perturbation]
            if perturbations is None
            else [str(name) for name in perturbations]
        )
        counts = {name: int(n_cells) for name in names}
    unknown = [name for name in names if name not in perturbation_classes]
    if unknown:
        raise ValueError(f"unknown perturbation {unknown}; choose from {perturbation_classes}")
    if not names or any(counts[name] < 1 for name in names):
        raise ValueError("every requested perturbation needs at least one cell")
    ode_steps = (
        int(flow_metadata.get("sampling", {}).get("default_steps", 50))
        if steps is None
        else int(steps)
    )

    reference = mu.read_h5mu(fold_file)
    model = load_autoencoder(ae_checkpoint, device=device)
    if (reference["rna"].n_vars, reference["atac"].n_vars) != (model.n_genes, model.n_peaks):
        raise ValueError("the training fold features do not match the autoencoder checkpoint")
    encoder_metadata = _encoder_metadata(reference)
    fold = _FoldLatents(
        reference,
        condition_key=condition_key,
        perturbation_key=perturbation_key,
        control_perturbation=control_perturbation,
    )
    rna_blocks: list[np.ndarray] = []
    atac_blocks: list[np.ndarray] = []
    for index, name in enumerate(names):
        rna, atac = sample_paired_latents(
            flow,
            counts[name],
            label=label_classes.index(held_out_cell_type),
            perturbation=perturbation_classes.index(name),
            steps=ode_steps,
            batch_size=batch_size,
            device=device,
            seed=seed + index,
            standardizer=standardizer,
        )
        latents = {"rna": rna.numpy(), "atac": atac.numpy()}
        if debias and name != control_perturbation:
            target_mask, heldout_control_mask = fold.masks(held_out_cell_type, name, min_cells)
            latents = {
                modality: fold.shift(
                    values.astype(np.float32, copy=False),
                    modality=modality,
                    target_mask=target_mask,
                    heldout_control_mask=heldout_control_mask,
                    alpha=alpha,
                )
                for modality, values in latents.items()
            }
        rna_blocks.append(latents["rna"])
        atac_blocks.append(latents["atac"])

    classes = sorted(set(fold.cell_types.tolist()))
    library_mean, library_sd = _log_library_stats_by_class(
        reference["rna"], condition_key=condition_key, classes=classes
    )
    total = int(sum(counts[name] for name in names))
    labels = np.full(total, classes.index(held_out_cell_type), dtype=np.int64)
    rna_matrix, atac_matrix = _decode_latents(
        model,
        np.concatenate(rna_blocks),
        np.concatenate(atac_blocks),
        labels,
        task="perturbation",
        rna_std10=float(encoder_metadata["rna_std10"]),
        atac_std10=float(encoder_metadata["atac_std10"]),
        library_mean=library_mean,
        library_sd=library_sd,
        device=device,
        batch_size=2048,
        seed=seed,
    )
    obs = pd.DataFrame(
        {
            condition_key: np.asarray([held_out_cell_type] * total, dtype=object),
            perturbation_key: np.asarray(
                [name for name in names for _ in range(counts[name])], dtype=object
            ),
        },
        index=pd.Index(
            np.asarray([f"multiflow_{index:08d}" for index in range(total)], dtype=object),
            dtype=object,
        ),
    )
    predicted = _profile_mudata(rna_matrix, atac_matrix, obs, reference, task="perturbation")
    predicted.uns["multiflow_prediction"] = json.dumps(
        {
            "schema_version": 1,
            "held_out_cell_type": held_out_cell_type,
            "perturbations": names,
            "n_cells": {name: counts[name] for name in names},
            "debias": "training_only_perturbation_mean_shift" if debias else "none",
            "alpha": float(alpha),
            "ode_steps": ode_steps,
            "seed": int(seed),
            "training_fold": str(fold_file),
            "flow_checkpoint_sha256": sha256_file(flow_checkpoint),
            "autoencoder_sha256": sha256_file(ae_checkpoint),
        },
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_h5mu_for_write(predicted).write_h5mu(output)
    return output
