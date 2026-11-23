"""Evaluation metrics of the MultiFlow paper.

Two tables can be reproduced with this module::

    import multiflow as mf

    # cell-state generation: generated against observed profiles, per modality
    mf.metrics.evaluate_generation("test.h5mu", "generated_profiles.h5mu")

    # perturbation prediction: one row per perturbation of a held-out cell type
    mf.metrics.evaluate_perturbation("observed.h5mu", "predicted.h5mu", cell_type="Pro-E")

RNA is compared on the ``log1p`` of counts per 10,000; ATAC is compared as
binary accessibility. All functions take cells in rows and features in columns.
"""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse, stats
from scipy.sparse.linalg import svds

__all__ = [
    "evaluate_generation",
    "evaluate_perturbation",
    "generation_metrics",
    "knn_mixing",
    "log1p_cp10k",
    "multiscale_rbf_mmd",
    "pca_embedding",
    "pearson",
    "perturbation_metrics",
    "rank_genes_by_t_test",
    "rbf_mmd",
    "scib_available",
    "scib_ilisi",
    "spearman",
    "stable_seed",
]


# ----------------------------------------------------------------------------- basics
def log1p_cp10k(counts: Any) -> np.ndarray:
    """Dense ``log1p`` of counts scaled to 10,000 per cell (cells with no counts stay zero).

    Same result as ``scanpy.pp.normalize_total(target_sum=1e4)`` followed by
    ``scanpy.pp.log1p``.
    """
    dense = _dense(counts)
    if dense.ndim != 2 or not np.isfinite(dense).all():
        raise ValueError("expected a finite two-dimensional matrix")
    totals = dense.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(dense / (totals / np.float32(1e4)))


def _mean(matrix: Any) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0), dtype=np.float64).reshape(-1)


def _dense(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def pearson(x: Any, y: Any) -> float:
    """Pearson correlation; ``nan`` when either vector is constant or too short."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: Any, y: Any) -> float:
    """Spearman rank correlation; ``nan`` when either vector is constant or too short."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).correlation)


def stable_seed(*parts: Any) -> int:
    """A seed that depends only on the given names (same on every machine and run)."""
    text = "||".join(str(part) for part in parts)
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % (2**32 - 1)


def _subsample(matrix: np.ndarray, max_cells: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_cells is None or matrix.shape[0] <= max_cells:
        return matrix
    return matrix[rng.choice(matrix.shape[0], size=max_cells, replace=False)]


def _match_rows(matrix: np.ndarray, target: int, seed: int) -> np.ndarray:
    """Resample rows to ``target`` cells (without replacement when enough cells exist)."""
    if matrix.shape[0] == target or matrix.shape[0] == 0 or target <= 0:
        return matrix
    rng = np.random.default_rng(int(seed))
    index = rng.choice(matrix.shape[0], size=target, replace=matrix.shape[0] < target)
    return matrix[index]


# ----------------------------------------------------------------------------- population metrics
def pca_embedding(
    real: Any, generated: Any, *, n_components: int = 50, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Principal components of the two cell sets, fitted on both together.

    Returns the scores of the real and of the generated cells. The components
    are exact for small matrices and computed by a truncated SVD otherwise.
    """
    real_dense = _dense(real)
    split = real_dense.shape[0]
    combined = np.vstack([real_dense, _dense(generated)])
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    centered = combined - combined.mean(axis=0, dtype=np.float64).astype(np.float32)
    rank = min(centered.shape)
    k = min(int(n_components), rank - 1)
    if k < 1:
        scores = centered.astype(np.float64)
    elif rank <= 512 or k >= rank - 1:
        u, s, _ = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
        scores = u[:, :k] * s[:k]
    else:
        start = np.random.default_rng(int(seed)).normal(size=rank).astype(np.float32)
        u, s, _ = svds(centered, k=k, v0=start)
        order = np.argsort(-s)
        scores = (u[:, order] * s[order]).astype(np.float64)
    return scores[:split], scores[split:]


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    distances = (x * x).sum(axis=1)[:, None] + (y * y).sum(axis=1)[None, :] - 2.0 * (x @ y.T)
    return np.maximum(distances, 0.0)


def rbf_mmd(real: Any, generated: Any, *, unbiased: bool = True) -> float:
    """Squared maximum mean discrepancy with a Gaussian kernel.

    The kernel width is the median squared distance between all cells of both
    sets. ``unbiased=True`` leaves out each cell's distance to itself (used for
    perturbation prediction); ``unbiased=False`` keeps it (used for cell-state
    generation).
    """
    x = np.asarray(real, dtype=np.float64)
    y = np.asarray(generated, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    combined = np.vstack([x, y])
    distances = _squared_distances(combined, combined)
    upper = distances[np.triu_indices_from(distances, k=1)]
    upper = upper[upper > 0]
    sigma2 = float(np.median(upper)) if upper.size else float(max(combined.shape[1], 1))
    gamma = 1.0 / max(2.0 * sigma2, 1e-12)
    n = x.shape[0]
    kernel = np.exp(-gamma * distances)
    k_xx, k_yy, k_xy = kernel[:n, :n], kernel[n:, n:], kernel[:n, n:]
    if unbiased:
        m = y.shape[0]
        term_x = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        term_y = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    else:
        term_x, term_y = k_xx.mean(), k_yy.mean()
    return float(term_x + term_y - 2.0 * k_xy.mean())


def multiscale_rbf_mmd(
    real: Any, generated: Any, *, kernel_mul: float = 2.0, kernel_num: int = 5
) -> float:
    """Squared MMD with a sum of Gaussian kernels of several widths.

    This is the estimator of the scDiffusion-X evaluation: the middle width is
    the mean squared distance between all cells, the others are that width
    times ``kernel_mul`` to the power -2 ... 2, and the kernels are summed.
    """
    x = np.asarray(real, dtype=np.float64)
    y = np.asarray(generated, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    combined = np.vstack([x, y])
    total = combined.shape[0]
    distances = _squared_distances(combined, combined)
    np.fill_diagonal(distances, 0.0)
    bandwidth = distances.sum() / (total**2 - total)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return 0.0
    bandwidth /= kernel_mul ** (kernel_num // 2)
    kernel = np.zeros_like(distances)
    for index in range(kernel_num):
        kernel += np.exp(-distances / (bandwidth * kernel_mul**index))
    n = x.shape[0]
    return float(kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2.0 * kernel[:n, n:].mean())


def knn_mixing(real: Any, generated: Any, *, k: int = 90, chunk: int = 2048) -> float:
    """How well real and generated cells mix among each cell's ``k`` nearest neighbours.

    For every cell, the inverse Simpson index of the two sources among its
    neighbours is computed and rescaled to 0 (neighbours all from one source)
    to 1 (half from each); the mean over cells is returned. This is a
    dependency-free relative of iLISI, not the scIB value itself.
    """
    x = np.asarray(real, dtype=np.float64)
    y = np.asarray(generated, dtype=np.float64)
    combined = np.vstack([x, y])
    total = combined.shape[0]
    neighbours = min(int(k), total - 1)
    if neighbours < 2:
        return float("nan")
    is_real = np.arange(total) < x.shape[0]
    values = np.empty(total, dtype=np.float64)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        distances = _squared_distances(combined[start:stop], combined)
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = np.argpartition(distances, neighbours - 1, axis=1)[:, :neighbours]
        share = is_real[nearest].mean(axis=1)
        values[start:stop] = 1.0 / (share**2 + (1.0 - share) ** 2) - 1.0
    return float(values.mean())


def scib_available() -> bool:
    """Whether the optional packages for the scIB iLISI (``scib`` and ``scanpy``) are installed."""
    return all(importlib.util.find_spec(name) is not None for name in ("scib", "scanpy"))


def scib_ilisi(real: Any, generated: Any, *, n_neighbors: int = 10, k0: int = 90) -> float:
    """Graph iLISI of scIB between real and generated cells (0 = separate, 1 = mixed).

    Needs the optional packages ``scib`` and ``scanpy``. The two inputs are
    embeddings, for example the output of :func:`pca_embedding`. If scIB itself
    fails, a warning is issued and ``nan`` is returned.
    """
    if not scib_available():
        raise ImportError(
            "the scIB iLISI needs the optional packages scib and scanpy: "
            "`python -m pip install scib scanpy`"
        )
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    import scib

    embedding = np.vstack([np.asarray(real), np.asarray(generated)]).astype(np.float32)
    n_real = np.asarray(real).shape[0]
    if embedding.shape[0] < 3:
        return float("nan")
    adata = ad.AnnData(np.zeros((embedding.shape[0], 1), dtype=np.float32))
    adata.obsm["X_pca"] = embedding
    adata.obs["batch"] = pd.Categorical(
        ["real"] * n_real + ["generated"] * (embedding.shape[0] - n_real)
    )
    sc.pp.neighbors(
        adata, n_neighbors=min(int(n_neighbors), embedding.shape[0] - 1), use_rep="X_pca"
    )
    k0 = min(int(k0), embedding.shape[0] - 1)
    try:
        try:
            value = scib.me.ilisi_graph(
                adata, batch_key="batch", type_="knn", k0=k0, scale=True, n_cores=1
            )
        except TypeError:  # older scib releases have no k0 argument
            value = scib.me.ilisi_graph(adata, batch_key="batch", type_="knn", scale=True)
    except Exception as error:  # noqa: BLE001 - a failed optional metric must not stop a run
        warnings.warn(f"scIB iLISI failed and is reported as nan: {error!r}", stacklevel=2)
        return float("nan")
    return float(np.nanmean(np.asarray(value, dtype=np.float64)))


# ----------------------------------------------------------------------------- generation
def generation_metrics(
    real: Any,
    generated: Any,
    *,
    n_components: int = 50,
    max_cells: int = 1500,
    k: int = 90,
    ilisi: bool = False,
    seed: int = 0,
) -> dict[str, float]:
    """Compare generated with observed cells of one modality.

    ``real`` and ``generated`` are on the same scale (``log1p`` CP10k for RNA,
    binary for ATAC) and have the same features.

    Returns the Pearson and Spearman correlation of the mean profiles (``pcc``,
    ``scc``), two MMD estimates on the first ``n_components`` principal
    components with at most ``max_cells`` cells per set (``mmd``: one Gaussian
    kernel of median width; ``mmd_multiscale``: the five-kernel estimate of
    :func:`multiscale_rbf_mmd`) and the real/generated mixing of the nearest
    neighbours (``knn_mixing``). With ``ilisi=True`` the scIB iLISI is added
    (optional packages).
    """
    real_dense, generated_dense = _dense(real), _dense(generated)
    if real_dense.shape[1] != generated_dense.shape[1]:
        raise ValueError("real and generated cells need the same features")
    real_mean, generated_mean = _mean(real_dense), _mean(generated_dense)
    real_pcs, generated_pcs = pca_embedding(
        real_dense, generated_dense, n_components=n_components, seed=seed
    )
    rng = np.random.default_rng(int(seed))
    real_sample = _subsample(real_pcs, max_cells, rng)
    generated_sample = _subsample(generated_pcs, max_cells, rng)
    result = {
        "n_real": int(real_dense.shape[0]),
        "n_generated": int(generated_dense.shape[0]),
        "pcc": pearson(real_mean, generated_mean),
        "scc": spearman(real_mean, generated_mean),
        "mmd": rbf_mmd(real_sample, generated_sample, unbiased=False),
        "mmd_multiscale": multiscale_rbf_mmd(real_sample, generated_sample),
        "knn_mixing": knn_mixing(real_pcs, generated_pcs, k=k),
    }
    if ilisi:
        result["ilisi"] = scib_ilisi(real_pcs[:, :20], generated_pcs[:, :20])
    return result


# ----------------------------------------------------------------------------- perturbation
def rank_genes_by_t_test(group: Any, reference: Any) -> np.ndarray:
    """Gene indices ordered by decreasing absolute Welch t statistic.

    Same statistic as ``scanpy.tl.rank_genes_groups(method="t-test")`` with a
    reference group; genes without variation in both groups get a score of 0.
    """

    def moments(matrix: Any) -> tuple[np.ndarray, np.ndarray, int]:
        n = matrix.shape[0]
        mean = _mean(matrix)
        if sparse.issparse(matrix):
            mean_of_squares = _mean(matrix.multiply(matrix))
        else:
            dense = np.asarray(matrix)
            mean_of_squares = np.einsum("ij,ij->j", dense, dense, dtype=np.float64) / n
        if n < 2:
            return mean, np.full_like(mean, np.nan), n
        return mean, (mean_of_squares - mean**2) * (n / (n - 1)), n

    mean_a, var_a, n_a = moments(group)
    mean_b, var_b, n_b = moments(reference)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = (mean_a - mean_b) / np.sqrt(var_a / n_a + var_b / n_b)
    scores[~np.isfinite(scores)] = 0.0
    return np.argsort(-np.abs(scores), kind="stable")


def perturbation_metrics(
    real_control: Any,
    real_perturbed: Any,
    predicted: Any,
    *,
    top_k: int = 100,
    n_components: int = 20,
    max_cells: int = 2000,
    ilisi: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare predicted with observed perturbed cells of one cell type.

    All three matrices are RNA on the ``log1p`` CP10k scale with the same genes.
    The ``top_k`` genes are those with the largest absolute t statistic between
    the observed perturbed and the observed control cells.

    Returns, for these genes: ``pcc_delta`` (Pearson correlation between the
    predicted and the observed change of the mean, both taken from the observed
    control mean), ``delta_mse`` (mean squared difference of the two changes)
    and ``mmd`` (between predicted and observed perturbed cells on
    ``n_components`` principal components); ``pcc_delta_all`` and
    ``delta_mse_all`` use all genes. With ``ilisi=True`` the scIB iLISI is added.
    """
    n_real = real_perturbed.shape[0]
    ranked = rank_genes_by_t_test(real_perturbed, real_control)
    top = ranked[: min(int(top_k), ranked.size)]
    control_mean = _mean(real_control)
    real_mean = _mean(real_perturbed)
    predicted_dense = _match_rows(_dense(predicted), n_real, seed)
    predicted_mean = _mean(predicted_dense)
    real_delta = real_mean - control_mean
    predicted_delta = predicted_mean - control_mean

    real_top = _dense(real_perturbed[:, top])
    predicted_top = predicted_dense[:, top]
    rng = np.random.default_rng(int(seed))
    real_sample = _subsample(real_top, max_cells, rng)
    predicted_sample = _subsample(predicted_top, max_cells, rng)
    real_pcs, predicted_pcs = pca_embedding(
        real_sample, predicted_sample, n_components=n_components, seed=seed
    )
    result: dict[str, Any] = {
        "n_real": int(n_real),
        "n_predicted": int(predicted.shape[0]),
        "pcc_delta": pearson(predicted_delta[top], real_delta[top]),
        "delta_mse": float(np.mean((real_delta[top] - predicted_delta[top]) ** 2)),
        "mmd": rbf_mmd(real_pcs, predicted_pcs, unbiased=True),
        "pcc_delta_all": pearson(predicted_delta, real_delta),
        "delta_mse_all": float(np.mean((real_delta - predicted_delta) ** 2)),
        "top_genes": top,
    }
    if ilisi:
        result["ilisi"] = scib_ilisi(real_pcs, predicted_pcs)
    return result


# ----------------------------------------------------------------------------- file-level tables
@contextmanager
def _opened(source: Any) -> Iterator[Any]:
    """Yield a MuData or AnnData; a path is opened without loading the matrices.

    :func:`_rows` then reads only the cells that are needed, and the file is
    closed on exit.
    """
    if not isinstance(source, (str, Path)):
        yield source
        return
    import mudata as mu

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read_h5mu(path, backed="r")
    try:
        yield mdata
    finally:
        close = getattr(getattr(mdata, "file", None), "close", None)
        if callable(close):
            close()


def _modality(source: Any, modality: str) -> Any:
    """The AnnData of one modality of a MuData; an AnnData is returned as it is."""
    return source[modality] if hasattr(source, "mod") else source


def _rows(adata: Any, index: np.ndarray) -> Any:
    """The chosen cells as an in-memory AnnData."""
    subset = adata[np.sort(np.asarray(index))]
    return subset.to_memory() if getattr(adata, "isbacked", False) else subset


def _rna_log(adata: Any) -> np.ndarray:
    scale = str(adata.uns.get("matrix_scale", "raw_counts"))
    if scale.startswith("log1p"):
        return _dense(adata.X)
    return log1p_cp10k(adata.X)


def _matched_indices(
    real_labels: np.ndarray,
    generated_labels: np.ndarray,
    *,
    max_per_label: int,
    max_total: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The same number of real and generated cells for every shared label."""
    rng = np.random.default_rng(int(seed))
    shared = sorted(set(real_labels.tolist()) & set(generated_labels.tolist()))
    if not shared:
        raise ValueError("real and generated cells share no label")
    cap = min(int(max_per_label), max(1, int(max_total) // len(shared)))
    real_index, generated_index = [], []
    for name in shared:
        real_pool = np.flatnonzero(real_labels == name)
        generated_pool = np.flatnonzero(generated_labels == name)
        take = min(real_pool.size, generated_pool.size, cap)
        real_index.append(rng.choice(real_pool, size=take, replace=False))
        generated_index.append(rng.choice(generated_pool, size=take, replace=False))
    return np.sort(np.concatenate(real_index)), np.sort(np.concatenate(generated_index))


def evaluate_generation(
    real: Any,
    generated: Any,
    *,
    condition_key: str = "cell_type",
    max_per_label: int = 200,
    max_total: int = 4000,
    ilisi: bool = False,
    seed: int = 0,
) -> Any:
    """Generation metrics for RNA and ATAC, as in Fig. 2 of the paper.

    ``real`` holds observed raw profiles and ``generated`` the output of
    :func:`multiflow.decode`; each is a path to an H5MU file or a MuData.
    The same number of observed and generated cells is drawn for every cell
    type (at most ``max_per_label``, and ``max_total`` in total) before the
    metrics of :func:`generation_metrics` are computed. Returns a data frame
    with one row per modality.
    """
    import pandas as pd

    rows = []
    with _opened(real) as real_data, _opened(generated) as generated_data:
        for modality in ("rna", "atac"):
            real_adata = _modality(real_data, modality)
            generated_adata = _modality(generated_data, modality)
            real_index, generated_index = _matched_indices(
                real_adata.obs[condition_key].astype(str).to_numpy(),
                generated_adata.obs[condition_key].astype(str).to_numpy(),
                max_per_label=max_per_label,
                max_total=max_total,
                seed=seed,
            )
            real_subset = _rows(real_adata, real_index)
            generated_subset = _rows(generated_adata, generated_index)
            if modality == "rna":
                real_matrix = _rna_log(real_subset)
                generated_matrix = _rna_log(generated_subset)
            else:
                real_matrix = (_dense(real_subset.X) > 0).astype(np.float32)
                generated_matrix = (_dense(generated_subset.X) > 0).astype(np.float32)
            values = generation_metrics(real_matrix, generated_matrix, ilisi=ilisi, seed=seed)
            rows.append(
                {
                    "modality": modality,
                    "n_cell_types": int(real_subset.obs[condition_key].astype(str).nunique()),
                    **values,
                }
            )
    return pd.DataFrame(rows).set_index("modality")


def evaluate_perturbation(
    real: Any,
    predicted: Any,
    *,
    cell_type: str,
    condition_key: str = "cell_type",
    perturbation_key: str = "perturbation",
    control_perturbation: str = "control",
    min_cells: int = 5,
    top_k: int = 100,
    ilisi: bool = False,
) -> Any:
    """Perturbation metrics of one held-out cell type, as in Fig. 3 of the paper.

    ``real`` holds the observed raw RNA counts (path, MuData or the RNA
    AnnData) and ``predicted`` the output of
    :func:`multiflow.predict_perturbations`. For every perturbation with
    at least ``min_cells`` observed and predicted cells, the predicted cells
    are compared with the observed ones by :func:`perturbation_metrics`.
    Returns a data frame with one row per perturbation; the column ``top_genes``
    lists the ``top_k`` genes used.
    """
    import pandas as pd

    with _opened(real) as real_data, _opened(predicted) as predicted_data:
        real_rna = _modality(real_data, "rna")
        predicted_rna = _modality(predicted_data, "rna")
        in_type = (real_rna.obs[condition_key].astype(str) == cell_type).to_numpy()
        if not in_type.any():
            raise ValueError(f"no observed cells of cell type {cell_type!r}")
        if predicted_rna.n_vars != real_rna.n_vars:
            raise ValueError("observed and predicted RNA need the same genes")
        real_type = _rows(real_rna, np.flatnonzero(in_type))
        real_labels = real_type.obs[perturbation_key].astype(str).to_numpy()
        real_log = log1p_cp10k(real_type.X)
        control = real_labels == control_perturbation
        if int(control.sum()) < min_cells:
            raise ValueError(f"cell type {cell_type!r} has fewer than {min_cells} control cells")
        predicted_obs = predicted_rna.obs
        predicted_type = (predicted_obs[condition_key].astype(str) == cell_type).to_numpy()
        predicted_labels = predicted_obs[perturbation_key].astype(str).to_numpy()
        gene_names = np.asarray(real_rna.var_names.astype(str))

        rows = []
        for name in sorted(set(real_labels.tolist()) - {control_perturbation}):
            observed = real_labels == name
            chosen = np.flatnonzero(predicted_type & (predicted_labels == name))
            if int(observed.sum()) < min_cells or chosen.size < min_cells:
                continue
            values = perturbation_metrics(
                real_log[control],
                real_log[observed],
                _rna_log(_rows(predicted_rna, chosen)),
                top_k=top_k,
                ilisi=ilisi,
                seed=stable_seed(cell_type, name, "population"),
            )
            values["top_genes"] = ",".join(gene_names[values["top_genes"]].tolist())
            rows.append({"cell_type": cell_type, "perturbation": name, **values})
    return pd.DataFrame(rows)
