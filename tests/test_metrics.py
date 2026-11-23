import numpy as np
import pytest
from scipy import sparse

import multiflow as mf
from multiflow import metrics


def _cells(n: int, shift: float = 0.0, seed: int = 0, n_genes: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rate = np.linspace(0.5, 4.0, n_genes)[None, :] * np.exp(shift * np.linspace(-1, 1, n_genes))
    return metrics.log1p_cp10k(rng.poisson(rate, size=(n, n_genes)).astype(np.float32))


def test_correlations_handle_constant_vectors() -> None:
    assert np.isnan(metrics.pearson([1, 1, 1], [1, 2, 3]))
    assert np.isnan(metrics.spearman([1, 2, 3], [5, 5, 5]))
    assert metrics.pearson([1, 2, 3], [2, 4, 6.5]) > 0.99
    assert metrics.spearman([1, 2, 3], [9, 4, 1]) == pytest.approx(-1.0)


def test_stable_seed_depends_only_on_the_names() -> None:
    assert metrics.stable_seed("Pro-E", "GATA1", "population") == metrics.stable_seed(
        "Pro-E", "GATA1", "population"
    )
    assert metrics.stable_seed("Pro-E", "GATA1") != metrics.stable_seed("Pro-E", "TAL1")


def test_mmd_is_zero_for_one_population_and_grows_with_the_shift() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 5)), rng.normal(size=(300, 5))
    same = metrics.rbf_mmd(a, b)
    shifted = metrics.rbf_mmd(a, b + 1.0)
    far = metrics.rbf_mmd(a, b + 3.0)
    assert abs(same) < 0.01 < shifted < far
    assert metrics.rbf_mmd(a, a, unbiased=False) == pytest.approx(0.0, abs=1e-12)


def test_unbiased_mmd_matches_a_direct_calculation() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(9, 3)) + 0.5
    z = np.vstack([x, y])
    d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    gamma = 1.0 / (2.0 * np.median(d[np.triu_indices(16, k=1)]))
    k = np.exp(-gamma * d)
    expected = (
        sum(k[i, j] for i in range(7) for j in range(7) if i != j) / (7 * 6)
        + sum(k[i, j] for i in range(7, 16) for j in range(7, 16) if i != j) / (9 * 8)
        - 2 * k[:7, 7:].mean()
    )
    assert metrics.rbf_mmd(x, y) == pytest.approx(expected, rel=1e-9)


def test_knn_mixing_separates_mixed_from_disjoint_populations() -> None:
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(200, 4)), rng.normal(size=(200, 4))
    assert metrics.knn_mixing(a, b, k=30) > 0.85
    assert metrics.knn_mixing(a, b + 50.0, k=30) == pytest.approx(0.0)


def test_pca_embedding_matches_scikit_learn_distances() -> None:
    decomposition = pytest.importorskip("sklearn.decomposition")
    rng = np.random.default_rng(3)
    for shape in ((60, 30), (700, 600)):  # exact and truncated branch
        base = rng.normal(size=(shape[0], 8)) @ rng.normal(size=(8, shape[1]))
        data = (base + 0.01 * rng.normal(size=shape)).astype(np.float32)
        real, generated = data[: shape[0] // 2], data[shape[0] // 2 :]
        ours = np.vstack(metrics.pca_embedding(real, generated, n_components=5))
        reference = decomposition.PCA(n_components=5, svd_solver="full").fit_transform(data)
        ours_d = ((ours[:, None] - ours[None]) ** 2).sum(-1)
        reference_d = ((reference[:, None] - reference[None]) ** 2).sum(-1)
        assert np.allclose(ours_d, reference_d, rtol=1e-3, atol=1e-3 * reference_d.max())


def test_t_test_ranking_matches_scanpy() -> None:
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    import pandas as pd

    control, perturbed = _cells(80, seed=4), _cells(50, shift=0.6, seed=5)
    control[:, 7] = 0.0
    perturbed[:, 7] = 0.0  # a gene without variation
    adata = ad.AnnData(np.vstack([control, perturbed]))
    adata.obs["group"] = pd.Categorical(["control"] * 80 + ["KO"] * 50)
    adata.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(adata, "group", groups=["KO"], reference="control", method="t-test")
    result = adata.uns["rank_genes_groups"]
    scores = np.abs(np.asarray(result["scores"]["KO"], dtype=float))
    names = np.asarray(result["names"]["KO"]).astype(int)
    expected = names[np.argsort(-scores, kind="stable")]
    ours = metrics.rank_genes_by_t_test(perturbed, control)
    assert set(ours[:20]) == set(expected[:20])
    assert np.array_equal(ours[:10], expected[:10])
    sparse_rank = metrics.rank_genes_by_t_test(
        sparse.csr_matrix(perturbed), sparse.csr_matrix(control)
    )
    assert np.array_equal(sparse_rank[:20], ours[:20])


def test_perturbation_metrics_reward_the_right_prediction() -> None:
    control = _cells(200, seed=6)
    observed = _cells(150, shift=0.8, seed=7)
    good = _cells(150, shift=0.8, seed=8)
    wrong = _cells(150, shift=-0.8, seed=9)
    right = metrics.perturbation_metrics(control, observed, good, top_k=20, seed=0)
    bad = metrics.perturbation_metrics(control, observed, wrong, top_k=20, seed=0)
    assert right["pcc_delta"] > 0.9 > 0 > bad["pcc_delta"]
    assert right["delta_mse"] < bad["delta_mse"]
    assert right["mmd"] < bad["mmd"]
    assert len(right["top_genes"]) == 20 and right["n_real"] == 150
    # the definition: both changes are taken from the observed control mean
    top = right["top_genes"]
    expected = np.corrcoef(
        good.mean(0)[top] - control.mean(0)[top], observed.mean(0)[top] - control.mean(0)[top]
    )[0, 1]
    assert right["pcc_delta"] == pytest.approx(expected, rel=1e-6)


def test_generation_metrics_reward_matching_cells() -> None:
    real, same, other = _cells(300, seed=10), _cells(300, seed=11), _cells(300, shift=1.0, seed=12)
    good = metrics.generation_metrics(real, same, n_components=10, k=20)
    bad = metrics.generation_metrics(real, other, n_components=10, k=20)
    assert good["pcc"] > bad["pcc"] and 0.9 < good["scc"] <= 1.0
    assert good["mmd"] < bad["mmd"] and good["knn_mixing"] > bad["knn_mixing"]


def test_scib_ilisi_reports_the_missing_optional_packages() -> None:
    if metrics.scib_available():
        pytest.skip("scib is installed")
    with pytest.raises(ImportError, match="scib"):
        metrics.scib_ilisi(np.zeros((4, 2)), np.ones((4, 2)))


def test_file_level_tables(tmp_path) -> None:
    import anndata as ad
    import mudata as mu
    import pandas as pd

    rng = np.random.default_rng(13)

    def profiles(n, shift, perturbation, cell_type, expectation=False):
        rate = np.linspace(0.5, 4.0, 30)[None, :] * np.exp(shift * np.linspace(-1, 1, 30))
        counts = rng.poisson(rate, size=(n, 30)).astype(np.float32)
        obs = pd.DataFrame(
            {"cell_type": cell_type, "perturbation": perturbation},
            index=[f"{cell_type}_{perturbation}_{expectation}_{i}" for i in range(n)],
        )
        rna = ad.AnnData(sparse.csr_matrix(counts), obs=obs)
        peaks = sparse.csr_matrix((counts[:, :12] > 2).astype(np.float32))
        atac = ad.AnnData(peaks, obs=obs.copy())
        if expectation:
            rna.uns["matrix_scale"] = "raw_count_sample"
        return rna, atac

    def stack(parts):
        rna = ad.concat([p[0] for p in parts])
        atac = ad.concat([p[1] for p in parts])
        rna.uns.update(parts[0][0].uns)
        return mu.MuData({"rna": rna, "atac": atac})

    real = stack(
        [
            profiles(60, 0.0, "control", "A"),
            profiles(40, 0.7, "KO1", "A"),
            profiles(40, -0.7, "KO2", "A"),
            profiles(3, 0.2, "KO3", "A"),  # too few cells: skipped
            profiles(50, 0.0, "control", "B"),
        ]
    )
    predicted = stack(
        [profiles(40, 0.7, "KO1", "A", True), profiles(40, 0.7, "KO2", "A", True)]
    )
    table = metrics.evaluate_perturbation(real, predicted, cell_type="A", top_k=10)
    assert table["perturbation"].tolist() == ["KO1", "KO2"]
    assert table.loc[0, "pcc_delta"] > 0.8 and table.loc[1, "pcc_delta"] < 0
    assert len(table.loc[0, "top_genes"].split(",")) == 10
    with pytest.raises(ValueError, match="no observed cells"):
        metrics.evaluate_perturbation(real, predicted, cell_type="missing")

    real_path, predicted_path = tmp_path / "real.h5mu", tmp_path / "predicted.h5mu"
    real.write_h5mu(real_path)
    predicted.write_h5mu(predicted_path)
    from_files = metrics.evaluate_perturbation(real_path, predicted_path, cell_type="A", top_k=10)
    assert np.allclose(from_files["pcc_delta"], table["pcc_delta"])

    generated = stack([profiles(45, 0.0, "control", "A"), profiles(45, 0.0, "control", "B")])
    summary = metrics.evaluate_generation(real, generated, max_per_label=30)
    assert summary.index.tolist() == ["rna", "atac"]
    assert summary.loc["rna", "n_real"] == summary.loc["rna", "n_generated"] == 60
    assert summary.loc["rna", "n_cell_types"] == 2 and summary.loc["rna", "pcc"] > 0.9
    assert mf.metrics is metrics
