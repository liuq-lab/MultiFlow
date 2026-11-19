from __future__ import annotations

import json

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from multiflow_omics import cli
from multiflow_omics.h5mu import prepare_h5mu_for_write
from multiflow_omics.paper import (
    _log_library_stats_by_class,
    debias_perturbation_h5mu,
    prepare_perturbation_fold,
)
from multiflow_omics.paper_encoders import RNAExpressionVAE


def test_cli_routes_perturbation_debias_to_paper_workflow(
    monkeypatch, tmp_path, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_debias(**kwargs):
        captured.update(kwargs)
        return tmp_path / "corrected.h5mu"

    monkeypatch.setattr(cli, "debias_perturbation_h5mu", fake_debias)
    cli.main(
        [
            "paper",
            "debias-perturbation",
            "--input",
            "generated.h5mu",
            "--training-fold",
            "fold.h5mu",
            "--output",
            "corrected.h5mu",
            "--held-out-cell-type",
            "A",
            "--perturbation",
            "p1",
            "--min-cells",
            "2",
        ]
    )
    assert captured["held_out_cell_type"] == "A"
    assert captured["perturbation"] == "p1"
    assert captured["min_cells"] == 2
    assert "saved debiased perturbation latents" in capsys.readouterr().out


def test_cli_routes_decode_to_paper_workflow(monkeypatch, tmp_path, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        return tmp_path / "profiles.h5mu"

    monkeypatch.setattr(cli, "decode_paper_h5mu", fake_decode)
    cli.main(
        [
            "paper",
            "decode",
            "--task",
            "generation",
            "--input",
            "generated.h5mu",
            "--reference",
            "encoded.h5mu",
            "--output",
            "profiles.h5mu",
            "--autoencoder",
            "autoencoder.pt",
            "--rna-vae",
            "rna.pt",
        ]
    )
    assert captured["task"] == "generation"
    assert captured["autoencoder_checkpoint"] == "autoencoder.pt"
    assert captured["rna_vae_checkpoint"] == "rna.pt"
    assert captured["generated_path"] == "generated.h5mu"
    assert captured["reference_path"] == "encoded.h5mu"
    assert "saved decoded RNA/ATAC profiles" in capsys.readouterr().out


def test_sparse_library_statistics_use_sample_standard_deviation() -> None:
    obs = pd.DataFrame(
        {"cell_type": ["B", "B", "T", "T"]},
        index=pd.Index(np.asarray(["a", "b", "c", "d"], dtype=object)),
    )
    rna = ad.AnnData(
        sparse.csr_matrix(
            np.asarray(
                [[1, 1, 0], [2, 2, 0], [1, 2, 5], [2, 4, 10]],
                dtype=np.float32,
            )
        ),
        obs=obs,
    )
    mean, sd = _log_library_stats_by_class(
        rna,
        condition_key="cell_type",
        classes=["B", "T"],
    )
    expected = [np.log([2.0, 4.0]), np.log([8.0, 16.0])]
    assert np.allclose(mean, [values.mean() for values in expected])
    assert np.allclose(sd, [values.std(ddof=1) for values in expected])


def test_rna_vae_matches_paper_dimensions_and_unit_latent() -> None:
    model = RNAExpressionVAE(n_genes=7, latent_dim=128).eval()
    x = torch.rand(4, 7)
    with torch.no_grad():
        latent = model(x, return_latent=True)
        decoded = model(latent, return_decoded=True)
    assert latent.shape == (4, 128)
    assert decoded.shape == (4, 7)
    assert torch.allclose(torch.linalg.vector_norm(latent, dim=1), torch.ones(4))
    assert torch.all(decoded >= 0)


def test_prepare_perturbation_fold_matches_notebook_split(tmp_path) -> None:
    names = pd.Index(np.asarray(["a", "b", "c", "d", "e", "f"], dtype=object))
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "A", "A", "B", "B", "B"],
            "perturbation": ["control", "p1", "p2", "control", "p1", "p2"],
        },
        index=names,
    )
    rna = ad.AnnData(np.ones((6, 3), dtype=np.float32), obs=obs.copy())
    atac = ad.AnnData(np.ones((6, 2), dtype=np.float32), obs=obs.copy())
    rna.obsm["X_multiflow"] = np.arange(12, dtype=np.float32).reshape(6, 2)
    atac.obsm["X_multiflow"] = np.arange(6, dtype=np.float32).reshape(6, 1)
    source = tmp_path / "encoded.h5mu"
    prepare_h5mu_for_write(mu.MuData({"rna": rna, "atac": atac})).write_h5mu(source)
    output = tmp_path / "fold.h5mu"

    prepare_perturbation_fold(
        input_path=source,
        output_path=output,
        held_out_cell_type="A",
        condition_key="cell_type",
        perturbation_key="perturbation",
        control_perturbation="control",
    )
    fold = mu.read_h5mu(output)
    assert list(fold["rna"].obs_names.astype(str)) == ["a", "d", "e", "f"]
    assert list(map(str, fold.uns["multiflow_context_classes"])) == ["A", "B"]
    expected_a = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    expected_b = np.array([8.0, 9.0, 4.0], dtype=np.float32)
    assert np.allclose(fold.uns["multiflow_context_matrix"], [expected_a, expected_b])
    metadata = json.loads(fold.uns["multiflow_fold"])
    assert metadata["n_held_out_perturbed"] == 2
    assert metadata["n_train"] == 4


def test_perturbation_debias_matches_training_only_mean_shift(tmp_path) -> None:
    names = pd.Index(np.asarray(["a", "b", "c", "d", "e", "f"], dtype=object))
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B", "B", "B"],
            "perturbation": ["control", "control", "control", "p1", "control", "p1"],
        },
        index=names,
    )
    rna = ad.AnnData(np.ones((6, 1), dtype=np.float32), obs=obs.copy())
    atac = ad.AnnData(np.ones((6, 1), dtype=np.float32), obs=obs.copy())
    rna.obsm["X_multiflow"] = np.array([[1], [2], [3], [7], [5], [9]], dtype=np.float32)
    atac.obsm["X_multiflow"] = np.array([[2], [3], [4], [10], [6], [12]], dtype=np.float32)
    fold_path = tmp_path / "fold.h5mu"
    prepare_h5mu_for_write(mu.MuData({"rna": rna, "atac": atac})).write_h5mu(
        fold_path
    )

    generated_obs = pd.DataFrame(
        index=pd.Index(np.asarray(["g1", "g2"], dtype=object))
    )
    generated = mu.MuData(
        {
            "rna": ad.AnnData(
                np.array([[20], [30]], dtype=np.float32), obs=generated_obs.copy()
            ),
            "atac": ad.AnnData(
                np.array([[40], [60]], dtype=np.float32), obs=generated_obs.copy()
            ),
        }
    )
    generated_path = tmp_path / "generated.h5mu"
    prepare_h5mu_for_write(generated).write_h5mu(generated_path)
    output = tmp_path / "debiased.h5mu"

    debias_perturbation_h5mu(
        generated_path=generated_path,
        training_fold_path=fold_path,
        output_path=output,
        held_out_cell_type="A",
        perturbation="p1",
        condition_key="cell_type",
        perturbation_key="perturbation",
        control_perturbation="control",
        alpha=1.0,
        min_cells=1,
    )
    result = mu.read_h5mu(output)
    # RNA delta=8-(1+2+3+5)/4=5.25; held-out A control mean=1.5.
    assert np.allclose(np.asarray(result["rna"].X).mean(axis=0), [6.75])
    # ATAC delta=11-(2+3+4+6)/4=7.25; held-out A control mean=2.5.
    assert np.allclose(np.asarray(result["atac"].X).mean(axis=0), [9.75])
    metadata = json.loads(result.uns["multiflow_debias"])
    assert metadata["method"] == "training_only_perturbation_mean_shift"


def test_subset_cells_keeps_global_uns():
    """mudata before 0.3 drops the global uns in views; the training fold needs it."""
    import json

    import anndata as ad
    import mudata as md
    import numpy as np

    from multiflow_omics.paper import _subset_cells

    names = [f"c{i}" for i in range(6)]
    rna = ad.AnnData(np.ones((6, 3), dtype=np.float32))
    atac = ad.AnnData(np.ones((6, 4), dtype=np.float32))
    rna.obs_names = atac.obs_names = names
    atac.var_names = [f"p{i}" for i in range(4)]
    mdata = md.MuData({"rna": rna, "atac": atac})
    mdata.uns["multiflow_encoder"] = json.dumps({"schema_version": 1})
    subset = _subset_cells(mdata, names[:3])
    assert subset.n_obs == 3
    assert json.loads(subset.uns["multiflow_encoder"]) == {"schema_version": 1}
