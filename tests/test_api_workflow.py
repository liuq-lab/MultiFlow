"""End-to-end run of the Python API on tiny synthetic raw profiles (CPU, seconds)."""

import json

import mudata as mu
import numpy as np
import pytest

import multiflow_omics as mf


@pytest.fixture()
def raw(tmp_path):
    path = mf.write_toy_h5mu(tmp_path / "raw.h5mu", seed=1)
    mdata = mu.read_h5mu(path)
    for modality in ("rna", "atac"):
        del mdata[modality].obsm["X_multiflow"]
        mdata[modality].obs["perturbation"] = np.asarray(
            ["control", "control", "KO1", "KO2"] * 12, dtype=object
        )
    mdata.update()
    mdata.write_h5mu(path)
    return path


def test_generation_workflow_from_raw_profiles(raw, tmp_path) -> None:
    train_path, test_path = mf.split_h5mu(raw, tmp_path / "split", train_fraction=0.75, seed=0)
    assert mu.read_h5mu(train_path).n_obs == 36 and mu.read_h5mu(test_path).n_obs == 12
    autoencoder = mf.train_autoencoder(
        train_path, tmp_path / "ae", epochs=2, batch_size=8, device="cpu", verbose=False
    )
    summary = json.loads((tmp_path / "ae" / "autoencoder_training.json").read_text())
    assert summary["n_cells"] == 36 and len(summary["history"]) == 2
    rna_vae = mf.train_rna_vae(
        train_path, tmp_path / "vae", max_steps=3, batch_size=8, device="cpu", verbose=False
    )
    encoded = mf.encode(
        train_path,
        tmp_path / "encoded.h5mu",
        task="generation",
        autoencoder=autoencoder,
        rna_vae=rna_vae,
        device="cpu",
    )
    latents = mu.read_h5mu(encoded)
    assert latents["rna"].obsm["X_multiflow"].shape == (36, 128)
    assert latents["atac"].obsm["X_multiflow"].shape == (36, 128)
    run = mf.train(encoded, tmp_path / "run", epochs=2, batch_size=16, hidden_dim=16, device="cpu")
    assert (run / "model.pt").is_file() and (run / "run.json").is_file()
    generated = mf.generate(
        run, tmp_path / "generated.h5mu", n=5, cell_type="T cell", steps=2, device="cpu"
    )
    profiles = mf.decode(
        generated,
        tmp_path / "profiles.h5mu",
        task="generation",
        reference=encoded,
        autoencoder=autoencoder,
        rna_vae=rna_vae,
        device="cpu",
    )
    decoded = mu.read_h5mu(profiles)
    assert decoded["rna"].shape == (5, 12) and decoded["atac"].shape == (5, 20)
    assert set(np.unique(decoded["atac"].X.toarray())) <= {0.0, 1.0}
    assert float(np.asarray(decoded["rna"].X).min()) >= 0

    # several cell types in one call; the first one repeats the single-type result
    several = mf.generate(
        run, tmp_path / "several.h5mu", n={"T cell": 5, "B cell": 3}, steps=2, device="cpu"
    )
    latents = mu.read_h5mu(several)
    assert latents["rna"].obs["cell_type"].tolist() == ["T cell"] * 5 + ["B cell"] * 3
    assert np.allclose(latents["rna"].X[:5], mu.read_h5mu(generated)["rna"].X)
    table = mf.metrics.evaluate_generation(
        test_path,
        mf.decode(
            several,
            tmp_path / "several_profiles.h5mu",
            task="generation",
            reference=encoded,
            autoencoder=autoencoder,
            rna_vae=rna_vae,
            device="cpu",
        ),
    )
    assert table.loc["rna", "n_real"] == table.loc["rna", "n_generated"] == 7  # 4 T + 3 B
    with pytest.raises(ValueError, match="either one cell_type"):
        mf.generate(run, tmp_path / "x.h5mu", n={"T cell": 2}, cell_type="T cell")


def test_perturbation_workflow_from_raw_profiles(raw, tmp_path) -> None:
    autoencoder = mf.train_autoencoder(
        raw,
        tmp_path / "ae",
        epochs=2,
        batch_size=8,
        rna_dims=(16, 8),
        atac_dims=(24, 8),
        device="cpu",
        verbose=False,
    )
    encoded = mf.encode(
        raw, tmp_path / "encoded.h5mu", task="perturbation", autoencoder=autoencoder, device="cpu"
    )
    metadata = json.loads(mu.read_h5mu(encoded).uns["multiflow_encoder"])
    assert metadata["rna_std10"] > 0 and metadata["atac_std10"] > 0
    scaled = mu.read_h5mu(encoded)["rna"].obsm["X_multiflow"]
    assert np.isclose(scaled.std(axis=0, ddof=1).mean(), 0.1, rtol=1e-3)
    fold = mf.prepare_perturbation_fold(
        encoded, tmp_path / "fold.h5mu", held_out_cell_type="T cell"
    )
    run = mf.train(
        fold,
        tmp_path / "run",
        model="perturbation",
        perturbation_key="perturbation",
        epochs=2,
        batch_size=16,
        hidden_dim=16,
        device="cpu",
    )
    generated = mf.generate(
        run, tmp_path / "generated.h5mu", n=6, cell_type="T cell", perturbation="KO1", device="cpu"
    )
    corrected = mf.debias_perturbation(
        generated,
        tmp_path / "corrected.h5mu",
        training_fold=fold,
        held_out_cell_type="T cell",
        perturbation="KO1",
        min_cells=2,
    )
    profiles = mf.decode(
        corrected,
        tmp_path / "profiles.h5mu",
        task="perturbation",
        reference=fold,
        autoencoder=autoencoder,
        device="cpu",
    )
    decoded = mu.read_h5mu(profiles)
    assert decoded["rna"].shape == (6, 12) and decoded["atac"].shape == (6, 20)
    assert decoded["rna"].uns["matrix_scale"] == "raw_count_expectation"

    # one call for the same perturbation gives the same cells as the three steps
    predicted = mf.predict_perturbations(
        run,
        tmp_path / "predicted.h5mu",
        training_fold=fold,
        held_out_cell_type="T cell",
        autoencoder=autoencoder,
        n_cells={"KO1": 6},
        min_cells=2,
        batch_size=512,
        device="cpu",
    )
    one_call = mu.read_h5mu(predicted)
    assert np.allclose(one_call["rna"].X, decoded["rna"].X, rtol=1e-5, atol=1e-6)
    assert np.array_equal(one_call["atac"].X.toarray(), decoded["atac"].X.toarray())

    # all perturbations of the run, then the paper's table for the held-out cell type
    everything = mf.predict_perturbations(
        run,
        tmp_path / "all.h5mu",
        training_fold=fold,
        held_out_cell_type="T cell",
        autoencoder=autoencoder,
        n_cells=8,
        min_cells=2,
        device="cpu",
    )
    obs = mu.read_h5mu(everything)["rna"].obs
    assert obs["perturbation"].value_counts().to_dict() == {"KO1": 8, "KO2": 8}
    assert set(obs["cell_type"]) == {"T cell"}
    table = mf.metrics.evaluate_perturbation(
        raw, everything, cell_type="T cell", min_cells=2, top_k=5
    )
    assert table["perturbation"].tolist() == ["KO1", "KO2"]
    assert np.isfinite(table[["pcc_delta", "delta_mse", "mmd"]].to_numpy()).all()
    with pytest.raises(ValueError, match="unknown perturbation"):
        mf.predict_perturbations(
            run,
            tmp_path / "bad.h5mu",
            training_fold=fold,
            held_out_cell_type="T cell",
            autoencoder=autoencoder,
            n_cells={"missing": 3},
            device="cpu",
        )


def test_api_rejects_inconsistent_arguments(raw, tmp_path) -> None:
    with pytest.raises(ValueError, match="requires perturbation_key"):
        mf.train(raw, tmp_path / "run", model="perturbation")
    with pytest.raises(ValueError, match="cell-state, perturbation or concat"):
        mf.train(raw, tmp_path / "run", model="other")
