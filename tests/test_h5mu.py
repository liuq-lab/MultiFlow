from __future__ import annotations

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import pytest

from multiflow.h5mu import (
    prepare_h5mu_for_write,
    read_paired_latents,
    validate_h5mu,
)


def _paired(tmp_path, *, mismatch=False, add_latents=True, nonbinary_atac=False):
    rna_obs = pd.DataFrame(
        {"cell_type": np.asarray(["T", "B", "T"], dtype=object)},
        index=pd.Index(np.asarray(["a", "b", "c"], dtype=object), dtype=object),
    )
    atac_index = ["a", "c", "b"] if mismatch else ["a", "b", "c"]
    atac_obs = pd.DataFrame(
        {
            "cell_type": np.asarray(
                ["T", "T", "B"] if mismatch else ["T", "B", "T"], dtype=object
            )
        },
        index=pd.Index(np.asarray(atac_index, dtype=object), dtype=object),
    )
    rna = ad.AnnData(X=np.array([[1, 0], [0, 2], [3, 1]], dtype=np.float32), obs=rna_obs)
    atac_x = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.float32)
    if nonbinary_atac:
        atac_x[0, 0] = 0.5
    atac = ad.AnnData(X=atac_x, obs=atac_obs)
    if add_latents:
        rna.obsm["X_multiflow"] = np.arange(6, dtype=np.float32).reshape(3, 2)
        atac.obsm["X_multiflow"] = np.arange(6, dtype=np.float32).reshape(3, 2) + 1
    path = tmp_path / "paired.h5mu"
    mdata = mu.MuData({"rna": rna, "atac": atac})
    prepare_h5mu_for_write(mdata).write_h5mu(path)
    return path


def test_read_latents_preserves_pairing_and_fixed_sorted_classes(tmp_path):
    data = read_paired_latents(_paired(tmp_path))
    assert data.rna.shape == (3, 2)
    assert data.atac.shape == (3, 2)
    assert data.label_classes == ["B", "T"]
    assert data.labels.tolist() == [1, 0, 1]


def test_pair_order_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="same order"):
        read_paired_latents(_paired(tmp_path, mismatch=True))


def test_missing_latents_are_reported_without_using_raw_x(tmp_path):
    path = _paired(tmp_path, add_latents=False)
    report = validate_h5mu(path)
    assert report["paired_raw_contract_ok"] is True
    assert report["latent_ready"] is False
    with pytest.raises(ValueError, match="Raw X is not silently treated"):
        read_paired_latents(path)


def test_nonbinary_atac_fails_raw_contract(tmp_path):
    report = validate_h5mu(_paired(tmp_path, nonbinary_atac=True))
    assert report["atac"]["contract_ok"] is False
    assert report["paired_raw_contract_ok"] is False


def test_perturbation_mapping_reserves_control_and_checks_context_order(tmp_path):
    obs = pd.DataFrame(
        {
            "cell_type": np.asarray(["T", "B", "T"], dtype=object),
            "perturbation": np.asarray(["ATF4", "control", "ATF4"], dtype=object),
        },
        index=pd.Index(np.asarray(["a", "b", "c"], dtype=object), dtype=object),
    )
    rna = ad.AnnData(X=np.ones((3, 2), dtype=np.float32), obs=obs.copy())
    atac = ad.AnnData(X=np.ones((3, 2), dtype=np.float32), obs=obs.copy())
    rna.obsm["X_multiflow"] = np.ones((3, 2), dtype=np.float32)
    atac.obsm["X_multiflow"] = np.ones((3, 2), dtype=np.float32)
    mdata = mu.MuData({"rna": rna, "atac": atac})
    mdata.uns["multiflow_context_matrix"] = np.eye(2, dtype=np.float32)
    mdata.uns["multiflow_context_classes"] = np.asarray(["B", "T"], dtype=object)
    path = tmp_path / "perturbation.h5mu"
    prepare_h5mu_for_write(mdata).write_h5mu(path)

    data = read_paired_latents(path, perturbation_key="perturbation")
    assert data.perturbation_classes == ["control", "ATF4"]
    assert data.perturbations.tolist() == [1, 0, 1]

    mdata.uns["multiflow_context_classes"] = np.asarray(["T", "B"], dtype=object)
    bad_path = tmp_path / "bad_context_order.h5mu"
    prepare_h5mu_for_write(mdata).write_h5mu(bad_path)
    with pytest.raises(ValueError, match="class order differs"):
        read_paired_latents(bad_path, perturbation_key="perturbation")


def test_validation_scans_rows_after_the_first_256(tmp_path):
    n_cells = 300
    obs = pd.DataFrame(
        {"cell_type": np.asarray(["B"] * n_cells, dtype=object)},
        index=pd.Index(
            np.asarray([f"cell_{index}" for index in range(n_cells)], dtype=object),
            dtype=object,
        ),
    )
    rna = ad.AnnData(X=np.zeros((n_cells, 2), dtype=np.float32), obs=obs.copy())
    atac_x = np.zeros((n_cells, 2), dtype=np.float32)
    atac_x[299, 1] = 0.5
    atac = ad.AnnData(X=atac_x, obs=obs.copy())
    rna.obsm["X_multiflow"] = np.zeros((n_cells, 2), dtype=np.float32)
    atac.obsm["X_multiflow"] = np.zeros((n_cells, 2), dtype=np.float32)
    path = tmp_path / "late_nonbinary.h5mu"
    prepare_h5mu_for_write(mu.MuData({"rna": rna, "atac": atac})).write_h5mu(path)

    report = validate_h5mu(path)
    assert report["atac"]["audit_mode"] == "full_chunked"
    assert report["atac"]["contract_ok"] is False


def test_validation_rejects_nonfinite_latents_and_closes_file(tmp_path):
    path = _paired(tmp_path)
    mdata = mu.read_h5mu(path)
    mdata.mod["rna"].obsm["X_multiflow"][2, 1] = np.nan
    prepare_h5mu_for_write(mdata).write_h5mu(path)

    report = validate_h5mu(path)
    assert report["rna_latent"]["finite"] is False
    assert report["latent_ready"] is False

    renamed = tmp_path / "validated_and_closed.h5mu"
    path.rename(renamed)
    assert renamed.is_file()


def test_validation_rejects_empty_matrices(tmp_path):
    obs = pd.DataFrame(
        {"cell_type": np.asarray([], dtype=object)},
        index=pd.Index(np.asarray([], dtype=object), dtype=object),
    )
    rna = ad.AnnData(X=np.empty((0, 2), dtype=np.float32), obs=obs.copy())
    atac = ad.AnnData(X=np.empty((0, 3), dtype=np.float32), obs=obs.copy())
    rna.obsm["X_multiflow"] = np.empty((0, 2), dtype=np.float32)
    atac.obsm["X_multiflow"] = np.empty((0, 2), dtype=np.float32)
    path = tmp_path / "empty.h5mu"
    prepare_h5mu_for_write(mu.MuData({"rna": rna, "atac": atac})).write_h5mu(path)

    report = validate_h5mu(path)
    assert report["paired_raw_contract_ok"] is False
    assert report["latent_ready"] is False
