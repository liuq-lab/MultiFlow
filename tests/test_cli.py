from __future__ import annotations

import json

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from multiflow import __version__
from multiflow.cli import _default_sampling_steps, _parser, main
from multiflow.h5mu import prepare_h5mu_for_write
from multiflow.legacy import migrate_legacy_checkpoint
from multiflow.models import ConditionalConcatFlow


def _write_toy_h5mu(path) -> None:
    obs = pd.DataFrame(
        {"cell_type": np.asarray(["B cell", "T cell", "B cell", "T cell"], dtype=object)},
        index=pd.Index(
            np.asarray([f"cell_{index}" for index in range(4)], dtype=object), dtype=object
        ),
    )
    generator = np.random.default_rng(4)
    rna = ad.AnnData(
        X=sparse.csr_matrix(generator.poisson(1, (4, 3)).astype(np.float32)),
        obs=obs.copy(),
    )
    atac = ad.AnnData(
        X=sparse.csr_matrix(generator.binomial(1, 0.3, (4, 5)).astype(np.float32)),
        obs=obs.copy(),
    )
    rna.obsm["X_multiflow"] = generator.normal(size=(4, 2)).astype(np.float32)
    atac.obsm["X_multiflow"] = generator.normal(size=(4, 2)).astype(np.float32)
    mdata = mu.MuData({"rna": rna, "atac": atac})
    prepare_h5mu_for_write(mdata).write_h5mu(path)


def test_cli_help_uses_concise_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "data" in output
    assert "train" in output
    assert "generate" in output
    assert "train-latents" not in output


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_sampling_defaults_match_research_protocol():
    parser = _parser()
    generation = parser.parse_args(
        ["train", "--input", "paired.h5mu", "--output", "run"]
    )
    perturbation = parser.parse_args(
        [
            "train",
            "--input",
            "paired.h5mu",
            "--output",
            "run",
            "--model",
            "perturbation",
        ]
    )
    assert generation.sampling_steps is None
    assert perturbation.sampling_steps is None
    assert _default_sampling_steps(generation.model) == 100
    assert _default_sampling_steps(perturbation.model) == 50


def test_cli_creates_valid_h5mu_example(tmp_path, capsys):
    destination = tmp_path / "example.h5mu"
    main(["data", "example", "--output", str(destination), "--seed", "11"])
    assert destination.is_file()
    assert "saved paired H5MU example" in capsys.readouterr().out
    output = mu.read_h5mu(destination)
    assert output.mod["rna"].n_obs == 48
    assert output.mod["rna"].obsm["X_multiflow"].shape == (48, 4)
    assert output.mod["atac"].obsm["X_multiflow"].shape == (48, 4)


def test_cli_h5mu_train_inspect_generate_round_trip(tmp_path, capsys):
    source = tmp_path / "paired.h5mu"
    run = tmp_path / "run"
    generated = tmp_path / "generated.h5mu"
    _write_toy_h5mu(source)

    main(["data", "validate", str(source)])
    report = json.loads(capsys.readouterr().out)
    assert report["latent_ready"] is True
    assert report["paired_raw_contract_ok"] is True

    main(
        [
            "train",
            "--input",
            str(source),
            "--output",
            str(run),
            "--model",
            "concat",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-dim",
            "2",
            "--num-layers",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert (run / "model.pt").is_file()
    assert (run / "history.csv").is_file()
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["label_classes"] == ["B cell", "T cell"]

    main(["inspect", "--checkpoint", str(run / "model.pt")])
    assert '"model_type": "concat"' in capsys.readouterr().out

    main(
        [
            "generate",
            "--run",
            str(run),
            "--output",
            str(generated),
            "--n",
            "2",
            "--cell-type",
            "B cell",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )
    output = mu.read_h5mu(generated)
    assert output.mod["rna"].shape == (2, 2)
    assert output.mod["atac"].shape == (2, 2)
    assert output.mod["rna"].obs["cell_type"].tolist() == ["B cell", "B cell"]
    assert output.mod["rna"].uns["matrix_scale"] == "latent"
    assert np.isfinite(output.mod["rna"].X).all()
    assert bool(output.uns["multiflow"]["standardizer_present"])
    assert bool(output.uns["multiflow"]["standardization_inverted"])


def test_cli_rejects_unknown_cell_type(tmp_path):
    source = tmp_path / "paired.h5mu"
    run = tmp_path / "run"
    _write_toy_h5mu(source)
    main(
        [
            "train",
            "--input",
            str(source),
            "--output",
            str(run),
            "--model",
            "concat",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-dim",
            "2",
            "--num-layers",
            "2",
            "--device",
            "cpu",
        ]
    )
    with pytest.raises(ValueError, match="unknown cell_type"):
        main(
            [
                "generate",
                "--run",
                str(run),
                "--output",
                str(tmp_path / "bad.h5mu"),
                "--n",
                "1",
                "--cell-type",
                "not-a-cell",
                "--device",
                "cpu",
            ]
        )


def test_cli_no_standardize_records_truthful_generation_metadata(tmp_path):
    source = tmp_path / "paired.h5mu"
    run = tmp_path / "run"
    generated = tmp_path / "generated.h5mu"
    _write_toy_h5mu(source)
    main(
        [
            "train",
            "--input",
            str(source),
            "--output",
            str(run),
            "--model",
            "concat",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-dim",
            "2",
            "--num-layers",
            "2",
            "--no-standardize",
            "--device",
            "cpu",
        ]
    )
    main(
        [
            "generate",
            "--run",
            str(run),
            "--output",
            str(generated),
            "--n",
            "2",
            "--cell-type",
            "B cell",
            "--steps",
            "1",
            "--device",
            "cpu",
        ]
    )
    output = mu.read_h5mu(generated)
    assert not bool(output.uns["multiflow"]["standardizer_present"])
    assert not bool(output.uns["multiflow"]["standardization_inverted"])


def test_migrated_condition_mapping_supports_named_generation(tmp_path):
    legacy = tmp_path / "legacy.pt"
    migrated = tmp_path / "migrated.pt"
    generated = tmp_path / "generated.h5mu"
    model = ConditionalConcatFlow(
        2,
        2,
        hidden_dim=4,
        num_layers=2,
        num_classes=2,
        dropout=0.0,
    )
    torch.save(
        {
            "format_version": 5,
            "model_config": model.get_config(),
            "model_state": model.state_dict(),
            "label_classes": ["B cell", "T cell"],
        },
        legacy,
    )
    migrate_legacy_checkpoint(legacy, migrated)

    main(
        [
            "generate",
            "--run",
            str(migrated),
            "--output",
            str(generated),
            "--n",
            "2",
            "--cell-type",
            "T cell",
            "--steps",
            "1",
            "--device",
            "cpu",
        ]
    )
    output = mu.read_h5mu(generated)
    assert output.mod["rna"].obs["cell_type"].tolist() == ["T cell", "T cell"]
    assert not bool(output.uns["multiflow"]["standardizer_present"])
    assert not bool(output.uns["multiflow"]["standardization_inverted"])
