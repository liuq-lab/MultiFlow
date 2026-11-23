"""User-facing command line interface for MultiFlow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import api
from ._version import __version__
from .checkpoint import load_checkpoint
from .datasets import DATASETS, download_dataset
from .h5mu import (
    DEFAULT_ATAC_REPRESENTATION,
    DEFAULT_RNA_REPRESENTATION,
    validate_h5mu,
    write_toy_h5mu,
)
from .legacy import migrate_legacy_checkpoint
from .normalization import LatentStandardizer
from .paper import (
    debias_perturbation_h5mu,
    decode_paper_h5mu,
    encode_paper_h5mu,
    prepare_perturbation_fold,
    split_h5mu,
    train_autoencoder_h5mu,
    train_rna_vae_h5mu,
)


def _add_representation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rna-representation",
        default=DEFAULT_RNA_REPRESENTATION,
        help=f"RNA obsm key (default: {DEFAULT_RNA_REPRESENTATION})",
    )
    parser.add_argument(
        "--atac-representation",
        default=DEFAULT_ATAC_REPRESENTATION,
        help=f"ATAC obsm key (default: {DEFAULT_ATAC_REPRESENTATION})",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multiflow",
        description="Coupled flow matching for paired single-cell multiomics",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="download or validate paired H5MU data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    download = data_commands.add_parser("download", help="download a version-pinned dataset")
    download.add_argument("dataset", choices=sorted(DATASETS))
    download.add_argument("--output", required=True)
    download.add_argument("--accept-license", action="store_true")
    download.add_argument("--force", action="store_true")
    validate = data_commands.add_parser("validate", help="audit pairing and matrix scales")
    validate.add_argument("input")
    validate.add_argument("--condition-key", default="cell_type")
    _add_representation_arguments(validate)
    validate.add_argument("--json", dest="json_output", default=None)
    split = data_commands.add_parser(
        "split", help="write a condition-stratified train/test split of raw profiles"
    )
    split.add_argument("--input", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--condition-key", default="cell_type")
    split.add_argument("--train-fraction", type=float, default=0.8)
    split.add_argument("--seed", type=int, default=20260717)
    example = data_commands.add_parser("example", help="create a tiny paired H5MU file")
    example.add_argument("--output", default="toy_multiflow.h5mu")
    example.add_argument("--seed", type=int, default=7)

    train = subparsers.add_parser("train", help="train MultiFlow from paired H5MU latents")
    train.add_argument("--input", required=True, help="prepared paired .h5mu file")
    train.add_argument("--output", required=True, help="new run directory")
    _add_representation_arguments(train)
    train.add_argument("--condition-key", default="cell_type")
    train.add_argument("--unconditional", action="store_true")
    train.add_argument("--perturbation-key", default=None)
    train.add_argument("--context-matrix-key", default="multiflow_context_matrix")
    train.add_argument("--context-classes-key", default="multiflow_context_classes")
    train.add_argument(
        "--control-perturbation",
        default="control",
        help="perturbation name reserved as the zero vector (default: control)",
    )
    train.add_argument(
        "--model",
        choices=["cell-state", "perturbation", "concat"],
        default="cell-state",
    )
    train.add_argument("--epochs", type=int, default=600)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--hidden-dim", type=int, default=512)
    train.add_argument("--num-blocks", type=int, default=2)
    train.add_argument("--num-layers", type=int, default=8)
    train.add_argument("--cross-attention-dim", type=int, default=64)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--no-standardize", action="store_true")
    train.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help=(
            "default midpoint ODE steps saved with the run "
            "(default: 100 for generation, 50 for perturbation)"
        ),
    )
    train.add_argument(
        "--hash-input",
        action="store_true",
        help="record SHA256 of the input H5MU (can take time for large files)",
    )

    generate = subparsers.add_parser("generate", help="generate paired states as H5MU")
    generate.add_argument("--run", required=True, help="run directory created by train")
    generate.add_argument("--output", required=True, help="generated .h5mu file")
    generate.add_argument("--n", type=int, required=True)
    generate.add_argument("--cell-type", default=None)
    generate.add_argument("--perturbation", default=None)
    generate.add_argument("--steps", type=int, default=None)
    generate.add_argument("--batch-size", type=int, default=512)
    generate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--keep-standardized", action="store_true")
    generate.add_argument(
        "--rng-mode",
        choices=["joint", "legacy_interleaved", "batch_invariant"],
        default="joint",
        help=(
            "joint samples one full-rank RNA-ATAC Gaussian source; "
            "legacy_interleaved reproduces the historical draw order"
        ),
    )

    inspect = subparsers.add_parser("inspect", help="print checkpoint metadata")
    inspect.add_argument("--checkpoint", required=True)

    migrate = subparsers.add_parser(
        "migrate",
        help="convert a trusted historical checkpoint to the portable format",
    )
    migrate.add_argument("--source", required=True)
    migrate.add_argument("--output", required=True)
    migrate.add_argument(
        "--model-type",
        choices=["cell-state", "perturbation", "concat"],
        default=None,
    )
    migrate.add_argument(
        "--standardizer-npz",
        default=None,
        help="optional legacy NPZ containing rna_mean/rna_std/atac_mean/atac_std",
    )
    migrate.add_argument(
        "--label-classes",
        nargs="+",
        default=None,
        help="ordered biological names for legacy cell-type/context indices",
    )
    migrate.add_argument(
        "--perturbation-classes",
        nargs="+",
        default=None,
        help="ordered perturbation names; index 0 must be the control",
    )
    migrate.add_argument("--trust-source", action="store_true")

    paper = subparsers.add_parser(
        "paper",
        help="train encoders, encode raw profiles and decode generated latents",
    )
    paper_commands = paper.add_subparsers(dest="paper_command", required=True)
    train_ae = paper_commands.add_parser(
        "train-autoencoder",
        help="train the paired RNA/ATAC autoencoder on raw profiles",
    )
    train_ae.add_argument("--input", required=True, help="raw paired .h5mu file")
    train_ae.add_argument("--output-dir", required=True)
    train_ae.add_argument("--condition-key", default="cell_type")
    train_ae.add_argument("--epochs", type=int, default=300)
    train_ae.add_argument("--batch-size", type=int, default=256)
    train_ae.add_argument("--seed", type=int, default=20260717)
    train_ae.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_vae = paper_commands.add_parser(
        "train-rna-vae",
        help="train the RNA VAE used for cell-state generation",
    )
    train_vae.add_argument("--input", required=True, help="raw paired .h5mu file")
    train_vae.add_argument("--output-dir", required=True)
    train_vae.add_argument(
        "--scimilarity-model-dir",
        default=None,
        help="SCimilarity annotation model used to initialise the hidden layers (paper setting)",
    )
    train_vae.add_argument("--max-steps", type=int, default=200_000)
    train_vae.add_argument("--checkpoint-every", type=int, default=50_000)
    train_vae.add_argument("--batch-size", type=int, default=128)
    train_vae.add_argument("--seed", type=int, default=1234)
    train_vae.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    encode = paper_commands.add_parser(
        "encode",
        help="encode raw RNA/ATAC profiles with the paper encoders",
    )
    encode.add_argument("--task", choices=["generation", "perturbation"], required=True)
    encode.add_argument("--input", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument(
        "--autoencoder", "--multimodal-ae-checkpoint", dest="autoencoder", required=True
    )
    encode.add_argument("--rna-vae", "--rna-vae-checkpoint", dest="rna_vae", default=None)
    encode.add_argument("--condition-key", default="cell_type")
    encode.add_argument("--batch-size", type=int, default=2048)
    encode.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    fold = paper_commands.add_parser(
        "prepare-perturbation-fold",
        help="remove held-out non-control cells and build the notebook context matrix",
    )
    fold.add_argument("--input", required=True)
    fold.add_argument("--output", required=True)
    fold.add_argument("--held-out-cell-type", required=True)
    fold.add_argument("--condition-key", default="cell_type")
    fold.add_argument("--perturbation-key", default="perturbation")
    fold.add_argument("--control-perturbation", default="control")

    debias = paper_commands.add_parser(
        "debias-perturbation",
        help="apply the paper training-only perturbation mean-shift correction",
    )
    debias.add_argument("--input", required=True, help="generated latent H5MU")
    debias.add_argument("--training-fold", required=True)
    debias.add_argument("--output", required=True)
    debias.add_argument("--held-out-cell-type", required=True)
    debias.add_argument("--perturbation", required=True)
    debias.add_argument("--condition-key", default="cell_type")
    debias.add_argument("--perturbation-key", default="perturbation")
    debias.add_argument("--control-perturbation", default="control")
    debias.add_argument("--alpha", type=float, default=1.0)
    debias.add_argument("--min-cells", type=int, default=5)

    decode = paper_commands.add_parser(
        "decode",
        help="decode generated latents with the matching paper decoders",
    )
    decode.add_argument("--task", choices=["generation", "perturbation"], required=True)
    decode.add_argument("--input", required=True, help="generated latent H5MU")
    decode.add_argument("--reference", required=True, help="encoded training H5MU")
    decode.add_argument("--output", required=True)
    decode.add_argument(
        "--autoencoder", "--multimodal-ae-checkpoint", dest="autoencoder", required=True
    )
    decode.add_argument("--rna-vae", "--rna-vae-checkpoint", dest="rna_vae", default=None)
    decode.add_argument("--condition-key", default="cell_type")
    decode.add_argument("--batch-size", type=int, default=2048)
    decode.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    decode.add_argument("--seed", type=int, default=0)
    return parser


def _default_sampling_steps(model_type: str) -> int:
    """Return the notebook-aligned midpoint solver budget."""
    return api.default_sampling_steps(model_type)


def _train(args: argparse.Namespace) -> None:
    run = api.train(
        args.input,
        args.output,
        model=args.model,
        condition_key=None if args.unconditional else args.condition_key,
        perturbation_key=args.perturbation_key,
        control_perturbation=args.control_perturbation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_layers=args.num_layers,
        cross_attention_dim=args.cross_attention_dim,
        sampling_steps=args.sampling_steps,
        standardize=not args.no_standardize,
        seed=args.seed,
        device=args.device,
        hash_input=args.hash_input,
        rna_representation=args.rna_representation,
        atac_representation=args.atac_representation,
        context_matrix_key=args.context_matrix_key,
        context_classes_key=args.context_classes_key,
        verbose=False,
    )
    print(f"saved run: {run}")
    print("  checkpoint: model.pt")
    print("  history: history.csv")
    print("  manifest: run.json")


def _generate(args: argparse.Namespace) -> None:
    destination = api.generate(
        args.run,
        args.output,
        n=args.n,
        cell_type=args.cell_type,
        perturbation=args.perturbation,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        keep_standardized=args.keep_standardized,
        rng_mode=args.rng_mode,
    )
    print(f"saved paired generated H5MU: {destination}")


def _split_data(args: argparse.Namespace) -> None:
    train, test = split_h5mu(
        input_path=args.input,
        output_dir=args.output_dir,
        condition_key=args.condition_key,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    print(f"saved {train}")
    print(f"saved {test}")


def _validate_data(args: argparse.Namespace) -> None:
    report = validate_h5mu(
        args.input,
        condition_key=args.condition_key,
        rna_representation=args.rna_representation,
        atac_representation=args.atac_representation,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        destination = Path(args.json_output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    if not report["paired_raw_contract_ok"]:
        raise ValueError("H5MU raw matrix scale contract failed")
    if not report["latent_ready"]:
        print(
            "note: raw paired data are valid, but encoder latents are absent; "
            "run `multiflow paper encode` (or `multiflow.encode`) before flow training"
        )


def _download_data(args: argparse.Namespace) -> None:
    dataset = DATASETS[args.dataset]
    print(f"source DOI: https://doi.org/{dataset.doi}")
    print(f"license: {dataset.license}")
    print(f"download size: {dataset.size_bytes / 1e9:.2f} GB")
    destination = download_dataset(
        args.dataset,
        args.output,
        accept_license=args.accept_license,
        force=args.force,
    )
    print(f"saved verified dataset: {destination}")


def _example_data(args: argparse.Namespace) -> None:
    destination = write_toy_h5mu(args.output, seed=args.seed)
    print(f"saved paired H5MU example: {destination}")


def _inspect(args: argparse.Namespace) -> None:
    model, standardizer, payload = load_checkpoint(args.checkpoint)
    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "format_version": payload["format_version"],
        "package_version": payload.get("package_version"),
        "model_config": model.get_config(),
        "epoch": payload.get("epoch"),
        "has_standardizer": standardizer is not None,
        "metadata": payload.get("metadata", {}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _migrate(args: argparse.Namespace) -> None:
    standardizer = None
    if args.standardizer_npz is not None:
        with np.load(args.standardizer_npz) as source:
            required = {"rna_mean", "rna_std", "atac_mean", "atac_std"}
            missing = sorted(required.difference(source.files))
            if missing:
                raise ValueError(f"standardizer NPZ is missing: {missing}")
            standardizer = LatentStandardizer.from_state_dict(
                {name: torch.as_tensor(source[name]) for name in required}
            )
    migration_metadata: dict[str, object] = {}
    if args.label_classes is not None:
        migration_metadata["label_classes"] = list(args.label_classes)
    if args.perturbation_classes is not None:
        migration_metadata["perturbation_classes"] = list(args.perturbation_classes)
    output = migrate_legacy_checkpoint(
        args.source,
        args.output,
        model_type=None if args.model_type is None else args.model_type.replace("-", "_"),
        standardizer=standardizer,
        trust_source=args.trust_source,
        metadata=migration_metadata,
    )
    print(f"saved migrated checkpoint: {output}")
    if standardizer is None:
        _, recovered, _ = load_checkpoint(output)
        if recovered is None:
            print(
                "warning: no latent standardizer was available; supply the original "
                "training-only statistics before decoding generated latents"
            )


def _paper(args: argparse.Namespace) -> None:
    if args.paper_command == "train-autoencoder":
        output = train_autoencoder_h5mu(
            input_path=args.input,
            output_dir=args.output_dir,
            condition_key=args.condition_key,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        print(f"saved autoencoder: {output}")
    elif args.paper_command == "train-rna-vae":
        output = train_rna_vae_h5mu(
            input_path=args.input,
            output_dir=args.output_dir,
            scimilarity_model_dir=args.scimilarity_model_dir,
            max_steps=args.max_steps,
            checkpoint_every=args.checkpoint_every,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        print(f"saved RNA VAE: {output}")
    elif args.paper_command == "encode":
        output = encode_paper_h5mu(
            input_path=args.input,
            output_path=args.output,
            task=args.task,
            autoencoder_checkpoint=args.autoencoder,
            rna_vae_checkpoint=args.rna_vae,
            condition_key=args.condition_key,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"saved encoded paper input: {output}")
    elif args.paper_command == "prepare-perturbation-fold":
        output = prepare_perturbation_fold(
            input_path=args.input,
            output_path=args.output,
            held_out_cell_type=args.held_out_cell_type,
            condition_key=args.condition_key,
            perturbation_key=args.perturbation_key,
            control_perturbation=args.control_perturbation,
        )
        print(f"saved perturbation training fold: {output}")
    elif args.paper_command == "debias-perturbation":
        output = debias_perturbation_h5mu(
            generated_path=args.input,
            training_fold_path=args.training_fold,
            output_path=args.output,
            held_out_cell_type=args.held_out_cell_type,
            perturbation=args.perturbation,
            condition_key=args.condition_key,
            perturbation_key=args.perturbation_key,
            control_perturbation=args.control_perturbation,
            alpha=args.alpha,
            min_cells=args.min_cells,
        )
        print(f"saved debiased perturbation latents: {output}")
    elif args.paper_command == "decode":
        output = decode_paper_h5mu(
            generated_path=args.input,
            reference_path=args.reference,
            output_path=args.output,
            task=args.task,
            autoencoder_checkpoint=args.autoencoder,
            rna_vae_checkpoint=args.rna_vae,
            condition_key=args.condition_key,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(f"saved decoded RNA/ATAC profiles: {output}")
    else:  # pragma: no cover - argparse restricts the accepted subcommands.
        raise ValueError(f"unsupported paper command: {args.paper_command}")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "data" and args.data_command == "download":
        _download_data(args)
    elif args.command == "data" and args.data_command == "validate":
        _validate_data(args)
    elif args.command == "data" and args.data_command == "split":
        _split_data(args)
    elif args.command == "data":
        _example_data(args)
    elif args.command == "train":
        _train(args)
    elif args.command == "generate":
        _generate(args)
    elif args.command == "inspect":
        _inspect(args)
    elif args.command == "paper":
        _paper(args)
    else:
        _migrate(args)


if __name__ == "__main__":
    main()
