# Changelog

All notable changes to MultiFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow [Semantic Versioning](https://semver.org/)
after its first public release.

## [0.2.0]

First release on PyPI. The package is now self-contained: `pip install
multiflow-omics` is all that is needed for both workflows of the paper.

### Added

- `multiflow_omics.autoencoder`: the paired RNA/ATAC autoencoder in plain
  PyTorch, a port of the scDiffusion-X multimodal autoencoder. Parameter names
  are unchanged, so checkpoints written by scDiffusion-X load without
  conversion.
- A Python API with one function per step, exported from the package:
  `split_h5mu`, `train_autoencoder`, `train_rna_vae`, `encode`, `train`,
  `generate`, `decode`, `prepare_perturbation_fold`, `debias_perturbation` and
  `predict_perturbations`. Each function writes files and returns the path.
- `predict_perturbations`: sampling, training-only mean shift and decoding of
  several perturbations of a held-out cell type in one call.
- `generate` accepts a mapping from cell type to number of cells.
- `multiflow_omics.metrics`: the metrics of the paper for cell-state generation
  (`evaluate_generation`) and perturbation prediction
  (`evaluate_perturbation`), without Scanpy or scikit-learn.
- `simulate_h5mu` (a small simulated perturbation dataset) and `read_obs`.
- Two tutorial notebooks in `tutorials/` that reproduce the generation and the
  perturbation benchmark from the raw data.
- Commands `multiflow paper train-autoencoder`, `multiflow paper train-rna-vae`
  and `multiflow data split`.

### Changed

- scDiffusion-X, scvi-tools, PyTorch Lightning, Hydra, PyYAML and Scanpy are no
  longer needed. The extras `paper` and `analysis` are gone; `notebooks`
  (figures in the tutorials) and `ilisi` (scIB iLISI) are new.
- `multiflow paper encode` and `multiflow paper decode` take `--autoencoder`
  and `--rna-vae`. `--scdiffusion-x-root` and `--encoder-config` were removed;
  `--multimodal-ae-checkpoint` and `--rna-vae-checkpoint` remain as aliases.
- The encoder metadata in H5MU files records `autoencoder_sha256` instead of
  `multimodal_ae_sha256`.
- The canonical source is one full-rank joint Gaussian random variable in the
  concatenated RNA-ATAC latent space. Training and default sampling draw this
  source once and split it into modality coordinate blocks; the old two-call
  draw order remains available as `rng_mode="legacy_interleaved"`.

### Removed

- The scripts in `workflows/`; their steps are functions of the package now.

## [0.1.0]

Initial public version on GitHub (not released on PyPI).

### Added

- Source-layout Python package for paired latent flow matching.
- Cell-state, perturbation-conditioned, and concatenation-ablation models.
- Feature-wise latent standardization, ODE sampling, checkpoint utilities, and
  a command-line interface.
- Unit tests and release checks for Python 3.10 through 3.12.
- Paired H5MU training and generated-output contracts with string cell-type
  labels retained in portable checkpoints.
- Version-pinned, checksum-verified download command for the scDiffusion-X
  OpenProblem dataset.
- Version-pinned, checksum-verified download command for the processed
  GSE274113 perturbation dataset on Zenodo.
- A concise H5MU-first tutorial and an implementation audit linking the public
  models to the executed research references.
- End-to-end paper workflows for training the generation RNA VAE and
  multimodal AE, encoding raw paired profiles, and decoding generated states.
- Leave-one-cell-type-out perturbation preparation and the executed
  training-only post-sampling latent mean-shift correction.
- Public perturbation tutorial commands using the published GSE274113 H5MU.

### Changed

- The concise user-facing command is now `multiflow`; the historical
  `multiflow-omics` command remains as a compatibility alias.
- `multiflow paper decode` now computes perturbation library-size statistics
  from sparse RNA row sums and avoids densifying the complete reference H5MU.
- The main CLI no longer uses anonymous NPZ files for training or generation.
- Default midpoint-solver steps now match the research protocols: 100 for
  cell-state generation and 50 for perturbation prediction.
