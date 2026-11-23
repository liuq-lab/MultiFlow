# MultiFlow

[![CI](https://github.com/liuq-lab/MultiFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/liuq-lab/MultiFlow/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/multiflow-omics.svg)](https://pypi.org/project/multiflow-omics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Documentation, tutorials and the attention explorer:
<https://liuq-lab.github.io/MultiFlow/>**

MultiFlow is a coupled flow-matching model for paired single-cell RNA and ATAC
profiles. The two modalities are generated together, each one attending to the
other at every step. It has two uses:

- **Cell-state generation.** Generate paired RNA and ATAC profiles of a chosen
  cell type.
- **Perturbation prediction.** Predict the paired RNA and ATAC response of a
  cell type to a perturbation, including cell types whose perturbed cells were
  never seen in training.

## Install

```bash
pip install multiflow-omics
```

Python 3.10 or newer and PyTorch 2.1 or newer. Nothing else has to be cloned or
installed. The package is imported as `multiflow` and its command is
`multiflow` (the name `multiflow` on PyPI belongs to an unrelated project).

The development version installs from GitHub:

```bash
pip install "git+https://github.com/liuq-lab/MultiFlow.git"
```

## Quick start

This runs in about half a minute on a laptop CPU. It simulates a small
perturbation dataset, holds out the perturbed cells of one cell type, predicts
them and scores the prediction.

```python
import multiflow as mf

data = mf.simulate_h5mu("demo/cells.h5mu")                      # 2,400 simulated cells
small = dict(rna_dims=(128, 64, 16), atac_dims=(128, 64, 16))   # the paper uses the defaults

autoencoder = mf.train_autoencoder(data, "demo/autoencoder", epochs=40, **small)
encoded = mf.encode(data, "demo/encoded.h5mu", task="perturbation", autoencoder=autoencoder)

fold = mf.prepare_perturbation_fold(encoded, "demo/fold.h5mu", held_out_cell_type="type A")
run = mf.train(fold, "demo/multiflow", model="perturbation", perturbation_key="perturbation",
               epochs=15, hidden_dim=64, batch_size=64)
predicted = mf.predict_perturbations(run, "demo/predicted.h5mu", training_fold=fold,
                                     held_out_cell_type="type A", autoencoder=autoencoder,
                                     n_cells=200)
print(mf.metrics.evaluate_perturbation(data, predicted, cell_type="type A"))
```

Every function reads and writes files and returns the path it wrote, so the
steps can be run one at a time and resumed.

## Tutorials

Two notebooks reproduce the main benchmarks of the paper from the raw data,
with every model trained from scratch:

| Notebook | Task | Data |
|---|---|---|
| [`tutorials/generation_openproblems.ipynb`](tutorials/generation_openproblems.ipynb) | Cell-state generation for 22 cell types, scored against held-out cells (Fig. 2) | OpenProblems multiome, 8.4 GB |
| [`tutorials/perturbation_gse274113.ipynb`](tutorials/perturbation_gse274113.ipynb) | Knockout responses of held-out cell types, leave-one-cell-type-out (Fig. 3a) | GSE274113 Perturb-multiome, 31 GB |

Both need a GPU for the full run and have a quick mode that finishes in about
a minute on a CPU. See [`tutorials/README.md`](tutorials/README.md).

## The steps

```text
raw profiles ──► encoders ──► latent states ──► MultiFlow ──► decoders ──► profiles
```

| Step | Python | Command line |
|---|---|---|
| Download a dataset | `mf.download_dataset` | `multiflow data download` |
| Check an input file | `mf.validate_h5mu` | `multiflow data validate` |
| Train/test split | `mf.split_h5mu` | `multiflow data split` |
| Train the paired autoencoder | `mf.train_autoencoder` | `multiflow paper train-autoencoder` |
| Train the RNA VAE (generation) | `mf.train_rna_vae` | `multiflow paper train-rna-vae` |
| Encode raw profiles | `mf.encode` | `multiflow paper encode` |
| Hold out one cell type | `mf.prepare_perturbation_fold` | `multiflow paper prepare-perturbation-fold` |
| Train MultiFlow | `mf.train` | `multiflow train` |
| Generate latent states | `mf.generate` | `multiflow generate` |
| Decode to RNA and ATAC | `mf.decode` | `multiflow paper decode` |
| Predict perturbations in one call | `mf.predict_perturbations` | |
| Metrics of the paper | `mf.metrics.evaluate_generation`, `mf.metrics.evaluate_perturbation` | |

## Input

One H5MU file with paired cells:

```text
paired.h5mu
├── rna.X                         raw RNA counts
├── atac.X                        binary ATAC accessibility
├── rna.obs["cell_type"]          cell type of every cell
└── rna.obs["perturbation"]       only for perturbation prediction; controls are "control"
```

RNA and ATAC must list the same cells in the same order. `mf.validate_h5mu`
checks a file before any training. Details are in
[docs/h5mu_contract.md](docs/h5mu_contract.md).

## Data

`mf.download_dataset` fetches the two datasets of the paper, resumes
interrupted downloads and verifies size and checksum.

| Name | Content | Size | Source |
|---|---|---|---|
| `openproblem` | OpenProblems bone-marrow multiome, 69,249 cells | 8.4 GB | [Figshare](https://doi.org/10.6084/m9.figshare.28582061.v3), CC BY 4.0 |
| `gse274113` | Processed GSE274113 Perturb-multiome | 31 GB | [Zenodo](https://doi.org/10.5281/zenodo.21986866) |

## More documentation

- [docs/tutorial.md](docs/tutorial.md): both workflows step by step.
- [docs/model_contract.md](docs/model_contract.md): objective, normalization and sampling.
- [docs/implementation_audit.md](docs/implementation_audit.md): how the package
  relates to the code that produced the paper.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/releasing.md](docs/releasing.md).

## Citation and acknowledgements

Citation metadata are in [CITATION.cff](CITATION.cff); the paper DOI will be
added when available.

The paired autoencoder is a port of the multimodal autoencoder of
[scDiffusion-X](https://github.com/EperLuo/scDiffusion-X), and the RNA VAE
follows [scDiffusion](https://github.com/EperLuo/scDiffusion) and
[SCimilarity](https://github.com/Genentech/scimilarity). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
