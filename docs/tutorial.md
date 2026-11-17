# MultiFlow tutorial

MultiFlow has two tasks:

1. **cell-state generation**: generate paired RNA and ATAC profiles of a cell
   type; and
2. **perturbation prediction**: predict the paired RNA and ATAC response of a
   held-out cell type to a perturbation.

Both start from raw profiles and use the same sequence of steps:

```text
raw paired H5MU
    -> encoders (trained on the same data)
    -> paired latent states
    -> MultiFlow
    -> the matching decoders
    -> RNA and ATAC profiles
```

The two notebooks in [`tutorials/`](../tutorials/README.md) run these steps end
to end and compute the metrics of the paper. This page explains each step. All
code below is Python; the equivalent commands are listed in section 6.

## 1. Install

```bash
pip install multiflow-omics
```

Python 3.10 or newer and PyTorch 2.1 or newer. No other repository is needed.
The full runs below were made on Linux with a CUDA GPU; every function takes
`device="cuda"` or `device="cpu"` and uses the GPU by default when there is one.

```python
import multiflow_omics as mf
```

## 2. Input

One H5MU file with paired raw profiles and labels:

```text
paired.h5mu
├── rna.X                    nonnegative integer RNA counts
├── atac.X                   binary ATAC accessibility
├── rna.obs["cell_type"]     cell-type name
└── matching RNA/ATAC obs_names in the same order
```

Perturbation data also need `rna.obs["perturbation"]`, with unperturbed cells
labelled `control`. Check a file before training:

```python
report = mf.validate_h5mu("paired.h5mu")
assert report["paired_raw_contract_ok"]
```

The latent representation `X_multiflow` is written by `mf.encode`; it is not
something to construct by hand.

## 3. Task 1: cell-state generation

Notebook: [`tutorials/generation_openproblems.ipynb`](../tutorials/generation_openproblems.ipynb).

### 3.1 Data and split

```python
data = mf.download_dataset("openproblem", "data/openproblem_filtered.h5mu",
                           accept_license=True)          # 8.4 GB, CC BY 4.0
train, test = mf.split_h5mu(data, "data/openproblem", train_fraction=0.8)
```

The split is stratified by cell type and made before any encoder is fitted, so
the test cells influence neither the encoders nor the latent normalization.

### 3.2 Encoders

For generation, RNA is encoded by a VAE on `log1p` counts per 10,000 (three
1,024-unit layers, 128-dimensional latent state), and ATAC by the ATAC branch
of the paired autoencoder (128-dimensional latent state, Bernoulli decoder).

```python
autoencoder = mf.train_autoencoder(train, "runs/openproblem/autoencoder", epochs=300)
rna_vae = mf.train_rna_vae(train, "runs/openproblem/rna_vae", max_steps=200_000,
                           scimilarity_model_dir="external/annotation_model_v1")
```

In the paper the hidden layers of the RNA VAE start from the public SCimilarity
annotation model. Download it once (the folder must contain `encoder.ckpt`,
`decoder.ckpt` and `gene_order.tsv`):

```bash
mkdir -p external
curl -L "https://zenodo.org/records/8286452/files/annotation_model_v1.tar.gz?download=1" \
  -o external/annotation_model_v1.tar.gz
tar -xzf external/annotation_model_v1.tar.gz -C external
```

Without `scimilarity_model_dir` the VAE starts from random weights, which is a
different setting from the paper.

### 3.3 Encode and train MultiFlow

```python
encoded = mf.encode(train, "data/openproblem/train_encoded.h5mu", task="generation",
                    autoencoder=autoencoder, rna_vae=rna_vae)
run = mf.train(encoded, "runs/openproblem/multiflow", model="cell-state", epochs=1600)
```

`mf.encode` stores the latent states, the checksums of the two encoders and
their scale factors in the output file. `mf.train` writes `model.pt`,
`history.csv` and `run.json` into the run folder; the mean and standard
deviation of the latent states are estimated on the training cells, saved with
the model and undone automatically after sampling.

MultiFlow starts from one Gaussian random variable in the joint RNA-ATAC latent
space. The RNA and the ATAC branch exchange information through three
bidirectional cross-attention modules, and the two velocity losses are added.

### 3.4 Generate and decode

```python
latent = mf.generate(run, "runs/openproblem/generated_latent.h5mu",
                     n={"CD14+ Mono": 200, "NK": 200, "HSC": 200})
profiles = mf.decode(latent, "runs/openproblem/generated_profiles.h5mu", task="generation",
                     reference=encoded, autoencoder=autoencoder, rna_vae=rna_vae)
```

`n` is a number for one `cell_type`, or a mapping from cell type to number of
cells. Decoded RNA is a nonnegative `log1p` CP10k reconstruction; decoded ATAC
is a Bernoulli sample and contains only 0 and 1.

### 3.5 Metrics

```python
mf.metrics.evaluate_generation(test, profiles)
```

The same number of held-out and generated cells is drawn for every cell type.
The table has one row per modality: Pearson and Spearman correlation of the
mean profiles, two MMD estimates on 50 principal components and the mixing of
real and generated cells among nearest neighbours. With the optional package
`scib` (`pip install "multiflow-omics[ilisi]"`), `ilisi=True` adds the scIB
iLISI.

## 4. Task 2: perturbation prediction

Notebook: [`tutorials/perturbation_gse274113.ipynb`](../tutorials/perturbation_gse274113.ipynb).

### 4.1 Data

```python
data = mf.download_dataset("gse274113", "data/GSE274113_filtered.h5mu",
                           accept_license=True)          # 31 GB
cells = (mf.read_obs(data).astype({"cell_type": str, "perturbation": str})
         .groupby(["cell_type", "perturbation"]).size().unstack(fill_value=0))
```

This is the processed Perturb-multiome data of
[GSE274113](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274113),
version [10.5281/zenodo.21986866](https://doi.org/10.5281/zenodo.21986866).
Review the Zenodo record and the terms of the source data before passing
`accept_license=True`. `cells` lists the number of cells per cell type and
perturbation; its row names are the cell types that can be held out.

### 4.2 Autoencoder

Perturbation prediction does not use the RNA VAE. Both modalities go through
the paired autoencoder: RNA with a negative-binomial decoder on raw counts,
ATAC with a Bernoulli decoder.

```python
autoencoder = mf.train_autoencoder(data, "runs/perturbation/autoencoder", epochs=100)
encoded = mf.encode(data, "data/GSE274113_encoded.h5mu", task="perturbation",
                    autoencoder=autoencoder)
```

As in the paper, one autoencoder is trained on all cells and the hold-out
applies to the flow. To keep the held-out cells out of the autoencoder as well,
build the fold from the raw data first and train one autoencoder per fold.

### 4.3 Hold out one cell type, train and predict

```python
cell_type = cells.index[0]                       # the held-out cell type
fold = mf.prepare_perturbation_fold(encoded, "runs/perturbation/fold.h5mu",
                                    held_out_cell_type=cell_type)
run = mf.train(fold, "runs/perturbation/multiflow", model="perturbation",
               perturbation_key="perturbation", epochs=600)

observed = cells.loc[cell_type].drop("control")
predicted = mf.predict_perturbations(run, "runs/perturbation/predicted.h5mu",
                                     training_fold=fold, held_out_cell_type=cell_type,
                                     autoencoder=autoencoder,
                                     n_cells=observed[observed >= 5].to_dict())
```

`prepare_perturbation_fold` removes every perturbed cell of the held-out cell
type; its control cells stay. The context of a cell type is the mean latent
state of its training cells. The control perturbation has index 0 and a fixed
zero embedding.

`predict_perturbations` samples the requested number of cells per perturbation
with 50 integration steps, applies the training-only mean shift of the paper
and decodes both modalities into one file. The mean shift works on the latent
states of each modality: the mean effect of the perturbation in the training
cells is added to the control mean of the held-out cell type, and the generated
cells are moved so that their mean equals this target (`alpha=1` uses the full
effect; `debias=False` switches the shift off). No perturbed cell of the
held-out cell type is used. RNA is decoded as expected counts and ATAC as a
binary sample.

The three steps are also available one perturbation at a time as
`mf.generate`, `mf.debias_perturbation` and `mf.decode`.

### 4.4 Metrics

```python
mf.metrics.evaluate_perturbation(data, predicted, cell_type=cell_type)
```

One row per perturbation. The 100 genes with the largest absolute t statistic
between observed perturbed and observed control cells are used for:
`pcc_delta`, the Pearson correlation between the predicted and the observed
change of the mean expression, both measured from the observed control mean;
`delta_mse`, the mean squared difference of the two changes; and `mmd`, the
maximum mean discrepancy between predicted and observed cells on 20 principal
components. `pcc_delta_all` and `delta_mse_all` use all genes.

## 5. Paper settings

| Setting | Generation | Perturbation |
|---|---:|---:|
| RNA encoder | 128-d RNA VAE | 128-d paired autoencoder, RNA branch |
| RNA encoder input | `log1p` CP10k | raw counts |
| ATAC encoder | 128-d paired autoencoder, ATAC branch | same |
| ATAC encoder input | binary | binary |
| Autoencoder epochs | 300 | 100 |
| RNA VAE steps | 200,000 | not used |
| Flow hidden width | 512 | 512 |
| Residual blocks per branch | 2 | 2 |
| Cross-attention feature width | 64 | 64 |
| Flow epochs | 1600 | 600 |
| Batch size | 512 | 512 |
| Adam learning rate | 1e-4 | 1e-4 |
| Midpoint ODE steps | 100 | 50 |
| Sampling batch size | 512 | 256 |
| Latent mean shift after sampling | none | training-only, alpha = 1 |

These are the defaults of the functions above, except for the number of epochs
and steps, which are passed explicitly in this tutorial.

## 6. Command line

Every file-level step is also a command, with the same defaults:

```bash
multiflow data download openproblem --output data/openproblem_filtered.h5mu --accept-license
multiflow data validate data/openproblem_filtered.h5mu
multiflow data split --input data/openproblem_filtered.h5mu --output-dir data/openproblem
multiflow paper train-autoencoder --input data/openproblem/train.h5mu \
  --output-dir runs/openproblem/autoencoder --epochs 300
multiflow paper train-rna-vae --input data/openproblem/train.h5mu \
  --output-dir runs/openproblem/rna_vae --max-steps 200000
multiflow paper encode --task generation --input data/openproblem/train.h5mu \
  --output data/openproblem/train_encoded.h5mu \
  --autoencoder runs/openproblem/autoencoder/autoencoder.pt \
  --rna-vae runs/openproblem/rna_vae/rna_vae.pt
multiflow train --input data/openproblem/train_encoded.h5mu \
  --output runs/openproblem/multiflow --model cell-state --epochs 1600
multiflow generate --run runs/openproblem/multiflow \
  --output runs/openproblem/generated_latent.h5mu --cell-type "CD14+ Mono" --n 200
multiflow paper decode --task generation --input runs/openproblem/generated_latent.h5mu \
  --reference data/openproblem/train_encoded.h5mu \
  --output runs/openproblem/generated_profiles.h5mu \
  --autoencoder runs/openproblem/autoencoder/autoencoder.pt \
  --rna-vae runs/openproblem/rna_vae/rna_vae.pt
```

`multiflow paper prepare-perturbation-fold` and
`multiflow paper debias-perturbation` cover the perturbation workflow; run any
command with `--help` for its options.

## 7. Reproducibility

Each step records what it used: `split_metadata.json`,
`autoencoder_training.json`, `rna_vae_training.json`, `run.json`, and the
metadata inside every H5MU output hold the input checksums, seeds, settings and
checkpoint checksums. Keep them with the results, and decode generated states
only with the encoders that the flow was trained on.

## 8. Latent-only use

`mf.train` and `mf.generate` also work on latent states produced by other
encoders. The input contract is in [h5mu_contract.md](h5mu_contract.md).
