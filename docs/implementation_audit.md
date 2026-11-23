# Implementation audit

This document records the source-to-release checks performed before writing the
public tutorial. It separates the executed model graph from inactive notebook
experiments and documents the raw-profile encoder/decoder boundary.

## Reference hierarchy

1. Cell-state architecture: executed cells 10, 14, 15, and 19 in
   `MultiFlow/generation/best_record/Joint_model_VAE_AE_encoder_cross_attention.ipynb`.
2. Perturbation architecture: the production leave-one-cell-type-out runner
   `run_leave_one_out_all_celltypes.py`, which incorporates the intended zero
   control embedding missing from the earlier exploratory notebook cell.
3. Concatenation ablation: the active graph in
   `Joint_model_VAE_diffusion_encoder_cross_attention_concat.ipynb`, represented
   by the corrected research module `multiflow_generation/model.py`.
4. Protocol defaults and encoder boundary: manuscript Methods plus the executed
   training and sampling cells.

## Numerical graph checks

The release models were initialized with the reference state tensors after a
semantic key mapping and evaluated on identical random inputs. The largest
absolute output difference was `0.0` for all three checks:

| Release model | Reference | State entries | Max. absolute difference |
|---|---|---:|---:|
| `CellStateFlow` | executed cross-attention notebook module | 87 | 0.0 |
| `PerturbationFlow` | production leave-one-out runner | 88 | 0.0 |
| `ConditionalConcatFlow` | corrected concat notebook module | 47 | 0.0 |

The executable audit is kept outside the package test suite because it points
to the local research tree. The package tests independently cover model
construction, training, sampling, checkpoint migration, H5MU validation, and
CLI round trips.

## Training and sampling checks

- target path: linear interpolation between Gaussian source and encoded target;
- objective: RNA MSE plus ATAC MSE with equal weight;
- canonical source: one full-rank joint Gaussian state in the concatenated
  RNA-ATAC latent space, sampled once and split into modality coordinate blocks;
- concat ablation source: one replicated Gaussian block retained only for
  historical reproducibility;
- optimization: Adam, learning rate `1e-4`, batch size 512, 600 epochs;
- latent normalization: per-feature mean and standard deviation estimated from
  training cells and inverted after sampling;
- generation solver: midpoint ODE integration with 100 steps;
- perturbation solver: midpoint ODE integration with 50 steps; and
- perturbation control: fixed class index 0 with an exact zero embedding.

## Explicit boundaries and corrected exploratory cells

- The user-facing paper workflow begins with raw-count RNA and binary ATAC in
  H5MU. `multiflow paper encode` creates the latent representations; the core
  flow still consumes those explicit derived representations.
- Generation RNA uses the scDiffusion/SCimilarity VAE after
  `normalize_total(1e4)+log1p`. Generation ATAC uses the binary branch of the
  paired autoencoder. Perturbation uses the paired autoencoder for both raw
  RNA counts and binary ATAC. These task-specific scales are not interchangeable.
- The release includes the audited RNA VAE architecture and a port of the
  scDiffusion-X multimodal autoencoder (`multiflow.autoencoder`). The
  port keeps the layer layout and parameter names, so a checkpoint written by
  scDiffusion-X loads without conversion. With the same seed and inputs, the
  port and the original modules give identical initial weights, latent states
  and decoder outputs, and the training loss is assembled in the same way; the
  negative-binomial log-likelihood, which the original takes from scvi-tools,
  was checked against `torch.distributions`. scDiffusion-X itself is neither
  installed nor imported. The release does not bundle pretrained weights or
  large data files. The processed GSE274113 H5MU is distributed separately through
  the version-pinned Zenodo record
  [10.5281/zenodo.21986866](https://doi.org/10.5281/zenodo.21986866).
- `multiflow paper prepare-perturbation-fold` reproduces the executed notebook:
  it removes held-out non-control cells from flow training, retains held-out
  controls, and computes each cell-type context from all remaining flow-training
  rows. The original run used one fixed multimodal AE before this flow-level
  split; stricter fold-specific encoder refitting must be labeled as a different
  protocol.
- The executed perturbation workflow applies a post-sampling latent mean shift
  before decoding. For each modality, the training perturbation-minus-control
  mean is added to the held-out cell type's training control mean, and the
  generated group is translated to that target (alpha 1). The correction uses
  flow-training cells and held-out controls only; held-out perturbed cells are
  never used.
- The exploratory concat notebook assigns `x1_atac = rna_latent` in one cell.
  This is inconsistent with paired RNA/ATAC training and with the surrounding
  code. The corrected research module and release use `atac_latent`; the concat
  model is labeled as an ablation rather than the canonical paper model.
- The earliest perturbation notebook initialized the control embedding as a
  trainable random vector. The later production runner fixes and masks row 0,
  matching the manuscript definition `p = 0`; the release follows that
  production behavior.
- The previous CLI fallback of 200 ODE steps did not match the paper protocol.
  The release now selects 100 steps for cell-state generation and 50 for
  perturbation unless an explicit override is recorded.

These boundaries are also stated in the README and tutorial. Generated latent
states are an intermediate artifact; only `multiflow paper decode` writes the
task-specific biological profile scale.
