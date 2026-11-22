# H5MU data contract

MultiFlow uses H5MU as the public container for paired cells. The user-facing
paper workflow starts from raw profiles. A task-specific encoder command then
creates a derived H5MU for latent flow training.

## Required structure

An initial input must contain `rna` and `atac` modalities. Their `obs_names`
must be unique and match exactly in value and order:

- `rna.X` contains nonnegative integer raw counts;
- `atac.X` contains only 0 and 1;
- `rna.obs["cell_type"]` contains a non-missing cell-type name.

Perturbation input also contains `rna.obs["perturbation"]`, including a named
`control` condition. Run `multiflow paper encode` to create the derived
representations:

- generation RNA: 128-dimensional RNA VAE latent after
  `normalize_total(1e4)+log1p`;
- generation ATAC: 128-dimensional multimodal-AE ATAC latent from binary X;
- perturbation RNA: 128-dimensional multimodal-AE RNA latent from raw counts;
  and
- perturbation ATAC: 128-dimensional multimodal-AE ATAC latent from binary X.

The derived values are stored in `rna.obsm["X_multiflow"]` and
`atac.obsm["X_multiflow"]`. They are intermediate data, not fields a new user
is expected to supply manually.

Use `--rna-representation` and `--atac-representation` to select another
explicit `obsm` key. `X` can be selected explicitly, but is never used as an
implicit fallback when a latent key is absent.

## Scale boundaries

The core package does not guess preprocessing from numeric ranges. The paper
RNA VAE and multimodal AE use different input and decoder scales, so an encoder
release must state all of the following:

1. input matrix/layer and scale;
2. feature identifiers in exact order;
3. checkpoint and configuration hashes;
4. latent dimension and encoder-specific scale factors;
5. decoder output scale; and
6. whether the encoder was trained only on the training split.

MultiFlow then fits an additional feature-wise latent standardizer using the
training cells. That standardizer and the string-to-integer condition mapping
are stored in `model.pt` and `run.json`.

For perturbation flow training, `obs["perturbation"]` (or the selected column) must
contain the named control condition. The CLI reserves that condition as index
0 and therefore as the exact zero perturbation vector. The context embedding
matrix is stored in `uns["multiflow_context_matrix"]`; its row names must be
stored in `uns["multiflow_context_classes"]` and must exactly match the fixed
cell-type class order. MultiFlow rejects missing or reordered row labels.

## Generated files

The low-level generator writes paired generated latent states to `rna.X` and
`atac.X` with `uns["matrix_scale"] = "latent"`. `mdata.uns["multiflow"]` records the
checkpoint SHA256, model configuration, seed, ODE steps, batch size, label
mapping, RNG mode, and whether latent standardization was inverted.

`multiflow paper decode` then uses the exact matching checkpoint bundle to
write biological profile matrices. Generation RNA is normalized-log
reconstruction; perturbation RNA is raw counts drawn from the negative binomial
of the autoencoder; ATAC is binary in both tasks. A decoder must never be inferred from an unrelated H5MU file.
