# Model and data contract

## Canonical architectures

`MultiFlow` is an alias of `CellStateFlow`, the paired cell-state generator
used by the cross-attention experiments. RNA and ATAC are processed in two
streams. Three bidirectional feature-level cross-attention modules allow each
modality's velocity to depend on the other modality's current state.

`MultiFlowPerturbation` adds cell-context embeddings and perturbation
embeddings to the same coupled vector field. Perturbation index `0` is the
control condition and contributes an exactly zero perturbation embedding.

`MultiFlowConcat` is the concatenation ablation used by later generation
experiments. It is retained for reproducibility but is not the canonical
cross-attention architecture.

## Latent inputs

All training inputs must satisfy the following contract:

- RNA and ATAC are finite `float32`-compatible matrices;
- rows are paired cells in identical order;
- label arrays use zero-based, contiguous integer indices;
- the context matrix row order equals the cell-context label order;
- encoder feature order and preprocessing scale are fixed outside this
  package; and
- normalization statistics are estimated on training cells only.

The package standardizes each latent feature independently by default. The
standardizer is stored in the portable checkpoint and inverted after sampling.
It does not replace the encoder-specific scale factors required by an external
decoder.

## Flow objective

For target paired latent state `x1 = (x1_R, x1_A)`, MultiFlow draws one joint
Gaussian source

```text
x0 = (x0_R, x0_A) ~ N(0, I_(d_R + d_A))
```

and a time `t ~ U(0, 1)`. It then trains on the straight-line path

```text
x_t = (1 - t) x0 + t x1
v_t = x1 - x0
```

and minimizes the sum of RNA and ATAC mean-squared velocity errors. The RNA
and ATAC blocks of `x0` are independent coordinate blocks of one full-rank
joint random variable. Their dependence at the endpoint is learned by the
coupled vector field and its bidirectional cross-attention modules. The concat
ablation retains the historical replicated 128-dimensional source only for
legacy reproducibility; it is not the canonical MultiFlow prior.

## Sampling protocol

The default `rng_mode="joint"` samples one concatenated joint source per
batch and splits it into RNA and ATAC coordinate blocks. Record `seed`,
`batch_size`, ODE `steps`, device, and package version because all are part of
exact stochastic reproduction. Use `rng_mode="legacy_interleaved"` only to
reproduce the historical two-call RNA/ATAC random-draw order.
The cell-state generation experiments used 100 midpoint steps and the
perturbation experiments used 50. The CLI selects the corresponding default
from the model type and records the chosen value in every output. Use
`--sampling-steps` during training or `--steps` during generation only when an
explicitly documented sensitivity analysis requires a different value.

`rng_mode="batch_invariant"` makes a fixed seed independent of batch size by
allocating the complete source state first. This option can use substantially
more device memory for very large sample counts.

## Encoder boundary

MultiFlow itself is trained on latent states. Since version 0.2 the package
also contains the encoders of the paper: the paired RNA/ATAC autoencoder
(`mf.train_autoencoder`) and, for cell-state generation, the RNA VAE
(`mf.train_rna_vae`). `mf.encode` writes the latent states into the explicit
H5MU `obsm` keys together with the encoder checksums and scale factors, and
`mf.decode` refuses a reference whose features do not match the autoencoder.
No pretrained encoder weights are bundled; the encoders are trained on the
user's data. The flow never performs hidden preprocessing and never reads raw
`X`: latent states from other encoders can be placed in the same `obsm` keys.
