# Third-party notices

MultiFlow depends on NumPy and PyTorch, which are distributed under their own
licenses. Dependency packages are not vendored in this repository; their
license texts accompany their respective distributions.

The optional OpenProblem tutorial download is the version 3 scDiffusion-X
dataset hosted on Figshare (DOI `10.6084/m9.figshare.28582061.v3`, file ID
`52945733`). The data are distributed separately under CC BY 4.0 and are not
included in this repository or Python distribution.

The paired residual and bidirectional cross-attention architecture in
`src/multiflow_omics/models.py` was adapted from the scDiffusion-X multimodal
backbone and the corresponding MultiFlow research implementation. The audited
upstream revision was:

- scDiffusion-X, commit `d60d928b635ab7a52ace030a641efd02137c193f`
- <https://github.com/EperLuo/scDiffusion-X>
- relevant package source:
  `scdiffusionX/src/scdiffusionX/DiffusionBackbone/`

The paired RNA/ATAC autoencoder in `src/multiflow_omics/autoencoder.py` is a
port of the multimodal autoencoder of the same scDiffusion-X revision (relevant
source: `scdiffusionX/src/scdiffusionX/Autoencoder/`, in particular
`models/base/encoder_model.py`, `models/base/utils.py` and
`data/scrnaseq_loader.py`). Layer layout, parameter names, likelihoods and
training settings are unchanged, so checkpoints written by scDiffusion-X load
without conversion; the PyTorch Lightning, Hydra and scvi-tools dependencies of
the original were replaced by plain PyTorch. scDiffusion-X is not installed
and not imported by MultiFlow.

The audited scDiffusion-X checkout contains an MIT license at the repository
root and a BSD 3-Clause license inside the Python package. Their precise scope
should be confirmed with the upstream maintainers. Until then, both notices are
retained for the adapted portions.

The RNA VAE architecture and SCimilarity checkpoint-loading rules in
`src/multiflow_omics/paper_encoders.py` and its training loop in
`src/multiflow_omics/paper.py` were adapted from the scDiffusion repository:

- scDiffusion, <https://github.com/EperLuo/scDiffusion>
- relevant source: `VAE/VAE_model.py`, `VAE/VAE_train.py` and
  `guided_diffusion/cell_datasets_loader.py`
- upstream license: MIT, copyright (c) 2023 Erpai Luo

The adaptation keeps the three 1,024-unit layers, 128-dimensional
L2-normalized latent state, MSE reconstruction objective, AdamW settings and
`normalize_total(1e4)+log1p` input transform used by the audited notebook. The
wrapper replaces server-specific paths and records the effective random seed
and input checksum.

## scDiffusion-X repository license

MIT License

Copyright (c) 2026 Erpai Luo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## scDiffusion-X package license

BSD 3-Clause License

Copyright (c) 2024, Erpai Luo

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Release verification

Before release, the maintainers should compare the final implementation with
all research notebooks and upstream files from which it was derived. Add any
missing copyright notices here and confirm that this file is included in both
the source distribution and wheel.
