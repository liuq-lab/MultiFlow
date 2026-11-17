# Tutorials

Two notebooks reproduce the main benchmarks of the MultiFlow paper from the raw
data. Every model is trained from scratch with the Python API; nothing besides
`pip install multiflow-omics` is required.

| Notebook | What it does | Data |
|---|---|---|
| [`generation_openproblems.ipynb`](generation_openproblems.ipynb) | Trains the encoders and MultiFlow on OpenProblems, generates paired RNA and ATAC profiles for 22 cell types and scores them against held-out cells (Fig. 2). | 8.4 GB, downloaded by the notebook |
| [`perturbation_gse274113.ipynb`](perturbation_gse274113.ipynb) | Leave-one-cell-type-out prediction of knockout responses on the GSE274113 Perturb-multiome data, with the top-100 PCC-delta, delta MSE and MMD of Fig. 3a. | 31 GB, downloaded by the notebook |

## Quick check

Each notebook has a switch `QUICK` in its first code cell. With `QUICK = True`
the same steps run on a small simulated dataset with small networks, in about a
minute on a CPU. Use it to check an installation; its numbers say nothing about
the real data.

```bash
pip install "multiflow-omics[notebooks]" nbconvert
MULTIFLOW_QUICK=1 jupyter nbconvert --to notebook --execute \
  --output quick_check.ipynb tutorials/perturbation_gse274113.ipynb
```

## Full run

The full runs need a GPU and several hours, so run them as a batch job rather
than in an interactive session. Each step writes a file and is skipped when
that file exists; an interrupted job continues where it stopped when it is
submitted again.

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 tutorials/generation_openproblems.ipynb
```

On a Slurm cluster, for example:

```bash
#!/bin/bash
#SBATCH --job-name=multiflow-tutorial
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

cd /path/to/MultiFlow
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 tutorials/perturbation_gse274113.ipynb
```

Results are written under `runs/` in the folder the job is started from, and
the datasets under `data/`. The perturbation notebook reads the 31 GB dataset
into memory, so give it enough RAM. To shorten that notebook, set `CELL_TYPES`
to a list of a few cell types; each held-out cell type trains one flow.

The optional scIB iLISI is reported when `scib` is installed
(`pip install "multiflow-omics[ilisi]"`).
