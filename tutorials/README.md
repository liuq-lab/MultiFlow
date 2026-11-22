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

A notebook keeps its files next to itself: datasets under `data/` and results
under `runs/` in the folder that holds the notebook, also when it is executed
from another folder. Copy the notebook to a disk with enough space before a
full run (about 30 GB for generation, 100 GB for perturbation).

## Full run on a Slurm cluster

The full runs need a GPU and several hours. One command sets everything up and
submits one job per notebook:

```bash
git clone https://github.com/liuq-lab/MultiFlow.git && cd MultiFlow
SBATCH_ARGS="--partition=gpu --gpus=1" bash tutorials/run_on_slurm.sh
```

`run_on_slurm.sh` builds a virtual environment in `tutorial_work/venv`,
installs MultiFlow there, links or downloads the two datasets and submits the
jobs. Each job first prints the library versions and runs its notebook in quick
mode, then with the settings of the paper, and copies the executed notebook,
the metric tables and the figures to `tutorial_work/results/`.

| Variable | Meaning | Default |
|---|---|---|
| `SBATCH_ARGS` | partition, account and GPU type for `sbatch` | `--gpus=1` |
| `PYTHON` or `CONDA_ENV` | a Python (or the name of a conda environment) that has PyTorch for the cluster's GPUs; nothing is installed into it | `python3` |
| `WORK` | folder for data, runs and results | `tutorial_work` in the repository |
| `OPENPROBLEM_H5MU`, `GSE274113_H5MU` | existing copies of the datasets, linked instead of downloaded | download |
| `SBATCH_TIME`, `SBATCH_MEM`, `SBATCH_CPUS` | job size | `1-00:00:00`, `200G`, `6` |
| `NOTEBOOKS` | which notebooks to run | both |

```bash
bash tutorials/run_on_slurm.sh status    # jobs, progress, and the metric tables once they exist
bash tutorials/run_on_slurm.sh check     # a few minutes, no GPU: versions, layout of the data files, quick runs
bash tutorials/run_on_slurm.sh quick     # only the quick runs, on this machine
```

Run `check` before waiting for a GPU: it prints the installed library versions,
compares the data files with what the notebooks expect (raw RNA counts in
`rna.X`, binary `atac.X`, the `cell_type` and `perturbation` annotations), and
runs both notebooks in quick mode in a fresh folder. `status` names the job it
reports on; while a job waits in the queue there is no log to show.

Every step of a notebook writes a file and is skipped when that file exists.
After a time-out or a failure, run the first command again: the new job
continues where the old one stopped. A notebook that has finished, or whose job
is still queued or running, is not submitted again. To shorten the perturbation notebook, set
`CELL_TYPES` in its first cell to a few cell types; each held-out cell type
trains one flow.

## Full run by hand

```bash
cp tutorials/generation_openproblems.ipynb /path/with/space/ && cd /path/with/space
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 generation_openproblems.ipynb
```

The perturbation notebook reads the 31 GB dataset into memory, so give it
enough RAM. The optional scIB iLISI is reported when `scib` is installed
(`pip install "multiflow-omics[ilisi]"`).
