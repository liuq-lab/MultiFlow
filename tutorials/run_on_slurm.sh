#!/usr/bin/env bash
# Run the MultiFlow tutorial notebooks on a Slurm cluster with one command.
#
#   bash tutorials/run_on_slurm.sh           set up, then submit one GPU job per notebook
#   bash tutorials/run_on_slurm.sh status    jobs, progress and results so far
#   bash tutorials/run_on_slurm.sh check     on this machine, a few minutes: installed versions, the layout of
#                                            the data files, and each notebook in quick mode (simulated data)
#   bash tutorials/run_on_slurm.sh quick     only the notebooks in quick mode
#
# Each job first runs the notebook in its quick mode (simulated data, about a minute, in a fresh folder),
# then with the settings of the paper. Every step of the full run writes a file and is skipped when the
# file exists, so running the same command again after a time-out or a failure continues where the job
# stopped. A notebook that has finished, or whose job is still queued or running, is not submitted again.
#
# Settings are environment variables; all are optional:
#   PYTHON             Python that has PyTorch for the cluster's GPUs          (default: python3)
#   CONDA_ENV          or: the name of a conda environment to take Python from
#   WORK               folder for data, runs and results, about 100 GB         (default: <repository>/tutorial_work)
#   SBATCH_ARGS        partition, account and GPU type for sbatch              (default: "--gpus=1")
#   SBATCH_TIME, SBATCH_MEM, SBATCH_CPUS                                       (default: 1-00:00:00, 200G, 6)
#   OPENPROBLEM_H5MU, GSE274113_H5MU   existing copies of the two datasets; they are linked, not downloaded
#   NOTEBOOKS          notebooks to run   (default: "generation_openproblems perturbation_gse274113")
#
# Nothing is installed into the Python you name: the script builds a virtual environment in $WORK/venv that
# can see its packages, and installs MultiFlow and the notebook tools there. Datasets that are not given are
# downloaded (8 GB and 31 GB); by running the script you accept their licences, see the tutorial pages.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-${REPO}/tutorial_work}"
NOTEBOOKS="${NOTEBOOKS:-generation_openproblems perturbation_gse274113}"
VENV="${WORK}/venv"
PY="${VENV}/bin/python"
KERNEL="multiflow-tutorial"
SELF="${REPO}/tutorials/run_on_slurm.sh"

say() { printf '[multiflow] %s\n' "$*"; }
die() { printf '[multiflow] ERROR: %s\n' "$*" >&2; exit 1; }

find_python() {
  if [[ -n "${PYTHON:-}" ]]; then echo "${PYTHON}"; return; fi
  if [[ -n "${CONDA_ENV:-}" ]]; then
    local base
    for base in "${HOME}/.conda/envs" "${HOME}/miniconda3/envs" "${HOME}/miniforge3/envs" \
                "${HOME}/mambaforge/envs" "${HOME}/anaconda3/envs"; do
      if [[ -x "${base}/${CONDA_ENV}/bin/python" ]]; then echo "${base}/${CONDA_ENV}/bin/python"; return; fi
    done
    die "conda environment '${CONDA_ENV}' not found; set PYTHON=/path/to/python instead"
  fi
  command -v python3 || die "no python3 on PATH; set PYTHON=/path/to/python"
}

dataset_of() {   # notebook -> dataset name, file name, variable with an existing copy
  case "$1" in
    generation_openproblems) echo "openproblem openproblem_filtered.h5mu ${OPENPROBLEM_H5MU:-}" ;;
    perturbation_gse274113)  echo "gse274113 GSE274113_filtered.h5mu ${GSE274113_H5MU:-}" ;;
    *) die "unknown notebook '$1'" ;;
  esac
}

setup() {
  mkdir -p "${WORK}/data" "${WORK}/logs" "${WORK}/results"
  if [[ ! -x "${PY}" ]]; then
    local base; base="$(find_python)"
    say "building ${VENV} on top of ${base}"
    "${base}" -m venv --system-site-packages "${VENV}"
  fi
  say "installing MultiFlow from ${REPO} and the notebook tools"
  "${PY}" -m pip install --quiet --disable-pip-version-check -e "${REPO}" nbconvert ipykernel matplotlib
  "${PY}" -m ipykernel install --prefix "${VENV}" --name "${KERNEL}" >/dev/null
  "${PY}" - <<'EOF'
import importlib.util
import multiflow as mf
import torch
extras = {name: importlib.util.find_spec(name) is not None for name in ("scanpy", "scib")}
print(f"[multiflow] multiflow-omics {mf.__version__}, torch {torch.__version__}; "
      f"scanpy {'yes' if extras['scanpy'] else 'no (no UMAP figure)'}, "
      f"scib {'yes' if extras['scib'] else 'no (no iLISI column)'}")
EOF
}

data() {
  local nb name file src target expected actual
  for nb in ${NOTEBOOKS}; do
    read -r name file src <<< "$(dataset_of "${nb}")"
    target="${WORK}/data/${file}"
    if [[ -e "${target}" ]]; then
      :
    elif [[ -n "${src}" ]]; then
      [[ -s "${src}" ]] || die "${src} does not exist"
      ln -s "${src}" "${target}"
      say "linked ${file} -> ${src}"
    else
      say "downloading ${name} to ${target}"
      "${PY}" -c "import multiflow as mf; mf.download_dataset('${name}', r'${target}', accept_license=True)"
    fi
    expected="$("${PY}" -c "from multiflow.datasets import DATASETS; print(DATASETS['${name}'].size_bytes)")"
    actual="$(stat -L -c %s "${target}")"
    [[ "${actual}" == "${expected}" ]] || die "${target} has ${actual} bytes, but the version used by the tutorial has ${expected}"
    say "${file}: size matches the pinned version (its checksum is verified inside the job)"
  done
}

versions() {   # the code and the libraries in use; printed at the start of every job
  say "code: $(git -C "${REPO}" log -1 --format='%h %s' 2>/dev/null || echo 'unknown (not a git checkout)')"
  "${PY}" - <<'EOF'
import platform
from importlib import metadata

parts = [f"python {platform.python_version()}"]
for name in ("multiflow-omics", "torch", "anndata", "mudata", "numpy", "pandas", "scipy", "h5py",
             "scanpy", "scib"):
    try:
        parts.append(f"{name} {metadata.version(name)}")
    except metadata.PackageNotFoundError:
        parts.append(f"{name} -")
print("[multiflow] versions: " + ", ".join(parts))
EOF
}

inspect_data() {   # layout of the data files against what the notebooks expect; reads a small sample only
  local nb name file src target bad=0
  for nb in ${NOTEBOOKS}; do
    read -r name file src <<< "$(dataset_of "${nb}")"
    target="${WORK}/data/${file}"
    if [[ ! -e "${target}" ]]; then target="${src}"; fi
    if [[ -z "${target}" || ! -e "${target}" ]]; then
      say "${file}: not here yet (it is linked or downloaded when the jobs are submitted)"
      continue
    fi
    "${PY}" - "${target}" "${nb}" <<'EOF' || bad=1
import sys

import h5py
import numpy as np

path, notebook = sys.argv[1], sys.argv[2]
ok = True
total_gb = 0.0


def text(values):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def sample(node, n=200_000):
    """The first stored values of a matrix (sparse group or dense dataset)."""
    data = node["data"] if isinstance(node, h5py.Group) else node
    if data.ndim == 1:
        return np.asarray(data[:n], dtype=np.float64)
    rows = max(1, n // max(1, data.shape[1]))
    return np.asarray(data[:rows], dtype=np.float64).ravel()


def size_gb(node):
    """Memory the matrix takes once it is read."""
    if isinstance(node, h5py.Group):
        return sum(node[k].size * node[k].dtype.itemsize for k in ("data", "indices", "indptr")) / 1e9
    return node.size * node.dtype.itemsize / 1e9


def describe(node):
    global total_gb
    values = sample(node)
    gb = size_gb(node)
    total_gb += gb
    stored = "sparse" if isinstance(node, h5py.Group) else "dense"
    if values.size == 0:
        return f"empty, {stored}", False, False
    integer = bool(np.all(np.abs(values - np.round(values)) < 1e-6) and values.min() >= 0)
    binary = integer and bool(values.max() <= 1)
    kind = "binary" if binary else ("integer counts" if integer else "not integer")
    return f"{kind} (largest value in the sample: {values.max():g}), {stored}, {gb:.1f} GB in memory", integer, binary


def column(obs, key):
    """One annotation column as strings, read without loading anything else."""
    node = obs[key]
    if isinstance(node, h5py.Group) and "categories" in node:  # categorical: categories and codes
        categories, codes = text(node["categories"][...]), np.asarray(node["codes"][...])
        return np.where(codes >= 0, categories[np.clip(codes, 0, None)], "nan")
    if isinstance(node, h5py.Group):  # nullable array: values and mask
        return text(node["values"][...])
    values = node[...]
    legacy = obs.get("__categories")
    if legacy is not None and key in legacy:
        categories = text(legacy[key][...])
        return np.where(values >= 0, categories[np.clip(values, 0, None)], "nan")
    return text(values)


print(f"[multiflow] {path}")
with h5py.File(path, "r") as handle:
    for modality, need in (("rna", "raw counts"), ("atac", "binary values")):
        group = handle[f"mod/{modality}"]
        shape = group["X"].attrs.get("shape", getattr(group["X"], "shape", "?"))
        found, integer, binary = describe(group["X"])
        good = integer if modality == "rna" else binary
        ok = ok and good
        print(f"[multiflow]   {modality}.X: {tuple(int(v) for v in shape)}, {found}; "
              f"needed: {need}: {'ok' if good else 'NOT OK'}")
        layers = group.get("layers")
        for name in sorted(layers.keys()) if layers is not None else ():
            print(f"[multiflow]   {modality}.layers[{name!r}]: {describe(layers[name])[0]}")
    print(f"[multiflow]   all matrices together: about {total_gb:.0f} GB in memory when the file is read")
    obs = handle["mod/rna/obs"]
    columns = [k for k in obs.keys() if k not in ("_index", "__categories")]
    print(f"[multiflow]   rna.obs columns: {', '.join(columns)}")
    keys = ("cell_type", "perturbation") if notebook.startswith("perturbation") else ("cell_type",)
    for key in keys:
        if key not in obs:
            ok = False
            print(f"[multiflow]   rna.obs[{key!r}]: missing: NOT OK")
            continue
        labels, numbers = np.unique(column(obs, key), return_counts=True)
        order = np.argsort(-numbers)
        listed = ", ".join(f"{labels[i]} ({numbers[i]:,})" for i in order)
        print(f"[multiflow]   rna.obs[{key!r}]: {int(numbers.sum()):,} cells, {len(labels)} values: {listed}")
        if key == "perturbation" and "control" not in set(labels.tolist()):
            ok = False
            print("[multiflow]   rna.obs['perturbation'] has no 'control' label: NOT OK")
print(f"[multiflow]   layout as the notebook expects: {'yes' if ok else 'NO'}")
sys.exit(0 if ok else 3)
EOF
  done
  return "${bad}"
}

execute() {   # execute <notebook file> <timeout in seconds, -1 for none>; runs in $WORK, results land there
  "${PY}" -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name="${KERNEL}" --ExecutePreprocessor.timeout="$2" "$1"
}

quick_check() {   # the notebook in quick mode, in a fresh folder: no file of an earlier run or version is reused
  local nb="$1" dir
  mkdir -p "${WORK}"
  dir="$(mktemp -d "${WORK}/quick_check_${nb%%_*}.XXXXXX")"
  cp "${REPO}/tutorials/${nb}.ipynb" "${dir}/quick_${nb}.ipynb"
  say "quick check of ${nb} on simulated data"
  if (cd "${dir}" && MULTIFLOW_QUICK=1 execute "quick_${nb}.ipynb" 3600); then
    rm -rf "${dir}"
    say "quick check of ${nb}: passed"
  else
    say "ERROR: the quick check of ${nb} failed; its files are kept in ${dir}" >&2
    return 1
  fi
}

check() {   # everything that can be checked in a few minutes without a GPU
  local nb failed=0
  versions
  inspect_data || failed=1
  for nb in ${NOTEBOOKS}; do
    quick_check "${nb}" || failed=1
  done
  if [[ "${failed}" == 0 ]]; then
    say "CHECK PASSED: versions, data layout and quick runs are fine"
  else
    die "CHECK FAILED: see the lines marked NOT OK or ERROR above"
  fi
}

run() {   # inside the job
  local nb="$1" task="${1%%_*}"
  [[ -x "${PY}" ]] || die "${VENV} is missing; run 'bash tutorials/run_on_slurm.sh' on a login node first"
  cd "${WORK}"
  say "$(date) on $(hostname): ${nb}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
  "${PY}" - <<'EOF'
import os, sys, torch
if torch.cuda.is_available():
    print("[multiflow] GPU:", torch.cuda.get_device_name(0))
elif os.environ.get("ALLOW_CPU") != "1":
    sys.exit("[multiflow] ERROR: PyTorch sees no GPU in this job. Check SBATCH_ARGS and the Python in use, "
             "or set ALLOW_CPU=1 to run on the CPU anyway.")
EOF
  versions
  quick_check "${nb}"
  cp "${REPO}/tutorials/${nb}.ipynb" "${nb}.ipynb"
  say "full run of ${nb}"
  execute "${nb}.ipynb" -1
  mkdir -p "results/${task}"
  local file
  for file in "runs/${task}"/*.csv "runs/${task}"/*.png; do
    if [[ -f "${file}" ]]; then cp "${file}" "results/${task}/"; fi
  done
  cp "${nb}.ipynb" "results/${nb}.ipynb"
  say "$(date): finished ${nb}; results are in ${WORK}/results"
}

submit() {
  command -v sbatch >/dev/null || die "sbatch not found: this command submits Slurm jobs"
  local nb job name
  # shellcheck disable=SC2086
  for nb in ${NOTEBOOKS}; do
    name="mf-${nb%%_*}"
    if [[ -s "${WORK}/results/${nb}.ipynb" ]]; then
      say "${nb}: already finished (${WORK}/results/${nb}.ipynb); delete that file to run it again"
      continue
    fi
    if command -v squeue >/dev/null \
        && [[ -n "$(squeue -h -u "${USER:-$(id -un)}" -n "${name}" -o %i 2>/dev/null)" ]]; then
      say "${nb}: job ${name} is already queued or running; not submitted again"
      continue
    fi
    job="$(sbatch --parsable --job-name="${name}" \
      --cpus-per-task="${SBATCH_CPUS:-6}" --mem="${SBATCH_MEM:-200G}" --time="${SBATCH_TIME:-1-00:00:00}" \
      --output="${WORK}/logs/${nb}_%j.log" ${SBATCH_ARGS:---gpus=1} \
      --export=ALL,WORK="${WORK}" --wrap "bash '${SELF}' run ${nb}")"
    say "submitted ${nb} as job ${job}; log: ${WORK}/logs/${nb}_${job}.log"
  done
  say "check progress with: bash tutorials/run_on_slurm.sh status"
}

status() {
  local nb task name log file job state
  if command -v squeue >/dev/null; then
    squeue -u "${USER:-$(id -un)}" -o "%.10i %.16j %.9T %.11M %.11l %R" | awk 'NR == 1 || $2 ~ /^mf-/'
  fi
  for nb in ${NOTEBOOKS}; do
    task="${nb%%_*}"; name="mf-${task}"
    echo
    if [[ -s "${WORK}/results/${nb}.ipynb" ]]; then
      say "${nb}: FINISHED; executed notebook: ${WORK}/results/${nb}.ipynb"
      for file in "${WORK}/results/${task}"/*_metrics.csv; do
        if [[ -f "${file}" ]]; then echo "--- ${file}"; head -n 25 "${file}" | cut -c1-160; fi
      done
      continue
    fi
    job=""; state=""
    if command -v squeue >/dev/null; then
      read -r job state <<< "$(squeue -h -u "${USER:-$(id -un)}" -n "${name}" -o '%i %T' 2>/dev/null | head -n 1)" || true
    fi
    if [[ -z "${job}" ]]; then
      say "${nb}: NOT FINISHED, and no job is queued or running; run the submit command again to continue"
    elif [[ "${state}" == PENDING ]]; then
      say "${nb}: job ${job} is WAITING in the queue; it has not started, so it has no log and no result yet"
    else
      say "${nb}: job ${job} is ${state}"
    fi
    if [[ "${task}" == perturbation ]]; then
      say "held-out cell types done: $(find "${WORK}/runs/${task}/folds" -name metrics.csv 2>/dev/null | wc -l)"
    fi
    if [[ -n "${job}" ]]; then   # only the log of the current job; an older log would describe an older run
      log="${WORK}/logs/${nb}_${job}.log"
      if [[ -s "${log}" ]]; then say "last lines of the log of job ${job}:"; tail -n 12 "${log}" | cut -c1-200; fi
    else
      log="$(ls -t "${WORK}/logs/${nb}"_*.log 2>/dev/null | head -n 1 || true)"
      if [[ -n "${log}" ]]; then
        job="${log##*_}"; job="${job%.log}"
        say "last lines of the log of the last job (${job}, no longer running):"; tail -n 12 "${log}" | cut -c1-200
      fi
    fi
  done
}

case "${1:-submit}" in
  submit) setup; data; submit ;;
  status) status ;;
  check)  if [[ ! -x "${PY}" ]]; then setup; fi; check ;;
  quick)  setup; for nb in ${NOTEBOOKS}; do quick_check "${nb}"; done; say "quick check passed" ;;
  run)    [[ $# -eq 2 ]] || die "usage: run <notebook>"; run "$2" ;;
  *)      die "unknown command '$1' (use: submit, status, check, quick)" ;;
esac
