#!/usr/bin/env bash
# Run the MultiFlow tutorial notebooks on a Slurm cluster with one command.
#
#   bash tutorials/run_on_slurm.sh           set up, then submit one GPU job per notebook
#   bash tutorials/run_on_slurm.sh status    jobs, progress and results so far
#   bash tutorials/run_on_slurm.sh quick     only the one-minute check on simulated data, on this machine
#
# Each job first runs the notebook in its quick mode (simulated data, about a minute), then with the
# settings of the paper. Every step of a notebook writes a file and is skipped when the file exists, so
# running the same command again after a time-out or a failure continues where the job stopped. A notebook
# that has finished, or whose job is still queued or running, is not submitted again.
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
import multiflow_omics as mf
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
      "${PY}" -c "import multiflow_omics as mf; mf.download_dataset('${name}', r'${target}', accept_license=True)"
    fi
    expected="$("${PY}" -c "from multiflow_omics.datasets import DATASETS; print(DATASETS['${name}'].size_bytes)")"
    actual="$(stat -L -c %s "${target}")"
    [[ "${actual}" == "${expected}" ]] || die "${target} has ${actual} bytes, but the version used by the tutorial has ${expected}"
    say "${file}: size matches the pinned version (its checksum is verified inside the job)"
  done
}

execute() {   # execute <notebook file> <timeout in seconds, -1 for none>; runs in $WORK, results land there
  "${PY}" -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name="${KERNEL}" --ExecutePreprocessor.timeout="$2" "$1"
}

quick_check() {
  local nb="$1"
  cd "${WORK}"
  cp "${REPO}/tutorials/${nb}.ipynb" "quick_${nb}.ipynb"
  say "quick check of ${nb} on simulated data"
  MULTIFLOW_QUICK=1 execute "quick_${nb}.ipynb" 3600
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
  local nb task log file
  if command -v squeue >/dev/null; then
    squeue -u "${USER:-$(id -un)}" -o "%.10i %.16j %.9T %.11M %.11l %R" | awk 'NR == 1 || $2 ~ /^mf-/'
  fi
  for nb in ${NOTEBOOKS}; do
    task="${nb%%_*}"
    echo
    if [[ -s "${WORK}/results/${nb}.ipynb" ]]; then
      say "${nb}: FINISHED; executed notebook: ${WORK}/results/${nb}.ipynb"
      for file in "${WORK}/results/${task}"/*_metrics.csv; do
        if [[ -f "${file}" ]]; then echo "--- ${file}"; head -n 25 "${file}" | cut -c1-160; fi
      done
      continue
    fi
    say "${nb}: not finished"
    if [[ "${task}" == perturbation ]]; then
      say "held-out cell types done: $(find "${WORK}/runs/${task}/folds" -name metrics.csv 2>/dev/null | wc -l)"
    fi
    log="$(ls -t "${WORK}/logs/${nb}"_*.log 2>/dev/null | head -n 1 || true)"
    if [[ -n "${log}" ]]; then say "last lines of ${log}:"; tail -n 12 "${log}" | cut -c1-200; fi
  done
}

case "${1:-submit}" in
  submit) setup; data; submit ;;
  status) status ;;
  quick)  setup; for nb in ${NOTEBOOKS}; do quick_check "${nb}"; done; say "quick check passed" ;;
  run)    [[ $# -eq 2 ]] || die "usage: run <notebook>"; run "$2" ;;
  *)      die "unknown command '$1' (use: submit, status, quick)" ;;
esac
