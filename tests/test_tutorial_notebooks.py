"""The tutorial notebooks must stay in step with the public API.

The cheap checks always run. Set ``MULTIFLOW_TEST_NOTEBOOKS=1`` to also execute
both notebooks in their quick mode (about two minutes on a CPU).
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import multiflow_omics as mf

TUTORIALS = Path(__file__).resolve().parents[1] / "tutorials"
NOTEBOOKS = sorted(TUTORIALS.glob("*.ipynb"))


def _code(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def test_both_tutorials_are_present() -> None:
    assert [path.name for path in NOTEBOOKS] == [
        "generation_openproblems.ipynb",
        "perturbation_gse274113.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_code_compiles_and_uses_existing_api(path: Path) -> None:
    code = _code(path)
    compile(code, str(path), "exec")
    for name in sorted(set(re.findall(r"\bmf\.metrics\.(\w+)", code))):
        assert hasattr(mf.metrics, name), f"mf.metrics.{name} does not exist"
    for name in sorted(set(re.findall(r"\bmf\.(\w+)", code)) - {"metrics"}):
        assert hasattr(mf, name), f"mf.{name} does not exist"


@pytest.mark.skipif(
    os.environ.get("MULTIFLOW_TEST_NOTEBOOKS") != "1",
    reason="set MULTIFLOW_TEST_NOTEBOOKS=1 to execute the notebooks in quick mode",
)
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_runs_in_quick_mode(path: Path, tmp_path: Path) -> None:
    script = tmp_path / "notebook.py"
    script.write_text(_code(path), encoding="utf-8")
    environment = {**os.environ, "MULTIFLOW_QUICK": "1", "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    metrics = list((tmp_path / "runs").glob("*/*_metrics.csv"))
    assert len(metrics) == 1 and metrics[0].stat().st_size > 0
