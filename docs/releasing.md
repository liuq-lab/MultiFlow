# Release guide

The package is published on PyPI as `multiflow-omics` by the GitHub workflow
`.github/workflows/publish.yml`, which runs when a GitHub release is published.
No API token is stored anywhere: PyPI trusts the workflow directly ("Trusted
Publishing").

## 1. One-time setup on PyPI

1. Sign in at <https://pypi.org> with the account that should own the project.
2. Open *Your account → Publishing* and add a **pending publisher** with:

   | Field | Value |
   |---|---|
   | PyPI project name | `multiflow-omics` |
   | Owner | `liuq-lab` |
   | Repository name | `MultiFlow` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

For a rehearsal, do the same at <https://test.pypi.org> with workflow
`testpypi.yml` and environment `testpypi`.

## 2. Before a release

1. Set the version in `src/multiflow_omics/_version.py` and `CITATION.cff`,
   and move the entries of `CHANGELOG.md` under that version.
2. Check locally:

   ```bash
   python -m pip install -e ".[dev]"
   ruff check .
   pytest
   python -m build
   twine check dist/*
   ```

3. Install the wheel into a new environment and run the quick start of the
   README.
4. Commit and push to `main`; wait for the CI workflow to pass.

## 3. Rehearsal on TestPyPI (optional)

Run the workflow *Publish to TestPyPI* from the Actions tab, then:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ multiflow-omics
```

A version number can be uploaded only once, on TestPyPI as on PyPI.

## 4. Release

On GitHub, *Releases → Draft a new release*: create the tag `v<version>` (for
example `v0.2.0`) on `main`, paste the changelog entry and publish. The
workflow *Publish to PyPI* builds the distributions, runs the tests and
uploads them. Afterwards:

```bash
python -m pip install multiflow-omics
```

## 5. Large files

H5AD/H5MU files, latent arrays, encoder checkpoints, generated results and
trained weights are excluded from Git and from the distributions. Publish
permitted data or weights through Zenodo or a GitHub release, with checksums.
