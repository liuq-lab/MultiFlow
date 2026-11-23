from __future__ import annotations

import hashlib

import pytest

from multiflow.datasets import DATASETS, DatasetFile, verify_dataset_file


def test_dataset_file_verification(tmp_path):
    payload = b"paired-h5mu-test"
    path = tmp_path / "file.h5mu"
    path.write_bytes(payload)
    dataset = DatasetFile(
        name="test",
        filename=path.name,
        url="https://example.invalid/file",
        size_bytes=len(payload),
        md5=hashlib.md5(payload).hexdigest(),  # noqa: S324 - integrity fixture
        doi="10.example/test",
        license="CC0",
    )
    verify_dataset_file(path, dataset)
    with pytest.raises(ValueError, match="size mismatch"):
        verify_dataset_file(
            path,
            DatasetFile(**{**dataset.__dict__, "size_bytes": len(payload) + 1}),
        )


def test_public_dataset_registry_is_version_pinned():
    perturbation = DATASETS["gse274113"]
    assert perturbation.filename == "GSE274113_filtered.h5mu"
    assert perturbation.doi == "10.5281/zenodo.21986866"
    assert perturbation.url.endswith(
        "/records/21986866/files/GSE274113_filtered.h5mu?download=1"
    )
    assert perturbation.size_bytes == 31_154_968_237
    assert perturbation.md5 == "b1aecf4d4cacf3328c0b3147426f5d16"
