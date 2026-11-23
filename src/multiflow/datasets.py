"""Version-pinned public dataset downloads used in MultiFlow tutorials."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetFile:
    name: str
    filename: str
    url: str
    size_bytes: int
    md5: str
    doi: str
    license: str


OPENPROBLEM = DatasetFile(
    name="openproblem",
    filename="openproblem_filtered.h5mu",
    url="https://ndownloader.figshare.com/files/52945733",
    size_bytes=8_377_659_018,
    md5="88d5b9906e984678d170bb84c91a2c4f",
    doi="10.6084/m9.figshare.28582061.v3",
    license="CC BY 4.0",
)

GSE274113 = DatasetFile(
    name="gse274113",
    filename="GSE274113_filtered.h5mu",
    url=(
        "https://zenodo.org/records/21986866/files/"
        "GSE274113_filtered.h5mu?download=1"
    ),
    size_bytes=31_154_968_237,
    md5="b1aecf4d4cacf3328c0b3147426f5d16",
    doi="10.5281/zenodo.21986866",
    license="See the Zenodo record and the GSE274113 source-data terms",
)

DATASETS = {
    OPENPROBLEM.name: OPENPROBLEM,
    GSE274113.name: GSE274113,
}


def md5_file(path: str | os.PathLike[str], *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - used only for upstream file integrity
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_file(path: str | os.PathLike[str], dataset: DatasetFile) -> None:
    source = Path(path)
    observed_size = source.stat().st_size
    if observed_size != dataset.size_bytes:
        raise ValueError(
            f"downloaded size mismatch: expected {dataset.size_bytes}, observed {observed_size}"
        )
    observed_md5 = md5_file(source)
    if observed_md5.lower() != dataset.md5.lower():
        raise ValueError(
            f"downloaded MD5 mismatch: expected {dataset.md5}, observed {observed_md5}"
        )


def download_dataset(
    name: str,
    output: str | os.PathLike[str],
    *,
    accept_license: bool,
    force: bool = False,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Download, resume, verify, and atomically publish a known dataset file."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    dataset = DATASETS[name]
    if not accept_license:
        raise ValueError(
            f"{name} is distributed under {dataset.license}; pass --accept-license "
            "after reviewing the source and license"
        )
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        verify_dataset_file(destination, dataset)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() and not force else 0
    if force and partial.exists():
        partial.unlink()
    request = urllib.request.Request(dataset.url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:  # noqa: S310 - pinned HTTPS URL
        status = getattr(response, "status", None)
        append = bool(offset and status == 206)
        mode = "ab" if append else "wb"
        if not append:
            offset = 0
        downloaded = offset
        with partial.open(mode) as handle:
            while True:
                block = response.read(chunk_size)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                percent = 100.0 * downloaded / dataset.size_bytes
                print(
                    f"\rdownloaded {downloaded / 1e9:.2f}/{dataset.size_bytes / 1e9:.2f} GB "
                    f"({percent:.1f}%)",
                    end="",
                    flush=True,
                )
    print()
    verify_dataset_file(partial, dataset)
    os.replace(partial, destination)
    return destination
