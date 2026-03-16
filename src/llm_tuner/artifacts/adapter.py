from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import certifi
import requests
import tinker


@dataclass
class AdapterDownloadResult:
    checkpoint_path: str
    archive_path: str
    extract_dir: str


def download_adapter_from_checkpoint(
    *,
    checkpoint_path: str,
    archive_path: str | Path,
    extract_dir: str | Path,
    base_url: str | None = None,
) -> AdapterDownloadResult:
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    sc = tinker.ServiceClient(base_url=base_url)
    rc = sc.create_rest_client()
    signed = rc.get_checkpoint_archive_url_from_tinker_path(checkpoint_path).result()

    with requests.get(signed.url, stream=True, timeout=300, verify=certifi.where()) as response:
        response.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=extract_dir, filter="data")

    return AdapterDownloadResult(
        checkpoint_path=checkpoint_path,
        archive_path=str(archive_path),
        extract_dir=str(extract_dir),
    )
