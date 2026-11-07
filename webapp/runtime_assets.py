"""Load private NLP model and corpus assets from Vercel Blob when deployed."""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

REQUIRED_ASSETS = (
    "models/pos_char_bilstm_crf.pt",
    "models/ner_char_bilstm_crf.pt",
    "results/pos_char_bilstm_crf_results.json",
    "results/ner_char_bilstm_crf_results.json",
    "data/Phase_2B8_Final_Corpus.csv",
)


async def _download_private_blob(pathname: str) -> bytes:
    from vercel.blob import AsyncBlobClient

    client = AsyncBlobClient()
    result = await client.get(pathname, access="private")
    if result is None or result.status_code != 200 or result.stream is None:
        raise FileNotFoundError(f"Private Blob asset not found: {pathname}")

    chunks = []
    async for chunk in result.stream:
        chunks.append(chunk)
    return b"".join(chunks)


def get_asset_root(project_root: Path | str) -> Path:
    """
    Local development reads assets from the project root (or RONGLAT_ASSET_DIR).
    Vercel downloads missing assets from a private Blob store into /tmp.
    """
    configured_root = os.getenv("RONGLAT_ASSET_DIR")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
    elif os.getenv("VERCEL"):
        root = Path("/tmp/ronglat-private-assets")
    else:
        root = Path(project_root).resolve()

    missing = [name for name in REQUIRED_ASSETS if not (root / name).is_file()]
    if not missing:
        return root

    if not os.getenv("VERCEL") and not configured_root:
        raise RuntimeError(
            "Missing local NLP assets: " + ", ".join(missing)
        )

    if not os.getenv("BLOB_READ_WRITE_TOKEN") and not os.getenv("VERCEL_OIDC_TOKEN") and not os.getenv("VERCEL"):
        raise RuntimeError(
            "Set BLOB_READ_WRITE_TOKEN or connect the private Blob store to Vercel."
        )

    for relative_name in missing:
        destination = root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            payload = asyncio.run(_download_private_blob(relative_name))
        except Exception as exc:
            raise RuntimeError(
                f"Could not download private asset '{relative_name}' from Vercel Blob. "
                "Check the private Blob store, pathname and project connection."
            ) from exc

        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, delete=False
            ) as temporary:
                temporary.write(payload)
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    return root
