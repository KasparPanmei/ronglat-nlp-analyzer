"""Load private NLP model and corpus assets from Vercel Blob when deployed."""
from __future__ import annotations

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

    # On a local machine, do not attempt remote downloads: show the missing paths.
    if not os.getenv("VERCEL") and not configured_root:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing local NLP assets: {missing_text}")

    if not (os.getenv("BLOB_STORE_ID") or os.getenv("BLOB_READ_WRITE_TOKEN")):
        raise RuntimeError(
            "Private Vercel Blob storage is not connected. Set up a private Blob "
            "store and connect it to this Vercel project."
        )

    try:
        from vercel.blob import sync as blob
    except ImportError as exc:
        raise RuntimeError(
            "The vercel-blob package is missing. Add it to requirements.txt."
        ) from exc

    for relative_name in missing:
        destination = root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloaded = blob.get_blob(relative_name, access="private")
            payload = downloaded.body
        except Exception as exc:
            raise RuntimeError(
                f"Could not download private asset '{relative_name}' from Vercel Blob. "
                "Check that the file exists and that the Blob store is connected."
            ) from exc

        # Write atomically so concurrent function requests do not see partial files.
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, delete=False
            ) as temporary:
                temporary.write(payload)
                temp_name = temporary.name
            os.replace(temp_name, destination)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    return root
