
"""Load Ronglat model and corpus assets from local storage or Vercel Blob."""

import os
from pathlib import Path

from vercel.blob import sync as blob

REQUIRED_ASSETS = (
    "models/pos_char_bilstm_crf.pt",
    "models/ner_char_bilstm_crf.pt",
    "results/pos_char_bilstm_crf_results.json",
    "results/ner_char_bilstm_crf_results.json",
    "data/Phase_2B8_Final_Corpus.csv",
)

def get_asset_root(project_root: Path | str) -> Path:
    configured_root = os.getenv("RONGLAT_ASSET_DIR")

    if configured_root:
        root = Path(configured_root).expanduser().resolve()
    elif os.getenv("VERCEL"):
        root = Path("/tmp/ronglat-private-assets")
    else:
        root = Path(project_root).resolve()

    missing = [
        name for name in REQUIRED_ASSETS
        if not (root / name).is_file()
    ]

    if not missing:
        return root

    # Local development should use the original local assets.
    if not os.getenv("VERCEL") and not configured_root:
        raise RuntimeError(
            "Missing local Ronglat assets: " + ", ".join(missing)
        )

    if not os.getenv("BLOB_READ_WRITE_TOKEN") and not os.getenv("VERCEL_OIDC_TOKEN"):
        raise RuntimeError(
            "Vercel Blob credentials are unavailable. "
            "Configure BLOB_READ_WRITE_TOKEN or Vercel OIDC."
        )

    for relative_name in missing:
        destination = root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Downloading private asset: {relative_name}")

            result = blob.get_blob(
                relative_name,
                access="private",
            )

            if result is None or result.body is None:
                raise FileNotFoundError(relative_name)

            temporary = destination.with_suffix(
                destination.suffix + ".tmp"
            )

            try:
                temporary.write_bytes(result.body)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

        except Exception as exc:
            raise RuntimeError(
                f"Unable to download private asset: {relative_name}. "
                "Check the Blob store, credentials and pathname."
            ) from exc

    return root
