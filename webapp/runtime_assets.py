"""Resolve Ronglat runtime assets for local development and Vercel production.

Local development reads the five files from the project (or RONGLAT_ASSET_DIR).
Vercel downloads missing files from the connected private Blob store into /tmp.
"""

from __future__ import annotations

import os
from pathlib import Path

REQUIRED_ASSETS = (
    "models/pos_char_bilstm_crf.pt",
    "models/ner_char_bilstm_crf.pt",
    "results/pos_char_bilstm_crf_results.json",
    "results/ner_char_bilstm_crf_results.json",
    "data/Phase_2B8_Final_Corpus.csv",
)


def _missing_assets(root: Path) -> list[str]:
    return [name for name in REQUIRED_ASSETS if not (root / name).is_file()]


def get_asset_root(project_root: Path | str) -> Path:
    """Return a directory containing every required inference asset."""
    configured_root = os.getenv("RONGLAT_ASSET_DIR")

    if os.getenv("VERCEL"):
        root = Path("/tmp/ronglat-private-assets")
    elif configured_root:
        root = Path(configured_root).expanduser().resolve()
    else:
        root = Path(project_root).resolve()

    missing = _missing_assets(root)
    if not missing:
        return root

    # Local development must remain independent of Vercel and Blob.
    if not os.getenv("VERCEL"):
        raise RuntimeError(
            "Missing local Ronglat assets in "
            f"'{root}': {', '.join(missing)}. "
            "Restore the files or set RONGLAT_ASSET_DIR to their parent directory."
        )

    if not os.getenv("BLOB_READ_WRITE_TOKEN") and not os.getenv("VERCEL_OIDC_TOKEN"):
        raise RuntimeError(
            "Vercel Blob credentials are unavailable. Connect the private Blob "
            "store to this Vercel project and enable its read/write token."
        )

    # Import only in production: local startup does not need the Vercel SDK.
    try:
        from vercel.blob import sync as blob
    except ImportError as exc:
        raise RuntimeError(
            "The Vercel Blob SDK is not installed in the production function."
        ) from exc

    for relative_name in missing:
        destination = root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")

        try:
            print(f"Downloading private Ronglat asset: {relative_name}")
            result = blob.get_blob(relative_name, access="private")
            if result is None or result.body is None:
                raise FileNotFoundError(
                    f"Private Blob asset returned no content: {relative_name}"
                )

            temporary.write_bytes(result.body)
            os.replace(temporary, destination)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download private Blob asset '{relative_name}'. "
                "Check the Blob pathname, store connection and credentials."
            ) from exc
        finally:
            if temporary.exists():
                temporary.unlink()

    remaining = _missing_assets(root)
    if remaining:
        raise RuntimeError(
            "Runtime assets are still missing after Blob download: "
            + ", ".join(remaining)
        )

    return root
