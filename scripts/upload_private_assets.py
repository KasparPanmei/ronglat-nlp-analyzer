"""Upload the five runtime assets to a PRIVATE Vercel Blob store.

Run locally after setting BLOB_READ_WRITE_TOKEN:
    python scripts/upload_private_assets.py

The files are uploaded with stable pathnames expected by webapp/runtime_assets.py.
"""
from pathlib import Path
import os
import sys

from vercel.blob import sync as blob

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    "models/pos_char_bilstm_crf.pt",
    "models/ner_char_bilstm_crf.pt",
    "results/pos_char_bilstm_crf_results.json",
    "results/ner_char_bilstm_crf_results.json",
    "data/Phase_2B8_Final_Corpus.csv",
)


def main():
    if not (os.getenv("BLOB_READ_WRITE_TOKEN") or os.getenv("BLOB_STORE_ID")):
        raise SystemExit(
            "Set BLOB_READ_WRITE_TOKEN (or configure local Vercel Blob OIDC credentials) first."
        )

    for relative_name in ASSETS:
        source = ROOT / relative_name
        if not source.is_file():
            raise SystemExit(f"Required local asset is missing: {source}")

        with source.open("rb") as file_handle:
            result = blob.put_blob(
                relative_name,
                file_handle,
                access="private",
                add_random_suffix=False,
                allow_overwrite=True,
                content_length=source.stat().st_size,
            )
        print(f"Uploaded {relative_name} -> {result.pathname}")

    print("\nPrivate runtime assets uploaded successfully.")


if __name__ == "__main__":
    main()
