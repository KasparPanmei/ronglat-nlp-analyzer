"""Upload the runtime assets to a PRIVATE Vercel Blob store.

Run locally after setting BLOB_READ_WRITE_TOKEN:
    python scripts/upload_private_assets.py
"""
from pathlib import Path
import os

from vercel.blob import BlobClient

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    "models/pos_char_bilstm_crf.pt",
    "models/ner_char_bilstm_crf.pt",
    "results/pos_char_bilstm_crf_results.json",
    "results/ner_char_bilstm_crf_results.json",
    "data/Phase_2B8_Final_Corpus.csv",
)


def main():
    if not os.getenv("BLOB_READ_WRITE_TOKEN"):
        raise SystemExit("Set BLOB_READ_WRITE_TOKEN before uploading.")

    client = BlobClient()
    for relative_name in ASSETS:
        source = ROOT / relative_name
        if not source.is_file():
            raise SystemExit(f"Required local asset is missing: {source}")

        with source.open("rb") as file_handle:
            result = client.put(
                relative_name,
                file_handle.read(),
                access="private",
                add_random_suffix=False,
                overwrite=True,
            )
        print(f"Uploaded {relative_name} -> {result.pathname}")

    print("\nPrivate runtime assets uploaded successfully.")


if __name__ == "__main__":
    main()
