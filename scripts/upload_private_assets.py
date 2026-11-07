
"""Upload Ronglat runtime assets to a private Vercel Blob store."""

import os
from pathlib import Path

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
    if not os.getenv("BLOB_READ_WRITE_TOKEN"):
        raise SystemExit(
            "ERROR: BLOB_READ_WRITE_TOKEN is not set. "
            "Configure it before uploading."
        )

    for relative_name in ASSETS:
        source = ROOT / relative_name

        if not source.is_file():
            raise SystemExit(f"Missing required asset: {source}")

        print(f"Uploading: {relative_name}")

        uploaded = blob.put_blob(
            relative_name,
            source.read_bytes(),
            access="private",
            allow_overwrite=True,
        )

        print(f"Uploaded successfully: {uploaded.pathname}")

    print("\nAll five private assets uploaded successfully.")


if __name__ == "__main__":
    main()
