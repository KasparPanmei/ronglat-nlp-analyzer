# Ronglat NLP Analyzer

HTML/CSS/JavaScript frontend with a Python Flask backend for Ronglat POS tagging, NER and corpus vocabulary exploration.

## Local development

Keep the private runtime assets on your computer in this layout:

```text
models/
  pos_char_bilstm_crf.pt
  ner_char_bilstm_crf.pt
results/
  pos_char_bilstm_crf_results.json
  ner_char_bilstm_crf_results.json
data/
  Phase_2B8_Final_Corpus.csv
```

Install dependencies and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py
```

Open http://127.0.0.1:5000.

## Vercel deployment

1. Import this GitHub repository into Vercel. Use the repository root as the Root Directory.
2. Vercel detects the Flask WSGI application exported by the root-level `server.py`.
3. Create a **Private** Vercel Blob store in the Vercel project and connect it to this project for Production (and Preview if needed).
4. On your local computer, set `BLOB_READ_WRITE_TOKEN` to the store's read/write token. Do not commit this token.
5. Install the requirements locally and run `python scripts/upload_private_assets.py`. This uploads only the two Character-BiLSTM-CRF checkpoints, their two evaluation JSON files and the corpus CSV.
6. Redeploy. The Flask application downloads missing assets from the private Blob store into Vercel's temporary `/tmp` storage on first use.

The runtime uses the private Blob SDK with `access="private"`. The Blob store must be connected to the Vercel project so the Python SDK can authenticate with the project's Vercel credentials.

## Important

- Do not commit `models/`, `data/`, `results/`, model checkpoints, corpus files, or environment secrets.
- The repository's current branch excludes private data files. Removing them from the current branch does not erase earlier Git history. If this repository was public, consider making it private and cleaning Git history if you need to revoke access to previously committed assets.
- Vercel Functions have a 500 MB standard uncompressed bundle limit. PyTorch may require Vercel Large Functions support depending on the final dependency bundle size.
- Vercel Functions have a read-only application filesystem. Runtime assets are cached only in temporary `/tmp` storage and may need to be downloaded again after a cold start.
- English glosses are corpus-derived token lookups, not fluent sentence translations.
