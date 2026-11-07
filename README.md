# Ronglat NLP Analyzer — HTML/CSS frontend with Python backend

This version separates the interface from the NLP processing:

- HTML: `webapp/templates/index.html`
- CSS: `webapp/static/css/style.css`
- Browser interactions: `webapp/static/js/app.js`
- Python / Flask API and model inference: `webapp/server.py`
- Existing `inference.py`, trained model files, and corpus remain in your project.

## Expected project layout

```text
YourProject/
├── data/
│   └── Phase_2B8_Final_Corpus.csv
├── models/                       # keep your existing model files here
├── webapp/
│   ├── inference.py              # keep your existing inference.py here
│   ├── server.py                 # included in this package
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
└── requirements-web.txt
```

If your current `inference.py` is in a different directory, keep it where it is and adjust the import in `server.py` or move `server.py` alongside it.

## Setup

From your project root in PowerShell:

```powershell
# Activate your existing virtual environment
.venv\Scripts\Activate.ps1

# Install the web server dependencies
pip install -r requirements-web.txt

# Start the app
python webapp/server.py
```

Open `http://127.0.0.1:5000` in your browser.

## Important

1. Keep your trained model checkpoints and configuration files unchanged.
2. Put `Phase_2B8_Final_Corpus.csv` in either the project root or `data/`.
3. This uses the same `Predictor(ROOT)` interface and expects `predictor.analyze(text)` to return rows containing `token`, `pos`, `ner`, and `known_word`.
4. The displayed English items are corpus-derived word glosses, not fluent sentence translations.
5. The Flask debug server is for local development only. Do not expose it directly to the public internet.
