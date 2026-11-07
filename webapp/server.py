import csv
import io
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request
)

from inference import Predictor

ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__)

_predictor = None
_dictionary = None
_dictionary_source = None
_vocabulary_rows = None
_vocabulary_stats = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor(ROOT)
    return _predictor


def get_dictionary():
    """Load token/gloss pairs from the project corpus."""
    global _dictionary, _dictionary_source

    if _dictionary is not None:
        return _dictionary

    filename = "Phase_2B8_Final_Corpus.csv"
    candidates = [
        ROOT / "data" / filename,
        ROOT / filename,
    ]
    corpus_path = next((p for p in candidates if p.is_file()), None)

    if corpus_path is None:
        _dictionary = {}
        _dictionary_source = None
        return _dictionary

    try:
        df = pd.read_csv(
            corpus_path,
            usecols=["token", "gloss"],
            encoding="utf-8-sig",
        )
        df = df.dropna(subset=["token"]).copy()
        df["token"] = df["token"].astype(str).str.strip()
        df["gloss"] = df["gloss"].fillna("").astype(str).str.strip()
        df = df[(df["token"] != "")]
        df["lookup_token"] = df["token"].str.casefold()

        dictionary = {}
        for token, group in df.groupby("lookup_token", sort=False):
            dictionary[token] = " / ".join(
                dict.fromkeys(group["gloss"].tolist())
            )

        _dictionary = dictionary
        _dictionary_source = str(corpus_path)
    except Exception as exc:
        app.logger.exception("Could not load corpus dictionary")
        _dictionary = {}
        _dictionary_source = f"ERROR: {exc}"

    return _dictionary

def get_vocabulary_data():

    global _vocabulary_rows, _vocabulary_stats

    if _vocabulary_rows is not None:
        return _vocabulary_rows, _vocabulary_stats

    filename = "Phase_2B8_Final_Corpus.csv"

    candidates = [
        ROOT / "data" / filename,
        ROOT / filename,
        ROOT / "data" / "raw" / filename
    ]

    corpus_path = next(
        (p for p in candidates if p.is_file()),
        None
    )

    if corpus_path is None:

        _vocabulary_rows = []

        _vocabulary_stats = {
            "corpus_rows": 0,
            "unique_entries": 0,
            "entries_with_gloss": 0
        }

        return _vocabulary_rows, _vocabulary_stats

    try:

        df = pd.read_csv(
            corpus_path,
            usecols=["token", "gloss"],
            encoding="utf-8-sig"
        )

        corpus_rows = len(df)

        df = df.dropna(
            subset=["token", "gloss"]
        ).copy()

        df["token"] = (
            df["token"]
            .astype(str)
            .str.strip()
        )

        df["gloss"] = (
            df["gloss"]
            .astype(str)
            .str.strip()
        )

        df = df[df["token"] != ""]

        df["lookup_token"] = (
            df["token"].str.casefold()
        )

        grouped = []

        for lookup_token, group in df.groupby(
            "lookup_token",
            sort=False
        ):

            display_token = group["token"].iloc[0]

            glosses = list(
    dict.fromkeys(
        gloss
        for gloss in group["gloss"].tolist()
        if gloss
    )
)

            grouped.append({

                "token": display_token,

                "gloss": " / ".join(glosses),

                "frequency": int(len(group)),

                "lookup_token": lookup_token

            })

        # Most frequent tokens first
        grouped.sort(
            key=lambda item: (
                -item["frequency"],
                item["lookup_token"]
            )
        )

        _vocabulary_rows = grouped

        _vocabulary_stats = {

            "corpus_rows": int(corpus_rows),

            "unique_entries": len(grouped),

            "entries_with_gloss": sum(
                1 for item in grouped
                if item["gloss"]
            )

        }

    except Exception:

        app.logger.exception(
            "Could not prepare vocabulary data"
        )

        _vocabulary_rows = []

        _vocabulary_stats = {
            "corpus_rows": 0,
            "unique_entries": 0,
            "entries_with_gloss": 0
        }

    return _vocabulary_rows, _vocabulary_stats

def extract_entities(rows):
    entities = []
    current_tokens = []
    current_type = None

    def save_current():
        if current_tokens and current_type:
            entities.append({
                "text": " ".join(current_tokens),
                "type": current_type,
            })

    for row in rows:
        token = row["token"]
        tag = row["ner"]

        if tag == "O" or not tag:
            save_current()
            current_tokens = []
            current_type = None
        elif tag.startswith("B-"):
            save_current()
            current_tokens = [token]
            current_type = tag[2:]
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if current_tokens and current_type == entity_type:
                current_tokens.append(token)
            else:
                save_current()
                current_tokens = [token]
                current_type = entity_type
        else:
            save_current()
            current_tokens = []
            current_type = None

    save_current()
    return entities


@app.get("/")
def index():
    predictor = get_predictor()

    dictionary = get_dictionary()

    vocabulary_rows, vocabulary_stats = (
        get_vocabulary_data()
    )
    
    return render_template(
        "index.html",
        metrics={
            "pos_accuracy": round(predictor.pm["accuracy"] * 100, 2),
            "pos_macro_f1": round(predictor.pm["macro_f1"] * 100, 2),
            "ner_entity_f1": round(predictor.nm["entity_f1"] * 100, 2),
            "ner_token_accuracy": round(
                predictor.nm["token_accuracy"] * 100, 2
            ),
        },
        dictionary_count=len(dictionary),
        dictionary_available=bool(dictionary),
        dictionary_source=_dictionary_source,
        vocabulary_stats=vocabulary_stats,
    )


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    source = str(payload.get("text", "")).strip()

    if not source:
        return jsonify({"error": "Please enter Ronglat text."}), 400

    try:
        predictor = get_predictor()
        rows = predictor.analyze(source)
        if not rows:
            return jsonify({"error": "No tokens were returned."}), 422

        dictionary = get_dictionary()
        results = []

        for row in rows:
            token = row["token"]
            gloss = dictionary.get(token.strip().casefold(), "")
            results.append({
                "token": token,
                "gloss": gloss or "",
                "pos": row["pos"],
                "ner": row["ner"],
                "known": bool(row["known_word"]),
            })

        entities = extract_entities(rows)
        pos_counts = {}
        for row in rows:
            pos_counts[row["pos"]] = pos_counts.get(row["pos"], 0) + 1

        return jsonify({
            "results": results,
            "entities": entities,
            "pos_distribution": pos_counts,
            "summary": {
                "tokens": len(rows),
                "known": sum(1 for row in rows if row["known_word"]),
                "unseen": sum(1 for row in rows if not row["known_word"]),
                "entities": len(entities),
                "glossed": sum(1 for item in results if item["gloss"]),
            },
        })

    except Exception as exc:
        app.logger.exception("Analysis failed")
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

@app.get("/api/vocabulary")
def vocabulary():

    rows, stats = get_vocabulary_data()

    query = request.args.get(
        "q", ""
    ).strip().casefold()

    try:

        page = max(
            1,
            int(request.args.get("page", 1))
        )

        per_page = min(
            100,
            max(
                10,
                int(request.args.get("per_page", 25))
            )
        )

    except ValueError:

        return jsonify({
            "error": "Invalid pagination values."
        }), 400

    # Search Ronglat token or English gloss
    if query:

        filtered = [

            item for item in rows

            if (
                query in item["lookup_token"]
                or query in item["gloss"].casefold()
            )

        ]

    else:

        filtered = rows

    total = len(filtered)

    pages = max(
        1,
        (total + per_page - 1) // per_page
    )

    page = min(page, pages)

    start = (page - 1) * per_page

    page_items = [

        {
            "token": item["token"],
            "gloss": item["gloss"],
            "frequency": item["frequency"]
        }

        for item in filtered[
            start:start + per_page
        ]

    ]

    return jsonify({

        "items": page_items,

        "stats": stats,

        "pagination": {

            "page": page,

            "per_page": per_page,

            "total": total,

            "pages": pages

        }

    })
@app.get("/api/vocabulary/export")
def export_vocabulary():

    rows, _ = get_vocabulary_data()

    output = io.StringIO()

    writer = csv.DictWriter(
        output,
        fieldnames=[
            "Ronglat token",
            "English gloss",
            "Corpus frequency"
        ]
    )

    writer.writeheader()

    for item in rows:

        writer.writerow({

            "Ronglat token": item["token"],

            "English gloss": item["gloss"],

            "Corpus frequency": item["frequency"]

        })

    return Response(

        output.getvalue(),

        mimetype="text/csv; charset=utf-8",

        headers={

            "Content-Disposition":
            "attachment; filename=ronglat_corpus_vocabulary.csv"

        }

    )

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Local development server. Do not expose Flask's debug server publicly.
    app.run(host="127.0.0.1", port=5000, debug=True)
