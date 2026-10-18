"""Rebuild data/paper_trail.db from the pipeline outputs, inside one transaction.

The database is built in a temporary file and moved into place only after the commit,
so a failed run leaves the previous database untouched. Also writes data/samples.json:
the live function's analysis, cached for the gallery documents.
"""

import csv
import json
import os
import sqlite3

import cv2

import infer
from common import DB_PATH, DOCS, MODELS, RESULTS, SQL, VISION, ground_truth, read_json, write_json
from ocr import MODEL
from vision import gallery_ids, hog_features

TABLES = ["documents", "blocks", "entities", "ocr_runs", "metrics", "documents_fts"]


def metric_rows() -> list[tuple[str, float, str | None]]:
    doctype = read_json(RESULTS / "doctype.json")
    deskew = read_json(RESULTS / "deskew.json")
    ocr = read_json(RESULTS / "ocr.json")
    nlp = read_json(RESULTS / "nlp.json")

    def row(name, value, **detail):
        return name, float(value), json.dumps(detail) if detail else None

    rows = [
        row("doctype_cv_accuracy", doctype["cv_accuracy"], std=doctype["cv_std"],
            folds=doctype["fold_accuracy"], labels=doctype["labels"],
            confusion=doctype["confusion"], features=doctype["n_features"]),
        row("doctype_cv_accuracy_raw_scans", doctype["raw_cv_accuracy"]),
        row("category_cv_accuracy", nlp["category"]["cv_accuracy"], std=nlp["category"]["cv_std"],
            folds=nlp["category"]["fold_accuracy"], labels=nlp["category"]["labels"],
            confusion=nlp["category"]["confusion"], vocabulary=nlp["category"]["vocabulary_size"]),
    ]
    for setting, d in deskew.items():
        rows.append(row(f"deskew_error_deg_{setting}", d["mae_degrees"], max=d["max_error_degrees"],
                        documents=d["documents"]))
    for r in ocr["summary"]:
        for variant in ("raw", "deskewed"):
            rows.append(row(f"ocr_cer_{r['setting']}_{variant}", r[variant], model=ocr["model"]))
    for r in nlp["ner"]:
        rows.append(row(f"ner_f1_{r['label']}", r["with_rules_f1"], **{k: v for k, v in r.items() if k != "label"}))
    return rows


def sample_results(docs: list[dict], ocr_text: dict) -> list[dict]:
    """What /api/analyze returns, computed once for each gallery document."""
    doctype, category = infer.load_model(MODELS / "doctype.json"), infer.load_model(MODELS / "category.json")
    out = []
    for d in docs:
        gray = cv2.imread(str(DOCS / f"{d['id']}.png"), cv2.IMREAD_GRAYSCALE)
        text = ocr_text[d["id"]]["deskewed"]
        type_p = infer.predict_proba(doctype, hog_features(gray))[0]
        category_p = infer.predict_proba(category, infer.tfidf(category, [text]))[0]
        vision = read_json(VISION / f"{d['id']}.json")
        out.append({
            "sample_id": d["id"],
            "type": doctype["classes"][type_p.argmax()],
            "type_scores": {c: round(float(p), 4) for c, p in zip(doctype["classes"], type_p)},
            "category": category["classes"][category_p.argmax()],
            "category_scores": {c: round(float(p), 4) for c, p in zip(category["classes"], category_p)},
            "skew_deg": round(vision["angle"], 2),
            "blocks": len(vision["blocks"]),
            "text": text,
            "overlay": f"/samples/{d['id']}_overlay.png",
            "original": f"/samples/{d['id']}_original.png",
            "model": MODEL,
        })
    return out


def main() -> None:
    docs = ground_truth()
    ocr_text = read_json(RESULTS / "ocr_text.json")
    nlp_docs = read_json(RESULTS / "nlp_docs.json")
    with open(RESULTS / "ocr_cer.csv", newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    cer = {(r["id"], r["setting"], r["variant"]): float(r["cer"]) for r in runs}
    gallery = set(gallery_ids(docs))

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp, autocommit=True)
    con.execute("PRAGMA foreign_keys = ON")  # a no-op once a transaction is open
    con.autocommit = False  # opens the one transaction that everything below runs in
    try:
        con.executescript((SQL / "schema.sql").read_text(encoding="utf-8"))
        for d in docs:
            vision = read_json(VISION / f"{d['id']}.json")
            con.execute(
                "INSERT INTO documents (id, type, category, vendor, number, total, issued_on, ocr_text,"
                " summary, cer, skew_deg, in_gallery) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (d["id"], d["type"], d["category"], d["vendor"], d["number"], d["total"], d["issued_on"],
                 ocr_text[d["id"]]["deskewed"], nlp_docs[d["id"]]["summary"],
                 cer[(d["id"], "standard", "deskewed")], vision["angle"], int(d["id"] in gallery)),
            )
            con.executemany(
                "INSERT INTO blocks (doc_id, x, y, w, h) VALUES (?, ?, ?, ?, ?)",
                [(d["id"], *box) for box in vision["blocks"]],
            )
            con.executemany(
                "INSERT INTO entities (doc_id, label, text, start_char, end_char) VALUES (?, ?, ?, ?, ?)",
                [(d["id"], e["label"], e["text"], e["start"], e["end"]) for e in nlp_docs[d["id"]]["entities"]],
            )
        con.executemany(
            "INSERT INTO ocr_runs (doc_id, setting, variant, cer) VALUES (?, ?, ?, ?)",
            [(doc_id, setting, variant, value) for (doc_id, setting, variant), value in cer.items()],
        )
        con.executemany("INSERT INTO metrics (name, value, detail) VALUES (?, ?, ?)", metric_rows())
        con.execute("INSERT INTO documents_fts (documents_fts) VALUES ('rebuild')")
        problems = con.execute("PRAGMA foreign_key_check").fetchall()
        if problems:
            raise RuntimeError(f"Foreign key violations: {problems[:5]}")
        con.commit()
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()
    os.replace(tmp, DB_PATH)
    write_json(DB_PATH.parent / "samples.json", sample_results([d for d in docs if d["id"] in gallery], ocr_text))

    print(f"Rebuilt {DB_PATH}")
    for table, n in counts.items():
        print(f"  {table:<14} {n:>5} rows")
    print(f"Wrote {DB_PATH.parent / 'samples.json'} ({len(gallery)} cached analyses)")


if __name__ == "__main__":
    main()
