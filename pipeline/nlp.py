"""Entities, expense categories and summaries from the OCR text.

- Entities: spaCy's statistical NER for people and organisations, plus an EntityRuler for
  the structured fields (amounts, dates, document numbers, e-mails). Both setups are
  scored per entity type against the generated ground truth.
- Expense category: TF-IDF + LogisticRegression, scored with 5-fold cross-validation and
  exported to JSON for the live function.
- Summary: extractive TextRank over a TF-IDF cosine-similarity graph of the sentences.
"""

import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline

from common import MODELS, RESULTS, ground_truth, read_json, write_json

LABELS = ["ORG", "PERSON", "DATE", "MONEY", "DOC_ID", "EMAIL"]
STATISTICAL = {"ORG", "PERSON"}  # taken from spaCy's model; the rest come from the rules
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def regex(pattern: str) -> dict:
    return {"TEXT": {"REGEX": pattern}}


# Token patterns, written against how spaCy tokenises each format
# (for example "INV-2026-0042" -> "INV-2026", "-", "0042").
PATTERNS = [
    ("MONEY", [{"TEXT": "$"}, regex(r"^\d[\d,]*\.\d{2}$")]),
    ("DOC_ID", [regex(r"^INV-\d{4}$"), {"TEXT": "-"}, regex(r"^\d{4}$")]),
    ("DOC_ID", [regex(r"^(PO-\d{5}|RCPT-\d{6})$")]),
    ("DATE", [regex(r"^\d{4}$"), {"TEXT": "-"}, regex(r"^\d{2}$"), {"TEXT": "-"}, regex(r"^\d{2}$")]),
    ("DATE", [regex(r"^\d{1,2}$"), {"LOWER": {"IN": [m[:3] for m in MONTHS]}}, regex(r"^\d{4}$")]),
    ("DATE", [regex(r"^\d{2}[/.]\d{2}[/.]\d{4}$")]),
    ("DATE", [{"LOWER": {"IN": MONTHS}}, regex(r"^\d{1,2}$"), {"TEXT": ","}, regex(r"^\d{4}$")]),
    ("EMAIL", [{"LIKE_EMAIL": True}]),
]


def load_nlp(rules: bool = True) -> spacy.Language:
    nlp = spacy.load("en_core_web_sm")
    if rules:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns([{"label": label, "pattern": p, "id": "rule"} for label, p in PATTERNS])
    return nlp


def extract(nlp: spacy.Language, text: str) -> list[dict]:
    """Entities with character offsets; with rules on, only rule matches for structured types.

    Each OCR line is tagged on its own: lines are the page's layout units, and tagging the
    whole text let the statistical model run entities across them.
    """
    rules = "entity_ruler" in nlp.pipe_names
    lines = text.split("\n")
    starts = itertools.accumulate((len(line) + 1 for line in lines[:-1]), initial=0)
    out = []
    for start, doc in zip(starts, nlp.pipe(lines)):
        for ent in doc.ents:
            if ent.label_ not in LABELS:
                continue
            if rules and ent.ent_id_ != "rule" and ent.label_ not in STATISTICAL:
                continue
            out.append({"label": ent.label_, "text": ent.text,
                        "start": start + ent.start_char, "end": start + ent.end_char})
    return out


def norm(text: str) -> str:
    """Compare values, not formatting: '$1,234.56' and '1,234.56' are the same amount."""
    return " ".join(text.lower().replace("$", "").split()).strip(" .,;:")


def score(gold: list[list[dict]], predicted: list[list[dict]]) -> pd.DataFrame:
    """Per-type precision, recall and F1, matching (type, value) multisets per document."""
    counts = {label: Counter() for label in LABELS}
    for g_doc, p_doc in zip(gold, predicted):
        for label in LABELS:
            g = Counter(norm(e["text"]) for e in g_doc if e["label"] == label)
            p = Counter(norm(e["text"]) for e in p_doc if e["label"] == label)
            counts[label].update(tp=sum((g & p).values()), gold=sum(g.values()), pred=sum(p.values()))
    counts["ALL"] = sum(counts.values(), Counter())
    rows = []
    for label, c in counts.items():
        precision = c["tp"] / c["pred"] if c["pred"] else 0.0
        recall = c["tp"] / c["gold"] if c["gold"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": label, "support": c["gold"], "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows).set_index("label")


def make_category_model() -> Pipeline:
    return make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True), LogisticRegression(max_iter=5000))


def export_category(model: Pipeline) -> dict:
    vectorizer, clf = model[0], model[-1]
    return {
        "classes": [str(c) for c in clf.classes_],
        "vocabulary": {term: int(i) for term, i in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(),
        "ngram_range": list(vectorizer.ngram_range),
        "sublinear_tf": vectorizer.sublinear_tf,
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
    }


def is_prose(line: str) -> bool:
    """A line of running text: several plain words, mostly lower-case (unlike headers and addresses)."""
    words = [w for w in line.split() if w.isalpha()]
    return len(words) >= 6 and sum(w[0].islower() for w in words) >= len(words) / 2


def sentences(text: str) -> list[str]:
    """Split into sentences, first re-joining prose that the page wrapped onto several lines."""
    blocks: list[str] = []
    previous = ""
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        if blocks and is_prose(previous) and not previous.endswith((".", "!", "?", ":")):
            blocks[-1] = f"{blocks[-1]} {line}"
        else:
            blocks.append(line)
        previous = line
    return [s for block in blocks for s in re.split(r"(?<=[.!?])\s+", block) if s]


def textrank(text: str, k: int = 2, damping: float = 0.85) -> str:
    """The k most central sentences, in document order (PageRank by power iteration).

    Full sentences are preferred; forms without enough of them fall back to their lines.
    """
    candidates = [s for s in sentences(text) if len(re.findall(r"\w\w+", s)) >= 4]
    full = [s for s in candidates if s.endswith((".", "!", "?"))]
    if len(full) > k:
        candidates = full
    if len(candidates) <= k:
        return " ".join(candidates)
    vectors = TfidfVectorizer().fit_transform(candidates)
    similarity = (vectors @ vectors.T).toarray()
    np.fill_diagonal(similarity, 0)
    n = len(candidates)
    totals = similarity.sum(axis=1, keepdims=True)
    # Row-normalise; a sentence with no neighbours links to every sentence equally.
    transition = np.where(totals > 0, similarity / np.where(totals > 0, totals, 1), 1 / n)
    scores = np.full(n, 1 / n)
    for _ in range(200):
        updated = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(updated - scores).sum() < 1e-10:
            break
        scores = updated
    top = sorted(np.argsort(-scores, kind="stable")[:k])
    return " ".join(candidates[i] for i in top)


def main() -> None:
    docs = ground_truth()
    ocr_text = read_json(RESULTS / "ocr_text.json")
    texts = [ocr_text[d["id"]]["deskewed"] for d in docs]
    gold = [d["entities"] for d in docs]

    nlp_plain, nlp_rules = load_nlp(rules=False), load_nlp()
    plain = score(gold, [extract(nlp_plain, t) for t in texts])
    predicted = [extract(nlp_rules, t) for t in texts]
    ruled = score(gold, predicted)
    ner = pd.concat({"spacy_only": plain, "with_rules": ruled}, axis=1)

    y = [d["category"] for d in docs]
    cv_scores = cross_val_score(make_category_model(), texts, y, cv=FOLDS)
    cv_predicted = cross_val_predict(make_category_model(), texts, y, cv=FOLDS)
    labels = sorted(set(y))
    matrix = confusion_matrix(y, cv_predicted, labels=labels)
    model = make_category_model().fit(texts, y)
    write_json(MODELS / "category.json", export_category(model), indent=None)

    write_json(RESULTS / "nlp.json", {
        "ner": [{"label": label, **{f"{setup}_{metric}": float(ner.loc[label, (setup, metric)])
                                    for setup in ("spacy_only", "with_rules")
                                    for metric in ("precision", "recall", "f1")},
                 "support": int(ruled.loc[label, "support"])} for label in ner.index],
        "category": {"cv_accuracy": float(cv_scores.mean()), "cv_std": float(cv_scores.std()),
                     "fold_accuracy": cv_scores.tolist(), "labels": labels,
                     "confusion": matrix.tolist(), "vocabulary_size": len(model[0].vocabulary_)},
    })
    write_json(RESULTS / "nlp_docs.json", {
        d["id"]: {"entities": ents, "summary": textrank(text)}
        for d, ents, text in zip(docs, predicted, texts)
    })

    print("Entity extraction on OCR text (P = precision, R = recall):")
    table = ner.loc[:, (slice(None), ["precision", "recall", "f1"])].copy()
    table.columns = [f"{setup}_{metric[0].upper()}" for setup, metric in table.columns]
    table.insert(0, "support", ruled["support"])
    print(table.round(3).to_string())
    print(f"\nExpense category, TF-IDF + LogisticRegression ({len(model[0].vocabulary_)} terms):")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print("Confusion matrix (rows = true, columns = predicted):", labels)
    for label, row in zip(labels, matrix):
        print(f"  {label:<16} {row.tolist()}")
    print(f"\nExample summary ({docs[1]['id']}, {docs[1]['type']}): {textrank(texts[1])}")


if __name__ == "__main__":
    main()
