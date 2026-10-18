"""NumPy-only inference for models exported from scikit-learn to JSON.

The live function carries a copy of this logic (api/analyze.py), so it needs neither
scikit-learn nor SciPy; tests/test_parity.py checks both against scikit-learn.
"""

import json
import re
from pathlib import Path

import numpy as np


def load_model(path: Path) -> dict:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("mean", "scale", "coef", "intercept", "idf"):
        if key in model:
            model[key] = np.asarray(model[key], dtype=np.float64)
    return model


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def predict_proba(model: dict, x: np.ndarray) -> np.ndarray:
    """Multinomial logistic regression, with optional standardisation first."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if "mean" in model:
        x = (x - model["mean"]) / model["scale"]
    return softmax(x @ model["coef"].T + model["intercept"])


def predict(model: dict, x: np.ndarray) -> list[str]:
    return [model["classes"][i] for i in predict_proba(model, x).argmax(axis=1)]


TOKEN = re.compile(r"(?u)\b\w\w+\b")  # TfidfVectorizer's default token pattern


def tfidf(model: dict, texts: list[str]) -> np.ndarray:
    """TfidfVectorizer.transform with lowercase, word n-grams, smooth idf and l2 norm."""
    vocab, lo, hi = model["vocabulary"], *model["ngram_range"]
    out = np.zeros((len(texts), len(vocab)))
    for row, text in enumerate(texts):
        tokens = TOKEN.findall(text.lower())
        for n in range(lo, hi + 1):
            for i in range(len(tokens) - n + 1):
                j = vocab.get(" ".join(tokens[i : i + n]))
                if j is not None:
                    out[row, j] += 1
    if model.get("sublinear_tf"):
        seen = out > 0
        out[seen] = 1 + np.log(out[seen])
    out *= model["idf"]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)
