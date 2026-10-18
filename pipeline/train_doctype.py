"""Train the document-type classifier on HOG features and export it to JSON.

Scores StandardScaler + LogisticRegression with 5-fold stratified cross-validation, on
deskewed pages and, for comparison, on the raw scans.
"""

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DOCS, MODELS, RESULTS, ground_truth, write_json
from vision import HOG_SIZE, hog_features

FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def make_model() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def export(model: Pipeline) -> dict:
    """The fitted scaler and coefficients as plain lists, for NumPy-only inference."""
    scaler, clf = model[0], model[-1]
    return {
        "classes": [str(c) for c in clf.classes_],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "hog": {"size": HOG_SIZE, "block": 32, "stride": 16, "cell": 16, "bins": 9},
    }


def main() -> None:
    docs = ground_truth()
    y = np.array([d["type"] for d in docs])
    images = [cv2.imread(str(DOCS / f"{d['id']}.png"), cv2.IMREAD_GRAYSCALE) for d in docs]
    x_straight = np.array([hog_features(img) for img in images])
    x_raw = np.array([hog_features(img, straighten=False) for img in images])

    scores = cross_val_score(make_model(), x_straight, y, cv=FOLDS)
    raw_scores = cross_val_score(make_model(), x_raw, y, cv=FOLDS)
    predicted = cross_val_predict(make_model(), x_straight, y, cv=FOLDS)
    labels = sorted(str(label) for label in set(y))
    matrix = confusion_matrix(y, predicted, labels=labels)

    write_json(RESULTS / "doctype.json", {
        "cv_accuracy": float(scores.mean()),
        "cv_std": float(scores.std()),
        "fold_accuracy": scores.tolist(),
        "raw_cv_accuracy": float(raw_scores.mean()),
        "labels": labels,
        "confusion": matrix.tolist(),
        "n_features": int(x_straight.shape[1]),
    })
    model = make_model().fit(x_straight, y)
    write_json(MODELS / "doctype.json", export(model), indent=None)

    print(f"HOG features: {x_straight.shape[1]} per page, {len(docs)} pages")
    print(f"5-fold CV accuracy (deskewed): {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"5-fold CV accuracy (raw scans): {raw_scores.mean():.3f} +/- {raw_scores.std():.3f}")
    print("Confusion matrix (rows = true, columns = predicted):", labels)
    for label, row in zip(labels, matrix):
        print(f"  {label:<11} {row.tolist()}")
    print(f"Wrote {MODELS / 'doctype.json'}")


if __name__ == "__main__":
    main()
