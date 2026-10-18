"""The NumPy-only inference used by the live function must match scikit-learn."""

import json

import numpy as np
from sklearn.datasets import make_classification

import infer
import train_doctype


def test_doctype_inference_matches_sklearn(tmp_path):
    x, y_idx = make_classification(
        n_samples=120, n_features=60, n_informative=20, n_classes=4, random_state=0
    )
    y = np.array(["invoice", "letter", "order_form", "receipt"])[y_idx]
    model = train_doctype.make_model().fit(x[:90], y[:90])
    path = tmp_path / "doctype.json"
    path.write_text(json.dumps(train_doctype.export(model)))

    exported = infer.load_model(path)
    np.testing.assert_allclose(infer.predict_proba(exported, x[90:]), model.predict_proba(x[90:]), atol=1e-9)
    assert infer.predict(exported, x[90:]) == model.predict(x[90:]).tolist()
