import json
import random

import numpy as np
import pytest

import infer
import nlp
from generate import ITEMS


@pytest.fixture(scope="module")
def ruled():
    return nlp.load_nlp()


def test_category_inference_matches_sklearn(tmp_path):
    rng = random.Random(0)
    texts, labels = [], []
    for category, items in ITEMS.items():
        for _ in range(8):
            names = [name for name, *_ in rng.sample(items, 3)]
            texts.append(f"Receipt {' '.join(names)} total ${rng.uniform(5, 500):.2f} thank you")
            labels.append(category)
    model = nlp.make_category_model().fit(texts[::2], labels[::2])
    path = tmp_path / "category.json"
    path.write_text(json.dumps(nlp.export_category(model)))

    exported = infer.load_model(path)
    test = texts[1::2] + ["words the vocabulary has never seen", ""]
    np.testing.assert_allclose(infer.tfidf(exported, test), model[0].transform(test).toarray(), atol=1e-12)
    np.testing.assert_allclose(
        infer.predict_proba(exported, infer.tfidf(exported, test)), model.predict_proba(test), atol=1e-9
    )


def test_rules_find_structured_fields(ruled):
    text = (
        "Invoice No: INV-2026-0042\nInvoice date: 14 Mar 2026\nDue date: 2026-04-13\n"
        "Ordered 03/14/2026, delivered on March 20, 2026 (order PO-58231)\n"
        "Total due $1,234.56\nWrite to billing@acme.example."
    )
    found = {(e["label"], e["text"]) for e in nlp.extract(ruled, text)}
    assert {
        ("DOC_ID", "INV-2026-0042"), ("DOC_ID", "PO-58231"), ("DATE", "14 Mar 2026"),
        ("DATE", "2026-04-13"), ("DATE", "03/14/2026"), ("DATE", "March 20, 2026"),
        ("MONEY", "$1,234.56"), ("EMAIL", "billing@acme.example"),
    } <= found


def test_entity_offsets_point_into_the_text(ruled):
    text = "Bill to: Jane Smith\nTotal $12.00 due 14.03.2026"
    for e in nlp.extract(ruled, text):
        assert text[e["start"] : e["end"]] == e["text"]


def test_score_counts_matches_per_type():
    gold = [[{"label": "MONEY", "text": "$5.00"}, {"label": "ORG", "text": "Acme Deli"}]]
    predicted = [[{"label": "MONEY", "text": "5.00"}, {"label": "PERSON", "text": "Acme Deli"}]]
    table = nlp.score(gold, predicted)
    assert table.loc["MONEY", "f1"] == 1.0
    assert table.loc["ORG", "recall"] == 0.0
    assert table.loc["PERSON", "precision"] == 0.0
    assert table.loc["ALL", "precision"] == 0.5


def test_sentences_rejoin_wrapped_prose():
    text = "ACME DELI\nThank you for choosing Acme Deli to cater your team\nlunch on May 2. The bill is due.\nTotal $5.00"
    assert nlp.sentences(text) == [
        "ACME DELI",
        "Thank you for choosing Acme Deli to cater your team lunch on May 2.",
        "The bill is due.",
        "Total $5.00",
    ]


def test_sentences_keep_address_lines_apart():
    text = "956 Edward Glens, Leonardmouth, AR 34206\nTel (555) 744-8268 accounts@acme.example\nRe: Catering for your team event"
    assert nlp.sentences(text) == text.split("\n")


def test_textrank_returns_two_input_sentences_in_order():
    text = (
        "The team lunch was catered by Acme Deli on Friday. "
        "Acme Deli served sandwiches and salads for the team lunch. "
        "Parking is available behind the building for visitors. "
        "The invoice for the team lunch from Acme Deli is attached."
    )
    summary = nlp.textrank(text)
    picked = [s for s in nlp.sentences(text) if s in summary]
    assert len(picked) == 2
    assert summary == " ".join(picked)
    assert "Parking" not in summary
