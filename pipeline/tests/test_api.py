"""The live function (api/analyze.py) must preprocess and infer exactly like the pipeline,
validate uploads, and fall back to a cached example when limited or failing."""

import importlib.util
import random

import cv2
import numpy as np
import pytest
from faker import Faker
from PIL import Image

import infer
import vision
from common import MODELS, ROOT
from generate import ITEMS, make_record, make_vendors, render

spec = importlib.util.spec_from_file_location("analyze", ROOT / "api" / "analyze.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def scan(doc_type: str, angle: float) -> np.ndarray:
    fake = Faker("en_US")
    fake.seed_instance(3)
    rng = random.Random(3)
    vendor = make_vendors(fake, rng)["meals"][0]
    img, *_ = render(make_record(0, doc_type, "meals", vendor, "Jane Doe", rng), rng)
    return np.asarray(img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=245))


def png(gray: np.ndarray) -> bytes:
    return cv2.imencode(".png", gray)[1].tobytes()


@pytest.mark.parametrize("doc_type,angle", [("receipt", -6.0), ("invoice", 3.5), ("letter", 12.0), ("order_form", -1.0)])
def test_preprocessing_matches_pipeline(doc_type, angle):
    gray = scan(doc_type, angle)
    straight, a = api.deskew(gray)
    expected, b = vision.deskew(gray)
    assert a == b
    assert np.array_equal(straight, expected)
    assert api.text_blocks(straight) == vision.text_blocks(expected)
    assert np.array_equal(api.hog_features(straight), vision.hog_features(gray))


def test_inference_matches_pipeline():
    rng = np.random.default_rng(0)
    doctype = infer.load_model(MODELS / "doctype.json")
    x = rng.normal(size=(5, len(doctype["mean"])))
    np.testing.assert_allclose(api.predict_proba(api.DOCTYPE, x), infer.predict_proba(doctype, x))

    category = infer.load_model(MODELS / "category.json")
    texts = [" ".join(name for name, *_ in items) for items in ITEMS.values()] + [""]
    for text in texts:
        np.testing.assert_allclose(api.tfidf(api.CATEGORY, text), infer.tfidf(category, [text])[0])


def test_fallback_sample_exists():
    assert api.FALLBACK_ID in api.SAMPLES


@pytest.mark.parametrize("body,status", [
    (b"", 400),
    (b"GIF89a" + bytes(100), 415),
    (b"\x89PNG\r\n\x1a\n" + bytes(api.MAX_BYTES), 413),
    (b"\x89PNG\r\n\x1a\n" + b"not really a png", 400),
], ids=["empty", "gif", "too-large", "corrupt-png"])
def test_rejects_bad_uploads(body, status):
    code, payload = api.analyze(body, "203.0.113.1")
    assert code == status
    assert payload["error"]


def test_sixth_upload_in_an_hour_gets_the_cached_example(monkeypatch):
    monkeypatch.setattr(api, "LIMITS", api.Limits())
    body = png(scan("receipt", 4.0))
    results = [api.analyze(body, "203.0.113.7", now=1000.0 + i, read_text=lambda _: "Coffee large lunch")[1]
               for i in range(6)]
    assert [r["cached"] for r in results] == [False] * 5 + [True]
    assert results[0]["type"] == "receipt"
    assert results[0]["category"] in api.CATEGORY["classes"]
    assert "Rate limit" in results[5]["note"]
    # another visitor is still served, and the window reopens after an hour
    assert api.analyze(body, "203.0.113.8", now=1010.0, read_text=lambda _: "")[1]["cached"] is False
    assert api.analyze(body, "203.0.113.7", now=4700.0, read_text=lambda _: "")[1]["cached"] is False


def test_daily_cap_applies_across_visitors():
    limits = api.Limits(per_ip=100, daily=2)
    assert limits.check("a", 0.0) is None
    assert limits.check("b", 1.0) is None
    assert "daily cap" in limits.check("c", 2.0)
    assert limits.check("c", 86400.0 + 5) is None  # a new UTC day


def test_ocr_failure_falls_back(monkeypatch):
    monkeypatch.setattr(api, "LIMITS", api.Limits())

    def broken(_):
        raise RuntimeError("quota")

    code, payload = api.analyze(png(scan("letter", 2.0)), "203.0.113.9", read_text=broken)
    assert code == 200
    assert payload["cached"] is True
    assert "failed" in payload["note"]
