"""Live analysis of an uploaded document image (Vercel Python function, POST /api/analyze).

The request body is the raw PNG or JPEG file, 1 MB at most. The function deskews the page
and finds its text blocks with OpenCV, classifies the document type from models/doctype.json,
reads the text with Gemini and classifies the expense category from models/category.json.
The models were trained offline with scikit-learn and exported to JSON, so only OpenCV,
NumPy and google-genai are needed here. pipeline/tests/test_api.py checks this copy of the
preprocessing and inference against the pipeline's own.

Visitors can't use up the API key's free quota: each IP address gets 5 analyses an hour and
the instance a fixed number a day (counted in memory, per instance). Past a limit, or when
a step fails, the response is a cached example with "cached": true and a note saying why.
"""

import base64
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-3.1-flash-lite"
PROMPT = (
    "Transcribe all text in this scanned document exactly as printed. Return plain text "
    "only: one line per printed line, from top to bottom; separate items on the same line "
    "with a single space. No markdown, no table borders, no commentary."
)
MAX_BYTES = 1024 * 1024
PER_IP_PER_HOUR = 5
DAILY_CAP = 50
MAX_SIDE = 2000  # larger uploads are scaled down before processing
FALLBACK_ID = "doc_006"


def _load(name: str) -> dict:
    model = json.loads((ROOT / "models" / name).read_text(encoding="utf-8"))
    for key in ("mean", "scale", "coef", "intercept", "idf"):
        if key in model:
            model[key] = np.asarray(model[key], dtype=np.float64)
    return model


DOCTYPE = _load("doctype.json")
CATEGORY = _load("category.json")
SAMPLES = {s["sample_id"]: s for s in json.loads((ROOT / "data" / "samples.json").read_text(encoding="utf-8"))}
HOG = cv2.HOGDescriptor(tuple(DOCTYPE["hog"]["size"]), (32, 32), (16, 16), (16, 16), 9)

# ---- Image preprocessing (same steps as pipeline/vision.py) --------------------------------


def binarize(gray: np.ndarray) -> np.ndarray:
    smooth = cv2.medianBlur(gray, 3)
    bw = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def rect_angle(rect) -> float:
    box = cv2.boxPoints(rect)
    dx, dy = max([box[1] - box[0], box[2] - box[1]], key=lambda e: float(np.hypot(*e)))
    return float((-np.degrees(np.arctan2(dy, dx)) + 90) % 180 - 90)


def skew_angle(gray: np.ndarray) -> float:
    bw = binarize(gray)
    lines = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles, weights = [], []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        long_side, short_side = max(rect[1]), min(rect[1])
        if long_side >= 60 and long_side >= 5 * max(short_side, 1.0):
            angle = rect_angle(rect)
            if abs(angle) <= 30:
                angles.append(angle)
                weights.append(long_side)
    if len(angles) >= 3:
        order = np.argsort(angles)
        cumulative = np.cumsum(np.asarray(weights)[order])
        return float(np.asarray(angles)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
    points = cv2.findNonZero(bw)
    return 0.0 if points is None else (rect_angle(cv2.minAreaRect(points)) + 45) % 90 - 45


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=int(np.median(gray)))


def deskew(gray: np.ndarray, passes: int = 2) -> tuple[np.ndarray, float]:
    angle = 0.0
    for _ in range(passes):
        angle += skew_angle(rotate(gray, -angle) if angle else gray)
    return rotate(gray, -angle), angle


def text_blocks(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    merged = cv2.dilate(binarize(gray), cv2.getStructuringElement(cv2.MORPH_RECT, (31, 15)))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    return sorted((b for b in boxes if b[2] * b[3] >= 600 and b[3] >= 12), key=lambda b: (b[1], b[0]))


def hog_features(straight: np.ndarray) -> np.ndarray:
    bw = binarize(straight)
    points = cv2.findNonZero(bw)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        bw = bw[y : y + h, x : x + w]
    return HOG.compute(cv2.resize(bw, tuple(DOCTYPE["hog"]["size"]), interpolation=cv2.INTER_AREA)).ravel()


# ---- Inference with the exported models (same maths as pipeline/infer.py) ------------------


def predict_proba(model: dict, x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if "mean" in model:
        x = (x - model["mean"]) / model["scale"]
    z = x @ model["coef"].T + model["intercept"]
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tfidf(model: dict, text: str) -> np.ndarray:
    vocab, (lo, hi) = model["vocabulary"], model["ngram_range"]
    out = np.zeros(len(vocab))
    tokens = TOKEN.findall(text.lower())
    for n in range(lo, hi + 1):
        for i in range(len(tokens) - n + 1):
            j = vocab.get(" ".join(tokens[i : i + n]))
            if j is not None:
                out[j] += 1
    if model.get("sublinear_tf"):
        seen = out > 0
        out[seen] = 1 + np.log(out[seen])
    out *= model["idf"]
    norm = np.linalg.norm(out)
    return out / norm if norm else out


def scores(model: dict, x: np.ndarray) -> dict[str, float]:
    return {c: round(float(p), 4) for c, p in zip(model["classes"], predict_proba(model, x)[0])}


# ---- Gemini OCR ----------------------------------------------------------------------------

_client = None


def ocr(png: bytes) -> str:
    global _client
    from google import genai
    from google.genai import errors, types

    if _client is None:
        # A hung call times out and the request falls back to the cache.
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"], http_options=types.HttpOptions(timeout=25_000))
    for attempt in range(2):  # Gemini's 5xx ("overloaded") is usually gone a moment later
        try:
            response = _client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), PROMPT],
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                ),
            )
            return response.text or ""
        except errors.ServerError:
            if attempt:
                raise
            time.sleep(2)
    return ""


# ---- Limits --------------------------------------------------------------------------------


class Limits:
    """Per-IP sliding hour window and a daily cap, held in this instance's memory."""

    def __init__(self, per_ip: int = PER_IP_PER_HOUR, daily: int = DAILY_CAP):
        self.per_ip, self.daily = per_ip, daily
        self.recent: dict[str, deque] = defaultdict(deque)
        self.day, self.used_today = "", 0
        self.lock = threading.Lock()

    def check(self, ip: str, now: float) -> str | None:
        """Count one analysis for ip; returns why it is refused, or None if allowed."""
        with self.lock:
            window = self.recent[ip]
            while window and now - window[0] >= 3600:
                window.popleft()
            today = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
            if today != self.day:
                self.day, self.used_today = today, 0
            if len(window) >= self.per_ip:
                return f"Rate limit reached: {self.per_ip} analyses per hour per visitor."
            if self.used_today >= self.daily:
                return f"The demo's daily cap of {self.daily} live analyses is used up."
            window.append(now)
            self.used_today += 1
            return None


LIMITS = Limits()


def image_kind(body: bytes) -> str | None:
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if body.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return None


def cached(note: str) -> tuple[int, dict]:
    return 200, {**SAMPLES[FALLBACK_ID], "cached": True, "note": note}


def analyze(body: bytes, ip: str, now: float | None = None, read_text=None) -> tuple[int, dict]:
    """Validate, rate-limit and analyse one upload; returns (HTTP status, JSON payload)."""
    if not body:
        return 400, {"error": "Send the image file as the request body."}
    if len(body) > MAX_BYTES:
        return 413, {"error": "The image is larger than 1 MB."}
    if image_kind(body) is None:
        return 415, {"error": "Only PNG and JPEG images are accepted."}
    gray = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 400, {"error": "The file could not be read as an image."}

    refused = LIMITS.check(ip, time.time() if now is None else now)
    if refused:
        return cached(f"{refused} Showing a cached example instead.")
    try:
        scale = MAX_SIDE / max(gray.shape)
        if scale < 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        straight, angle = deskew(gray)
        blocks = text_blocks(straight)
        type_scores = scores(DOCTYPE, hog_features(straight))
        text = (read_text or ocr)(cv2.imencode(".png", straight)[1].tobytes())
        category_scores = scores(CATEGORY, tfidf(CATEGORY, text))
    except Exception as e:  # any failure, including Gemini errors and quota
        print(f"analysis failed: {e!r}"[:500], file=sys.stderr)
        return cached(f"Live analysis failed ({type(e).__name__}). Showing a cached example instead.")

    preview = cv2.cvtColor(straight, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in blocks:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (40, 110, 230), 3)
    preview = cv2.resize(preview, None, fx=600 / preview.shape[1], fy=600 / preview.shape[1], interpolation=cv2.INTER_AREA)
    jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    return 200, {
        "cached": False,
        "note": None,
        "sample_id": None,
        "type": max(type_scores, key=type_scores.get),
        "type_scores": type_scores,
        "category": max(category_scores, key=category_scores.get),
        "category_scores": category_scores,
        "skew_deg": round(angle, 2),
        "blocks": len(blocks),
        "text": text,
        "overlay": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
        "model": MODEL,
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        if length > MAX_BYTES:
            status, payload = 413, {"error": "The image is larger than 1 MB."}
        else:
            forwarded = self.headers.get("x-forwarded-for", "").split(",")[0].strip()
            ip = forwarded or self.headers.get("x-real-ip") or self.client_address[0]
            status, payload = analyze(self.rfile.read(length), ip)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send_response(405)
        self.send_header("Allow", "POST")
        self.end_headers()
