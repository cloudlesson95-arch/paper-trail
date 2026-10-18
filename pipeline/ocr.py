"""OCR every document with Gemini, raw and deskewed, and measure the character error rate.

Responses are cached in cache/ by image hash, so a rerun makes no API calls. Calls are
spaced out to stay under the free tier's per-minute limit.
"""

import hashlib
import re
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from common import CACHE, DOCS, PIPELINE, RESULTS, ROOT, VISION, ground_truth, read_json, write_json

# Flash-Lite: the free tier allows only 20 requests a day per model for gemini-3.8-flash.
MODEL = "gemini-3.1-flash-lite"
PROMPT = (
    "Transcribe all text in this scanned document exactly as printed. Return plain text "
    "only: one line per printed line, from top to bottom; separate items on the same line "
    "with a single space. No markdown, no table borders, no commentary."
)
CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level="low"),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)
MIN_INTERVAL = 5.0  # seconds between calls, under the per-minute limit
TIMEOUT_MS = 60_000  # a call that hangs is abandoned and retried


class Ocr:
    """Gemini OCR with an on-disk cache keyed by the SHA-256 of the image bytes.

    An entry made by a different model counts as a miss, so results never mix models.
    """

    def __init__(self):
        self.client = None
        self.calls = 0
        self.hits = 0
        self.last_call = 0.0

    def __call__(self, path: Path) -> str:
        data = path.read_bytes()
        cached = CACHE / f"{hashlib.sha256(data).hexdigest()}.json"
        if cached.exists() and (entry := read_json(cached))["model"] == MODEL:
            self.hits += 1
            return entry["text"]
        text = self.request(data)
        write_json(cached, {"model": MODEL, "source": path.relative_to(PIPELINE).as_posix(), "text": text})
        return text

    def request(self, data: bytes) -> str:
        if self.client is None:
            load_dotenv(ROOT / ".env.local")
            load_dotenv(ROOT / ".env")
            self.client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))
        for attempt in range(6):
            time.sleep(max(0.0, self.last_call + MIN_INTERVAL - time.monotonic()))
            self.last_call = time.monotonic()
            self.calls += 1
            try:
                response = self.client.models.generate_content(
                    model=MODEL,
                    contents=[types.Part.from_bytes(data=data, mime_type="image/png"), PROMPT],
                    config=CONFIG,
                )
                return response.text or ""
            except httpx.TransportError as e:  # timeouts and dropped connections
                print(f"  {type(e).__name__}; retrying", flush=True)
                continue
            except errors.APIError as e:
                if e.code != 429 and e.code < 500:
                    raise
                if "PerDay" in str(e):
                    limit = re.search(r"limit: (\d+)", str(e))
                    raise SystemExit(f"The daily free-tier quota for {MODEL} ({limit.group(1) if limit else '?'} "
                                     "requests) is used up; rerun later (results so far are cached).")
                hint = re.search(r"retryDelay\W+(\d+)", str(e))
                delay = int(hint.group(1)) + 1 if hint else 15 * (attempt + 1)
                print(f"  Gemini returned {e.code}; retrying in {delay}s", flush=True)
                time.sleep(delay)
        raise RuntimeError("Gemini kept failing; rerun later (results so far are cached).")


def normalize(text: str) -> str:
    """Drop formatting the model may add (markdown, rules) and collapse whitespace."""
    text = re.sub(r"```\w*", " ", text)
    text = re.sub(r"[|*#`]", " ", text)
    text = re.sub(r"[-_=]{3,}", " ", text)
    return " ".join(text.split())


def levenshtein(a: str, b: str) -> int:
    """Edit distance, one NumPy row at a time; insertions resolve with a running minimum."""
    if not a or not b:
        return len(a) + len(b)
    target = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    idx = np.arange(len(b) + 1)
    row = idx.copy()
    for ch in a:
        new = np.empty_like(row)
        new[0] = row[0] + 1
        new[1:] = np.minimum(row[:-1] + (target != ord(ch)), row[1:] + 1)
        row = np.minimum.accumulate(new - idx) + idx
    return int(row[-1])


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    return levenshtein(ref, hyp) / max(1, len(ref))


def main() -> None:
    ocr = Ocr()
    docs = ground_truth()
    hard_ids = {p.stem for p in (DOCS / "hard").glob("doc_*.png")}
    rows, texts = [], {}
    for n, d in enumerate(docs, 1):
        settings = [("standard", DOCS, VISION)]
        if d["id"] in hard_ids:
            settings.append(("hard", DOCS / "hard", VISION / "hard"))
        for setting, raw_dir, straight_dir in settings:
            for variant, folder in (("raw", raw_dir), ("deskewed", straight_dir)):
                text = ocr(folder / f"{d['id']}.png")
                rows.append({"id": d["id"], "type": d["type"], "setting": setting,
                             "variant": variant, "cer": cer(d["text"], text)})
                if setting == "standard":
                    texts.setdefault(d["id"], {})[variant] = text
        if n % 10 == 0:
            print(f"{n}/{len(docs)} documents ({ocr.calls} API calls, {ocr.hits} cache hits)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "ocr_cer.csv", index=False)
    write_json(RESULTS / "ocr_text.json", texts)
    summary = df.pivot_table(index="setting", columns="variant", values="cer", aggfunc="mean")
    by_type = df.pivot_table(index=["setting", "type"], columns="variant", values="cer", aggfunc="mean")
    for table in (summary, by_type):
        table["gain"] = table["raw"] - table["deskewed"]
    write_json(RESULTS / "ocr.json", {
        "model": MODEL,
        "summary": summary.reset_index().to_dict(orient="records"),
        "by_type": by_type.reset_index().to_dict(orient="records"),
    })

    print(f"\nCharacter error rate, {MODEL} (lower is better; gain = raw - deskewed):")
    print(summary[["raw", "deskewed", "gain"]].round(4).to_string())
    print("\nBy document type:")
    print(by_type[["raw", "deskewed", "gain"]].round(4).to_string())
    print(f"\nAPI calls: {ocr.calls}, cache hits: {ocr.hits}")


if __name__ == "__main__":
    main()
