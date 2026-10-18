"""Paths and small helpers shared by the pipeline scripts."""

import json
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parent

OUT = PIPELINE / "out"  # git-ignored intermediate outputs
DOCS = OUT / "docs"  # generated images and their ground truth
VISION = OUT / "vision"  # deskewed images and text-block boxes
RESULTS = OUT / "results"  # evaluation tables and metrics
CACHE = PIPELINE / "cache"  # OCR responses keyed by image hash

MODELS = ROOT / "models"  # exported JSON models, used by the live function
SAMPLES = ROOT / "public" / "samples"  # gallery images
DB_PATH = ROOT / "data" / "paper_trail.db"
SQL = ROOT / "sql"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent), encoding="utf-8")


def ground_truth() -> list[dict]:
    """All generated documents' ground truth, in id order."""
    return [read_json(p) for p in sorted(DOCS.glob("doc_*.json"))]
