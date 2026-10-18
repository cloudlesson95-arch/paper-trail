"""OpenCV preprocessing: binarise, deskew, find text blocks and compute HOG features.

Run as a script to process every generated document: it writes the deskewed images and
block boxes to out/vision/ and the before/after images of the gallery documents to
public/samples/.
"""

import cv2
import numpy as np
from PIL import Image

from common import DOCS, RESULTS, SAMPLES, VISION, ground_truth, read_json, write_json

HOG_SIZE = (128, 160)  # width, height of the image the HOG descriptor sees
HOG = cv2.HOGDescriptor(HOG_SIZE, (32, 32), (16, 16), (16, 16), 9)
GALLERY_PER_TYPE = 3
GALLERY_WIDTH = 600


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink as 255 on 0: adaptive threshold copes with uneven lighting, opening removes specks."""
    smooth = cv2.medianBlur(gray, 3)
    bw = cv2.adaptiveThreshold(
        smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def rect_angle(rect) -> float:
    """Angle of a minAreaRect's long side in degrees, counter-clockwise positive, in [-90, 90)."""
    box = cv2.boxPoints(rect)
    edges = [box[1] - box[0], box[2] - box[1]]
    dx, dy = max(edges, key=lambda e: float(np.hypot(*e)))
    angle = -np.degrees(np.arctan2(dy, dx))  # image y points down
    return float((angle + 90) % 180 - 90)


def skew_angle(gray: np.ndarray) -> float:
    """Estimate how far the page is rotated (counter-clockwise, degrees).

    Characters are smeared into line-shaped blobs; the minAreaRect of each long, thin
    blob follows its text line, and the length-weighted median over all lines is
    robust to stray blobs. Falls back to the minAreaRect of all ink pixels when few
    lines are found.
    """
    bw = binarize(gray)
    lines = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles, weights = [], []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        long_side, short_side = max(rect[1]), min(rect[1])
        if long_side >= 60 and long_side >= 5 * max(short_side, 1.0):
            angle = rect_angle(rect)
            if abs(angle) <= 30:  # skip vertical rules
                angles.append(angle)
                weights.append(long_side)
    if len(angles) >= 3:
        order = np.argsort(angles)
        cumulative = np.cumsum(np.asarray(weights)[order])
        middle = np.searchsorted(cumulative, cumulative[-1] / 2)
        return float(np.asarray(angles)[order][middle])
    points = cv2.findNonZero(bw)
    if points is None:
        return 0.0
    # The ink's bounding rectangle is the page's, so its angle is only known modulo 90.
    return (rect_angle(cv2.minAreaRect(points)) + 45) % 90 - 45


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by angle degrees about the centre, filling with paper colour."""
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    paper = int(np.median(gray))
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=paper)


def deskew(gray: np.ndarray, passes: int = 2) -> tuple[np.ndarray, float]:
    """Straighten the page. Smearing biases short lines towards 0 degrees, so a second
    pass measures the remaining (small, hence less biased) angle and adds it."""
    angle = 0.0
    for _ in range(passes):
        angle += skew_angle(rotate(gray, -angle) if angle else gray)
    return rotate(gray, -angle), angle


def text_blocks(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Boxes (x, y, w, h) of text blocks: dilation joins words and close lines into one blob."""
    bw = binarize(gray)
    merged = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 15)))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] >= 600 and b[3] >= 12]
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def hog_features(gray: np.ndarray, straighten: bool = True) -> np.ndarray:
    """HOG of the binarised page, cropped to its ink and resized to a fixed size."""
    if straighten:
        gray, _ = deskew(gray)
    bw = binarize(gray)
    points = cv2.findNonZero(bw)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        bw = bw[y : y + h, x : x + w]
    small = cv2.resize(bw, HOG_SIZE, interpolation=cv2.INTER_AREA)
    return HOG.compute(small).ravel()


def gallery_ids(docs: list[dict]) -> list[str]:
    """The first few documents of each type, shown in the web gallery."""
    picked: dict[str, list[str]] = {}
    for d in docs:
        ids = picked.setdefault(d["type"], [])
        if len(ids) < GALLERY_PER_TYPE:
            ids.append(d["id"])
    return sorted(i for ids in picked.values() for i in ids)


def overlay(gray: np.ndarray, boxes) -> np.ndarray:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 110, 230), 3)
    return img


def save_small(path, img: np.ndarray) -> None:
    """Downscale for the web and keep 16 colours: scan noise makes full-colour PNGs heavy."""
    scale = GALLERY_WIDTH / img.shape[1]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_GRAY2RGB if small.ndim == 2 else cv2.COLOR_BGR2RGB))
    rgb.quantize(colors=16, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


def main() -> None:
    docs = ground_truth()
    VISION.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    gallery = set(gallery_ids(docs))
    errors = []
    for d in docs:
        gray = cv2.imread(str(DOCS / f"{d['id']}.png"), cv2.IMREAD_GRAYSCALE)
        straight, angle = deskew(gray)
        boxes = text_blocks(straight)
        cv2.imwrite(str(VISION / f"{d['id']}.png"), straight)
        write_json(VISION / f"{d['id']}.json", {"angle": round(angle, 3), "blocks": boxes})
        errors.append(abs(angle - d["rotation"]))
        if d["id"] in gallery:
            save_small(SAMPLES / f"{d['id']}_original.png", gray)
            save_small(SAMPLES / f"{d['id']}_deskewed.png", straight)
            save_small(SAMPLES / f"{d['id']}_overlay.png", overlay(straight, boxes))

    hard_errors = []
    (VISION / "hard").mkdir(exist_ok=True)
    for path in sorted((DOCS / "hard").glob("doc_*.png")):
        straight, angle = deskew(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        cv2.imwrite(str(VISION / "hard" / path.name), straight)
        hard_errors.append(abs(angle - read_json(path.with_suffix(".json"))["rotation"]))

    summary = {
        setting: {"documents": len(errs), "mae_degrees": float(np.mean(errs)), "max_error_degrees": float(np.max(errs))}
        for setting, errs in (("standard", errors), ("hard", hard_errors))
    }
    write_json(RESULTS / "deskew.json", summary)
    print(f"Processed {len(docs)} documents; {len(gallery)} gallery samples in {SAMPLES}")
    for setting, row in summary.items():
        print(f"Deskew angle error ({setting}, {row['documents']} scans): "
              f"mean {row['mae_degrees']:.2f} deg, max {row['max_error_degrees']:.2f} deg")


if __name__ == "__main__":
    main()
