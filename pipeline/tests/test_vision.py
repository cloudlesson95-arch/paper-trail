import random

import numpy as np
import pytest
from faker import Faker
from PIL import Image

from generate import make_record, make_vendors, render
from vision import deskew, skew_angle, text_blocks


def clean_page(doc_type: str) -> Image.Image:
    fake = Faker("en_US")
    fake.seed_instance(1)
    rng = random.Random(1)
    vendor = make_vendors(fake, rng)["travel"][0]
    img, *_ = render(make_record(0, doc_type, "travel", vendor, "Jane Doe", rng), rng)
    return img


@pytest.mark.parametrize("doc_type", ["receipt", "invoice", "letter", "order_form"])
@pytest.mark.parametrize("angle", [-20.0, -7.5, -3.0, 0.0, 2.2, 6.0, 15.0])
def test_deskew_recovers_known_rotation(doc_type, angle):
    page = clean_page(doc_type)
    rotated = page.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    _, estimate = deskew(np.asarray(rotated))
    assert abs(estimate - angle) < 1.0


def test_deskewed_page_has_no_remaining_skew():
    rotated = clean_page("letter").rotate(5, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    straight, _ = deskew(np.asarray(rotated))
    assert abs(skew_angle(straight)) < 0.5


def test_text_blocks_cover_the_text():
    gray = np.asarray(clean_page("invoice"))
    boxes = text_blocks(gray)
    assert len(boxes) >= 5
    ink = gray < 128
    covered = np.zeros_like(ink)
    for x, y, w, h in boxes:
        covered[y : y + h, x : x + w] = True
    assert (ink & covered).sum() / ink.sum() > 0.95
