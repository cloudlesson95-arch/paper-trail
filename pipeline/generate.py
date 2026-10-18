"""Render synthetic small-business documents with ground truth.

Everything is fictional: vendors, people, addresses and amounts come from Faker with a
fixed seed, phone numbers use the 555 range and e-mail addresses the reserved .example
domain. Each image gets a JSON file with its type, expense category, full text in
reading order and entity spans in that text.
"""

import random
import re
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from common import DOCS, write_json

SEED = 20261018
N_DOCS = 120
HARD_PER_TYPE = 10  # the first documents of each type also get a harder scan, for the OCR comparison
TYPES = ["receipt", "invoice", "letter", "order_form"]
CATEGORIES = ["office_supplies", "travel", "meals", "utilities", "software"]
CUSTOMER = "Brightwater Design Studio"  # the small business whose paperwork this is
STAFF_SIZE = 4
VENDORS_PER_CATEGORY = 4
VENDOR_WEIGHTS = [5, 3, 2, 1]  # some vendors are used much more often than others
CUSTOMER_STREET = "48 Juniper Row"
CUSTOMER_CITY = "Millbrook, OR 97001"

VENDOR_KINDS = {
    "office_supplies": ["Office Supply", "Stationers", "Paper Company"],
    "travel": ["Travel", "Transit Services", "Car Hire"],
    "meals": ["Bistro", "Catering", "Deli"],
    "utilities": ["Power and Light", "Water Works", "Telecom"],
    "software": ["Software", "Cloud Services", "Digital Labs"],
}
# (item, lowest unit price, highest unit price, highest quantity)
ITEMS = {
    "office_supplies": [
        ("Printer paper A4 ream", 4, 9, 10), ("Toner cartridge black", 45, 95, 3),
        ("Ballpoint pens box", 3, 12, 5), ("Stapler heavy duty", 12, 30, 2),
        ("Sticky notes pack", 2, 8, 6), ("Ring binders set", 6, 18, 4),
        ("Manila folders box", 8, 22, 3), ("Desk organizer", 15, 40, 2),
        ("Envelopes C5 pack", 5, 14, 4), ("Whiteboard markers", 4, 11, 4),
    ],
    "travel": [
        ("Train ticket return", 40, 160, 2), ("Airport shuttle", 18, 45, 2),
        ("Hotel night standard room", 90, 220, 3), ("Taxi fare", 12, 60, 1),
        ("Car rental per day", 35, 90, 4), ("Parking garage", 8, 30, 2),
        ("Checked baggage fee", 25, 60, 2), ("Fuel unleaded", 30, 80, 1),
    ],
    "meals": [
        ("Lunch set menu", 9, 18, 8), ("Coffee large", 2, 5, 10),
        ("Club sandwich", 6, 12, 6), ("Garden salad", 5, 11, 6),
        ("Team dinner", 60, 240, 1), ("Pastries assorted", 3, 9, 12),
        ("Bottled water", 1, 3, 12), ("Catering platter", 45, 150, 2),
    ],
    "utilities": [
        ("Electricity usage", 60, 280, 1), ("Water service charge", 20, 90, 1),
        ("Gas supply", 30, 160, 1), ("Business internet plan", 45, 120, 1),
        ("Mobile phone plan", 25, 70, 4), ("Waste collection", 15, 50, 1),
        ("Meter service fee", 5, 20, 1),
    ],
    "software": [
        ("Design suite annual license", 240, 600, 3), ("Cloud storage 2 TB", 8, 25, 2),
        ("Project tracker seat", 7, 20, 8), ("Priority support plan", 50, 200, 1),
        ("Domain renewal", 12, 30, 3), ("Web hosting plan", 15, 60, 1),
        ("Antivirus suite", 30, 90, 5),
    ],
}
SHARED_ITEMS = [("Delivery charge", 4, 25, 1), ("Service fee", 2, 15, 1)]

LETTERS = {
    "office_supplies": (
        "Your stationery order",
        "Thank you for ordering office supplies from {vendor}. Your order of printer "
        "paper, toner and filing materials was dispatched on {date}.",
        "The amount of {amount} has been charged to the account of {customer}. Please "
        "quote reference {ref} if any item arrives damaged.",
    ),
    "travel": (
        "Travel booking confirmation",
        "We are pleased to confirm the travel arrangements booked with {vendor}. The "
        "itinerary includes rail tickets, airport transfers and hotel nights starting {date}.",
        "The total fare of {amount} will be invoiced to {customer} under reference {ref}.",
    ),
    "meals": (
        "Catering for your team event",
        "Thank you for choosing {vendor} to cater your team lunch on {date}. The menu "
        "covered sandwiches, salads, pastries and drinks for the whole team.",
        "The balance of {amount} is now due. Please include reference {ref} with your payment.",
    ),
    "utilities": (
        "Statement for your business account",
        "This letter summarises the electricity, water and internet charges for your "
        "premises for the billing period ending {date}.",
        "The balance of {amount} will be collected by direct debit from {customer} under "
        "reference {ref}.",
    ),
    "software": (
        "Renewal of your software subscription",
        "Your subscription to the design suite and cloud storage plan from {vendor} "
        "renews on {date}. All licensed seats will remain active without interruption.",
        "The renewal fee of {amount} will be billed to {customer} under reference {ref}.",
    ),
}
CLOSINGS = [
    "If you have any questions about this letter, please contact us at {email}.",
    "We appreciate your business and look forward to working with you again.",
    "Should you need a copy of the original invoice, write to {email} and we will send it.",
]
SIGNER_TITLES = ["Accounts Manager", "Customer Service Lead", "Billing Coordinator"]


@lru_cache
def font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow's bundled font renders the same on every machine, so image hashes are stable.
    return ImageFont.load_default(size=size)


def money(value: float) -> str:
    return f"${value:,.2f}"


def format_date(d: date, style: str) -> str:
    return {
        "iso": d.isoformat(),
        "dmy": f"{d.day} {d:%b %Y}",
        "us": f"{d:%m/%d/%Y}",
        "long": f"{d:%B} {d.day}, {d.year}",
        "eu": f"{d:%d.%m.%Y}",
    }[style]


def fill(template: str, values: dict) -> list:
    """Turn '{name}' placeholders into (text, label) pieces."""
    pieces = []
    for part in re.split(r"(\{\w+\})", template):
        if part.startswith("{"):
            pieces.append(values[part[1:-1]])
        elif part:
            pieces.append(part)
    return pieces


class Canvas:
    """Draws text and records the ground-truth text and entity spans in reading order."""

    def __init__(self, width: int, height: int, ink: int):
        self.img = Image.new("L", (width, height), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.ink = ink
        self.text = ""
        self.entities: list[dict] = []

    def line(self, y: int, *segments, size: int = 22) -> int:
        """Draw one text line made of (x, align, pieces) segments; returns the next y.

        Pieces are plain strings or (text, label) tuples for entities.
        """
        f = font(size)
        if self.text:
            self.text += "\n"
        for i, (x, align, pieces) in enumerate(segments):
            pieces = [p if isinstance(p, tuple) else (p, None) for p in pieces]
            s = "".join(t for t, _ in pieces)
            width = f.getlength(s)
            x0 = {"left": x, "right": x - width, "center": x - width / 2}[align]
            self.draw.text((x0, y), s, font=f, fill=self.ink)
            if i:
                self.text += " "
            for t, label in pieces:
                if label:
                    start = len(self.text)
                    self.entities.append(
                        {"label": label, "start": start, "end": start + len(t), "text": t}
                    )
                self.text += t
        return y + int(size * 1.5)

    def paragraph(self, x: int, y: int, width: int, pieces: list, size: int = 22) -> int:
        """Wrap pieces into lines no wider than width; entity pieces never break."""
        f = font(size)
        words, current = [], []
        for piece in pieces:
            if isinstance(piece, tuple):
                current.append(piece)
                continue
            for chunk in re.split(r"(\s+)", piece):
                if chunk.isspace():
                    if current:
                        words.append(current)
                        current = []
                elif chunk:
                    current.append(chunk)
        if current:
            words.append(current)

        def text_of(word):
            return "".join(p[0] if isinstance(p, tuple) else p for p in word)

        line: list = []
        for word in words:
            candidate = line + ([" "] if line else []) + word
            if line and f.getlength(text_of(candidate)) > width:
                y = self.line(y, (x, "left", line), size=size)
                candidate = list(word)
            line = candidate
        if line:
            y = self.line(y, (x, "left", line), size=size)
        return y

    def hline(self, x1: int, x2: int, y: int, width: int = 2) -> None:
        self.draw.line([(x1, y), (x2, y)], fill=self.ink, width=width)

    def vline(self, x: int, y1: int, y2: int, width: int = 2) -> None:
        self.draw.line([(x, y1), (x, y2)], fill=self.ink, width=width)

    def dashed(self, x1: int, x2: int, y: int) -> None:
        for x in range(x1, x2, 14):
            self.draw.line([(x, y), (min(x + 8, x2), y)], fill=self.ink, width=2)

    def rect(self, x1: int, y1: int, x2: int, y2: int, shade: int | None = None) -> None:
        self.draw.rectangle([x1, y1, x2, y2], outline=self.ink, fill=shade, width=2)

    def checkbox(self, x: int, y: int, checked: bool) -> None:
        self.rect(x, y + 2, x + 20, y + 22)
        if checked:
            self.draw.line([(x + 4, y + 12), (x + 9, y + 18), (x + 17, y + 5)], fill=self.ink, width=3)


def make_vendors(fake: Faker, rng: random.Random) -> dict[str, list[dict]]:
    """A few recurring vendors per category, each with fixed contact details."""
    vendors = {}
    for category, kinds in VENDOR_KINDS.items():
        vendors[category] = []
        for _ in range(VENDORS_PER_CATEGORY):
            surname = fake.unique.last_name()
            slug = re.sub(r"[^a-z]+", "", surname.lower())
            vendors[category].append({
                "vendor": f"{surname} {rng.choice(kinds)}",
                "email": f"{rng.choice(['billing', 'accounts', 'hello', 'orders'])}@{slug}.example",
                "street": fake.street_address(),
                "city": f"{fake.city()}, {fake.state_abbr(include_territories=False)} {fake.zipcode()}",
                "phone": f"(555) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
                "signer": f"{fake.first_name()} {fake.last_name()}",
                "signer_title": rng.choice(SIGNER_TITLES),
            })
    return vendors


def make_record(i: int, doc_type: str, category: str, vendor: dict, person: str,
                rng: random.Random) -> dict:
    """Choose the content of one document: items, amounts, dates and numbers."""
    issued = date(2025, 7, 1) + timedelta(days=rng.randrange(365))
    pool = ITEMS[category]
    picks = rng.sample(pool, rng.randint(2, min(5, len(pool))))
    if rng.random() < 0.3:
        picks.append(rng.choice(SHARED_ITEMS))
    items = []
    for name, low, high, max_qty in picks:
        qty = rng.randint(1, max_qty)
        unit = round(rng.uniform(low, high), 2)
        items.append({"name": name, "qty": qty, "unit": unit, "amount": round(qty * unit, 2)})
    subtotal = round(sum(it["amount"] for it in items), 2)
    tax_rate = rng.choice([0, 5, 8, 10, 20])
    tax = round(subtotal * tax_rate / 100, 2)

    return {
        **vendor,
        "id": f"doc_{i:03d}",
        "type": doc_type,
        "category": category,
        "person": person,
        "issued": issued,
        "due": issued + timedelta(days=rng.choice([14, 30])),
        "date_style": rng.choice(["iso", "dmy", "us", "long", "eu"]),
        "items": items,
        "subtotal": subtotal,
        "tax_rate": tax_rate,
        "tax": tax,
        "total": round(subtotal + tax, 2),
        "invoice_no": f"INV-{issued.year}-{rng.randint(1, 9999):04d}",
        "order_no": f"PO-{rng.randint(10000, 99999)}",
        "receipt_no": f"RCPT-{rng.randint(100000, 999999)}",
        "time": f"{rng.randint(8, 20):02d}:{rng.randint(0, 59):02d}",
        "payment": rng.choice([f"Paid by card ending {rng.randint(1000, 9999)}", "Paid in cash"]),
        "closing": rng.choice(CLOSINGS),
    }


def render_receipt(r: dict, c: Canvas) -> tuple[int, str]:
    left, right, cx, s = 36, c.img.width - 36, c.img.width // 2, 20
    date_text = format_date(r["issued"], r["date_style"])
    y = c.line(50, (cx, "center", [(r["vendor"], "ORG")]), size=30)
    y = c.line(y, (cx, "center", [r["street"]]), size=s)
    y = c.line(y, (cx, "center", [r["city"]]), size=s)
    y = c.line(y, (cx, "center", [f"Tel {r['phone']}"]), size=s)
    c.dashed(left, right, y + 6)
    y = c.line(y + 20, (left, "left", ["Date: ", (date_text, "DATE")]),
               (right, "right", [f"Time: {r['time']}"]), size=s)
    y = c.line(y, (left, "left", ["Receipt: ", (r["receipt_no"], "DOC_ID")]), size=s)
    c.dashed(left, right, y + 6)
    y += 20
    for it in r["items"]:
        name = f"{it['qty']} x {it['name']}" if it["qty"] > 1 else it["name"]
        y = c.line(y, (left, "left", [name]), (right, "right", [(money(it["amount"]), "MONEY")]), size=s)
    c.dashed(left, right, y + 6)
    y += 20
    y = c.line(y, (left, "left", ["Subtotal"]), (right, "right", [(money(r["subtotal"]), "MONEY")]), size=s)
    y = c.line(y, (left, "left", [f"Tax {r['tax_rate']}%"]), (right, "right", [(money(r["tax"]), "MONEY")]), size=s)
    y = c.line(y, (left, "left", ["TOTAL"]), (right, "right", [(money(r["total"]), "MONEY")]), size=26)
    y = c.line(y, (left, "left", [r["payment"]]), size=s)
    c.dashed(left, right, y + 6)
    y = c.paragraph(left, y + 20, right - left,
                    ["Thank you for shopping with us! Returns are accepted within 30 days "
                     "with this receipt."], size=s)
    return y, r["receipt_no"]


def render_invoice(r: dict, c: Canvas) -> tuple[int, str]:
    left, right = 70, c.img.width - 70
    date_text = format_date(r["issued"], r["date_style"])
    due_text = format_date(r["due"], r["date_style"])
    y = c.line(50, (right, "right", ["INVOICE"]), size=44)
    y = c.line(y + 10, (left, "left", [(r["vendor"], "ORG")]), size=30)
    y = c.line(y, (left, "left", [r["street"]]))
    y = c.line(y, (left, "left", [r["city"]]))
    y = c.line(y, (left, "left", [(r["email"], "EMAIL")]))
    y = c.line(y + 20, (left, "left", ["Invoice No: ", (r["invoice_no"], "DOC_ID")]))
    y = c.line(y, (left, "left", ["Invoice date: ", (date_text, "DATE")]))
    y = c.line(y, (left, "left", ["Due date: ", (due_text, "DATE")]))
    y = c.line(y + 20, (left, "left", ["Bill to: ", (r["person"], "PERSON")]))
    y = c.line(y, (left, "left", [(CUSTOMER, "ORG")]))
    y = c.line(y, (left, "left", [f"{CUSTOMER_STREET}, {CUSTOMER_CITY}"]))

    y += 25
    top = y - 8
    c.rect(left, top, right, y + 32, shade=225)
    cols = (left + 12, 500, 640, right - 12)
    y = c.line(y, (cols[0], "left", ["Description"]), (cols[1], "right", ["Qty"]),
               (cols[2], "right", ["Unit price"]), (cols[3], "right", ["Amount"]))
    for it in r["items"]:
        c.hline(left, right, y - 8, width=1)
        y = c.line(y, (cols[0], "left", [it["name"]]), (cols[1], "right", [str(it["qty"])]),
                   (cols[2], "right", [(money(it["unit"]), "MONEY")]),
                   (cols[3], "right", [(money(it["amount"]), "MONEY")]))
    c.rect(left, top, right, y - 4)

    y += 15
    y = c.line(y, (640, "right", ["Subtotal"]), (cols[3], "right", [(money(r["subtotal"]), "MONEY")]))
    y = c.line(y, (640, "right", [f"Tax {r['tax_rate']}%"]), (cols[3], "right", [(money(r["tax"]), "MONEY")]))
    y = c.line(y, (640, "right", ["Total due"]), (cols[3], "right", [(money(r["total"]), "MONEY")]), size=26)
    y = c.paragraph(left, y + 30, right - left,
                    ["Payment is due within 30 days of the invoice date. Please quote the "
                     "invoice number with your payment. Questions about this invoice? Write to ",
                     (r["email"], "EMAIL"), "."])
    return y, r["invoice_no"]


def render_letter(r: dict, c: Canvas) -> tuple[int, str]:
    left, right, cx = 80, c.img.width - 80, c.img.width // 2
    date_text = format_date(r["issued"], r["date_style"])
    values = {
        "vendor": (r["vendor"], "ORG"),
        "customer": (CUSTOMER, "ORG"),
        "date": (date_text, "DATE"),
        "amount": (money(r["total"]), "MONEY"),
        "ref": (r["invoice_no"], "DOC_ID"),
        "email": (r["email"], "EMAIL"),
    }
    subject, *paragraphs = LETTERS[r["category"]]
    paragraphs.append(r["closing"])

    y = c.line(50, (cx, "center", [(r["vendor"], "ORG")]), size=34)
    y = c.line(y, (cx, "center", [f"{r['street']}, {r['city']}"]), size=18)
    y = c.line(y, (cx, "center", [f"Tel {r['phone']}  ", (r["email"], "EMAIL")]), size=18)
    c.hline(left, right, y + 4)
    y = c.line(y + 30, (right, "right", [(date_text, "DATE")]))
    y = c.line(y + 10, (left, "left", [(r["person"], "PERSON")]))
    y = c.line(y, (left, "left", [(CUSTOMER, "ORG")]))
    y = c.line(y, (left, "left", [CUSTOMER_STREET]))
    y = c.line(y, (left, "left", [CUSTOMER_CITY]))
    y = c.line(y + 25, (left, "left", [f"Re: {subject}"]), size=24)
    y = c.line(y + 10, (left, "left", ["Dear ", (r["person"], "PERSON"), ","]))
    for p in paragraphs:
        y = c.paragraph(left, y + 12, right - left, fill(p, values))
    y = c.line(y + 20, (left, "left", ["Sincerely,"]))
    y = c.line(y + 30, (left, "left", [(r["signer"], "PERSON")]))
    y = c.line(y, (left, "left", [f"{r['signer_title']}, ", (r["vendor"], "ORG")]))
    return y, r["invoice_no"]


def render_order_form(r: dict, c: Canvas, rng: random.Random) -> tuple[int, str]:
    left, right, cx = 60, c.img.width - 60, c.img.width // 2
    date_text = format_date(r["issued"], r["date_style"])
    due_text = format_date(r["due"], r["date_style"])
    y = c.line(50, (cx, "center", ["PURCHASE ORDER FORM"]), size=40)
    y += 20
    fields = [
        ("Order No:", (r["order_no"], "DOC_ID")), ("Order date:", (date_text, "DATE")),
        ("Delivery by:", (due_text, "DATE")), ("Supplier:", (r["vendor"], "ORG")),
        ("Supplier email:", (r["email"], "EMAIL")), ("Ordered by:", (r["person"], "PERSON")),
        ("Company:", (CUSTOMER, "ORG")),
    ]
    for label, value in fields:
        c.rect(260, y - 6, right, y + 30)
        y = c.line(y, (left, "left", [label]), (272, "left", [value])) + 8

    y += 20
    xs = (left, 140, 460, 530, 655, right)
    top = y - 8
    c.rect(left, top, right, y + 32, shade=225)
    y = c.line(y, (xs[0] + 10, "left", ["Code"]), (xs[1] + 10, "left", ["Description"]),
               (xs[3] - 10, "right", ["Qty"]), (xs[4] - 10, "right", ["Unit price"]),
               (xs[5] - 10, "right", ["Line total"]))
    for it in r["items"]:
        c.hline(left, right, y - 8)
        code = f"{rng.choice('ABCDEFGH')}-{rng.randint(100, 999)}"
        y = c.line(y, (xs[0] + 10, "left", [code]), (xs[1] + 10, "left", [it["name"]]),
                   (xs[3] - 10, "right", [str(it["qty"])]),
                   (xs[4] - 10, "right", [(money(it["unit"]), "MONEY")]),
                   (xs[5] - 10, "right", [(money(it["amount"]), "MONEY")]))
    for _ in range(rng.randint(2, 4)):  # blank rows, as on a printed form
        c.hline(left, right, y - 8)
        y += 33
    for x in xs:
        c.vline(x, top, y - 8)
    c.hline(left, right, y - 8)

    c.rect(500, y + 10, right, y + 50)
    y = c.line(y + 16, (490, "right", ["Order total:"]),
               (right - 10, "right", [(money(r["subtotal"]), "MONEY")]), size=24)
    y += 20
    express = rng.random() < 0.4
    c.checkbox(left, y, not express)
    c.checkbox(left + 300, y, express)
    y = c.line(y, (left + 32, "left", ["Standard delivery"]), (left + 332, "left", ["Express delivery"]))
    y += 40
    y = c.line(y, (left, "left", ["Authorised signature"]), (470, "left", ["Date"]))
    c.hline(left + 230, 450, y - 12)
    c.hline(530, right, y - 12)
    return y, r["order_no"]


def degrade(img: Image.Image, rng: random.Random, nrng: np.random.Generator,
            hard: bool = False) -> tuple[Image.Image, float]:
    """Simulate a scan: rotation, blur, uneven lighting and noise.

    The standard setting rotates within +-8 degrees. The hard one rotates 12 to 25
    degrees either way, scans at 60% resolution and adds more blur and noise.
    """
    paper = rng.randint(238, 252)
    if hard:
        angle = round(rng.choice([-1, 1]) * rng.uniform(12, 25), 2)
    else:
        angle = round(rng.uniform(-8, 8), 2)
    img = Image.eval(img, lambda v: int(v * paper / 255))
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=paper)
    if hard:
        img = img.resize((int(img.width * 0.6), int(img.height * 0.6)), Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.0) if hard else rng.uniform(0.3, 1.1)))

    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    theta = rng.uniform(0, 2 * np.pi)
    ramp = np.cos(theta) * xx / w + np.sin(theta) * yy / h
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    arr *= 1 - rng.uniform(0.1, 0.35) * ramp
    arr += nrng.normal(0, rng.uniform(6, 12) if hard else rng.uniform(3, 9), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), angle


def render(r: dict, rng: random.Random) -> tuple[Image.Image, str, list, str]:
    """Draw a clean document; returns the image, its text, entities and document number."""
    ink = rng.randint(15, 70)
    if r["type"] == "receipt":
        c = Canvas(520, 1400, ink)
        bottom, number = render_receipt(r, c)
    else:
        c = Canvas(850, 1100, ink)
        if r["type"] == "invoice":
            bottom, number = render_invoice(r, c)
        elif r["type"] == "letter":
            bottom, number = render_letter(r, c)
        else:
            bottom, number = render_order_form(r, c, rng)
    img = c.img.crop((0, 0, c.img.width, min(c.img.height, bottom + 50)))
    return img, c.text, c.entities, number


def main() -> None:
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)
    fake = Faker("en_US")
    fake.seed_instance(SEED)

    types = [TYPES[i % len(TYPES)] for i in range(N_DOCS)]
    categories = [CATEGORIES[i % len(CATEGORIES)] for i in range(N_DOCS)]
    rng.shuffle(types)
    rng.shuffle(categories)
    vendors = make_vendors(fake, rng)
    staff = [f"{fake.first_name()} {fake.last_name()}" for _ in range(STAFF_SIZE)]

    # A separate random stream, so the hard scans leave the standard documents unchanged.
    hard_rng, hard_nrng = random.Random(SEED + 1), np.random.default_rng(SEED + 1)
    (DOCS / "hard").mkdir(parents=True, exist_ok=True)
    hard_counts = dict.fromkeys(TYPES, 0)
    for i, (doc_type, category) in enumerate(zip(types, categories)):
        vendor = rng.choices(vendors[category], weights=VENDOR_WEIGHTS)[0]
        r = make_record(i, doc_type, category, vendor, rng.choice(staff), rng)
        clean, text, entities, number = render(r, rng)
        scan, angle = degrade(clean, rng, nrng)
        scan.save(DOCS / f"{r['id']}.png", optimize=True)
        write_json(DOCS / f"{r['id']}.json", {
            "id": r["id"],
            "type": doc_type,
            "category": category,
            "vendor": r["vendor"],
            "number": number,
            "total": r["subtotal"] if doc_type == "order_form" else r["total"],
            "issued_on": r["issued"].isoformat(),
            "rotation": angle,
            "text": text,
            "entities": entities,
        })
        if hard_counts[doc_type] < HARD_PER_TYPE:
            hard_counts[doc_type] += 1
            hard, hard_angle = degrade(clean, hard_rng, hard_nrng, hard=True)
            hard.save(DOCS / "hard" / f"{r['id']}.png", optimize=True)
            write_json(DOCS / "hard" / f"{r['id']}.json", {"rotation": hard_angle})
    print(f"Wrote {N_DOCS} documents and {sum(hard_counts.values())} hard scans to {DOCS}")


if __name__ == "__main__":
    main()
