# Paper Trail

Document image intake for small-business paperwork. Scanned receipts, invoices, letters and
order forms go through an OpenCV clean-up and a document-type classifier, get read by Gemini,
then spaCy pulls out the entities and TextRank writes a two-sentence summary. Everything
lands in a SQLite database you can explore through named SQL queries in a Next.js demo.

**Live demo: https://paper-trail-azure.vercel.app**

| Page | What it shows |
|---|---|
| [Gallery](https://paper-trail-azure.vercel.app/) | Original scan next to the deskewed page with its text blocks outlined; document type, OCR text with highlighted entities, and the summary |
| [Try a document](https://paper-trail-azure.vercel.app/try) | Pick a sample (served from the cache) or upload your own PNG/JPEG for a live analysis |
| [SQL explorer](https://paper-trail-azure.vercel.app/sql) | Each named query's SQL next to its live result, plus full-text search over the OCR text |
| [Metrics](https://paper-trail-azure.vercel.app/metrics) | Cross-validation accuracy, confusion matrices, OCR error rates with and without deskewing, per-entity precision/recall/F1 |

All 120 documents are synthetic: rendered with Pillow and Faker from a fixed seed, with
fictional vendors, people, addresses and amounts (phone numbers use the 555 range, e-mail
domains the reserved `.example` TLD). Generating them is what makes every evaluation
possible: each image comes with its exact text, type, expense category and entity spans.

## What it shows

- **Computer vision.** Adaptive thresholding, deskewing from `cv2.minAreaRect` over
  line-shaped blobs of text pixels (length-weighted median, then a second pass on the
  residual angle), and text-block detection with dilation and contours. A document-type
  classifier on HOG features (`cv2.HOGDescriptor`) with StandardScaler + LogisticRegression,
  checked with stratified 5-fold cross-validation and a confusion matrix.
- **OCR evaluation.** Gemini reads every page twice, raw and deskewed. Character error
  rate against the ground truth measures what deskewing buys, on the standard scans and on
  a harder set (12–25° rotation, 60% resolution, more blur and noise).
- **Named-entity recognition.** spaCy's `en_core_web_sm` plus an EntityRuler for amounts,
  dates, invoice/order/receipt numbers and e-mails, scored per entity type for precision,
  recall and F1, with and without the rules.
- **Text classification.** TF-IDF + LogisticRegression assigns one of five expense
  categories from the OCR text, 5-fold cross-validated.
- **Extractive summarisation.** TextRank: a TF-IDF cosine-similarity graph over the
  sentences, PageRank by NumPy power iteration, the top two sentences in document order.
- **Relational SQL.** A normalised schema (documents, blocks, entities, OCR runs, metrics)
  with foreign keys, CHECK constraints and indexes, an external-content FTS5 index over the
  OCR text, and named queries using GROUP BY, CTEs, an anti-join, a self-join and window
  functions (`RANK() OVER`, `SUM() OVER (PARTITION BY …)`).

## Architecture

```
OFFLINE: pipeline/ (Python 3.12, uv), run once
  generate.py        120 synthetic scans + ground truth (Pillow, Faker)
  vision.py          binarise, deskew, text blocks ..................... public/samples/*.png
  train_doctype.py   HOG + LogisticRegression, 5-fold CV ............... models/doctype.json
  ocr.py             Gemini OCR of raw and deskewed scans (cached), CER
  nlp.py             spaCy + EntityRuler, TF-IDF category, TextRank .... models/category.json
  load_db.py         everything, in one transaction .................... data/paper_trail.db
                                                                         data/samples.json
ONLINE: Vercel
  Next.js pages      /, /try, /sql, /metrics, prerendered from data/paper_trail.db
  /api/query         route handler: better-sqlite3, read-only; named queries + FTS5 search
  /api/analyze       Python function: OpenCV deskew + blocks, NumPy inference with
                     models/*.json, Gemini OCR, per-IP limit + daily cap, cached fallback
```

Training, evaluation and the OCR of the corpus run offline. The models are exported from
scikit-learn to JSON, so the live function needs only OpenCV, NumPy and google-genai. Tests
check that its NumPy inference matches scikit-learn and that its copy of the preprocessing
matches the pipeline's.

## Results

Measured on the 120 generated documents (30 per type, 24 per expense category):

| Measure | Result |
|---|---|
| Document type, 5-fold CV accuracy (HOG + LogisticRegression, 2,268 features) | 1.000 ± 0.000 deskewed; 1.000 on raw scans |
| Expense category, 5-fold CV accuracy (TF-IDF, 640 terms + LogisticRegression) | 0.992 ± 0.017 |
| Deskew angle error vs. the known rotation, standard scans (±8°) | mean 0.02°, max 0.19° |
| Deskew angle error, hard scans (12–25°, 60% resolution) | mean 0.06°, max 0.23° |
| OCR character error rate (gemini-3.1-flash-lite), standard scans | 0.32% raw → 0.06% deskewed |
| OCR character error rate, hard scans | 1.13% raw → 0.06% deskewed (receipts: 2.90% → 0.03%) |
| Entity F1, all six types (1,740 entities) | 0.52 spaCy alone → 0.81 with the EntityRuler |

Entity extraction per type (spaCy + EntityRuler):

| Type | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Amount | 833 | 1.00 | 1.00 | 1.00 |
| Organisation | 284 | 0.42 | 0.50 | 0.46 |
| Date | 210 | 1.00 | 1.00 | 1.00 |
| Person | 150 | 0.25 | 0.93 | 0.40 |
| E-mail | 143 | 1.00 | 1.00 | 1.00 |
| Document number | 120 | 1.00 | 1.00 | 1.00 |

## Run it locally

Prerequisites: Node.js 20+, [uv](https://docs.astral.sh/uv/), and a Gemini API key in
`.env.local` (`GOOGLE_API_KEY=...`, see `.env.example`) if you want to redo the OCR.

The web app, using the committed database and models:

```powershell
npm install
npm run dev
```

The offline pipeline, in order (each step writes what the next one reads):

```powershell
uv run --directory pipeline python generate.py       # 120 scans + ground truth
uv run --directory pipeline python vision.py         # deskew, text blocks, gallery images
uv run --directory pipeline python train_doctype.py  # doc-type classifier -> models/doctype.json
uv run --directory pipeline python ocr.py            # Gemini OCR (cached in pipeline/cache/), CER
uv run --directory pipeline python nlp.py            # entities, category model, summaries
uv run --directory pipeline python load_db.py        # data/paper_trail.db + data/samples.json
uv run --directory pipeline pytest -q
```

`ocr.py` spaces its calls to stay under the free tier's per-minute limit, times out hung
requests and caches every response by image hash, so a second run makes no API calls.

To run the Python function next to the pages, use `vercel dev` (the function is
`api/analyze.py`; its dependencies are in `pyproject.toml`/`uv.lock` and `requirements.txt`).
`vercel dev` hands the function the variables from `.env`, not `.env.local`.

## Limits

- **Synthetic data.** Four fixed templates in one typeface, with fictional content. Real
  scans (phone photos, handwriting, folds, stamps, varied layouts) are much harder. The
  document-type task saturates because the templates look nothing alike, and the
  expense-category task is nearly as easy because each category has its own item vocabulary.
- **Gemini OCR.** A hosted model whose output can change between versions. Uploads on the
  Try page are sent to Google's API. Free-tier quotas are counted per model and per day:
  `gemini-3.8-flash` allowed 20 requests a day, too few for the 320-call evaluation, so
  OCR uses `gemini-3.1-flash-lite`.
- **In-memory rate limit.** The per-IP limit (5 an hour) and the daily cap are counted
  inside each function instance. They reset when an instance starts and aren't shared
  between instances, so they bound abuse rather than guarantee it.
- **Read-only database.** The database is built offline; live uploads are analysed but
  never stored or added to it.
- **Statistical NER on forms.** `en_core_web_sm` tags form labels and street names as
  people and organisations and reads "surname + trade" vendors as people, which shows in
  the Organisation and Person rows above and in the co-occurrence query. No rules were
  tuned on the evaluation set to hide this.
- **Extractive summaries.** TextRank reads well on letters. On receipts and invoices it
  picks their boilerplate sentences; order forms have none, so it falls back to lines.
- **CER** is measured after collapsing whitespace, so differences in reading order (for
  example a right-aligned date) count as errors.

## What was learned

- A hosted vision model reads skewed, noisy scans almost perfectly, so deskewing buys
  little OCR accuracy at mild angles (0.32% → 0.06% CER). It matters more at steep angles
  and low resolution (1.13% → 0.06%), and most for narrow receipts (2.90% → 0.03%).
- `minAreaRect` over all ink pixels is fragile: noise, page borders and a 90° ambiguity
  throw it off. Smearing characters into line blobs and taking a length-weighted median of
  their angles, then re-measuring the small residual, brought the mean error to 0.02°.
- Rules carry the structured fields. The EntityRuler took dates, amounts, document numbers
  and e-mails to 1.00 F1 and the overall F1 from 0.52 to 0.81. Tagging each OCR line on its
  own rather than the whole text raised the statistical model's organisation F1 from 0.26
  to 0.46, because entities stopped running across layout lines.
- Exporting scikit-learn models to JSON with a NumPy forward pass keeps the serverless
  function small; parity tests against scikit-learn make that safe to change.
- Free-tier LLM quotas differ by model and are counted per day. Caching by image hash,
  spacing the calls and setting a request timeout (one call hung for ten minutes) made
  the 320-call evaluation repeatable.
- With Cache Components, Next.js prerenders pages that read SQLite synchronously at build
  time, so only the query API needs the database at runtime.

---

Built with AI-assisted development (Claude Code).
