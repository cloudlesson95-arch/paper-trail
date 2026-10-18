-- Paper Trail database. pipeline/load_db.py rebuilds it from the pipeline outputs.
--
-- In documents, type, category, vendor, number, total and issued_on are the generated
-- ground truth; ocr_text, summary, cer and skew_deg, and the blocks and entities
-- tables, are what the pipeline produced from the scans.

CREATE TABLE documents (
    id         TEXT PRIMARY KEY,
    type       TEXT NOT NULL CHECK (type IN ('receipt', 'invoice', 'letter', 'order_form')),
    category   TEXT NOT NULL,
    vendor     TEXT NOT NULL,
    number     TEXT NOT NULL,                -- invoice, order or receipt number
    total      REAL NOT NULL CHECK (total >= 0),
    issued_on  TEXT NOT NULL CHECK (issued_on = date(issued_on)),
    ocr_text   TEXT NOT NULL,                -- Gemini OCR of the deskewed scan
    summary    TEXT NOT NULL,                -- two sentences picked by TextRank
    cer        REAL NOT NULL,                -- character error rate of ocr_text
    skew_deg   REAL NOT NULL,                -- rotation the deskew step measured and removed
    in_gallery INTEGER NOT NULL DEFAULT 0 CHECK (in_gallery IN (0, 1))
);

-- Text blocks found by dilation and contours, in deskewed-image pixels.
CREATE TABLE blocks (
    id     INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents (id),
    x      INTEGER NOT NULL,
    y      INTEGER NOT NULL,
    w      INTEGER NOT NULL CHECK (w > 0),
    h      INTEGER NOT NULL CHECK (h > 0)
);

-- Entities found by spaCy and the EntityRuler, with offsets into documents.ocr_text.
CREATE TABLE entities (
    id         INTEGER PRIMARY KEY,
    doc_id     TEXT NOT NULL REFERENCES documents (id),
    label      TEXT NOT NULL CHECK (label IN ('ORG', 'PERSON', 'DATE', 'MONEY', 'DOC_ID', 'EMAIL')),
    text       TEXT NOT NULL,
    start_char INTEGER NOT NULL CHECK (start_char >= 0),
    end_char   INTEGER NOT NULL CHECK (end_char > start_char)
);

-- OCR character error rate per document for each scan setting, with and without deskewing.
CREATE TABLE ocr_runs (
    doc_id  TEXT NOT NULL REFERENCES documents (id),
    setting TEXT NOT NULL CHECK (setting IN ('standard', 'hard')),
    variant TEXT NOT NULL CHECK (variant IN ('raw', 'deskewed')),
    cer     REAL NOT NULL CHECK (cer >= 0),
    PRIMARY KEY (doc_id, setting, variant)
);

-- Evaluation results: one headline value per row, details as JSON.
CREATE TABLE metrics (
    name   TEXT PRIMARY KEY,
    value  REAL NOT NULL,
    detail TEXT CHECK (detail IS NULL OR json_valid(detail))
);

CREATE INDEX blocks_doc_id ON blocks (doc_id);
CREATE INDEX entities_doc_id ON entities (doc_id);
CREATE INDEX entities_label_text ON entities (label, text);
CREATE INDEX documents_issued_on ON documents (issued_on);
CREATE INDEX documents_vendor ON documents (vendor);

-- Full-text index over the OCR text. External content: the text is stored once, in
-- documents, and the index is filled with the 'rebuild' command after loading.
CREATE VIRTUAL TABLE documents_fts USING fts5 (
    ocr_text,
    content = 'documents',
    tokenize = 'porter unicode61'
);
