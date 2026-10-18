-- Full-text search
-- FTS5 MATCH over the OCR text, ranked by BM25, with the matching words marked in a
-- snippet. The search box binds its words to :q.
SELECT
    d.id,
    d.type,
    d.vendor,
    d.issued_on,
    snippet(documents_fts, 0, '«', '»', ' … ', 12) AS snippet
FROM documents_fts
JOIN documents AS d ON d.rowid = documents_fts.rowid
WHERE documents_fts MATCH :q
ORDER BY rank
LIMIT 20;
