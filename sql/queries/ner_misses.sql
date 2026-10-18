-- Documents where NER missed the total or the vendor
-- Two anti-joins: each document is LEFT JOINed to the extracted amount equal to its known
-- total and to the ORG entity equal to its vendor; a NULL on the right means NER missed it.
WITH amounts AS (
    SELECT DISTINCT doc_id, ROUND(CAST(REPLACE(REPLACE(text, '$', ''), ',', '') AS REAL), 2) AS value
    FROM entities
    WHERE label = 'MONEY'
),
organisations AS (
    SELECT DISTINCT doc_id, text
    FROM entities
    WHERE label = 'ORG'
)
SELECT
    d.id,
    d.type,
    d.vendor,
    ROUND(d.total, 2) AS total,
    a.doc_id IS NULL AS missed_total,
    o.doc_id IS NULL AS missed_vendor
FROM documents AS d
LEFT JOIN amounts AS a ON a.doc_id = d.id AND a.value = ROUND(d.total, 2)
LEFT JOIN organisations AS o ON o.doc_id = d.id AND o.text = d.vendor
WHERE a.doc_id IS NULL OR o.doc_id IS NULL
ORDER BY d.type, d.id;
