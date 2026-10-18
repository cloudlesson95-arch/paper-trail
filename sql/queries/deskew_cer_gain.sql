-- OCR error with and without deskewing, by document type
-- Two CTEs: the first pivots each document's OCR runs into raw and deskewed columns,
-- the second averages them per scan setting and type; the gain is raw minus deskewed.
WITH per_document AS (
    SELECT
        doc_id,
        setting,
        MAX(CASE WHEN variant = 'raw' THEN cer END) AS raw_cer,
        MAX(CASE WHEN variant = 'deskewed' THEN cer END) AS deskewed_cer
    FROM ocr_runs
    GROUP BY doc_id, setting
),
per_type AS (
    SELECT
        p.setting,
        d.type,
        COUNT(*) AS documents,
        AVG(p.raw_cer) AS raw_cer,
        AVG(p.deskewed_cer) AS deskewed_cer,
        SUM(p.deskewed_cer < p.raw_cer) AS improved,
        SUM(p.deskewed_cer > p.raw_cer) AS worse
    FROM per_document AS p
    JOIN documents AS d ON d.id = p.doc_id
    GROUP BY p.setting, d.type
)
SELECT
    setting,
    type,
    documents,
    ROUND(100 * raw_cer, 2) AS raw_cer_pct,
    ROUND(100 * deskewed_cer, 2) AS deskewed_cer_pct,
    ROUND(100 * (raw_cer - deskewed_cer), 2) AS gain_points,
    improved,
    worse
FROM per_type
ORDER BY setting DESC, type;
