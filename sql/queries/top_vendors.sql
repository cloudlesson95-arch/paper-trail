-- Top vendors by spend
-- Vendors ranked by total spend with RANK() OVER, and each vendor's share of its category.
WITH vendor_totals AS (
    SELECT vendor, category, COUNT(*) AS documents, SUM(total) AS spend
    FROM documents
    GROUP BY vendor, category
)
SELECT
    RANK() OVER (ORDER BY spend DESC) AS rank,
    vendor,
    category,
    documents,
    ROUND(spend, 2) AS spend,
    ROUND(100.0 * spend / SUM(spend) OVER (PARTITION BY category), 1) AS pct_of_category
FROM vendor_totals
ORDER BY rank
LIMIT 10;
