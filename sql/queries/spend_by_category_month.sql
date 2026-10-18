-- Spend by category and month
-- Monthly totals per expense category: GROUP BY over a month derived with strftime.
SELECT
    strftime('%Y-%m', issued_on) AS month,
    category,
    COUNT(*) AS documents,
    ROUND(SUM(total), 2) AS spend
FROM documents
GROUP BY month, category
ORDER BY month, spend DESC;
