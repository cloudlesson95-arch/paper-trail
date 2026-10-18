-- People and organisations that appear together
-- Self-join of the entities table on the document: each PERSON paired with each ORG found
-- in the same document, counted across documents. The small spaCy model's false positives
-- show up here too (a street name or a form label tagged as a person); see the metrics page.
SELECT
    p.text AS person,
    o.text AS organisation,
    COUNT(DISTINCT p.doc_id) AS documents
FROM entities AS p
JOIN entities AS o ON o.doc_id = p.doc_id AND o.label = 'ORG'
WHERE p.label = 'PERSON'
GROUP BY p.text, o.text
HAVING COUNT(DISTINCT p.doc_id) >= 2
ORDER BY documents DESC, person, organisation
LIMIT 20;
