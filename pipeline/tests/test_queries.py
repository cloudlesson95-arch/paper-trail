"""Every named query runs against the committed database and returns rows."""

import sqlite3

import pytest

from common import DB_PATH, SQL

QUERIES = sorted((SQL / "queries").glob("*.sql"))


@pytest.fixture(scope="module")
def db():
    con = sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True)
    yield con
    con.close()


def test_there_are_named_queries():
    assert len(QUERIES) >= 6


@pytest.mark.parametrize("path", QUERIES, ids=lambda p: p.stem)
def test_query_returns_rows(db, path):
    sql = path.read_text(encoding="utf-8")
    assert sql.startswith("-- "), "a query file starts with its title comment"
    rows = db.execute(sql, {"q": '"invoice"'} if ":q" in sql else {}).fetchall()
    assert rows


def test_foreign_keys_hold(db):
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_full_text_index_covers_every_document(db):
    documents = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    matched = db.execute("SELECT COUNT(*) FROM documents_fts WHERE documents_fts MATCH 'total OR dear'").fetchone()[0]
    assert db.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0] == documents
    assert matched > 0
