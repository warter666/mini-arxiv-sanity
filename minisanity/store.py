"""SQLite-backed paper and vote storage."""

import json
import sqlite3
from dataclasses import dataclass, field


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    embedding: list = field(default_factory=list)  # reserved for future vector search

    @property
    def text(self):
        return f"{self.title} {self.abstract} {' '.join(self.tags)}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,
    tags TEXT NOT NULL,
    embedding TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS votes (
    arxiv_id TEXT PRIMARY KEY,
    up INTEGER NOT NULL
);
"""


class Store:
    def __init__(self, path=":memory:"):
        # FastAPI runs sync endpoints in a threadpool, so allow cross-thread use
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def upsert_paper(self, p: Paper):
        self.conn.execute(
            "INSERT INTO papers VALUES (?,?,?,?,?,?) ON CONFLICT(arxiv_id) DO "
            "UPDATE SET title=excluded.title, abstract=excluded.abstract, "
            "authors=excluded.authors, tags=excluded.tags",
            (p.arxiv_id, p.title, p.abstract,
             json.dumps(p.authors), json.dumps(p.tags), json.dumps(p.embedding)),
        )
        self.conn.commit()

    def all_papers(self):
        for row in self.conn.execute("SELECT * FROM papers"):
            yield Paper(row["arxiv_id"], row["title"], row["abstract"],
                        json.loads(row["authors"]), json.loads(row["tags"]),
                        json.loads(row["embedding"]))

    def vote(self, arxiv_id: str, up: bool):
        if up:
            self.conn.execute(
                "INSERT INTO votes VALUES (?,1) ON CONFLICT(arxiv_id) DO "
                "UPDATE SET up=1", (arxiv_id,))
        else:
            self.conn.execute("DELETE FROM votes WHERE arxiv_id=?", (arxiv_id,))
        self.conn.commit()

    def upvoted_ids(self):
        return {r["arxiv_id"] for r in self.conn.execute("SELECT arxiv_id FROM votes WHERE up=1")}
