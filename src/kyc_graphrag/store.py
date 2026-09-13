"""Durable SQLite property-graph snapshot with FTS5 provenance search."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import networkx as nx

from kyc_graphrag.communities import Community
from kyc_graphrag.corpus import Document

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, path TEXT NOT NULL, text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id),
  text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
  node_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, sources_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
  edge_id INTEGER PRIMARY KEY, source_node TEXT NOT NULL REFERENCES nodes(node_id),
  relation TEXT NOT NULL, target_node TEXT NOT NULL REFERENCES nodes(node_id),
  doc_id TEXT NOT NULL REFERENCES documents(doc_id), sent_id TEXT NOT NULL,
  evidence TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
CREATE TABLE IF NOT EXISTS communities (
  community_id INTEGER PRIMARY KEY, members_json TEXT NOT NULL,
  summary TEXT NOT NULL, source_docs_json TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
  chunk_id UNINDEXED, doc_id UNINDEXED, text
);
"""


class SQLiteGraphStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        return conn

    def replace(
        self,
        docs: Iterable[Document],
        chunks: list[dict[str, str]],
        graph: nx.MultiDiGraph,
        communities: list[Community],
    ) -> None:
        with self.connect() as conn:
            for table in ("chunk_fts", "edges", "communities", "nodes", "chunks", "documents"):
                conn.execute(f"DELETE FROM {table}")  # table names are fixed above
            conn.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                [(d.doc_id, d.title, d.path, d.text) for d in docs],
            )
            conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?)",
                [(c["chunk_id"], c["doc_id"], c["text"]) for c in chunks],
            )
            conn.executemany(
                "INSERT INTO chunk_fts VALUES (?, ?, ?)",
                [(c["chunk_id"], c["doc_id"], c["text"]) for c in chunks],
            )
            conn.executemany(
                "INSERT INTO nodes VALUES (?, ?, ?)",
                [
                    (node, data["entity_type"], json.dumps(sorted(data["sources"])))
                    for node, data in graph.nodes(data=True)
                ],
            )
            conn.executemany(
                """INSERT INTO edges(
                     source_node, relation, target_node, doc_id, sent_id, evidence
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        source,
                        data["relation"],
                        target,
                        data["source"],
                        data.get("sent_id", ""),
                        data["evidence"],
                    )
                    for source, target, data in graph.edges(data=True)
                ],
            )
            conn.executemany(
                "INSERT INTO communities VALUES (?, ?, ?, ?)",
                [
                    (
                        c.community_id,
                        json.dumps(c.members),
                        c.summary,
                        json.dumps(c.source_docs),
                    )
                    for c in communities
                ],
            )

    def neighbors(self, entity: str) -> list[dict[str, str]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT source_node, relation, target_node, doc_id, sent_id, evidence
                   FROM edges WHERE source_node = ? OR target_node = ?
                   ORDER BY relation, source_node, target_node""",
                (entity, entity),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            return {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("documents", "chunks", "nodes", "edges", "communities")
            }

    def lexical_search(self, query: str, limit: int = 5) -> list[dict[str, str | float]]:
        safe_query = " ".join(token for token in query.split() if token.isalnum())
        if not safe_query:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT chunk_id, doc_id, text, bm25(chunk_fts) AS score
                   FROM chunk_fts WHERE chunk_fts MATCH ?
                   ORDER BY score LIMIT ?""",
                (safe_query, limit),
            ).fetchall()
        return [dict(row) for row in rows]
