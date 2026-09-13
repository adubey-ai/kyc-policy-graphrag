"""Hybrid retrieval: dense/sparse chunks + entity linking + 2-hop graph + communities.

Fusion is Reciprocal Rank Fusion so neither channel has to share a score scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from kyc_graphrag.communities import Community
from kyc_graphrag.embed import Embedder, cosine_topk
from kyc_graphrag.schema import alias_table


@dataclass
class Hit:
    kind: str
    score: float
    text: str
    meta: dict[str, str] = field(default_factory=dict)
    rank: int = 0


def _rrf(rank_lists: list[tuple[list[str], float]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked, weight in rank_lists:
        for i, key in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + weight / (k + i + 1)
    return scores


def _link_entities(query: str, graph: nx.MultiDiGraph) -> list[str]:
    q = query.lower()
    hits: list[tuple[int, str]] = []
    for alias, canonical, _ in alias_table():
        if alias in q and canonical in graph:
            hits.append((len(alias), canonical))
    seen: list[str] = []
    for _, name in sorted(hits, reverse=True):
        if name not in seen:
            seen.append(name)
    return seen


class HybridIndex:
    def __init__(
        self,
        chunks: list[dict[str, str]],
        graph: nx.MultiDiGraph,
        communities: list[Community],
        embedder: Embedder,
    ) -> None:
        self.chunks = chunks
        self.graph = graph
        self.communities = communities
        self.embedder = embedder
        corpus = [c["text"] for c in chunks]
        corpus += [c.summary for c in communities]
        corpus += [f"{u} {d['relation']} {v} {d['evidence']}" for u, v, d in graph.edges(data=True)]
        embedder.fit(corpus)
        self.chunk_mat = (
            embedder.encode([c["text"] for c in chunks]) if chunks else np.zeros((0, 1))
        )
        self.comm_mat = (
            embedder.encode([c.summary for c in communities]) if communities else np.zeros((0, 1))
        )
        self.edge_records = [
            {
                "key": f"e:{u}|{d['relation']}|{v}|{d['source']}",
                "text": f"{u} -[{d['relation']}]-> {v}. {d['evidence']}",
                "u": u,
                "v": v,
                "relation": d["relation"],
                "source": d["source"],
                "sent_id": d.get("sent_id", ""),
            }
            for u, v, d in graph.edges(data=True)
        ]
        self.edge_mat = (
            embedder.encode([r["text"] for r in self.edge_records])
            if self.edge_records
            else np.zeros((0, 1))
        )

    def naive(self, query: str, k: int = 4) -> list[Hit]:
        qv = self.embedder.encode([query])[0]
        hits = []
        for idx, score in cosine_topk(qv, self.chunk_mat, k):
            chunk = self.chunks[idx]
            hits.append(
                Hit(
                    kind="chunk",
                    score=score,
                    text=chunk["text"],
                    meta={"doc_id": chunk["doc_id"], "chunk_id": chunk["chunk_id"]},
                )
            )
        return hits

    def search(self, query: str, k: int = 8) -> list[Hit]:
        qv = self.embedder.encode([query])[0]
        seeds = _link_entities(query, self.graph)

        chunk_rank = [self.chunks[i]["chunk_id"] for i, _ in cosine_topk(qv, self.chunk_mat, 8)]
        edge_rank: list[str] = []
        # Prefer edges touching linked entities, then dense score.
        seed_set = set(seeds)
        scored_edges: list[tuple[float, int]] = []
        if self.edge_mat.size:
            dense = self.edge_mat @ qv
            for i, rec in enumerate(self.edge_records):
                bonus = 1.5 if rec["u"] in seed_set or rec["v"] in seed_set else 0.0
                hop_bonus = 0.0
                if seeds:
                    nbrs: set[str] = set()
                    for s in seeds:
                        if s not in self.graph:
                            continue
                        nbrs.add(s)
                        nbrs.update(self.graph.successors(s))
                        nbrs.update(self.graph.predecessors(s))
                    if rec["u"] in nbrs or rec["v"] in nbrs:
                        hop_bonus = 1.0 if rec["u"] in seed_set or rec["v"] in seed_set else 0.55
                scored_edges.append((float(dense[i]) + bonus + hop_bonus, i))
            scored_edges.sort(reverse=True)
            edge_rank = [self.edge_records[i]["key"] for _, i in scored_edges[:12]]

        comm_rank = [
            f"c:{self.communities[i].community_id}" for i, _ in cosine_topk(qv, self.comm_mat, 4)
        ]

        # Edge evidence is the differentiating Graph RAG signal. Chunks retain
        # recall; broad community summaries are supporting context.
        fused = _rrf([(chunk_rank, 0.7), (edge_rank, 1.3), (comm_rank, 0.5)])
        ordered = sorted(fused.items(), key=lambda x: -x[1])[:k]

        by_chunk = {c["chunk_id"]: c for c in self.chunks}
        by_edge = {r["key"]: r for r in self.edge_records}
        by_comm = {f"c:{c.community_id}": c for c in self.communities}

        hits: list[Hit] = []
        for key, score in ordered:
            if key in by_chunk:
                ch = by_chunk[key]
                hits.append(
                    Hit("chunk", score, ch["text"], {"doc_id": ch["doc_id"], "chunk_id": key})
                )
            elif key in by_edge:
                rec = by_edge[key]
                hits.append(
                    Hit(
                        "triple",
                        score,
                        rec["text"],
                        {
                            "doc_id": rec["source"],
                            "sent_id": rec["sent_id"],
                            "relation": rec["relation"],
                        },
                    )
                )
            elif key in by_comm:
                comm = by_comm[key]
                hits.append(
                    Hit(
                        "community",
                        score,
                        f"Community {comm.community_id} ({', '.join(comm.members)}): {comm.summary}",
                        {"doc_id": ",".join(comm.source_docs)},
                    )
                )
        return hits
