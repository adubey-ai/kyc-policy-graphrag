"""Naive lexical RAG vs Graph RAG retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import networkx as nx

from kyc_graphrag.communities import Community
from kyc_graphrag.extract import ENTITY_TYPES


TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


def _tfidf_scores(query: str, corpus: list[str]) -> list[float]:
    q = tokenize(query)
    docs = [tokenize(c) for c in corpus]
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    n = max(len(docs), 1)
    idf = {t: math.log((n + 1) / (df[t] + 1)) + 1.0 for t in set(q)}
    scores: list[float] = []
    q_counts = Counter(q)
    for doc in docs:
        d_counts = Counter(doc)
        score = 0.0
        for term, qtf in q_counts.items():
            if term not in d_counts:
                continue
            tf = d_counts[term] / max(len(doc), 1)
            score += qtf * tf * idf.get(term, 0.0)
        scores.append(score)
    return scores


@dataclass
class Hit:
    kind: str
    score: float
    text: str
    meta: dict[str, str]


def naive_retrieve(query: str, chunks: list[dict[str, str]], k: int = 3) -> list[Hit]:
    scores = _tfidf_scores(query, [c["text"] for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    hits: list[Hit] = []
    for score, chunk in ranked[:k]:
        if score <= 0:
            continue
        hits.append(
            Hit(
                kind="chunk",
                score=score,
                text=chunk["text"],
                meta={"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"]},
            )
        )
    return hits


def _match_entities(query: str, graph: nx.MultiDiGraph) -> list[str]:
    q = query.lower()
    hits: list[tuple[int, str]] = []
    for node in graph.nodes:
        aliases = {node.lower(), ENTITY_TYPES.get(node, "").lower()}
        if any(alias and alias in q for alias in aliases if alias):
            hits.append((len(node), node))
        else:
            tokens = tokenize(node)
            if tokens and all(t in tokenize(query) for t in tokens):
                hits.append((len(node), node))
    # Prefer longer names (Politically Exposed vs PEP handled via aliases below)
    extra = []
    alias_map = {
        "pep": "PEP",
        "cbs": "CBS",
        "maker": "Maker",
        "checker": "Checker",
        "edd": "EDD",
        "ovd": "OVD",
        "iasw": "IASW Agent",
        "address": "Address Change",
        "current": "Current Account",
        "savings": "Savings Account",
        "freeze": "Financial Crime Unit",
        "hrc-22": "HRC-22",
        "hrc22": "HRC-22",
    }
    q_tokens = set(tokenize(query))
    for token, node in alias_map.items():
        if token in q_tokens or token in q:
            extra.append(node)
    ordered = [n for _, n in sorted(hits, reverse=True)]
    for n in extra:
        if n not in ordered:
            ordered.append(n)
    return ordered


def graph_retrieve(
    query: str,
    graph: nx.MultiDiGraph,
    communities: list[Community],
    hops: int = 2,
    k_local: int = 8,
    k_global: int = 2,
) -> list[Hit]:
    seeds = _match_entities(query, graph)
    q_tokens = set(tokenize(query))
    hits: list[Hit] = []

    # Local: ego neighbourhood of matched entities (Microsoft "local search").
    seen_edges: set[tuple[str, str, str]] = set()
    for seed in seeds:
        if seed not in graph:
            continue
        ego = nx.ego_graph(graph.to_undirected(), seed, radius=hops)
        for u, v, data in graph.edges(data=True):
            if u not in ego or v not in ego:
                continue
            key = (u, data["relation"], v)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            blob = tokenize(f"{u} {data['relation']} {v} {data['evidence']}")
            overlap = len(q_tokens.intersection(blob))
            seed_bonus = 2.0 if seed in (u, v) else 1.0
            hits.append(
                Hit(
                    kind="triple",
                    score=seed_bonus + 0.4 * overlap,
                    text=f"{u} -[{data['relation']}]-> {v}. Evidence: {data['evidence']}",
                    meta={"source": data["source"], "seed": seed},
                )
            )

    # Global: community summaries that overlap the query or the seed set.
    seed_set = set(seeds)
    comm_scores = []
    for comm in communities:
        overlap = len(seed_set.intersection(comm.members))
        lexical = _tfidf_scores(query, [comm.summary + " " + " ".join(comm.members)])[0]
        comm_scores.append((overlap * 2 + lexical, comm))
    comm_scores.sort(key=lambda x: x[0], reverse=True)
    for score, comm in comm_scores[:k_global]:
        if score <= 0:
            continue
        hits.append(
            Hit(
                kind="community",
                score=score,
                text=f"Community {comm.community_id} ({', '.join(comm.members)}): {comm.summary}",
                meta={"community_id": str(comm.community_id)},
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[: k_local + k_global]
