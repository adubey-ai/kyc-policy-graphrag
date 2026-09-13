"""Community detection and extractive community summaries.

Microsoft GraphRAG uses Leiden communities plus LLM summaries. This demo uses
greedy modularity (no extra C deps) and concatenates evidence strings so the
pipeline stays offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from kyc_graphrag.extract import Document


@dataclass
class Community:
    community_id: int
    members: list[str]
    summary: str
    source_docs: list[str]


def detect_communities(graph: nx.MultiDiGraph) -> list[Community]:
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes())
    for u, v, data in graph.edges(data=True):
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += 1
            undirected[u][v]["evidence"].append(data["evidence"])
            undirected[u][v]["sources"].add(data["source"])
        else:
            undirected.add_edge(
                u,
                v,
                weight=1,
                evidence=[data["evidence"]],
                sources={data["source"]},
            )

    raw = greedy_modularity_communities(undirected, weight="weight")
    communities: list[Community] = []
    for idx, members in enumerate(raw):
        member_list = sorted(members)
        evidence: list[str] = []
        sources: set[str] = set()
        for u, v, data in undirected.edges(data=True):
            if u in members and v in members:
                evidence.extend(data["evidence"])
                sources.update(data["sources"])
        summary = " ".join(dict.fromkeys(evidence)) or ", ".join(member_list)
        communities.append(
            Community(
                community_id=idx,
                members=member_list,
                summary=summary,
                source_docs=sorted(sources),
            )
        )
    return communities


def chunk_documents(docs: list[Document], max_chars: int = 420) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for doc in docs:
        paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
        buf = ""
        part = 0
        for para in paragraphs:
            if buf and len(buf) + len(para) > max_chars:
                chunks.append({"chunk_id": f"{doc.doc_id}:{part}", "doc_id": doc.doc_id, "text": buf})
                part += 1
                buf = para
            else:
                buf = f"{buf}\n\n{para}".strip()
        if buf:
            chunks.append({"chunk_id": f"{doc.doc_id}:{part}", "doc_id": doc.doc_id, "text": buf})
    return chunks
