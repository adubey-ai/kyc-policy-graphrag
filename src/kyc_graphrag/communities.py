"""Louvain communities and extractive summaries."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


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
    if undirected.number_of_edges() == 0:
        return []
    try:
        raw = nx.community.louvain_communities(undirected, weight="weight", seed=7)
    except Exception:
        raw = nx.community.greedy_modularity_communities(undirected, weight="weight")
    communities: list[Community] = []
    for idx, members in enumerate(raw):
        member_list = sorted(members)
        evidence: list[str] = []
        sources: set[str] = set()
        for u, v, data in undirected.edges(data=True):
            if u in members and v in members:
                evidence.extend(data["evidence"])
                sources.update(data["sources"])
        # Keep the most unique evidence snippets, not a wall of text.
        uniq = list(dict.fromkeys(evidence))[:6]
        summary = " ".join(uniq) or ", ".join(member_list)
        communities.append(
            Community(
                community_id=idx,
                members=member_list,
                summary=summary,
                source_docs=sorted(sources),
            )
        )
    return communities
