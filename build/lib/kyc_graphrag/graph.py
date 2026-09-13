"""Property graph over extracted triples."""

from __future__ import annotations

from typing import Any

import networkx as nx

from kyc_graphrag.extract import Triple
from kyc_graphrag.schema import ENTITIES

ENTITY_TYPES = {e.canonical: e.entity_type for e in ENTITIES}


def build_graph(triples: list[Triple]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for triple in triples:
        for name in (triple.head, triple.tail):
            if name not in graph:
                graph.add_node(
                    name,
                    entity_type=ENTITY_TYPES.get(name, "Unknown"),
                    sources=set(),
                )
            graph.nodes[name]["sources"].add(triple.source)
        graph.add_edge(
            triple.head,
            triple.tail,
            relation=triple.relation,
            source=triple.source,
            evidence=triple.evidence,
            sent_id=getattr(triple, "sent_id", ""),
        )
    return graph


def graph_stats(graph: nx.MultiDiGraph) -> dict[str, Any]:
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "entity_types": sorted({d["entity_type"] for _, d in graph.nodes(data=True)}),
    }


def export_graph(graph: nx.MultiDiGraph) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": n,
                "type": d["entity_type"],
                "sources": sorted(d["sources"]),
            }
            for n, d in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "relation": d["relation"],
                "doc": d["source"],
                "evidence": d["evidence"],
                "sent_id": d.get("sent_id", ""),
            }
            for u, v, d in graph.edges(data=True)
        ],
    }
