#!/usr/bin/env python3
"""Build the KYC policy graph and compare naive RAG vs Graph RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kyc_graphrag.communities import chunk_documents, detect_communities
from kyc_graphrag.eval import CASES
from kyc_graphrag.extract import extract_triples, load_documents
from kyc_graphrag.generate import answer_from_hits
from kyc_graphrag.graph import build_graph, export_graph, graph_stats
from kyc_graphrag.retrieve import graph_retrieve, naive_retrieve


def _contains(text: str, needles: tuple[str, ...]) -> bool:
    return all(n.lower() in text.lower() for n in needles)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, default=ROOT / "data" / "policies")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "graph.json")
    args = parser.parse_args()

    docs = load_documents(args.policies)
    triples = extract_triples(docs)
    graph = build_graph(triples)
    communities = detect_communities(graph)
    chunks = chunk_documents(docs)

    print("Documents:", len(docs))
    print("Triples:", len(triples))
    print("Graph:", graph_stats(graph))
    print("Communities:", len(communities))
    for comm in communities:
        print(f"  C{comm.community_id}: {', '.join(comm.members)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = export_graph(graph)
    payload["communities"] = [
        {
            "id": c.community_id,
            "members": c.members,
            "summary": c.summary,
            "source_docs": c.source_docs,
        }
        for c in communities
    ]
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

    print("\n=== Retrieval comparison ===")
    graph_hits_n = naive_hits_n = 0
    for case in CASES:
        naive = naive_retrieve(case.question, chunks)
        graphed = graph_retrieve(case.question, graph, communities)
        naive_ok = _contains(answer_from_hits(case.question, naive, "naive"), case.must_contain)
        graph_ok = _contains(answer_from_hits(case.question, graphed, "graph"), case.must_contain)
        graph_hits_n += int(graph_ok)
        naive_hits_n += int(naive_ok)
        print(f"\nQ: {case.question}")
        print(f"  naive RAG: {'HIT' if naive_ok else 'MISS'} | graph RAG: {'HIT' if graph_ok else 'MISS'}")
        print(f"  why graph: {case.why_graph}")
        print("  graph context:")
        for hit in graphed[:4]:
            print(f"    - {hit.text}")

    print(f"\nScore  naive={naive_hits_n}/{len(CASES)}  graph={graph_hits_n}/{len(CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
