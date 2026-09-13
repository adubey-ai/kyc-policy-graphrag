#!/usr/bin/env python3
"""Index KYC policies and compare naive chunk RAG vs hybrid Graph RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kyc_graphrag.eval import load_gold, score_case
from kyc_graphrag.generate import answer_from_hits
from kyc_graphrag.graph import export_graph, graph_stats
from kyc_graphrag.pipeline import build_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, default=ROOT / "data" / "policies")
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "gold.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "graph.json")
    parser.add_argument("--sparse", action="store_true", help="Skip MiniLM even if installed")
    args = parser.parse_args()

    pipe = build_pipeline(args.policies, prefer_dense=not args.sparse)
    gold = load_gold(args.gold)

    print("embedder:", pipe.embedder.kind)
    print("documents:", len(pipe.docs))
    print("triples:", len(pipe.triples))
    print("graph:", graph_stats(pipe.graph))
    print("communities:", len(pipe.communities))
    for comm in pipe.communities:
        print(f"  C{comm.community_id}: {', '.join(comm.members)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = export_graph(pipe.graph)
    payload["embedder"] = pipe.embedder.kind
    payload["communities"] = [
        {"id": c.community_id, "members": c.members, "summary": c.summary, "source_docs": c.source_docs}
        for c in pipe.communities
    ]
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", args.out)

    naive_n = graph_n = 0
    print("\n=== held-out retrieval ===")
    for case in gold:
        naive_hits = pipe.index.naive(case.question)
        graph_hits = pipe.index.search(case.question)
        n_ok = score_case(naive_hits, case)["hit"]
        g_ok = score_case(graph_hits, case)["hit"]
        naive_n += int(n_ok)
        graph_n += int(g_ok)
        print(f"\n[{case.id}] {case.question}")
        print(f"  naive {'HIT' if n_ok else 'MISS'} | graph {'HIT' if g_ok else 'MISS'} | hop={case.hop}")
        print(f"  {case.why_graph}")
        print("  graph:")
        for hit in graph_hits[:3]:
            print(f"    - ({hit.kind}) {hit.text[:220]}")

    print(f"\nheld-out  naive={naive_n}/{len(gold)}  graph={graph_n}/{len(gold)}")
    print("\n--- sample answer ---\n")
    print(answer_from_hits(gold[0].question, pipe.index.search(gold[0].question), "graph"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
