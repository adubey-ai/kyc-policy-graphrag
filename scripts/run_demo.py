#!/usr/bin/env python3
"""Index KYC policies and compare naive chunk RAG vs hybrid Graph RAG."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kyc_graphrag.eval import (
    extraction_metrics,
    load_gold,
    recall_at_k,
    reciprocal_rank,
    score_case,
)
from kyc_graphrag.generate import answer_from_hits
from kyc_graphrag.graph import export_graph, graph_stats
from kyc_graphrag.pipeline import build_pipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, default=ROOT / "data" / "policies")
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "gold.json")
    parser.add_argument(
        "--gold-triples",
        type=Path,
        default=ROOT / "data" / "gold_triples.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "graph.json")
    parser.add_argument("--sparse", action="store_true", help="Skip MiniLM even if installed")
    parser.add_argument("--k", type=int, default=6, help="Equal retrieval budget")
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
        {
            "id": c.community_id,
            "members": c.members,
            "summary": c.summary,
            "source_docs": c.source_docs,
        }
        for c in pipe.communities
    ]
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", args.out)
    extraction = extraction_metrics(pipe.triples, args.gold_triples)
    print(
        "extraction "
        f"precision={extraction['precision']:.3f} "
        f"recall={extraction['recall']:.3f} "
        f"f1={extraction['f1']:.3f} "
        f"forbidden={extraction['forbidden_emitted']}"
    )

    naive_n = graph_n = 0
    graph_paths = path_cases = 0
    naive_rr = graph_rr = graph_recall = 0.0
    naive_latency: list[float] = []
    graph_latency: list[float] = []
    print("\n=== held-out retrieval ===")
    for case in gold:
        started = time.perf_counter()
        naive_hits = pipe.index.naive(case.question, k=args.k)
        naive_latency.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        graph_hits = pipe.index.search(case.question, k=args.k)
        graph_latency.append((time.perf_counter() - started) * 1000)
        n_ok = score_case(naive_hits, case)["hit"]
        g_ok = score_case(graph_hits, case)["hit"]
        naive_n += int(n_ok)
        graph_n += int(g_ok)
        if case.required_path:
            path_cases += 1
            graph_paths += int(score_case(graph_hits, case)["path"])
        naive_rr += reciprocal_rank(naive_hits, case)
        graph_rr += reciprocal_rank(graph_hits, case)
        graph_recall += recall_at_k(graph_hits, case, k=args.k)
        print(f"\n[{case.id}] {case.question}")
        print(
            f"  naive {'HIT' if n_ok else 'MISS'} | graph {'HIT' if g_ok else 'MISS'} | hop={case.hop}"
        )
        print(f"  {case.why_graph}")
        print("  graph:")
        for hit in graph_hits[:3]:
            print(f"    - ({hit.kind}) {hit.text[:220]}")

    print(f"\nequal-k@{args.k}  naive={naive_n}/{len(gold)}  graph={graph_n}/{len(gold)}")
    if path_cases:
        print(f"path composition  graph={graph_paths}/{path_cases}")
    print(
        "ranking   "
        f"naive_mrr={naive_rr / len(gold):.3f}  "
        f"graph_mrr={graph_rr / len(gold):.3f}  "
        f"graph_recall@{args.k}={graph_recall / len(gold):.3f}"
    )
    print(
        "latency   "
        f"naive_mean_ms={sum(naive_latency) / len(naive_latency):.2f}  "
        f"graph_mean_ms={sum(graph_latency) / len(graph_latency):.2f}"
    )
    print("\n--- sample answer ---\n")
    print(
        answer_from_hits(
            gold[0].question,
            pipe.index.search(gold[0].question, k=args.k),
            "graph",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
