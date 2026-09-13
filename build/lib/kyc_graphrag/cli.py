"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from kyc_graphrag.generate import answer_from_hits
from kyc_graphrag.pipeline import build_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(prog="kyc-graphrag")
    parser.add_argument("question")
    parser.add_argument("--policies", type=Path, default=Path("data/policies"))
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    pipe = build_pipeline(args.policies, prefer_dense=not args.sparse)
    hits = pipe.index.search(args.question, k=args.top_k)
    print(answer_from_hits(args.question, hits, "hybrid-graph"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
