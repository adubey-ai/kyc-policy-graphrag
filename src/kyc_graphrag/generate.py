"""Answer construction from retrieved context. Offline extractive by default."""

from __future__ import annotations

from kyc_graphrag.retrieve import Hit


def answer_from_hits(query: str, hits: list[Hit], mode: str) -> str:
    if not hits:
        return f"[{mode}] No supporting context found for: {query}"
    lines = [f"[{mode}] {query}", ""]
    for i, hit in enumerate(hits, 1):
        lines.append(f"{i}. ({hit.kind}, score={hit.score:.3f}) {hit.text}")
    return "\n".join(lines)
