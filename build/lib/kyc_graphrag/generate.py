"""Cited extractive answers. Optional LLM rewrite if OPENAI_API_KEY is set."""

from __future__ import annotations

import os

from kyc_graphrag.retrieve import Hit


def answer_from_hits(query: str, hits: list[Hit], mode: str) -> str:
    if not hits:
        return f"[{mode}] No supporting context for: {query}"
    lines = [f"[{mode}] {query}", ""]
    cites: list[str] = []
    for i, hit in enumerate(hits, 1):
        cite = hit.meta.get("sent_id") or hit.meta.get("chunk_id") or hit.meta.get("doc_id", "")
        if cite:
            cites.append(cite)
        lines.append(f"{i}. ({hit.kind} | {cite}) {hit.text}")
    lines.append("")
    lines.append("Citations: " + ", ".join(dict.fromkeys(cites)))
    text = "\n".join(lines)
    return _maybe_llm(query, text, mode)


def _maybe_llm(query: str, grounded: str, mode: str) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return grounded
    try:
        import json
        from urllib.request import Request, urlopen

        body = json.dumps(
            {
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer only from the cited context. If the graph says no CBS write, say no. Quote citations.",
                    },
                    {"role": "user", "content": f"Question: {query}\n\nContext:\n{grounded}"},
                ],
            }
        ).encode()
        req = Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        reply = data["choices"][0]["message"]["content"]
        return f"[{mode} llm]\n{reply}\n\n--- grounded ---\n{grounded}"
    except Exception:
        return grounded
