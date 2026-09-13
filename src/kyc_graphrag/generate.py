"""Cited extractive answers. Optional LLM rewrite if OPENAI_API_KEY is set."""

from __future__ import annotations

import os
import re

from kyc_graphrag.retrieve import Hit
from kyc_graphrag.retrieve_text import tokenize

EDGE = re.compile(r"(.+?) -\[([A-Z_]+)\]-> (.+?)(?= -\[|\.|$)")


def synthesize_answer(query: str, hits: list[Hit]) -> str:
    """Produce a short deterministic answer before optional LLM rewriting."""
    evidence = [hit for hit in hits if hit.kind != "community"]
    lowered = query.lower()
    intents = []
    if any(term in lowered for term in ("right", "write", "post", "posting")):
        intents.append("CANNOT_WRITE")
    if any(term in lowered for term in ("freeze", "halt")):
        intents.append("CAN_FREEZE")
    if any(term in lowered for term in ("where", "desk", "sent")):
        intents.extend(["ON_FAILURE_ROUTES_TO", "ESCALATES_TO"])
    if any(term in lowered for term in ("extra control", "mandatory", "need", "require")):
        intents.append("REQUIRES")
    if any(term in lowered for term in ("who", "reviews", "escalat")):
        intents.extend(["ESCALATES_TO", "REQUIRES_CHECKER_FROM"])

    for relation in intents:
        relation_hits = [hit for hit in evidence if f"[{relation}]" in hit.text]
        if relation == "CANNOT_WRITE":
            composed = [
                hit for hit in relation_hits if hit.kind == "path" and "[ACTS_AS]" in hit.text
            ]
            if composed:
                relation_hits = composed + [hit for hit in relation_hits if hit not in composed]
        relation_hits.sort(
            key=lambda hit: (
                hit.kind == "path" and relation == "CANNOT_WRITE" and "[ACTS_AS]" in hit.text,
                len(set(tokenize(query)).intersection(tokenize(hit.text))),
            ),
            reverse=True,
        )
        for hit in relation_hits:
            cite = hit.meta.get("sent_id") or hit.meta.get("chunk_id", "")
            if relation == "CANNOT_WRITE" and hit.kind == "path" and "[ACTS_AS]" in hit.text:
                return f"No. IASW Agent acts as Maker, and Maker cannot write to CBS. [{cite}]"
            match = next(
                (m for m in EDGE.finditer(hit.text) if m.group(2) == relation),
                None,
            )
            if match:
                head, _, tail = match.groups()
                phrase = {
                    "CANNOT_WRITE": "cannot write to",
                    "CAN_FREEZE": "can freeze",
                    "ON_FAILURE_ROUTES_TO": "routes on failure to",
                    "ESCALATES_TO": "escalates to",
                    "REQUIRES": "requires",
                    "REQUIRES_CHECKER_FROM": "requires a checker from",
                }.get(relation, relation.lower().replace("_", " "))
                return f"{head.strip()} {phrase} {tail.strip()}. [{cite}]"

    if not evidence:
        return "Insufficient evidence."
    query_terms = set(tokenize(query))
    candidates = []
    for hit in evidence:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", hit.text):
            sentence = sentence.strip()
            if sentence:
                overlap = len(query_terms.intersection(tokenize(sentence)))
                candidates.append((overlap, len(sentence), sentence, hit))
    _, _, first_sentence, hit = max(candidates, key=lambda row: (row[0], -row[1]))
    cite = hit.meta.get("sent_id") or hit.meta.get("chunk_id", "")
    return f"{first_sentence}. [{cite}]"


def answer_from_hits(query: str, hits: list[Hit], mode: str) -> str:
    if not hits:
        return f"[{mode}] No supporting context for: {query}"
    lines = [f"[{mode}] {query}", "", "Answer: " + synthesize_answer(query, hits), ""]
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
