from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kyc_graphrag.retrieve import Hit


@dataclass
class GoldCase:
    id: str
    question: str
    answer_key: list[str]
    must_cite_any: list[str]
    hop: bool
    why_graph: str


def load_gold(path: Path) -> list[GoldCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GoldCase(**row) for row in raw]


def _doc_ids(hits: list[Hit]) -> set[str]:
    found: set[str] = set()
    for hit in hits:
        for part in hit.meta.get("doc_id", "").split(","):
            part = part.strip()
            if part:
                found.add(part)
    return found


def score_case(hits: list[Hit], case: GoldCase) -> dict[str, bool]:
    blob = "\n".join(h.text for h in hits).lower()
    keys_ok = all(k.lower() in blob for k in case.answer_key)
    cite_ok = bool(_doc_ids(hits).intersection(case.must_cite_any))
    return {"keys": keys_ok, "cite": cite_ok, "hit": keys_ok and cite_ok}


def reciprocal_rank(hits: list[Hit], case: GoldCase) -> float:
    relevant = set(case.must_cite_any)
    for rank, hit in enumerate(hits, 1):
        docs = {part.strip() for part in hit.meta.get("doc_id", "").split(",")}
        if docs.intersection(relevant):
            return 1.0 / rank
    return 0.0


def recall_at_k(hits: list[Hit], case: GoldCase, k: int) -> float:
    relevant = set(case.must_cite_any)
    found: set[str] = set()
    for hit in hits[:k]:
        found.update(part.strip() for part in hit.meta.get("doc_id", "").split(","))
    return len(found.intersection(relevant)) / len(relevant)
