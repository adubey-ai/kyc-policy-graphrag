from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from kyc_graphrag.generate import synthesize_answer
from kyc_graphrag.retrieve import Hit


@dataclass
class GoldCase:
    id: str
    question: str
    answer_key: list[str]
    must_cite_any: list[str]
    hop: bool
    why_graph: str
    required_path: list[str] | None = None


def load_gold(path: Path) -> list[GoldCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GoldCase(**row) for row in raw]


def _doc_ids(hits: list[Hit]) -> set[str]:
    found: set[str] = set()
    for hit in hits:
        if hit.kind == "community":
            continue
        for part in hit.meta.get("doc_id", "").split(","):
            part = part.strip()
            if part:
                found.add(part)
    return found


def score_case(hits: list[Hit], case: GoldCase) -> dict[str, bool]:
    evidence_hits = [hit for hit in hits if hit.kind != "community"]
    answer = synthesize_answer(case.question, evidence_hits)
    blob = answer.lower()
    keys_ok = all(k.lower() in blob for k in case.answer_key)
    cited_docs = {
        citation.strip().split(":", 1)[0]
        for group in re.findall(r"\[([^\]]+)\]", answer)
        for citation in group.split(",")
    }
    cite_ok = bool(cited_docs.intersection(case.must_cite_any))
    path_ok = True
    if case.required_path:
        path_ok = any(
            hit.kind == "path"
            and all(relation.lower() in hit.text.lower() for relation in case.required_path)
            for hit in hits
        )
    return {
        "keys": keys_ok,
        "cite": cite_ok,
        "path": path_ok,
        "hit": keys_ok and cite_ok,
    }


def reciprocal_rank(hits: list[Hit], case: GoldCase) -> float:
    relevant = set(case.must_cite_any)
    for rank, hit in enumerate(hits, 1):
        if hit.kind == "community":
            continue
        docs = {part.strip() for part in hit.meta.get("doc_id", "").split(",")}
        if docs.intersection(relevant):
            return 1.0 / rank
    return 0.0


def recall_at_k(hits: list[Hit], case: GoldCase, k: int) -> float:
    relevant = set(case.must_cite_any)
    found: set[str] = set()
    for hit in hits[:k]:
        if hit.kind == "community":
            continue
        found.update(part.strip() for part in hit.meta.get("doc_id", "").split(","))
    return len(found.intersection(relevant)) / len(relevant)


def extraction_metrics(triples: list, gold_path: Path) -> dict[str, float | int]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    expected = {tuple(row) for row in gold["expected"]}
    forbidden = {tuple(row) for row in gold["forbidden"]}
    predicted = {(triple.source, triple.head, triple.relation, triple.tail) for triple in triples}
    true_positive = len(predicted.intersection(expected))
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted": len(predicted),
        "gold": len(expected),
        "forbidden_emitted": len(predicted.intersection(forbidden)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
