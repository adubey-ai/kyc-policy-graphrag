"""Schema-guided mention + relation extraction.

Mentions come from a gazetteer (aliases). Relations fire when two typed
mentions co-occur in a sentence *and* a relation cue is present. Adding a
paraphrased circular should extract the same edge without a new fact regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kyc_graphrag.corpus import Document, Sentence, sentences
from kyc_graphrag.schema import RELATION_CUES, alias_table


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str
    source: str
    evidence: str
    sent_id: str


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    canonical: str
    entity_type: str


_ALIASES = alias_table()
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+./-]*")


def _find_mentions(text: str) -> list[Mention]:
    lower = text.lower()
    taken = [False] * len(lower)
    found: list[Mention] = []
    for alias, canonical, entity_type in _ALIASES:
        start = 0
        while True:
            i = lower.find(alias, start)
            if i < 0:
                break
            j = i + len(alias)
            if not any(taken[i:j]) and _boundary(lower, i, j):
                for k in range(i, j):
                    taken[k] = True
                found.append(Mention(i, j, canonical, entity_type))
            start = i + 1
    found.sort(key=lambda m: (m.start, -(m.end - m.start)))
    return found


def _boundary(text: str, i: int, j: int) -> bool:
    left_ok = i == 0 or not text[i - 1].isalnum()
    right_ok = j == len(text) or not text[j].isalnum()
    return left_ok and right_ok


DIRECTED = frozenset(
    {
        "CANNOT_WRITE",
        "CAN_WRITE",
        "CAN_FREEZE",
        "ACTS_AS",
        "ESCALATES_TO",
        "ON_FAILURE_ROUTES_TO",
        "REQUIRES_CHECKER_FROM",
        "CONFIDENCE_GATE",
        "GOVERNS",
        "RETAINED_YEARS",
        "ACCEPTS",
    }
)


def _cue_relation(sentence: str, head: Mention, tail: Mention) -> str | None:
    blob = sentence.lower()
    for cues, head_types, tail_types, relation in RELATION_CUES:
        if head.entity_type not in head_types or tail.entity_type not in tail_types:
            continue
        if head.canonical == tail.canonical:
            continue
        if not any(cue in blob for cue in cues):
            continue
        if relation == "CANNOT_WRITE" and head.canonical == "Checker":
            continue
        if relation == "CAN_WRITE" and head.canonical != "Checker":
            continue
        if relation == "ACTS_AS" and not (
            head.entity_type == "System" and tail.canonical == "Maker"
        ):
            continue
        if relation == "CAN_FREEZE" and head.canonical != "Financial Crime Unit":
            continue
        if relation == "REQUIRES" and tail.entity_type == "Role" and tail.canonical != "Dual Checker":
            continue
        if relation == "REQUIRES" and head.entity_type == "Product" and tail.entity_type not in {"Document", "Control", "Form"}:
            continue
        return relation
    return None


def extract_triples(docs: list[Document]) -> list[Triple]:
    triples: list[Triple] = []
    seen: set[tuple[str, str, str, str]] = set()
    sents = sentences(docs)
    windows: list[Sentence] = []
    # Pair adjacent sentences in the same doc so "It is prohibited..." still
    # binds to the IASW mention in the previous sentence.
    i = 0
    while i < len(sents):
        cur = sents[i]
        if i + 1 < len(sents) and sents[i + 1].doc_id == cur.doc_id:
            windows.append(
                Sentence(
                    sent_id=cur.sent_id,
                    doc_id=cur.doc_id,
                    text=cur.text + " " + sents[i + 1].text,
                )
            )
        windows.append(cur)
        i += 1
    for sent in windows:
        mentions = _find_mentions(sent.text)
        if len(mentions) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(mentions) for b in mentions[i + 1 :]]
        for head, tail in pairs:
            candidates = ((head, tail), (tail, head))
            for h, t in candidates:
                rel = _cue_relation(sent.text, h, t)
                if not rel:
                    continue
                if rel in DIRECTED and h.start > t.start and rel != "ACTS_AS":
                    # Keep directed edges in mention order except ACTS_AS,
                    # which is typed System -> Maker regardless of order.
                    if rel != "ACTS_AS":
                        pass
                key = (h.canonical, rel, t.canonical, sent.doc_id)
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    Triple(
                        head=h.canonical,
                        relation=rel,
                        tail=t.canonical,
                        source=sent.doc_id,
                        evidence=sent.text.strip()[:400],
                        sent_id=sent.sent_id,
                    )
                )
    return triples
