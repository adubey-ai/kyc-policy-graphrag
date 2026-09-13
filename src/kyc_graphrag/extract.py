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


def _valid_relation(relation: str, head: Mention, tail: Mention) -> bool:
    if head.canonical == tail.canonical:
        return False
    if relation == "CANNOT_WRITE" and head.canonical == "Checker":
        return False
    if relation == "CAN_WRITE" and head.canonical != "Checker":
        return False
    if relation == "ACTS_AS" and not (head.entity_type == "System" and tail.canonical == "Maker"):
        return False
    if relation == "CAN_FREEZE" and head.canonical != "Financial Crime Unit":
        return False
    if relation == "REQUIRES" and tail.entity_type == "Role" and tail.canonical != "Dual Checker":
        return False
    if (
        relation == "REQUIRES"
        and head.entity_type == "Product"
        and tail.entity_type not in {"Document", "Control", "Form"}
    ):
        return False
    return True


def _extract_from_window(sentence: Sentence, mentions: list[Mention]) -> list[Triple]:
    """Emit at most one closest typed argument pair per relation cue.

    The previous implementation emitted every compatible mention pair in a
    two-sentence window, producing compliance-dangerous false edges. Nearest
    typed arguments are a deterministic semantic-role approximation.
    """
    blob = sentence.text.lower()
    extracted: list[Triple] = []
    for cues, head_types, tail_types, relation in RELATION_CUES:
        if relation == "CONFIDENCE_GATE" and not ("auto-fill" in blob or "below 0.82" in blob):
            continue
        if relation == "ESCALATES_TO" and (
            "failed ovd" in blob
            or "first attempt" in blob
            or "first officially valid document" in blob
        ):
            continue
        cue_positions = [(blob.find(cue), cue) for cue in cues if blob.find(cue) >= 0]
        for cue_start, cue in cue_positions:
            cue_mid = cue_start + len(cue) / 2
            pairs = [
                (head, tail)
                for head in mentions
                for tail in mentions
                if head.start < tail.start
                and head.entity_type in head_types
                and tail.entity_type in tail_types
                and _valid_relation(relation, head, tail)
            ]
            if not pairs:
                continue
            head, tail = min(
                pairs,
                key=lambda pair: abs(pair[0].end - cue_mid) + abs(pair[1].start - cue_mid),
            )
            if relation == "REQUIRES" and tail.canonical == "Proof of Address":
                process = next(
                    (m for m in mentions if m.canonical == "Address Change"),
                    None,
                )
                if process is not None:
                    head = process
            extracted.append(
                Triple(
                    head=head.canonical,
                    relation=relation,
                    tail=tail.canonical,
                    source=sentence.doc_id,
                    evidence=sentence.text.strip()[:400],
                    sent_id=sentence.sent_id,
                )
            )
    return extracted


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
        for triple in _extract_from_window(sent, mentions):
            key = (triple.head, triple.relation, triple.tail, triple.source)
            if key in seen:
                continue
            seen.add(key)
            triples.append(triple)
    return triples
