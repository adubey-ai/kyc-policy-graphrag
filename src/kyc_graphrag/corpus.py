"""Load policy files and split into sentence-sized units."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    path: str


@dataclass(frozen=True)
class Sentence:
    sent_id: str
    doc_id: str
    text: str


def load_documents(policy_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(policy_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), path.stem)
        docs.append(Document(doc_id=path.stem, title=title, text=text, path=str(path)))
    return docs


def sentences(docs: list[Document]) -> list[Sentence]:
    out: list[Sentence] = []
    for doc in docs:
        parts = [p.strip() for p in SENT_SPLIT.split(doc.text) if p.strip() and p.strip() != "#"]
        for i, part in enumerate(parts):
            if len(part) < 12:
                continue
            out.append(Sentence(sent_id=f"{doc.doc_id}:{i}", doc_id=doc.doc_id, text=part))
    return out


def chunk_documents(docs: list[Document], max_chars: int = 480) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for doc in docs:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
        buf = ""
        part = 0
        for para in paragraphs:
            if buf and len(buf) + len(para) > max_chars:
                chunks.append({"chunk_id": f"{doc.doc_id}:{part}", "doc_id": doc.doc_id, "text": buf})
                part += 1
                buf = para
            else:
                buf = f"{buf}\n\n{para}".strip()
        if buf:
            chunks.append({"chunk_id": f"{doc.doc_id}:{part}", "doc_id": doc.doc_id, "text": buf})
    return chunks
