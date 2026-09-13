"""Schema-guided entity and relation extraction for KYC policy Graph RAG.

The extractor is deterministic so the demo runs without an LLM API. Optional
LLM extraction can be layered later; the graph schema stays the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str
    source: str
    evidence: str


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    path: str


# (pattern, head, relation, tail) — first match wins per rule, per document.
RULES: list[tuple[str, str, str, str]] = [
    (r"Maker cannot post the account to the Core Banking System \(CBS\)",
     "Maker", "CANNOT_WRITE", "CBS"),
    (r"Only the Checker role can approve the KYC packet and trigger the CBS write",
     "Checker", "CAN_WRITE", "CBS"),
    (r"Checker is the only role with write access to the CBS customer-master address",
     "Checker", "CAN_WRITE", "CBS"),
    (r"neither Maker nor branch Checker can write KYC or address changes",
     "Financial Crime Unit", "CAN_FREEZE", "CBS"),
    (r"Politically Exposed Person \(PEP\), Enhanced Due Diligence \(EDD\) is mandatory",
     "PEP", "REQUIRES", "EDD"),
    (r"PEP customer requesting an address change also triggers EDD",
     "Address Change", "REQUIRES", "EDD"),
    (r"high-risk pin codes in list HRC-22",
     "HRC-22", "REQUIRES", "EDD"),
    (r"If the new address is in a high-risk pin code list HRC-22",
     "Address Change", "ESCALATES_TO", "Financial Crime Unit"),
    (r"one of the two Checkers must be from the Financial Crime Unit",
     "EDD", "REQUIRES_CHECKER_FROM", "Financial Crime Unit"),
    (r"Failed OVD verification on the first attempt must be routed to the Central KYC Desk",
     "OVD", "ON_FAILURE_ROUTES_TO", "Central KYC Desk"),
    (r"savings account requires Officially Valid Document \(OVD\)",
     "Savings Account", "REQUIRES", "OVD"),
    (r"current account additionally requires a GST certificate",
     "Current Account", "REQUIRES", "GST Certificate"),
    (r"Utility bill, Passport, and Aadhaar e-KYC are accepted",
     "Address Change", "ACCEPTS", "Proof of Address"),
    (r"confidence below 0.82 cannot auto-fill the CBS address block",
     "Maker", "CONFIDENCE_GATE", "CBS"),
    (r"IASW-style agentic pipelines that auto-extract fields are treated as Maker actions",
     "IASW Agent", "ACTS_AS", "Maker"),
    (r"They inherit Maker restrictions: no CBS write path",
     "IASW Agent", "CANNOT_WRITE", "CBS"),
    (r"Retention is 8 years after account closure for OVD images",
     "OVD", "RETAINED_YEARS", "8"),
    (r"10 years for EDD files",
     "EDD", "RETAINED_YEARS", "10"),
    (r"Policy ID: POL-KYC-001",
     "POL-KYC-001", "GOVERNS", "Savings Account"),
    (r"Policy ID: POL-KYC-001",
     "POL-KYC-001", "GOVERNS", "Current Account"),
    (r"Policy ID: POL-CHG-014",
     "POL-CHG-014", "GOVERNS", "Address Change"),
    (r"Policy ID: POL-AML-007",
     "POL-AML-007", "GOVERNS", "EDD"),
    (r"Policy ID: POL-AUD-003",
     "POL-AUD-003", "GOVERNS", "OVD"),
    (r"dual Checker sign-off",
     "EDD", "REQUIRES", "Dual Checker"),
    (r"form SOF-Q1",
     "EDD", "REQUIRES", "SOF-Q1"),
]


ENTITY_TYPES = {
    "Maker": "Role",
    "Checker": "Role",
    "Dual Checker": "Role",
    "IASW Agent": "System",
    "CBS": "System",
    "Financial Crime Unit": "Unit",
    "Central KYC Desk": "Unit",
    "PEP": "CustomerType",
    "EDD": "Control",
    "OVD": "Document",
    "GST Certificate": "Document",
    "Proof of Address": "Document",
    "Savings Account": "Product",
    "Current Account": "Product",
    "Address Change": "Process",
    "HRC-22": "RiskList",
    "SOF-Q1": "Form",
    "POL-KYC-001": "Policy",
    "POL-CHG-014": "Policy",
    "POL-AML-007": "Policy",
    "POL-AUD-003": "Policy",
    "8": "Duration",
    "10": "Duration",
}


def load_documents(policy_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        docs.append(Document(doc_id=path.stem, title=title, text=text, path=str(path)))
    return docs


def extract_triples(docs: list[Document]) -> list[Triple]:
    triples: list[Triple] = []
    seen: set[tuple[str, str, str, str]] = set()
    for doc in docs:
        for pattern, head, relation, tail in RULES:
            match = re.search(pattern, doc.text)
            if not match:
                continue
            key = (head, relation, tail, doc.doc_id)
            if key in seen:
                continue
            seen.add(key)
            triples.append(
                Triple(
                    head=head,
                    relation=relation,
                    tail=tail,
                    source=doc.doc_id,
                    evidence=match.group(0),
                )
            )
    return triples
