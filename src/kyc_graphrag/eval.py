"""Gold multi-hop questions where lexical RAG is expected to miss a constraint."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    question: str
    must_contain: tuple[str, ...]
    why_graph: str


CASES: list[Case] = [
    Case(
        question="Can an IASW agent post an address change directly to CBS?",
        must_contain=("IASW Agent", "CANNOT_WRITE", "CBS"),
        why_graph="Needs the ACTS_AS Maker hop plus Maker CANNOT_WRITE CBS.",
    ),
    Case(
        question="A PEP customer wants an address change. What extra control fires?",
        must_contain=("Address Change", "REQUIRES", "EDD"),
        why_graph="Address-change policy points at EDD; onboarding PEP rule is in another doc.",
    ),
    Case(
        question="Who can freeze CBS posting while EDD is pending?",
        must_contain=("Financial Crime Unit", "CAN_FREEZE", "CBS"),
        why_graph="Freeze authority lives in the AML doc, not the address-change doc.",
    ),
    Case(
        question="If OVD verification fails once, where must the case go?",
        must_contain=("OVD", "ON_FAILURE_ROUTES_TO", "Central KYC Desk"),
        why_graph="Single-hop but easy for chunk RAG to bury in onboarding prose.",
    ),
    Case(
        question="Does a current-account opening need a GST certificate?",
        must_contain=("Current Account", "REQUIRES", "GST Certificate"),
        why_graph="Product-specific requirement next to savings OVD rules.",
    ),
]
