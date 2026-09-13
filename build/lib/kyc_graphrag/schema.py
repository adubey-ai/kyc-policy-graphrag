"""Domain schema: entity types, aliases, and relation cues.

Cues are schema-level (who may connect to whom, which verbs mean what).
They are not one regex per gold fact. A paraphrased policy should still extract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MentionSpec:
    canonical: str
    entity_type: str
    aliases: tuple[str, ...]


ENTITIES: tuple[MentionSpec, ...] = (
    MentionSpec("Maker", "Role", ("maker", "branch maker", "maker staff", "maker role")),
    MentionSpec("Checker", "Role", ("checker", "branch checker", "checker role")),
    MentionSpec(
        "Dual Checker",
        "Role",
        ("dual checker", "two checkers", "dual sign-off", "dual checker sign-off"),
    ),
    MentionSpec(
        "IASW Agent",
        "System",
        (
            "iasw",
            "iasw agent",
            "iasw pipeline",
            "iasw packet",
            "field-extraction bot",
            "extraction microservice",
            "extraction bot",
            "extraction service",
            "packet builder",
        ),
    ),
    MentionSpec(
        "CBS",
        "System",
        (
            "cbs",
            "core banking",
            "core-banking",
            "core banking system",
            "customer-master",
            "customer master",
        ),
    ),
    MentionSpec("Financial Crime Unit", "Unit", ("financial crime unit", "fcu", "financial crime")),
    MentionSpec("Central KYC Desk", "Unit", ("central kyc desk", "central kyc")),
    MentionSpec(
        "PEP", "CustomerType", ("pep", "politically exposed person", "politically exposed")
    ),
    MentionSpec("EDD", "Control", ("edd", "enhanced due diligence")),
    MentionSpec("OVD", "Document", ("ovd", "officially valid document")),
    MentionSpec("GST Certificate", "Document", ("gst certificate", "gst")),
    MentionSpec("Proof of Address", "Document", ("proof of address", "poa", "utility bill")),
    MentionSpec("Savings Account", "Product", ("savings account", "savings")),
    MentionSpec("Current Account", "Product", ("current account", "current")),
    MentionSpec(
        "Address Change", "Process", ("address change", "address update", "change of address")
    ),
    MentionSpec("HRC-22", "RiskList", ("hrc-22", "hrc22", "high-risk pin", "high risk pin")),
    MentionSpec("SOF-Q1", "Form", ("sof-q1", "sof q1", "source-of-funds", "source of funds")),
    MentionSpec("POL-KYC-001", "Policy", ("pol-kyc-001",)),
    MentionSpec("POL-CHG-014", "Policy", ("pol-chg-014",)),
    MentionSpec("POL-AML-007", "Policy", ("pol-aml-007",)),
    MentionSpec("POL-AUD-003", "Policy", ("pol-aud-003",)),
    MentionSpec("8", "Duration", ("8 years",)),
    MentionSpec("10", "Duration", ("10 years",)),
)


# (cue substrings, head types, tail types, relation)
RELATION_CUES: tuple[tuple[tuple[str, ...], frozenset[str], frozenset[str], str], ...] = (
    (
        (
            "cannot post",
            "cannot write",
            "must not post",
            "barred from",
            "prohibited from",
            "forbidden from",
            "no write",
            "no cbs write",
            "inherit maker restriction",
        ),
        frozenset({"Role", "System", "Unit"}),
        frozenset({"System"}),
        "CANNOT_WRITE",
    ),
    (
        (
            "only the checker",
            "checker is the only",
            "write access",
            "can approve",
            "trigger the cbs",
            "may post",
            "can write",
        ),
        frozenset({"Role"}),
        frozenset({"System"}),
        "CAN_WRITE",
    ),
    (
        ("can freeze", "freeze cbs", "freeze posting", "freeze authority"),
        frozenset({"Unit", "Role"}),
        frozenset({"System"}),
        "CAN_FREEZE",
    ),
    (
        ("treated as", "classified as", "acts as", "inherits maker", "same as a maker"),
        frozenset({"System", "Role"}),
        frozenset({"Role"}),
        "ACTS_AS",
    ),
    (
        (
            "requires",
            "require",
            "mandatory",
            "must submit",
            "need a",
            "needs a",
            "additionally requires",
            "triggers edd",
            "trigger edd",
        ),
        frozenset({"Product", "Process", "CustomerType", "Control", "RiskList", "Role"}),
        frozenset({"Document", "Control", "Form", "Role", "Product"}),
        "REQUIRES",
    ),
    (
        ("escalat", "routed to", "route to", "must go to", "refer to"),
        frozenset({"Document", "Process", "Product", "RiskList"}),
        frozenset({"Unit"}),
        "ESCALATES_TO",
    ),
    (
        ("on failure", "failed ovd", "first attempt"),
        frozenset({"Document"}),
        frozenset({"Unit"}),
        "ON_FAILURE_ROUTES_TO",
    ),
    (
        ("governs", "policy id", "owner:"),
        frozenset({"Policy"}),
        frozenset({"Product", "Process", "Control", "Document"}),
        "GOVERNS",
    ),
    (
        ("retain", "retention", "8 years", "10 years"),
        frozenset({"Document", "Control"}),
        frozenset({"Duration"}),
        "RETAINED_YEARS",
    ),
    (
        ("checker from", "from the financial crime"),
        frozenset({"Control"}),
        frozenset({"Unit"}),
        "REQUIRES_CHECKER_FROM",
    ),
    (
        ("confidence", "auto-fill", "below 0.82"),
        frozenset({"Role", "System"}),
        frozenset({"System"}),
        "CONFIDENCE_GATE",
    ),
    (
        ("accepted", "accepts"),
        frozenset({"Process"}),
        frozenset({"Document"}),
        "ACCEPTS",
    ),
)


def alias_table() -> list[tuple[str, str, str]]:
    """Longest alias first for greedy matching."""
    rows: list[tuple[str, str, str]] = []
    for spec in ENTITIES:
        names = (spec.canonical, *spec.aliases)
        for alias in names:
            rows.append((alias.lower(), spec.canonical, spec.entity_type))
    rows.sort(key=lambda r: len(r[0]), reverse=True)
    return rows
