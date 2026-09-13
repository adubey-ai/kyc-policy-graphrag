from pathlib import Path

from kyc_graphrag.corpus import load_documents
from kyc_graphrag.eval import extraction_metrics, load_gold, score_case
from kyc_graphrag.extract import Document, extract_triples
from kyc_graphrag.pipeline import build_pipeline
from kyc_graphrag.retrieve import Hit

ROOT = Path(__file__).resolve().parents[1]

POLICIES = ROOT / "data" / "policies"
GOLD = ROOT / "data" / "gold.json"
GOLD_TRIPLES = ROOT / "data" / "gold_triples.json"


def test_paraphrase_extracts_iasw_without_original_sentence():
    docs = [d for d in load_documents(POLICIES) if d.doc_id == "ops_circular_q3"]
    rels = {(t.head, t.relation, t.tail) for t in extract_triples(docs)}
    assert ("IASW Agent", "ACTS_AS", "Maker") in rels
    assert ("Maker", "CANNOT_WRITE", "CBS") in rels


def test_ocr_scan_extracts_hrc22_escalation():
    docs = [d for d in load_documents(POLICIES) if d.doc_id == "scanned_address_policy"]
    rels = {(t.head, t.relation, t.tail) for t in extract_triples(docs)}
    assert ("Address Change", "ESCALATES_TO", "Financial Crime Unit") in rels or (
        "HRC-22",
        "ESCALATES_TO",
        "Financial Crime Unit",
    ) in rels


def test_held_out_graph_beats_naive():
    pipe = build_pipeline(POLICIES, prefer_dense=False)
    gold = load_gold(GOLD)
    naive_n = graph_n = 0
    hop_naive = hop_graph = hop_total = 0
    for case in gold:
        n_ok = score_case(pipe.index.naive(case.question, k=6), case)["hit"]
        g_ok = score_case(pipe.index.search(case.question, k=6), case)["hit"]
        naive_n += int(n_ok)
        graph_n += int(g_ok)
        if case.hop:
            hop_total += 1
            hop_naive += int(n_ok)
            hop_graph += int(g_ok)
    assert naive_n == 5
    assert graph_n == len(gold)
    assert hop_graph >= hop_naive
    assert hop_graph >= 2


def test_flagship_returns_composed_permission_path():
    pipe = build_pipeline(POLICIES, prefer_dense=False)
    case = load_gold(GOLD)[0]
    result = score_case(pipe.index.search(case.question, k=6), case)
    assert result["path"]


def test_community_metadata_alone_cannot_pass_gold():
    case = load_gold(GOLD)[0]
    leaked = Hit(
        kind="community",
        score=1.0,
        text="IASW Agent Maker CBS",
        meta={"doc_id": "ops_circular_q3"},
    )
    assert not score_case([leaked], case)["hit"]


def test_negative_sentences_do_not_create_compliance_edges():
    doc = Document(
        doc_id="negative",
        title="Negative controls",
        path="memory",
        text=(
            "Maker identity and confidence scores are retained for audit. "
            "Checker staff cannot delete archived scans from CBS."
        ),
    )
    rels = {(t.head, t.relation, t.tail) for t in extract_triples([doc])}
    assert ("Maker", "CONFIDENCE_GATE", "CBS") not in rels
    assert ("Checker", "CANNOT_WRITE", "CBS") not in rels


def test_extraction_matches_hand_annotated_regression_set():
    triples = extract_triples(load_documents(POLICIES))
    metrics = extraction_metrics(triples, GOLD_TRIPLES)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["forbidden_emitted"] == 0
