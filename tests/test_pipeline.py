from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kyc_graphrag.eval import load_gold, score_case
from kyc_graphrag.extract import extract_triples
from kyc_graphrag.corpus import load_documents
from kyc_graphrag.pipeline import build_pipeline

POLICIES = ROOT / "data" / "policies"
GOLD = ROOT / "data" / "gold.json"


def test_paraphrase_extracts_iasw_without_original_sentence():
    docs = [d for d in load_documents(POLICIES) if d.doc_id == "ops_circular_q3"]
    rels = {(t.head, t.relation, t.tail) for t in extract_triples(docs)}
    assert ("IASW Agent", "ACTS_AS", "Maker") in rels
    assert ("IASW Agent", "CANNOT_WRITE", "CBS") in rels


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
        n_ok = score_case(pipe.index.naive(case.question), case)["hit"]
        g_ok = score_case(pipe.index.search(case.question), case)["hit"]
        naive_n += int(n_ok)
        graph_n += int(g_ok)
        if case.hop:
            hop_total += 1
            hop_naive += int(n_ok)
            hop_graph += int(g_ok)
    assert graph_n >= naive_n
    assert graph_n >= 5
    assert hop_graph >= hop_naive
    assert hop_graph >= 2
