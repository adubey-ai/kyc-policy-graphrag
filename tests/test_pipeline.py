from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kyc_graphrag.communities import chunk_documents, detect_communities
from kyc_graphrag.eval import CASES
from kyc_graphrag.extract import extract_triples, load_documents
from kyc_graphrag.generate import answer_from_hits
from kyc_graphrag.graph import build_graph
from kyc_graphrag.retrieve import graph_retrieve, naive_retrieve

POLICIES = ROOT / "data" / "policies"


def test_extracts_multi_hop_iasw_rule():
    triples = extract_triples(load_documents(POLICIES))
    rels = {(t.head, t.relation, t.tail) for t in triples}
    assert ("IASW Agent", "ACTS_AS", "Maker") in rels
    assert ("IASW Agent", "CANNOT_WRITE", "CBS") in rels
    assert ("Maker", "CANNOT_WRITE", "CBS") in rels


def test_graph_beats_or_matches_naive_on_gold():
    docs = load_documents(POLICIES)
    graph = build_graph(extract_triples(docs))
    communities = detect_communities(graph)
    chunks = chunk_documents(docs)
    graph_n = naive_n = 0
    for case in CASES:
        naive_text = answer_from_hits(case.question, naive_retrieve(case.question, chunks), "naive")
        graph_text = answer_from_hits(
            case.question, graph_retrieve(case.question, graph, communities), "graph"
        )
        naive_n += int(all(x.lower() in naive_text.lower() for x in case.must_contain))
        graph_n += int(all(x.lower() in graph_text.lower() for x in case.must_contain))
    assert graph_n == len(CASES)
    assert graph_n >= naive_n
