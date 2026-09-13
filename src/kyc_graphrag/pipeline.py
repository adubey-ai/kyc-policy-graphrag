from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kyc_graphrag.communities import detect_communities
from kyc_graphrag.corpus import chunk_documents, load_documents
from kyc_graphrag.embed import Embedder
from kyc_graphrag.extract import extract_triples
from kyc_graphrag.graph import build_graph, export_graph, graph_stats
from kyc_graphrag.retrieve import HybridIndex


@dataclass
class Pipeline:
    docs: list
    triples: list
    graph: object
    communities: list
    index: HybridIndex
    embedder: Embedder


def build_pipeline(policy_dir: Path, prefer_dense: bool = True) -> Pipeline:
    docs = load_documents(policy_dir)
    triples = extract_triples(docs)
    graph = build_graph(triples)
    communities = detect_communities(graph)
    chunks = chunk_documents(docs)
    embedder = Embedder(prefer_dense=prefer_dense)
    index = HybridIndex(chunks, graph, communities, embedder)
    return Pipeline(docs, triples, graph, communities, index, embedder)
