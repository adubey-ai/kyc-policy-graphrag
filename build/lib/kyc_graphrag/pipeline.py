from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kyc_graphrag.communities import detect_communities
from kyc_graphrag.corpus import chunk_documents, load_documents
from kyc_graphrag.embed import Embedder
from kyc_graphrag.extract import extract_triples
from kyc_graphrag.graph import build_graph
from kyc_graphrag.llm_extract import OpenAIRelationExtractor
from kyc_graphrag.retrieve import HybridIndex
from kyc_graphrag.store import SQLiteGraphStore


@dataclass
class Pipeline:
    docs: list
    triples: list
    graph: object
    communities: list
    chunks: list[dict[str, str]]
    index: HybridIndex
    embedder: Embedder


def build_pipeline(
    policy_dir: Path,
    prefer_dense: bool = True,
    store_path: Path | None = None,
    extraction: str = "schema",
) -> Pipeline:
    docs = load_documents(policy_dir)
    if extraction == "schema":
        triples = extract_triples(docs)
    elif extraction == "openai":
        triples = OpenAIRelationExtractor().extract(docs)
    else:
        raise ValueError("extraction must be 'schema' or 'openai'")
    graph = build_graph(triples)
    communities = detect_communities(graph)
    chunks = chunk_documents(docs)
    embedder = Embedder(prefer_dense=prefer_dense)
    index = HybridIndex(chunks, graph, communities, embedder)
    if store_path is not None:
        SQLiteGraphStore(store_path).replace(docs, chunks, graph, communities)
    return Pipeline(docs, triples, graph, communities, chunks, index, embedder)
