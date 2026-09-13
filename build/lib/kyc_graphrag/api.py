"""FastAPI service for querying and inspecting the policy graph."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kyc_graphrag.generate import answer_from_hits
from kyc_graphrag.pipeline import Pipeline, build_pipeline


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    embedder: str
    latency_ms: float
    hits: list[dict]


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    root = _root()
    policy_dir = Path(os.environ.get("KYC_POLICY_DIR", root / "data" / "policies"))
    db_path = Path(os.environ.get("KYC_GRAPH_DB", root / "artifacts" / "graph.db"))
    sparse = os.environ.get("KYC_SPARSE", "").lower() in {"1", "true", "yes"}
    extraction = os.environ.get("KYC_EXTRACTION", "schema")
    return build_pipeline(
        policy_dir,
        prefer_dense=not sparse,
        store_path=db_path,
        extraction=extraction,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="KYC Policy Graph RAG",
        version="1.0.0",
        description="Hybrid chunk + graph + community retrieval with provenance.",
    )

    @app.get("/health")
    def health() -> dict:
        pipe = get_pipeline()
        return {
            "status": "ok",
            "embedder": pipe.embedder.kind,
            "documents": len(pipe.docs),
            "nodes": pipe.graph.number_of_nodes(),
            "edges": pipe.graph.number_of_edges(),
        }

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        pipe = get_pipeline()
        hits = pipe.index.search(request.question, k=request.top_k)
        return QueryResponse(
            answer=answer_from_hits(request.question, hits, "hybrid-graph"),
            embedder=pipe.embedder.kind,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            hits=[
                {
                    "kind": hit.kind,
                    "score": hit.score,
                    "text": hit.text,
                    "provenance": hit.meta,
                }
                for hit in hits
            ],
        )

    @app.get("/graph/neighbors/{entity}")
    def neighbors(entity: str) -> dict:
        pipe = get_pipeline()
        if entity not in pipe.graph:
            raise HTTPException(status_code=404, detail=f"Unknown entity: {entity}")
        edges = []
        for source, target, data in pipe.graph.edges(data=True):
            if entity not in {source, target}:
                continue
            edges.append(
                {
                    "source": source,
                    "relation": data["relation"],
                    "target": target,
                    "provenance": {
                        "doc_id": data["source"],
                        "sent_id": data.get("sent_id", ""),
                    },
                }
            )
        return {"entity": entity, "edges": edges}

    return app


app = create_app()
