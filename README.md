# KYC Policy Graph RAG

Graph RAG over Indian retail-banking **KYC / address-change / AML** policy. It answers *who may write, who must escalate, and which control hops from another circular* — not “stuff a PDF into a vector DB.”

Inspired by [microsoft/graphrag](https://github.com/microsoft/graphrag) (local + global search) and the LlamaIndex GraphRAG cookbook. **Not a fork.** Extraction, store, retrieval, and eval are original.

## Capabilities

| Layer | Implementation |
| --- | --- |
| Ingestion | Recursive Markdown, text, and **PDF** loading with stable document provenance |
| Extraction | Offline schema cues on paraphrases, or `KYC_EXTRACTION=openai` for constrained **JSON-Schema** model extraction |
| Graph | Typed NetworkX multigraph, Louvain communities, evidence on every edge |
| Retrieval | MiniLM/TF-IDF chunks + typed edges + explicit 2-edge paths + community summaries, fused with **RRF** |
| Persistence | Normalized **SQLite** snapshot (documents, chunks, nodes, edges, communities) plus FTS5 |
| Serving | Typed FastAPI `/query`, `/health`, and `/graph/neighbors/{entity}` endpoints |
| Evaluation | Equal-k answer/citation/path scoring, extraction P/R/F1, MRR, recall@6, and latency |
| Delivery | Installable package, CLI, Docker image, non-root runtime, healthcheck, and a GitHub Actions template |

The included corpus is synthetic. The six-question benchmark is a regression suite, not statistical evidence of superiority, and it is smaller than Microsoft GraphRAG (no open-domain extraction or production scale).

## Flagship hop

> Does maker-equivalent automation get posting rights on the ledger?

The question never says IASW or CBS. Chunk RAG typically returns “Maker cannot post.” The graph has to use `IASW Agent -ACTS_AS-> Maker` then `CANNOT_WRITE -> CBS`, including from the paraphrased Q3 circular (“field-extraction bot … classified as Maker … prohibited from updating core banking”).

Equal-budget evaluation (`scripts/run_demo.py --sparse --k 6`):

- grounded answer success at k=6: chunk **5/6**, graph **6/6**
- MRR: chunk **0.861**, graph **1.000**
- graph recall@6: **0.944**
- required path composition: **1/1**
- hand-annotated extraction set (26 triples): precision/recall/F1 **1.00**, zero forbidden edges
- mean sparse retrieval latency: chunk **0.16 ms**, graph **0.14 ms** (hardware-dependent)

Both systems receive exactly six results. Community names and joined source metadata are excluded from scoring; the generated answer must contain the answer atoms and cite the supporting span. The graph-only win is the required `ACTS_AS -> CANNOT_WRITE` composition.

## Pipeline

```
policies (PDF/Markdown/text)
  -> sentence windows (coref-light: adjacent sentences)
  -> mentions (aliases) + relation cues
  -> NetworkX property graph
  -> Louvain communities
  -> HybridIndex: chunk vectors + edge vectors + community vectors
  -> RRF fusion, explicit 2-edge path composition
  -> cited answer
```

MiniLM (`all-MiniLM-L6-v2`) is used when `sentence-transformers` loads; otherwise TF-IDF. Tests force TF-IDF so CI does not download weights.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dense]"
PYTHONPATH=src python scripts/run_demo.py --sparse   # no model download
PYTHONPATH=src python scripts/run_demo.py            # MiniLM if installed
PYTHONPATH=src python -m pytest -q
```

### Serve

```bash
KYC_SPARSE=1 uvicorn kyc_graphrag.api:app --reload
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"Who may post an address change to CBS?","top_k":8}'
```

Or:

```bash
docker build -t kyc-graphrag .
docker run --rm -p 8000:8000 kyc-graphrag
```

`ci/github-actions.yml` is the tested workflow template. Copy it to `.github/workflows/ci.yml` after authorizing a GitHub token with `workflow` scope.

For model extraction, set `KYC_EXTRACTION=openai` and `OPENAI_API_KEY`. Model output is constrained to the declared entities and relations and validated again before entering the graph.
