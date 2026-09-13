# KYC Policy Graph RAG

Graph RAG over Indian retail-banking **KYC / address-change / AML** policy. Built to answer *who may write, who must escalate, which control hops from another circular* — not “stuff a PDF into a vector DB.”

Inspired by [microsoft/graphrag](https://github.com/microsoft/graphrag) (local + global search) and the LlamaIndex GraphRAG cookbook. **Not a fork.** Extraction, store, retrieval, and eval are original.

## What “good” means here vs the first demo

| First version | This version |
| --- | --- |
| One regex per gold fact | Gazetteer mentions + **schema-level** relation cues (works on paraphrases) |
| Four clean markdown files | + Q3 circular, PEP SOP, OCR-noisy scan |
| Gold questions copied from the regex | Held-out questions that **do not name** the hop entity |
| Triple-string HIT/MISS | Keyword + **must-cite document** scoring |
| Ego-graph dump | Dense/TF-IDF chunks + typed edges + Louvain communities, fused with **RRF** |
| No citations | Cited extractive answers; optional OpenAI rewrite if `OPENAI_API_KEY` is set |

Still not Microsoft GraphRAG: no Leiden+LLM community summaries, no Neo4j, no GLiNER. It is a **domain Graph RAG you can run offline and defend in an interview**.

## Flagship hop

> Does maker-equivalent automation get posting rights on the ledger?

The question never says IASW or CBS. Chunk RAG typically returns “Maker cannot post.” The graph has to use `IASW Agent -ACTS_AS-> Maker` then `CANNOT_WRITE -> CBS`, including from the paraphrased Q3 circular (“field-extraction bot … classified as Maker … prohibited from updating core banking”).

Held-out (`scripts/run_demo.py --sparse`): **naive 5/6, graph 6/6**. The miss for chunk RAG is that hop.

## Pipeline

```
policies (md/txt)
  -> sentence windows (coref-light: adjacent sentences)
  -> mentions (aliases) + relation cues
  -> NetworkX property graph
  -> Louvain communities
  -> HybridIndex: chunk vectors + edge vectors + community vectors
  -> RRF fusion, 2-hop seed expansion
  -> cited answer
```

MiniLM (`all-MiniLM-L6-v2`) is used when `sentence-transformers` loads; otherwise TF-IDF. Tests force TF-IDF so CI does not download weights.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_demo.py --sparse   # no model download
PYTHONPATH=src python scripts/run_demo.py            # MiniLM if installed
PYTHONPATH=src python -m pytest -q
```

## Interview talking points

- Why Graph RAG: maker-checker constraints are **edges**, not chunks.
- Why schema extraction: new circulars should extract without a new fact regex (see `tests/test_pipeline.py`).
- Why eval is not circular: hop question omits the entity the graph must recover.
- Honest limit: gazetteer + cues, not open-domain LLM extraction. Next step would be GLiNER/LLM fill on the same schema.

## Resume line

Graph RAG over KYC/AML policy: schema-guided graph, RRF hybrid retrieval (chunks + 2-hop edges + communities). Held-out hop question that never names IASW: chunk RAG misses, graph recovers Maker-classed software cannot write CBS (5/6 vs 6/6).
