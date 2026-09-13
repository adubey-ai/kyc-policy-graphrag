# KYC Policy Graph RAG

Offline Graph RAG over a small **Indian retail-banking KYC / address-change / AML** policy corpus.

Naive chunk RAG answers “what does this paragraph say”. Graph RAG answers **who can write, who must escalate, and which control hops from another policy** — the questions that show up in maker-checker and agentic document-AI work.

This is **not** a fork of [microsoft/graphrag](https://github.com/microsoft/graphrag). That repo is the right reference for production Graph RAG (entity extraction, Leiden communities, local + global search). This repo is a small, dependency-light reimplementation of that *pattern* with a banking schema, a deterministic extractor (no API key), and a side-by-side eval against lexical RAG.

## What is different from a typical GitHub Graph RAG clone

| Typical demo | This repo |
| --- | --- |
| Wikipedia / news dump + OpenAI | Four synthetic bank policies, runs offline |
| Generic Person/Org entities | Roles, products, controls, CBS write path |
| LLM summaries required | Extractive community summaries |
| “It works on my notebook” | Five gold multi-hop questions + pytest |

The interesting gold case is: **Can an IASW agent post an address change directly to CBS?**  
Chunk RAG often stays in the address-change document. The graph has to hop `IASW Agent -ACTS_AS-> Maker` and `Maker -CANNOT_WRITE-> CBS`.

## Pipeline

```
policies/*.md
    -> schema-guided triples
    -> NetworkX property graph
    -> greedy-modularity communities
    -> local search (2-hop ego) + global search (community summaries)
    -> extractive answer
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_demo.py
PYTHONPATH=src python -m pytest -q
```

## Results on the gold set

`scripts/run_demo.py` prints HIT/MISS for naive TF-IDF chunk RAG vs graph retrieval. Graph RAG is expected to hit all five cases; naive RAG typically misses the IASW and freeze questions because the constraint lives one document away.

Graph JSON (nodes, edges, communities) is written to `artifacts/graph.json`.

## Resume one-liner

Graph RAG over KYC and AML policy: entity-relation graph, community summaries, and local+global retrieval that recovers multi-hop maker-checker constraints chunk RAG misses.

## References

- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, 2024. [microsoft/graphrag](https://github.com/microsoft/graphrag)
- LlamaIndex [GraphRAG cookbook v2](https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v2/) (Neo4j + PropertyGraph)
