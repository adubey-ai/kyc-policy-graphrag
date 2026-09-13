from pathlib import Path

from fastapi.testclient import TestClient

from kyc_graphrag.api import create_app, get_pipeline
from kyc_graphrag.corpus import Document
from kyc_graphrag.llm_extract import OpenAIRelationExtractor
from kyc_graphrag.pipeline import build_pipeline
from kyc_graphrag.store import SQLiteGraphStore

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "data" / "policies"


def test_sqlite_snapshot_has_provenance(tmp_path):
    db = tmp_path / "graph.db"
    pipe = build_pipeline(POLICIES, prefer_dense=False, store_path=db)
    store = SQLiteGraphStore(db)
    stats = store.stats()
    assert stats["documents"] == len(pipe.docs)
    assert stats["nodes"] == pipe.graph.number_of_nodes()
    assert stats["edges"] == pipe.graph.number_of_edges()
    assert any(edge["doc_id"] for edge in store.neighbors("CBS"))


def test_api_health_query_and_unknown_entity(monkeypatch, tmp_path):
    monkeypatch.setenv("KYC_POLICY_DIR", str(POLICIES))
    monkeypatch.setenv("KYC_GRAPH_DB", str(tmp_path / "api.db"))
    monkeypatch.setenv("KYC_SPARSE", "1")
    get_pipeline.cache_clear()
    client = TestClient(create_app())

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["edges"] > 0

    response = client.post(
        "/query",
        json={
            "question": "Does maker-equivalent automation get posting rights on the ledger?",
            "top_k": 8,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["hits"]
    assert any(hit["provenance"] for hit in payload["hits"])
    assert "Citations:" in payload["answer"]

    assert client.get("/graph/neighbors/does-not-exist").status_code == 404


def test_model_extractor_validates_structured_output():
    def fake_transport(_request):
        return b"""{
          "choices": [{"message": {"content":
            "{\\"triples\\":[{\\"head\\":\\"IASW Agent\\",\\"relation\\":\\"ACTS_AS\\",\\"tail\\":\\"Maker\\"},{\\"head\\":\\"invented\\",\\"relation\\":\\"ACTS_AS\\",\\"tail\\":\\"Maker\\"}]}"
          }}]
        }"""

    extractor = OpenAIRelationExtractor(transport=fake_transport)
    docs = [Document("circular", "Circular", "IASW is treated as Maker.", "memory")]
    triples = extractor.extract(docs)
    assert [(t.head, t.relation, t.tail) for t in triples] == [("IASW Agent", "ACTS_AS", "Maker")]
