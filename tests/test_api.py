"""API contract smoke tests (uses cached indices; offline generator)."""
import pytest
from fastapi.testclient import TestClient

from medrag.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["is_initialized"] is True and body["chunks_indexed"] > 0


def test_query_contract(client):
    r = client.post("/api/query", json={
        "question": "What are the symptoms of glaucoma?",
        "retriever_type": "bm25", "generator_model": "mock-offline"})
    assert r.status_code == 200
    body = r.json()
    assert {"answer", "refused", "retrieved_chunks", "citation_audit", "ir_diagnostics"} <= body.keys()


def test_ood_refused(client):
    r = client.post("/api/query", json={
        "question": "How do I bake chocolate chip cookies?",
        "retriever_type": "inverted_index", "generator_model": "mock-offline"})
    assert r.json()["refused"] is True


def test_validation(client):
    assert client.post("/api/query", json={"question": ""}).status_code == 422
