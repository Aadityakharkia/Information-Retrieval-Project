"""API contract smoke tests (uses cached indices; offline generator)."""
import pytest
from backend.app import app


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["chunks_count"] > 0


def test_ask_contract(client):
    r = client.post("/api/ask", json={
        "question": "headache relief and causes",
        "retriever": "hybrid",
        "model": "llama3.2:1b"
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "answered"
    assert len(body["sentences"]) > 0
    assert len(body["sources"]) > 0


def test_ood_abstained(client):
    r = client.post("/api/ask", json={
        "question": "how to build a rocket engine",
        "retriever": "hybrid",
        "model": "llama3.2:1b"
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "abstained"
    assert body["abstain"]["should_abstain"] is True

