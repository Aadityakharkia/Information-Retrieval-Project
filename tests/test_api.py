"""API contract tests (hermetic: uses the offline 'mock-demo' generator, no network/LLM needed)."""
import pytest

from backend.app import app
from backend.rag.generator import parse_citations
import backend.rag.generator as generator

OFFLINE = "mock-demo"


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def ask(client, question, **extra):
    return client.post("/api/ask", json={"question": question, "model": OFFLINE, **extra})


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["chunks_count"] > 0


def test_pages_are_served(client):
    for path in ("/", "/answer", "/library", "/about", "/js/api.js", "/css/styles.css"):
        assert client.get(path).status_code == 200, path


def test_ask_contract(client):
    r = ask(client, "headache relief and causes", retriever="hybrid")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "answered"
    assert body["generator"] == "mock"
    assert body["sentences"] and body["sources"]
    assert {"text", "citations", "support"} <= set(body["sentences"][0])
    assert {"n", "question", "text", "score"} <= set(body["sources"][0])
    assert {"sparse_top5", "dense_top5", "fused_top5"} <= set(body["inspector"])


@pytest.mark.parametrize("retriever", ["sparse", "bm25", "dense", "hybrid"])
def test_every_retriever_answers(client, retriever):
    r = ask(client, "what are the symptoms of diabetes", retriever=retriever)
    assert r.status_code == 200
    assert r.get_json()["sources"]


def test_off_topic_question_abstains(client):
    body = ask(client, "What is the capital of France?").get_json()
    assert body["status"] == "abstained"
    assert body["sentences"] == []


def test_self_harm_triggers_safety_message(client):
    body = ask(client, "I want to kill myself").get_json()
    assert body["status"] == "self_harm" and body["self_harm_message"]


def test_emergency_banner(client):
    body = ask(client, "sudden chest pain what should I do").get_json()
    assert body["emergency_banner"]


@pytest.mark.parametrize("payload,fragment", [
    ({}, "question"),
    ({"question": "   "}, "question"),
    ({"question": "x" * 501}, "too long"),
    ({"question": "fever", "retriever": "nope"}, "retriever"),
])
def test_ask_validation(client, payload, fragment):
    r = client.post("/api/ask", json=payload)
    assert r.status_code == 400
    assert fragment in r.get_json()["error"]


def test_suggest_and_search(client):
    assert client.get("/api/suggest?q=fev").get_json()["suggestions"]
    assert client.get("/api/suggest?q=").get_json()["suggestions"] == []
    res = client.get("/api/search?q=headache&retriever=bm25").get_json()
    assert res["count"] > 0
    assert client.get("/api/search").status_code == 400


def test_library_pagination_and_filter(client):
    page = client.get("/api/library?page=1&limit=5").get_json()
    assert len(page["chunks"]) == 5 and page["total_pages"] >= 1
    filtered = client.get("/api/library?search=diabetes").get_json()
    assert 0 < filtered["total_records"] <= page["total_records"]


def test_models_and_eval_summary(client):
    models = client.get("/api/models").get_json()
    assert models["default"] in models["models"] and OFFLINE in models["models"]
    assert client.get("/api/eval/summary").status_code == 200


def test_parse_citations_accepts_fullwidth_brackets():
    text = "The max dose is 4 g【1】."
    normalised = generator.re.sub(r"[【\[]\s*(\d+(?:\s*,\s*\d+)*)\s*[】\]]", r"[\1]", text)
    clean, ids, valid = parse_citations(normalised, max_k=5)
    assert ids == [1] and valid and "【" not in clean
