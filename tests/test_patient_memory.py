"""Prescription memory, search history and the permission-based Wikipedia fallback (no real network)."""
import io

import pytest
import requests

import backend.app as app_module
import backend.rag.llm_client as llm_client
import config
from backend.data.chunker import split_into_sentences
from backend.memory.prescription import PdfError, extract_pdf, medicines
from backend.web import wikipedia

app = app_module.app
RX = ["City Clinic  Dr. Rao", "Patient: Test Person", "Tab. Metformin 2.5 mg twice daily after food", "Cap Amoxicillin 500 mg three times a day for 5 days",
      "Fasting glucose 130 mg/dL", "Review after 14 days"]


def make_pdf(lines, encrypt=False):
    """Smallest valid one-page text PDF."""
    esc = [l.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)") for l in lines]
    stream = ("BT /F1 12 Tf 50 750 Td 16 TL " + " ".join(f"({l}) Tj T*" for l in esc) + " ET").encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream), b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out)); out += b"%d 0 obj\n%s\nendobj\n" % (i, o)
    xref = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offs)
    return out + b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % xref


@pytest.fixture
def client():
    return app.test_client()


@pytest.fixture
def net(monkeypatch):
    """Records every outgoing request; LLM calls fail (offline) and Wikipedia is faked."""
    calls = {"post": [], "get": []}

    def post(url, **kw):
        calls["post"].append((url, kw)); raise requests.ConnectionError("offline test")

    class Res:
        def __init__(self, body): self.body = body
        def raise_for_status(self): pass
        def json(self): return self.body

    def get(url, params=None, **kw):
        calls["get"].append((url, params))
        if params.get("list") == "search":
            return Res({"query": {"search": [{"title": "Zorblatt fever"}, {"title": "Zorblatt"}]}})
        links = {"Zorblatt fever": ["Zorblatt"], "Zorblatt": []}[params["titles"]]
        pg = {"title": params["titles"], "fullurl": "https://en.wikipedia.org/wiki/X", "links": [{"title": l} for l in links],
              "extract": f"{params['titles']} is a fictional zorblatt illness treated with rest. Zorblatt patients drink water every day.\n== References ==\nSECRETREF"}
        return Res({"query": {"pages": [pg]}})
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key-not-real")
    return calls


def upload(client, data=None, name="rx.pdf"):
    return client.post("/api/documents", data={"file": (io.BytesIO(data if data is not None else make_pdf(RX)), name)})


def ask(client, q, **kw):
    return client.post("/api/ask", json={"question": q, "model": "mock-demo", **kw}).get_json()


# --- PDF reading -----------------------------------------------------------
def test_pdf_text_and_medicines():
    text, pages = extract_pdf(make_pdf(RX))
    assert pages == 1 and "2.5 mg twice daily" in text
    assert medicines(text) == ["Metformin", "Amoxicillin"]  # not "glucose"


@pytest.mark.parametrize("data,msg", [(b"hello", "not a PDF"), (b"%PDF-1.4 garbage", "damaged"), (make_pdf([""]), "Scanned")])
def test_bad_pdfs_are_rejected_clearly(client, data, msg):
    r = upload(client, data)
    assert r.status_code == 422 and msg in r.get_json()["error"]


def test_upload_limits(client, monkeypatch):
    assert client.post("/api/documents").status_code == 400
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    assert upload(client).status_code == 413


def test_filename_is_reduced_to_a_basename(client):
    assert upload(client, name="../../etc/x.pdf").get_json()["filename"] == "x.pdf"


# --- privacy ----------------------------------------------------------------
def test_patients_cannot_see_or_delete_each_other(client, private_store):
    doc = upload(client).get_json()
    ask(client, "what is asthma")
    other = app.test_client()
    assert other.get("/api/documents").get_json()["documents"] == []
    assert other.get("/api/history").get_json()["history"] == []
    ask(other, "what is gout")  # gives the other browser an identity
    assert other.delete(f"/api/documents/{doc['id']}").status_code == 404
    hid = client.get("/api/history").get_json()["history"][0]["id"]
    assert other.delete(f"/api/history/{hid}").status_code == 404
    assert ask(other, "what medicines did my doctor prescribe")["memory_used"] is False
    assert len(client.get("/api/documents").get_json()["documents"]) == 1


def test_cookie_is_protected_and_token_is_not_stored(client, private_store):
    r = upload(client)
    cookie = r.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    token = cookie.split("=", 1)[1].split(";")[0]
    assert token.encode() not in private_store.path.read_bytes()
    assert private_store.path.stat().st_mode & 0o077 == 0


def test_cross_site_writes_are_blocked(client):
    assert client.delete("/api/history", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.delete("/api/history", headers={"Origin": "http://localhost"}).status_code == 200


def test_delete_everything(client, private_store):
    upload(client); ask(client, "what is asthma")
    assert client.delete("/api/me").get_json()["deleted"]
    assert b"Metformin" not in private_store.path.read_bytes()
    assert client.get("/api/documents").get_json()["documents"] == []


@pytest.mark.parametrize("q", ["what medicines did my doctor prescribe", "side effects of my medicines", "can I take metformin with alcohol",
                               "meri dawai ke nuksan kya hain"])
@pytest.mark.parametrize("model", ["openai/gpt-oss-120b", "llama3.2:1b"])
def test_prescription_never_leaves_this_computer(client, net, q, model):
    upload(client)
    body = ask(client, q, model=model)
    assert body["memory_used"] and body["local_only"]
    assert net["get"] == [] and all(url.startswith("http://localhost") for url, _ in net["post"])


def test_remote_ollama_is_not_used_for_prescriptions(client, net, monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "https://ollama.example.com")
    upload(client)
    ask(client, "side effects of my medicines", model="llama3.2:1b")
    assert net["post"] == []


# --- answers from memory ------------------------------------------------------
def test_prescription_lines_are_verbatim(client, net):
    upload(client)
    body = ask(client, "what medicines did my doctor prescribe?")
    assert body["status"] == "answered"
    texts = [s["text"] for s in body["sentences"]]
    assert "Tab. Metformin 2.5 mg twice daily after food" in texts
    assert all(s["origin"] == "prescription" and s["support"]["badge"] == "verbatim" for s in body["sentences"])
    assert body["wikipedia_offer"] is None


def test_knowledge_question_combines_prescription_and_database(client, net):
    upload(client)
    body = ask(client, "what are the side effects of my Amoxicillin")
    origins = {s["origin"] for s in body["sentences"]}
    assert origins == {"prescription", "database"}
    assert not any("Metformin" in s["text"] for s in body["sentences"] if s["origin"] == "prescription")
    for s in body["sentences"]:
        for c in s["citations"]:
            assert body["sources"][c - 1]["origin"] == s["origin"]


def test_unrelated_question_does_not_use_memory(client, net):
    upload(client)
    assert ask(client, "what are the symptoms of asthma")["memory_used"] is False


def test_memory_can_be_switched_off(client, net):
    upload(client)
    assert ask(client, "side effects of my medicines", use_memory=False)["memory_used"] is False


def test_unknown_medicine_offers_wikipedia_but_does_not_contact_it(client, net):
    upload(client, make_pdf(["Tab. Zorblattin 10 mg once daily", "Dr. Rao City Clinic"]))
    body = ask(client, "side effects of my medicine")
    assert body["missing_medicines"] == ["Zorblattin"] and "Zorblattin" in body["wikipedia_offer"]["query"]
    assert net["get"] == []


# --- Wikipedia permission -------------------------------------------------------
def test_unknown_question_asks_permission_first(client, net):
    body = ask(client, "what is the capital of France")
    assert body["status"] == "abstained" and body["wikipedia_offer"]["query"] and net["get"] == []


@pytest.mark.parametrize("flag", [None, False, "true", 1])
def test_only_explicit_true_is_permission(client, net, flag):
    ask(client, "what is zorblatt fever", allow_wikipedia=flag, wikipedia_query="zorblatt fever")
    assert net["get"] == []


def test_permission_needs_the_exact_term(client):
    r = client.post("/api/ask", json={"question": "x", "allow_wikipedia": True})
    assert r.status_code == 400


def test_wikipedia_answer_after_permission(client, net):
    body = ask(client, "how is zorblatt fever treated", allow_wikipedia=True, wikipedia_query="zorblatt fever")
    assert body["status"] == "answered"
    assert all(s["origin"] == "wikipedia" for s in body["sources"]) and body["sources"][0]["url"].startswith("https://en.wikipedia.org/")
    assert all(url == "https://en.wikipedia.org/w/api.php" for url, _ in net["get"])
    assert "SECRETREF" not in str(body)
    assert next(p for p in body["wikipedia_pages"] if p["title"] == "Zorblatt")["authority"] == 1.0


def test_wikipedia_with_prescription_sends_only_the_approved_term(client, net):
    upload(client)
    body = ask(client, "is zorblatt fever why my doctor prescribed this", model="openai/gpt-oss-120b", allow_wikipedia=True, wikipedia_query="zorblatt fever")
    assert body["memory_used"]
    assert "Metformin" not in str(net["get"]) and "doctor" not in str(net["get"])
    assert not any("groq" in url for url, _ in net["post"])


def test_wikipedia_answers_when_every_page_contains_the_term(client, net):
    """Regression: BM25 idf is tiny when all crawled pages share the term; the answer must not be rejected."""
    body = ask(client, "zorblatt", allow_wikipedia=True, wikipedia_query="zorblatt")
    assert body["sources"][0]["score"] < config.ABSTAIN_SCORE_THRESHOLD or body["status"] == "answered"
    assert body["status"] == "answered"


def test_wikipedia_rejects_unrelated_pages(client, net):
    assert ask(client, "quokka habitat", allow_wikipedia=True, wikipedia_query="quokka habitat")["status"] == "abstained"


@pytest.mark.parametrize("q,term", [("what is the capital of France?", "capital of France"), ("how do I renew my passport", "renew my passport"),
                                    ("biryani kaise banate hain", "biryani banate")])
def test_wikipedia_term_is_readable(client, net, q, term):
    assert ask(client, q)["wikipedia_offer"]["query"] == term


def test_wikipedia_failure_is_reported(client, monkeypatch):
    def boom(*a, **k): raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "get", boom)
    body = ask(client, "x y", allow_wikipedia=True, wikipedia_query="zorblatt")
    assert body["status"] == "abstained" and "could not be reached" in body["abstain"]["reason"]


def test_hindi_terms_use_hindi_wikipedia():
    assert wikipedia.host_for("डेंगू बुखार") == "hi.wikipedia.org" and wikipedia.host_for("dengue") == "en.wikipedia.org"


# --- search history ----------------------------------------------------------------
def test_history_newest_first_deduplicated_and_clearable(client):
    for q in ("what is asthma", "what is gout", "what is asthma"):
        ask(client, q)
    items = client.get("/api/history").get_json()["history"]
    assert [i["query"] for i in items] == ["what is asthma", "what is gout"]
    assert len(client.get("/api/history?limit=1").get_json()["history"]) == 1
    assert client.delete(f"/api/history/{items[0]['id']}").status_code == 200
    assert client.delete("/api/history").get_json()["deleted"] == 1
    assert client.get("/api/history").get_json()["history"] == []


def test_crisis_searches_are_not_saved(client):
    ask(client, "I want to kill myself")
    assert client.get("/api/history").get_json()["history"] == []


def test_history_is_capped(client, private_store, monkeypatch):
    monkeypatch.setattr(config, "HISTORY_LIMIT", 3)
    for i in range(5):
        private_store.add_search("p", f"q{i}", "answered")
    assert [h["query"] for h in private_store.history("p", 10)] == ["q4", "q3", "q2"]


# --- link modes and decimals ----------------------------------------------------------
@pytest.mark.parametrize("mode", config.LINK_MODES)
def test_link_modes(client, mode):
    body = ask(client, "what are the symptoms of diabetes", link_mode=mode)
    assert body["status"] == "answered" and body["sources"][0]["link"]["topic"]
    assert body["inspector"]["link_analysis"]["mode"] == mode


def test_bad_link_mode(client):
    assert client.post("/api/ask", json={"question": "x", "link_mode": "nope"}).status_code == 400


def test_decimal_doses_are_not_split():
    assert split_into_sentences("Take Metformin 2.5 mg daily [1]. Maximum 4.5 g [2].") == ["Take Metformin 2.5 mg daily [1].", "Maximum 4.5 g [2]."]
    msg = [{"role": "user", "content": "Context Pages:\n[1] Reference Page 1 (ID: x)\nQuestion: q\nContent: The usual metformin dose is 2.5 mg daily for adults.\n\n\nUser Health Question: metformin dose\n\nInstructions: x"}]
    assert "2.5 mg daily" in llm_client.generate_mock_rag_answer(msg)


# --- related topics -------------------------------------------------------------------
def test_related_topics_under_database_answers(client):
    body = ask(client, "how to prevent osteoporosis")
    names = [r["name"].lower() for r in body["related"]]
    assert body["status"] == "answered" and 1 <= len(names) <= config.RELATED_COUNT
    assert all(r["question"].startswith("What is ") for r in body["related"])
    assert "osteoporosis" not in names and "blood" not in names


def test_related_topics_follow_the_cited_passages_only(client):
    # the curated paracetamol page is not part of the topic graph, so nothing unrelated may be suggested
    body = ask(client, "what is the recommended dosage for paracetamol in adults")
    assert not any("blood pressure" in r["name"].lower() for r in body["related"])


def test_no_related_topics_when_not_answered_from_the_database(client, net):
    assert ask(client, "what is the capital of France")["related"] == []
    upload(client)
    assert ask(client, "what medicines did my doctor prescribe")["related"] == []
    assert ask(client, "I want to kill myself")["related"] == []


# --- Rocchio feedback through the API ------------------------------------------------------
def test_explicit_feedback_refines_and_hides_not_relevant(client, private_store):
    first = ask(client, "what are the symptoms of diabetes")
    ids = [s["chunk_id"] for s in first["sources"]]
    body = ask(client, "what are the symptoms of diabetes", feedback={"relevant": ids[:1], "non_relevant": ids[1:2]}, save_history=False)
    fb = body["inspector"]["feedback"]
    assert fb["applied"] and fb["mode"] == "explicit" and fb["added_terms"]
    assert ids[1] not in [s["chunk_id"] for s in body["sources"]]
    assert [h["query"] for h in client.get("/api/history").get_json()["history"]] == ["what are the symptoms of diabetes"]


def test_feedback_never_rescues_an_off_topic_question(client):
    any_id = ask(client, "what are the symptoms of diabetes")["sources"][0]["chunk_id"]
    body = ask(client, "what is the capital of France", feedback={"relevant": [any_id], "non_relevant": []})
    assert body["status"] == "abstained"


@pytest.mark.parametrize("fb", [{"relevant": ["nope"]}, {"relevant": "x"}, [1], {"relevant": ["qa_00001_c0"], "non_relevant": ["qa_00001_c0"]},
                                {"relevant": [f"qa_{i}_c0" for i in range(1, 12)], "non_relevant": []}])
def test_bad_feedback_is_rejected(client, fb):
    assert client.post("/api/ask", json={"question": "diabetes", "feedback": fb}).status_code == 400


def test_pseudo_feedback_is_off_by_default_and_optional(client):
    assert ask(client, "what are the symptoms of diabetes")["inspector"]["feedback"]["applied"] is False
    body = ask(client, "what are the symptoms of diabetes", pseudo_feedback=True)
    assert body["inspector"]["feedback"]["mode"] == "pseudo"
