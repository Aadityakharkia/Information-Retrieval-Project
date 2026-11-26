"""PageRank / HITS maths checked by hand on toy graphs."""
import pytest

from backend.ir.linkgraph import LinkGraph, hits, pagerank, topic_of


def test_pagerank_cycle_is_uniform_and_sums_to_one():
    pr = pagerank({"a": {"b"}, "b": {"c"}, "c": {"a"}}, "abc")
    assert all(v == pytest.approx(1 / 3) for v in pr.values())


def test_pagerank_dangling_node_by_hand():
    # a->b, b dangling, d=0.85: a = .075 + .425 b,  b = .075 + .85 a + .425 b  => a = 0.350877
    pr = pagerank({"a": {"b"}}, "ab")
    assert pr["a"] == pytest.approx(0.350877, abs=1e-5) and sum(pr.values()) == pytest.approx(1)


def test_hits_by_hand():
    auth, hub = hits({"a": {"c"}, "b": {"c"}}, "abc")
    assert auth == pytest.approx({"a": 0, "b": 0, "c": 1})
    assert hub["a"] == pytest.approx(2 ** -0.5) and hub["c"] == 0


def test_hits_without_links_is_zero_not_nan():
    assert hits({}, "ab") == ({"a": 0.0, "b": 0.0}, {"a": 0.0, "b": 0.0})


@pytest.mark.parametrize("title,topic", [
    ("What are the symptoms of Asthma ?", "asthma"), ("What is (are) Kidney Stones ?", "kidney stones"),
    ("Is Huntington disease inherited ?", "huntington disease"), ("Who is at risk for Stroke? ?", "stroke")])
def test_topic_of(title, topic):
    assert topic_of(title) == topic


def chunk(i, q, t):
    return {"chunk_id": str(i), "question": q, "text": t}


CHUNKS = [chunk(1, "What is (are) Stroke ?", "Stroke can follow high blood pressure."),
          chunk(2, "What is (are) High Blood Pressure ?", "Pressure in arteries."),
          chunk(3, "What is (are) Diabetes ?", "May cause high blood pressure and stroke."),
          chunk(4, "What is (are) Diabetes Insipidus ?", "A rare condition."),
          chunk(5, "What is (are) Thirst ?", "Seen in diabetes insipidus.")]


def test_mentions_become_links_and_longest_name_wins():
    g = LinkGraph(CHUNKS)
    assert g.out["diabetes"] == {"high blood pressure", "stroke"}
    assert g.out["thirst"] == {"diabetes insipidus"}  # not "diabetes"


def test_rerank_off_keeps_order_and_scores():
    g = LinkGraph(CHUNKS)
    res = [{"chunk_id": "3", "score": 0.03, "chunk": CHUNKS[2]}, {"chunk_id": "2", "score": 0.02, "chunk": CHUNKS[1]}]
    off, _ = g.rerank(res, "off")
    both, info = g.rerank(res, "both")
    assert [r["chunk_id"] for r in off] == ["3", "2"] and {r["score"] for r in both} == {0.03, 0.02}
    assert info["top_authorities"][0]["topic"] == "high blood pressure"


# --- topic-sensitive PageRank and related topics -------------------------------------
import numpy as np

import config
from backend.ir.linkgraph import display_name, personalized_pagerank


def test_uniform_restart_equals_global_pagerank():
    links = {"a": {"b"}, "b": {"c"}, "c": {"a", "b"}}
    nodes = "abc"
    src = np.array([nodes.index(u) for u in links for v in sorted(links[u])])
    dst = np.array([nodes.index(v) for u in links for v in sorted(links[u])])
    ppr = personalized_pagerank(src, dst, 3, np.ones(3))
    assert ppr == pytest.approx([pagerank(links, nodes)[x] for x in nodes], abs=1e-8)


def test_restart_pulls_rank_towards_the_seed():
    src, dst = np.array([0, 1, 2]), np.array([1, 2, 0])  # 3-cycle: global PR is uniform
    ppr = personalized_pagerank(src, dst, 3, np.array([1.0, 0, 0]))
    assert ppr.sum() == pytest.approx(1) and ppr[0] > ppr[1] > ppr[2]


@pytest.mark.parametrize("title,name", [("What is (are) Bone Density ?", "Bone Density"), ("How to prevent Kidney Stones in Adults ?", "Kidney Stones in Adults"),
                                        ("Is Huntington disease inherited ?", "Huntington disease")])
def test_display_name(title, name):
    assert display_name(title) == name


def related_graph(monkeypatch):
    monkeypatch.setattr(config, "RELATED_MIN_INDEGREE", 1)
    monkeypatch.setattr(config, "RELATED_MAX_INDEGREE_PCTL", 100)
    page = lambda i, topic, text: chunk(i, f"What is (are) {topic} ?", text)
    chunks = [page(1, "Asthma", "Asthma relates to Eczema and Allergy and Preeclampsia and Suicide and Asthma Attacks."),
              page(2, "Eczema", "Eczema and Asthma."), page(3, "Allergy", "Allergy may trigger Asthma."),
              page(4, "Preeclampsia", "Pregnancy problem."), page(5, "Suicide", "Crisis."), page(6, "Asthma Attacks", "Sudden."),
              page(7, "Hay Fever", "Allergy and Eczema.")]
    return LinkGraph(chunks)


def test_related_prefers_mutual_links_and_filters(monkeypatch):
    g = related_graph(monkeypatch)
    got = g.related({"asthma": 1.0}, k=3, similarity=lambda main, names: [0.5 if n in ("allergy", "eczema") else 0.0 for n in names])
    assert [r["name"] for r in got] == ["Allergy", "Eczema"] and {r["link"] for r in got} == {"mutual"}


def test_loose_mutual_links_are_dropped(monkeypatch):
    g = related_graph(monkeypatch)
    sim = lambda main, names: [0.1 if n == "eczema" else 0.5 for n in names]
    assert "Eczema" not in [r["name"] for r in g.related({"asthma": 1.0}, k=5, similarity=sim)]


def test_one_way_links_need_similar_meaning(monkeypatch):
    g = related_graph(monkeypatch)
    sim = lambda main, names: [0.9 if n == "asthma attacks" else 0.1 for n in names]
    names = [r["name"] for r in g.related({"asthma": 1.0}, k=5, similarity=sim)]
    assert "Asthma Attacks" in names and "Preeclampsia" not in names


def test_crisis_topics_are_never_suggested(monkeypatch):
    g = related_graph(monkeypatch)
    assert "Suicide" not in [r["name"] for r in g.related({"asthma": 1.0}, k=9, similarity=lambda m, n: [1.0] * len(n))]


def test_related_without_seeds_is_empty(monkeypatch):
    assert related_graph(monkeypatch).related({"unknown": 1.0}) == []


# --- Rocchio relevance feedback ---------------------------------------------------------
from backend.ir import rocchio
from backend.ir.inverted_index import InvertedIndex
from backend.ir.tfidf import TfIdfRetriever

DOCS = [{"chunk_id": "a", "qa_id": "a", "question": "Diabetes", "text": "diabetes insulin glucose sugar", "position": 0},
        {"chunk_id": "b", "qa_id": "b", "question": "Gout", "text": "gout uric acid toe", "position": 0},
        {"chunk_id": "c", "qa_id": "c", "question": "Insulin", "text": "insulin injection pancreas", "position": 0},
        {"chunk_id": "d", "qa_id": "d", "question": "Toe pain", "text": "toe pain shoes", "position": 0}]


@pytest.fixture
def small_index(monkeypatch):
    monkeypatch.setattr(config, "ROCCHIO_MIN_NEW_TERM_DF", 1)  # every word is rare in a 4-passage index
    idx = InvertedIndex()
    idx.index_chunks(DOCS)  # same tokenising (stop words, Porter stemming) as the app and as rocchio.py
    return idx


def test_rocchio_formula_by_hand(small_index):
    from backend.ir.text import tokenize
    q = rocchio.ltc_vector(tokenize("sugar"), small_index)
    assert q == {"sugar": pytest.approx(1.0)}
    w, added = rocchio.sparse_query("sugar", ["diabetes insulin glucose sugar"], ["gout uric acid toe"], small_index,
                                    alpha=1, beta=0.5, gamma=0.25, max_new_terms=2)
    rel = rocchio.ltc_vector(tokenize("diabetes insulin glucose sugar"), small_index)
    assert w["sugar"] == pytest.approx(1 + 0.5 * rel["sugar"])
    assert len(added) == 2 and set(added) <= set(tokenize("diabetes glucose insulin"))
    assert not set(tokenize("gout uric acid toe")) & set(w)  # pushed away, negative weights dropped


def test_feedback_brings_in_documents_without_the_typed_word(small_index):
    w, _ = rocchio.sparse_query("sugar", ["diabetes insulin glucose sugar"], [], small_index, max_new_terms=5)
    ids = [r["chunk_id"] for r in TfIdfRetriever(small_index).search_weighted(w, top_k=4)]
    assert ids[0] == "a" and "c" in ids  # "insulin" page found although the user typed only "sugar"


def test_dense_rocchio_moves_towards_relevant():
    q, rel, non = np.array([1.0, 0, 0]), [np.array([0, 1.0, 0])], [np.array([0, 0, 1.0])]
    v = rocchio.dense_query(q, rel, non, alpha=1, beta=0.75, gamma=0.15)
    assert np.linalg.norm(v) == pytest.approx(1) and v[1] > 0 > v[2]



def test_rare_fragments_are_not_added(small_index, monkeypatch):
    monkeypatch.setattr(config, "ROCCHIO_MIN_NEW_TERM_DF", 2)
    _, added = rocchio.sparse_query("sugar", ["diabetes insulin glucose sugar ah"], [], small_index, max_new_terms=9)
    assert added == ["insulin"]  # the only new word in 2+ passages; "ah" is too short anyway
