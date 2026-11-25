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
