"""
Metrics for the project report. Run: python -m eval.model_comparison.report_metrics
Writes eval/model_comparison/report_metrics.json.

Retrieval (40 health needs x English / Hinglish / Hindi, lexicon-expanded queries as in the app):
  P@5, P@10       share of the top passages that are relevant (grade >= 1)
  R@10            share of the relevant PAGES (titles) that appear in the top 10 passages
  F1@10           harmonic mean of P@10 and R@10
  MRR@10, nDCG@10 (graded: 2 = right condition and intent, 1 = right condition)
Answer / "I don't know" decision (120 answerable + 20 off-topic questions, full pipeline):
  accuracy, precision, recall, F1 with "answer" as the positive class.
"""
import json
import math
from pathlib import Path

import numpy as np

import config
from backend.ir.bm25 import BM25Retriever
from backend.ir.tfidf import TfIdfRetriever
from backend.pipeline import get_pipeline
from eval.model_comparison.compare import LANGS, rrf
from eval.model_comparison.encoders import BGEM3, BasePaperMiniLM, MedCPT

HERE, OUT = Path(__file__).parent, config.PROCESSED_DATA_DIR / "model_comparison"
K = 10


def scores(run, grades, ideal, rel_titles, title_of):
    rel = [grades.get(c, 0) for c in run[:K]]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(rel))
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:K]))
    first = next((i for i, g in enumerate(rel) if g), None)
    p5, p10 = sum(g > 0 for g in rel[:5]) / 5, sum(g > 0 for g in rel) / K
    r10 = len({title_of[c] for c in run[:K]} & set(rel_titles)) / len(rel_titles)
    return {"P@5": p5, "P@10": p10, "R@10": r10, "F1@10": 0.0 if p10 + r10 == 0 else 2 * p10 * r10 / (p10 + r10),
            "MRR@10": 0.0 if first is None else 1 / (first + 1), "nDCG@10": dcg / idcg}


if __name__ == "__main__":
    p = get_pipeline(); p.initialize()
    chunks = list(p.index.chunks_map.values())
    ids = [c["chunk_id"] for c in chunks]
    title_of = {c["chunk_id"]: c["question"] for c in chunks}
    by_title = {}
    for c in chunks:
        by_title.setdefault(c["question"], []).append(c["chunk_id"])
    data = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
    needs = data["answerable"]
    for n in needs:
        n["grades"] = {cid: g for t, g in n["relevant_titles"].items() for cid in by_title[t]}
        n["ideal"] = sorted(n["grades"].values(), reverse=True)
    queries = {li: [p.hinglish_expander.expand_query(n[li])[0] for n in needs] for li in LANGS}

    runs = {}
    tfidf, bm25 = TfIdfRetriever(p.index), BM25Retriever(p.index)
    for li in LANGS:
        runs[("TF-IDF (lnc.ltc)", li)] = [[r["chunk_id"] for r in tfidf.search(q, top_k=20)] for q in queries[li]]
        runs[("BM25", li)] = [[r["chunk_id"] for r in bm25.search(q, top_k=20)] for q in queries[li]]
        dense = [[r["chunk_id"] for r in p.dense_retriever.search(q, top_k=20)] for q in queries[li]]
        runs[("Multilingual MiniLM", li)] = dense
        runs[("Hybrid RRF (TF-IDF + MiniLM)", li)] = [rrf(a, b) for a, b in zip(runs[("TF-IDF (lnc.ltc)", li)], dense)]
        runs[("Hybrid RRF + PageRank/HITS", li)] = [[r["chunk_id"] for r in p._retrieve(q, "hybrid", K, "both")[0]] for q in queries[li]]
    for cls, emb in ((BasePaperMiniLM, OUT / "base-minilm.npy"), (MedCPT, OUT / "medcpt.npy"), (BGEM3, OUT / "bge-m3.npy")):
        enc, D = cls(), np.load(emb)
        S = __import__("scipy.sparse", fromlist=["x"]).load_npz(OUT / "bge-m3-sparse.npz") if cls is BGEM3 else None
        name = {"MiniLM-L6 (base paper)": "MiniLM-L6 (base paper [1])", "MedCPT": "MedCPT", "BGE-M3": "BGE-M3 dense+sparse"}[cls.name]
        for li in LANGS:
            Q, QS = enc.encode_queries(queries[li])
            sc = Q @ D.T + (0.3 * (QS @ S.T).toarray() if S is not None else 0)
            runs[(name, li)] = [[ids[j] for j in row] for row in np.argsort(-sc, axis=1)[:, :20]]
        del enc, D

    systems = list(dict.fromkeys(s for s, _ in runs))
    retrieval = {}
    for s in systems:
        retrieval[s] = {}
        for li in LANGS:
            ms = [scores(r, n["grades"], n["ideal"], list(n["relevant_titles"]), title_of) for r, n in zip(runs[(s, li)], needs)]
            retrieval[s][li] = {k: round(float(np.mean([m[k] for m in ms])), 4) for k in ms[0]}
        retrieval[s]["mean"] = {k: round(float(np.mean([retrieval[s][li][k] for li in LANGS])), 4) for k in retrieval[s]["en"]}
        print(f"{s:32s}", retrieval[s]["mean"], flush=True)

    # answer / I-don't-know decision of the full pipeline (default settings, offline generator)
    def decide(q):
        return p.ask(q, model="mock-demo")["status"] != "abstained"
    rows = [(decide(n[li]), True, li) for n in needs for li in LANGS] + [(decide(q), False, "off-topic") for q in data["unanswerable"]]
    def cls_metrics(rs):
        tp = sum(a and t for a, t, _ in rs); fp = sum(a and not t for a, t, _ in rs)
        fn = sum(not a and t for a, t, _ in rs); tn = sum(not a and not t for a, t, _ in rs)
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "accuracy": round((tp + tn) / len(rs), 4), "precision": round(prec, 4),
                "recall": round(rec, 4), "F1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0}
    off = [r for r in rows if r[2] == "off-topic"]
    abstention = {"all": cls_metrics(rows), **{li: cls_metrics([r for r in rows if r[2] == li] + off) for li in LANGS}}
    print("abstention", abstention, flush=True)
    (HERE / "report_metrics.json").write_text(json.dumps({"retrieval": retrieval, "abstention": abstention}, indent=1))
