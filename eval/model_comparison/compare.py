"""Compares retrievers on queries.json. Run: python -m eval.model_comparison.compare
Writes eval/model_comparison/results.json."""
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import config
from backend.ir.bm25 import BM25Retriever
from backend.ir.hinglish import HinglishExpander
from backend.ir.inverted_index import InvertedIndex
from backend.ir.tfidf import TfIdfRetriever
from eval.model_comparison.encoders import BGEM3, BasePaperMiniLM, MedCPT, MiniLM

HERE, OUT = Path(__file__).parent, config.PROCESSED_DATA_DIR / "model_comparison"
K, DEPTH, RRF_K = 10, 20, config.RRF_K
LANGS = ("en", "hinglish", "hindi")


def rrf(*runs):
    s = {}
    for run in runs:
        for rank, cid in enumerate(run[:DEPTH], 1):
            s[cid] = s.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(s, key=lambda c: -s[c])


def metrics(run, grades, ideal):
    g = [grades.get(c, 0) for c in run[:K]]
    dcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(g))
    idcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(ideal[:K]))
    first = next((i for i, x in enumerate(g) if x), None)
    return {"hit@1": float(bool(g) and g[0] > 0), "hit@5": float(any(g[:5])),
            "mrr@10": 0.0 if first is None else 1.0 / (first + 1), "ndcg@10": dcg / idcg}


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))


if __name__ == "__main__":
    index = InvertedIndex.load()
    chunks = list(index.chunks_map.values())
    ids = [c["chunk_id"] for c in chunks]
    data = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
    tfidf, bm25, exp = TfIdfRetriever(index), BM25Retriever(index), HinglishExpander()

    by_title = {}
    for c in chunks:
        by_title.setdefault(c["question"], []).append(c["chunk_id"])
    needs = data["answerable"]
    for n in needs:
        n["grades"] = {cid: g for t, g in n["relevant_titles"].items() for cid in by_title[t]}
        n["ideal"] = sorted(n["grades"].values(), reverse=True)

    # queries in two forms: raw, and with the app's Hinglish lexicon expansion
    texts = {}
    for li in LANGS:
        raw = [n[li] for n in needs]
        texts[(li, "raw")], texts[(li, "lexicon")] = raw, [exp.expand_query(q)[0] for q in raw]
    un = data["unanswerable"]
    texts[("un", "raw")], texts[("un", "lexicon")] = un, [exp.expand_query(q)[0] for q in un]

    runs, top, latency = {}, {}, {}
    for key, qs in texts.items():
        for name, r in (("TF-IDF", tfidf), ("BM25", bm25)):
            t = time.time(); res = [r.search(q, top_k=DEPTH) for q in qs]
            latency.setdefault(name, []).append((time.time() - t) / len(qs) * 1000)
            runs[(name, key)] = [[x["chunk_id"] for x in y] for y in res]
            top[(name, key)] = [y[0]["score"] if y else 0.0 for y in res]

    for cls, emb in ((BasePaperMiniLM, OUT / "base-minilm.npy"), (MiniLM, config.EMBEDDINGS_CACHE_PATH), (MedCPT, OUT / "medcpt.npy"), (BGEM3, OUT / "bge-m3.npy")):
        enc, D = cls(), np.load(emb)
        assert D.shape[0] == len(ids), f"{emb} does not match the index"
        S = sp.load_npz(OUT / "bge-m3-sparse.npz") if cls is BGEM3 else None
        enc.encode_queries(["warm up"])
        for key, qs in texts.items():
            t_all = time.time()
            for q in qs[:20]:
                v, _ = enc.encode_queries([q]); np.argpartition(-(D @ v[0]), DEPTH)
            latency.setdefault(cls.name + " dense", []).append((time.time() - t_all) / 20 * 1000)
            Q, QS = enc.encode_queries(qs)
            modes = {cls.name + " dense": Q @ D.T}
            if S is not None:
                modes["BGE-M3 sparse"] = (QS @ S.T).toarray()
                modes["BGE-M3 dense+sparse"] = modes["BGE-M3 dense"] + 0.3 * modes["BGE-M3 sparse"]  # paper's weights
            for m, sc in modes.items():
                order = np.argsort(-sc, axis=1, kind="stable")[:, :DEPTH]
                runs[(m, key)] = [[ids[j] for j in row] for row in order]
                top[(m, key)] = sc.max(axis=1).tolist()
        del enc, D

    for key in texts:
        for d in ("MiniLM (current) dense", "MedCPT dense", "BGE-M3 dense"):
            short = d.replace(" dense", "").replace(" (current)", "")
            runs[(f"RRF TF-IDF+{short}", key)] = [rrf(a, b) for a, b in zip(runs[("TF-IDF", key)], runs[(d, key)])]

    systems = list(dict.fromkeys(s for s, _ in runs))
    out, per_q = {}, {}
    for s in systems:
        for li in LANGS:
            for form in ("raw", "lexicon"):
                ms = [metrics(r, n["grades"], n["ideal"]) for r, n in zip(runs[(s, (li, form))], needs)]
                per_q[(s, li, form)] = np.array([m["ndcg@10"] for m in ms])
                out[f"{s}|{li}|{form}"] = {k: round(float(np.mean([m[k] for m in ms])), 4) for k in ms[0]}

    rng = np.random.default_rng(0)

    def compare(a, b):  # paired bootstrap 95% CI + sign-flip permutation p-value on nDCG@10
        d = a - b
        ci = np.percentile(rng.choice(d, (10000, len(d))).mean(1), [2.5, 97.5])
        flips = (rng.choice([-1, 1], (10000, len(d))) * d).mean(1)
        return {"diff": round(float(d.mean()), 4), "ci95": [round(float(x), 4) for x in ci],
                "p": round(float((np.abs(flips) >= abs(d.mean()) - 1e-12).mean()), 4)}

    tests = {}
    for li in LANGS:
        for form in ("raw", "lexicon"):
            for a in ("MiniLM (current) dense", "MedCPT dense", "BGE-M3 dense", "BGE-M3 dense+sparse", "RRF TF-IDF+MiniLM"):
                tests[f"{a} vs base paper|{li}|{form}"] = compare(per_q[(a, li, form)], per_q[("MiniLM-L6 (base paper) dense", li, form)])
            for a in ("MedCPT dense", "BGE-M3 dense", "BGE-M3 dense+sparse"):
                tests[f"{a} vs MiniLM|{li}|{form}"] = compare(per_q[(a, li, form)], per_q[("MiniLM (current) dense", li, form)])
            tests[f"BGE-M3 dense vs MedCPT|{li}|{form}"] = compare(per_q[("BGE-M3 dense", li, form)], per_q[("MedCPT dense", li, form)])
            for a in ("MedCPT", "BGE-M3"):
                tests[f"RRF {a} vs RRF MiniLM|{li}|{form}"] = compare(per_q[(f"RRF TF-IDF+{a}", li, form)], per_q[("RRF TF-IDF+MiniLM", li, form)])
            tests[f"BGE-M3 dense vs RRF TF-IDF+MiniLM|{li}|{form}"] = compare(per_q[("BGE-M3 dense", li, form)], per_q[("RRF TF-IDF+MiniLM", li, form)])

    # can the top-1 score alone separate answerable from unanswerable questions?
    sep = {s: {li: round(auroc(top[(s, (li, "raw"))], top[(s, ("un", "raw"))]), 4) for li in LANGS}
           for s in systems if (s, ("en", "raw")) in top}
    result = {"n_needs": len(needs), "n_unanswerable": len(un), "passages": len(ids), "metrics": out, "tests_ndcg@10": tests,
              "answerable_vs_unanswerable_auroc": sep, "latency_ms_per_query": {k: round(float(np.mean(v)), 1) for k, v in latency.items()}}
    result["per_query_ndcg@10"] = {f"{s}|{li}|{f}": [round(float(x), 4) for x in v] for (s, li, f), v in per_q.items()}
    (HERE / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    for li in LANGS:
        for form in ("raw", "lexicon"):
            print(f"\n== {li} / {form}")
            for s in systems:
                m = out[f"{s}|{li}|{form}"]; print(f"{s:28s} hit@1 {m['hit@1']:.3f} hit@5 {m['hit@5']:.3f} mrr {m['mrr@10']:.3f} ndcg {m['ndcg@10']:.3f}")
    print("\nAUROC", json.dumps(sep, indent=0)); print("latency", result["latency_ms_per_query"])
    for k, v in tests.items():
        print(k, v)
