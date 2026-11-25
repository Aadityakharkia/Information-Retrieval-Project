"""Does PageRank / HITS re-ranking help the app's hybrid search? Run: python -m eval.model_comparison.link_eval
Dev = even-numbered needs (choose the weight), test = odd-numbered needs (report). Lexicon-expanded queries, as in the app."""
import json
from pathlib import Path

import numpy as np

import config
from backend.ir.hybrid import HybridRetriever
from backend.ir.linkgraph import LinkGraph
from backend.ir.tfidf import TfIdfRetriever
from backend.pipeline import get_pipeline
from eval.model_comparison.compare import LANGS, metrics

HERE = Path(__file__).parent

if __name__ == "__main__":
    p = get_pipeline(); p.initialize()
    g = LinkGraph(p.chunks)
    needs = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))["answerable"]
    by_title = {}
    for c in p.chunks:
        by_title.setdefault(c["question"], []).append(c["chunk_id"])
    pools = {}
    for n in needs:
        n["grades"] = {cid: gr for t, gr in n["relevant_titles"].items() for cid in by_title[t]}
        n["ideal"] = sorted(n["grades"].values(), reverse=True)
        for li in LANGS:
            q = p.hinglish_expander.expand_query(n[li])[0]
            pools[(n["id"], li)] = HybridRetriever().fuse(TfIdfRetriever(p.index).search(q, top_k=20), p.dense_retriever.search(q, top_k=20), top_k=config.LINK_POOL)

    def score(mode, w, subset):
        config.LINK_WEIGHT = w
        return np.array([[metrics([x["chunk_id"] for x in g.rerank(pools[(n["id"], li)], mode)[0]], n["grades"], n["ideal"])["ndcg@10"]
                          for n in subset] for li in LANGS])
    dev, test = needs[::2], needs[1::2]
    base_dev, base_test = score("off", 0, dev), score("off", 0, test)
    out = {"graph": {"topics": len(g.nodes), "links": g.n_edges}, "baseline_dev": round(base_dev.mean(), 4), "baseline_test": round(base_test.mean(), 4), "modes": {}}
    rng = np.random.default_rng(0)
    print(f"graph {out['graph']}  no link analysis: dev {base_dev.mean():.4f}  test {base_test.mean():.4f} {dict(zip(LANGS, base_test.mean(1).round(3)))}")
    for mode in ("pagerank", "hits", "both"):
        dev_by_w = {w: float(score(mode, w, dev).mean()) for w in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)}
        best = max(dev_by_w, key=dev_by_w.get)
        t = score(mode, best, test); d = (t - base_test).mean(0)
        pval = float((np.abs((rng.choice([-1, 1], (10000, len(d))) * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())
        out["modes"][mode] = {"dev_by_weight": {str(k): round(v, 4) for k, v in dev_by_w.items()}, "best_weight": best, "test": round(float(t.mean()), 4),
                              "test_by_language": dict(zip(LANGS, t.mean(1).round(4).tolist())), "diff_vs_baseline": round(float(d.mean()), 4), "p": round(pval, 4)}
        print(mode, out["modes"][mode])
    (HERE / "link_results.json").write_text(json.dumps(out, indent=1))
