"""
Does Rocchio feedback help HealthNest's search? Run: python -m eval.model_comparison.rocchio_eval
Writes eval/model_comparison/rocchio_results.json.

1. Pseudo-relevance feedback (automatic). Settings (beta, confidence gate) are chosen on the even-numbered
   needs and reported on the odd-numbered ones, for English, Hinglish and Hindi, with the app's own path
   (lexicon expansion, hybrid retrieval, PageRank+HITS re-ranking).
2. Explicit feedback (the "More like this / Not relevant" buttons), simulated: the user marks the first 5
   results using the relevance labels. Scored on the RESIDUAL collection (passages the user has not seen
   yet), the standard way to evaluate relevance feedback (IIR ch. 9).
3. "I don't know" behaviour with the chosen settings, compared with no feedback.
"""
import itertools
import json
from pathlib import Path

import numpy as np

import config
from backend.pipeline import get_pipeline
from eval.model_comparison.compare import LANGS, metrics

HERE = Path(__file__).parent
K = 10


def perm_p(d, rng):
    return float((np.abs((rng.choice([-1, 1], (10000, len(d))) * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


if __name__ == "__main__":
    p = get_pipeline(); p.initialize()
    data = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
    needs = data["answerable"]
    by_title = {}
    for c in p.chunks:
        by_title.setdefault(c["question"], []).append(c["chunk_id"])
    for n in needs:
        n["grades"] = {cid: g for t, g in n["relevant_titles"].items() for cid in by_title[t]}
        n["ideal"] = sorted(n["grades"].values(), reverse=True)
    expanded = {(n["id"], li): p.hinglish_expander.expand_query(n[li])[0] for n in needs for li in LANGS}

    def run(n, li, fb, k=K):
        return [r["chunk_id"] for r in p._retrieve(expanded[(n["id"], li)], "hybrid", k, config.LINK_DEFAULT_MODE, feedback=fb)[0]]

    def ndcg(subset, fb):
        return np.array([[metrics(run(n, li, fb), n["grades"], n["ideal"])["ndcg@10"] for n in subset] for li in LANGS])

    rng = np.random.default_rng(0)
    dev, test = needs[::2], needs[1::2]
    out = {"setup": "hybrid + lexicon + link_mode=" + config.LINK_DEFAULT_MODE, "prf": {}, "explicit": {}, "abstention": {}}

    # 1. pseudo-relevance feedback
    base_dev, base_test = ndcg(dev, None), ndcg(test, None)
    grid = {}
    for beta, gate in itertools.product((0.25, 0.5, 0.75), (0.45, 0.55, 0.65)):
        config.ROCCHIO_BETA, config.ROCCHIO_PRF_MIN_DENSE = beta, gate
        grid[(beta, gate)] = float(ndcg(dev, {"pseudo": True}).mean())
        print(f"dev beta={beta} gate={gate}: {grid[(beta, gate)]:.4f} (no feedback {base_dev.mean():.4f})", flush=True)
    beta, gate = max(grid, key=grid.get)
    config.ROCCHIO_BETA, config.ROCCHIO_PRF_MIN_DENSE = beta, gate
    prf_test = ndcg(test, {"pseudo": True})
    d = (prf_test - base_test).mean(0)
    out["prf"] = {"dev_grid": {f"beta={b},gate={g}": round(v, 4) for (b, g), v in grid.items()}, "dev_no_feedback": round(float(base_dev.mean()), 4),
                  "chosen": {"beta": beta, "gate": gate},
                  "test_no_feedback": dict(zip(LANGS, base_test.mean(1).round(4).tolist())), "test_prf": dict(zip(LANGS, prf_test.mean(1).round(4).tolist())),
                  "test_mean": [round(float(base_test.mean()), 4), round(float(prf_test.mean()), 4)], "diff": round(float(d.mean()), 4), "p": perm_p(d, rng),
                  "helped_hurt_same": [int((d > 1e-9).sum()), int((d < -1e-9).sum()), int((abs(d) <= 1e-9).sum())]}
    print("PRF test:", out["prf"], flush=True)

    # 2. explicit feedback on the residual collection (all 40 needs; textbook beta)
    config.ROCCHIO_BETA = 0.75
    rows = {li: [] for li in LANGS}
    for n in needs:
        for li in LANGS:
            first = run(n, li, None, k=K + 5)
            seen = first[:5]
            rel = [c for c in seen if n["grades"].get(c, 0) > 0]
            non = [c for c in seen if n["grades"].get(c, 0) == 0]
            if not rel and not non:
                continue
            resid = lambda ranking: [c for c in ranking if c not in seen][:K]
            ideal = sorted((g for c, g in n["grades"].items() if c not in seen), reverse=True)
            if not ideal:
                continue
            before = metrics(resid(first), n["grades"], ideal)["ndcg@10"]
            after = metrics(resid(run(n, li, {"relevant": rel, "non_relevant": non}, k=K + 5)), n["grades"], ideal)["ndcg@10"]
            rows[li].append((before, after, bool(rel)))
    for li in LANGS:
        b, a, has_rel = map(np.array, zip(*rows[li]))
        out["explicit"][li] = {"queries": len(b), "with_a_relevant_mark": int(has_rel.sum()), "residual_ndcg_before": round(float(b.mean()), 4),
                               "residual_ndcg_after": round(float(a.mean()), 4), "p": perm_p(a - b, rng)}
    allb = np.array([x[0] for li in LANGS for x in rows[li]]); alla = np.array([x[1] for li in LANGS for x in rows[li]])
    out["explicit"]["all"] = {"queries": len(allb), "before": round(float(allb.mean()), 4), "after": round(float(alla.mean()), 4), "p": perm_p(alla - allb, rng)}
    print("explicit:", out["explicit"], flush=True)

    # 3. abstention with the chosen PRF settings
    config.ROCCHIO_BETA = beta
    for name, prf in (("no_feedback", False), ("prf", True)):
        ab = {li: sum(p.ask(n[li], model="mock-demo", pseudo_feedback=prf)["status"] == "abstained" for n in needs) for li in LANGS}
        ab["off_topic_abstained"] = sum(p.ask(q, model="mock-demo", pseudo_feedback=prf)["status"] == "abstained" for q in data["unanswerable"])
        out["abstention"][name] = ab
    print("abstention:", out["abstention"], flush=True)
    (HERE / "rocchio_results.json").write_text(json.dumps(out, indent=1))
