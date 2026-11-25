"""
backend/ir/linkgraph.py
Hand-written link analysis for HealthNest: PageRank and HITS.

The corpus has no hyperlinks, so links are inferred: every condition/topic ("Asthma", "Stroke", ...)
is a node, and topic A links to topic B when A's passages mention B by name (longest match wins, so
"diabetes insipidus" is not counted as a mention of "diabetes").

  PageRank - query independent: random surfer with damping d, dangling mass spread uniformly.
  HITS     - query dependent: root set = topics of the retrieved passages, expanded by in/out
             neighbours to a base set, then mutual reinforcement of hubs and authorities.

Lecture Concepts: Link Analysis, PageRank (power iteration), HITS (hubs & authorities).
Strict Compliance: no graph library used.
"""

import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple

import config

_WORD = re.compile(r"[a-z0-9']+")
_MAX_NAME_WORDS = 8
_TITLE_PREFIX = re.compile(
    r"^(?:what (?:is|are) \(are\)|what is \(are\)|what are the symptoms of|what are the treatments for|what causes|"
    r"how to diagnose|how to prevent|who is at risk for|how many people are affected by|is|what to do for|"
    r"what are the genetic changes related to|what are the complications of|what is the outlook for|"
    r"what are the stages of|what are the brand names of|do you have information about|"
    r"what research \(or clinical trials\) is being done for)\s+", re.I)


def topic_of(title: str) -> str:
    """'What are the symptoms of Asthma ?' -> 'asthma'. Titles with no known template are their own topic."""
    t = _TITLE_PREFIX.sub("", title.strip())
    t = re.sub(r"\binherited\s*\?*\s*$", "", t, flags=re.I) if title.lower().startswith("is ") else t
    return " ".join(_WORD.findall(t.lower()))


def pagerank(links: Dict[str, Set[str]], nodes: Iterable[str], d: float = config.PAGERANK_DAMPING,
             tol: float = 1e-9, max_iter: int = 200) -> Dict[str, float]:
    """PR(v) = (1-d)/N + d * ( sum_{u->v} PR(u)/out(u) + dangling/N ). Scores sum to 1."""
    nodes = sorted(nodes)
    n = len(nodes)
    if n == 0:
        return {}
    pr = {v: 1.0 / n for v in nodes}
    for _ in range(max_iter):
        dangling = sum(pr[v] for v in nodes if not links.get(v))
        nxt = {v: (1.0 - d) / n + d * dangling / n for v in nodes}
        for u in nodes:
            out = links.get(u)
            if out:
                share = d * pr[u] / len(out)
                for v in out:
                    nxt[v] += share
        delta = sum(abs(nxt[v] - pr[v]) for v in nodes)
        pr = nxt
        if delta < tol:
            break
    return pr


def hits(links: Dict[str, Set[str]], nodes: Iterable[str], tol: float = 1e-9,
         max_iter: int = 200) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Returns (authority, hub), L2-normalised, for the subgraph induced by `nodes`."""
    nodes = sorted(nodes)
    keep = set(nodes)
    out = {u: sorted(links.get(u, set()) & keep) for u in nodes}
    if not any(out.values()):
        return {v: 0.0 for v in nodes}, {v: 0.0 for v in nodes}
    auth = {v: 1.0 for v in nodes}
    hub = {v: 1.0 for v in nodes}
    for _ in range(max_iter):
        new_auth = {v: 0.0 for v in nodes}
        for u in nodes:
            for v in out[u]:
                new_auth[v] += hub[u]
        norm = math.sqrt(sum(x * x for x in new_auth.values())) or 1.0
        new_auth = {v: x / norm for v, x in new_auth.items()}
        new_hub = {u: sum(new_auth[v] for v in out[u]) for u in nodes}
        norm = math.sqrt(sum(x * x for x in new_hub.values())) or 1.0
        new_hub = {u: x / norm for u, x in new_hub.items()}
        delta = sum(abs(new_auth[v] - auth[v]) + abs(new_hub[v] - hub[v]) for v in nodes)
        auth, hub = new_auth, new_hub
        if delta < tol:
            break
    return auth, hub


class LinkGraph:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.out: Dict[str, Set[str]] = defaultdict(set)
        self.inc: Dict[str, Set[str]] = defaultdict(set)
        self.topic_by_title: Dict[str, str] = {}
        for c in chunks:
            if c["question"] not in self.topic_by_title:
                self.topic_by_title[c["question"]] = topic_of(c["question"])
        self.nodes: Set[str] = {t for t in self.topic_by_title.values() if t}
        # Only named topics can be mentioned; very short names ("als") are too ambiguous.
        names = {t for t in self.nodes if len(t) >= 4 and len(t.split()) <= _MAX_NAME_WORDS}
        first = defaultdict(int)
        for t in names:
            w = t.split()
            first[w[0]] = max(first[w[0]], len(w))
        for c in chunks:
            src = self.topic_by_title[c["question"]]
            toks = _WORD.findall(c["text"].lower())
            i = 0
            while i < len(toks):
                for k in range(first.get(toks[i], 0), 0, -1):
                    cand = " ".join(toks[i:i + k])
                    if cand in names:
                        if cand != src and src:
                            self.out[src].add(cand)
                            self.inc[cand].add(src)
                        i += k - 1
                        break
                i += 1
        self.pr = pagerank(self.out, self.nodes)
        self._max_pr = max(self.pr.values(), default=1.0)

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.out.values())

    def base_set(self, root: Iterable[str], cap: int = config.HITS_NEIGHBOUR_CAP) -> Set[str]:
        """Root topics plus up to `cap` in- and out-neighbours each (highest PageRank first)."""
        root = [t for t in root if t in self.nodes]
        base = set(root)
        for t in root:
            for neigh in (self.out.get(t, ()), self.inc.get(t, ())):
                base.update(sorted(neigh, key=lambda x: (-self.pr[x], x))[:cap])
        return base

    def rerank(self, results: List[Dict[str, Any]], mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Re-orders retrieved results. Each item keeps its retrieval `score`; ordering uses
        (1-w)*norm(retrieval) + w*norm(link evidence), w = config.LINK_WEIGHT.
        """
        if not results:
            return results, {}
        topics = [self.topic_by_title.get(r["chunk"]["question"], "") for r in results]
        base = self.base_set(dict.fromkeys(topics))
        auth, hub = hits(self.out, base)
        max_auth = max(auth.values(), default=0.0) or 1.0
        hi = max(r["score"] for r in results)
        lo = min(r["score"] for r in results)
        out = []
        for r, t in zip(results, topics):
            rel = (r["score"] - lo) / (hi - lo) if hi > lo else 1.0
            pr_n = math.log1p(1000 * self.pr.get(t, 0.0) / self._max_pr) / math.log1p(1000)
            au_n = auth.get(t, 0.0) / max_auth
            link = {"pagerank": pr_n, "hits": au_n, "both": 0.5 * (pr_n + au_n)}.get(mode, 0.0)
            w = 0.0 if mode == "off" else config.LINK_WEIGHT
            item = dict(r)
            item["link"] = {"topic": t, "pagerank": round(self.pr.get(t, 0.0), 6), "authority": round(auth.get(t, 0.0), 4),
                            "hub": round(hub.get(t, 0.0), 4), "final": round((1 - w) * rel + w * link, 4)}
            out.append(item)
        if mode != "off":
            out.sort(key=lambda x: -x["link"]["final"])  # stable: ties keep retrieval order
        for i, item in enumerate(out, 1):
            item["rank"] = i

        def top(scores):
            return [{"topic": t, "score": round(s, 4)} for t, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:5] if s > 0]
        return out, {"mode": mode, "base_set_size": len(base), "top_authorities": top(auth), "top_hubs": top(hub),
                     "top_pagerank": [{"topic": t, "score": round(self.pr[t], 6)}
                                      for t in sorted({t for t in topics if t}, key=lambda t: -self.pr.get(t, 0))[:5]]}
