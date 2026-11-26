"""
backend/ir/linkgraph.py
Hand-written link analysis for HealthNest: PageRank and HITS.

The corpus has no hyperlinks, so links are inferred: every condition/topic ("Asthma", "Stroke", ...)
is a node, and topic A links to topic B when A's passages mention B by name (longest match wins, so
"diabetes insipidus" is not counted as a mention of "diabetes").

  PageRank - query independent: random surfer with damping d, dangling mass spread uniformly.
  HITS     - query dependent: root set = topics of the retrieved passages, expanded by in/out
             neighbours to a base set, then mutual reinforcement of hubs and authorities.
  Topic-sensitive PageRank - the random surfer restarts at the answer's own topics; used to suggest
             related topics, scored by "lift" over global PageRank so that generic topics do not win.

Lecture Concepts: Link Analysis, PageRank (power iteration), HITS (hubs & authorities),
Topic-sensitive / Personalized PageRank (Haveliwala, 2002).
Strict Compliance: no graph library used.
"""

import math
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

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


def personalized_pagerank(src: np.ndarray, dst: np.ndarray, n: int, restart: np.ndarray, d: float = config.PAGERANK_DAMPING,
                          tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """
    Topic-sensitive (personalised) PageRank over an edge list (Haveliwala, 2002):

        PR = (1 - d) * t  +  d * ( M·PR  +  dangling_mass * t )

    t is the restart (teleport) distribution: with t uniform this is ordinary PageRank; with t on the
    answer's topics the surfer keeps returning there, so nearby topics score higher. Pages with no
    outgoing links ("dangling") hand their score back to t. Scores always sum to 1.
    One iteration is a single numpy.bincount over the 36,791 edges, so a full run takes a few ms.
    """
    out_deg = np.bincount(src, minlength=n).astype(float)
    dangling = out_deg == 0
    t = restart / restart.sum()
    pr = t.copy()
    for _ in range(max_iter):
        flow = np.bincount(dst, weights=pr[src] / out_deg[src], minlength=n)
        nxt = (1 - d) * t + d * (flow + pr[dangling].sum() * t)
        done = np.abs(nxt - pr).sum() < tol
        pr = nxt
        if done:
            break
    return pr


def display_name(title: str) -> str:
    """'What are the symptoms of Bone Density ?' -> 'Bone Density' (original capitals kept)."""
    t = _TITLE_PREFIX.sub("", title.strip())
    if title.lower().startswith("is "):
        t = re.sub(r"\binherited\s*\?*\s*$", "", t, flags=re.I)
    return re.sub(r"[\s?]+$", "", t)


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
        # Array form of the graph for topic-sensitive PageRank (one run per answer).
        self._order = sorted(self.nodes)
        self._ix = {t: i for i, t in enumerate(self._order)}
        edges = [(self._ix[u], self._ix[v]) for u, vs in self.out.items() for v in vs]
        self._src = np.array([e[0] for e in edges], dtype=np.int64)
        self._dst = np.array([e[1] for e in edges], dtype=np.int64)
        self._in_deg = np.bincount(self._dst, minlength=len(self._order))
        self._global = np.array([self.pr[t] for t in self._order])
        self._max_in = float(np.percentile(self._in_deg, config.RELATED_MAX_INDEGREE_PCTL)) if len(self._order) else 0.0
        self.display: Dict[str, str] = {}
        for title, t in self.topic_by_title.items():
            # prefer the plain overview page title for the name shown to users
            if t and (t not in self.display or title.lower().startswith("what is (are)")):
                self.display[t] = display_name(title)

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

    def related(self, seed_weights: Dict[str, float], k: int = config.RELATED_COUNT,
                similarity: Optional[Callable[[str, List[str]], List[float]]] = None) -> List[Dict[str, Any]]:
        """
        Suggest up to `k` topics related to an answer (the "Related: A · B · C" line under it).

        Parameters
          seed_weights  topics of the passages the answer CITES, weighted 1/rank (built in
                        HealthNestPipeline._related). Uncited search results are left out on purpose:
                        otherwise "paracetamol dosage" suggested blood-pressure topics.
          k             how many topics to return (config.RELATED_COUNT).
          similarity    fn(main_topic, [names]) -> [cosine]; the pipeline passes the dense MiniLM model.

        How a topic is picked
          1. Topic-sensitive PageRank: a random surfer who keeps restarting at the seed topics.
             Its score alone favours generic hubs ("blood", "medicines", "surgery" for every answer),
             so topics are ranked by LIFT = personalised PageRank / global PageRank, i.e. how much more
             often they are reached from THIS answer than from a random start.
          2. Candidates are topics the seed pages mention. MUTUAL links (each page mentions the other)
             come first; ONE-WAY links (only the seed page mentions it) are used only to fill up.
          3. Filters (usable() below): not one of the most-linked 1% of topics, mentioned by at least
             RELATED_MIN_INDEGREE topics (drops most rare genetic conditions), not an administrative or
             crisis topic (_EXCLUDED), and not just a variant of the seed's own name.
          4. Meaning check with the dense model, because links alone can be loose (Asthma <-> Preeclampsia
             mention each other): mutual links need cosine >= RELATED_MIN_SIMILARITY_MUTUAL, one-way links
             >= RELATED_MIN_SIMILARITY.
          5. Near-duplicates are skipped ("Aneurysms" after "Aneurysm").

        Returns [{topic, name, link: "mutual"|"one_way", via: [seed names], lift}], possibly empty
        (e.g. glaucoma, gout) when no candidate passes every filter.

        Known limits
          - Settings were tuned by inspecting about 20 real questions, not by a benchmark with correct
            answers, so this is a qualitative feature.
          - Some suggestions are still loose (migraine -> epilepsy, anxiety; psoriasis -> burns).
          - Links come from topic names found in passage text, not real hyperlinks; a page that names a
            topic only in passing still creates a link.
          - Every setting lives in config.py (RELATED_*), so it can be re-tuned without code changes.
        """
        seeds = {t: w for t, w in seed_weights.items() if t in self._ix and w > 0}
        if not seeds or k <= 0:
            return []  # e.g. the cited page is a curated Q&A that is not part of the topic graph

        # Step 1: topic-sensitive PageRank, restarting at the seeds in proportion to their weights.
        restart = np.zeros(len(self._order))
        for t, w in seeds.items():
            restart[self._ix[t]] += w
        lift = personalized_pagerank(self._src, self._dst, len(self._order), restart) / self._global

        # Step 3: filters applied to every candidate.
        def usable(t: str) -> bool:
            i = self._ix[t]
            words = set(t.split())
            return (config.RELATED_MIN_INDEGREE <= self._in_deg[i] <= self._max_in
                    and not _EXCLUDED.search(t)
                    and not any(words <= set(s.split()) for s in seeds))

        # Step 2: candidates. mentioned = topics the seed pages name; mentioning = topics whose pages name a seed.
        mentioned = set().union(*(self.out.get(s, set()) for s in seeds)) - set(seeds)
        mentioning = set().union(*(self.inc.get(s, set()) for s in seeds)) - set(seeds)
        by_lift = lambda ts: sorted((t for t in ts if usable(t)), key=lambda t: (-lift[self._ix[t]], t))
        mutual = by_lift(mentioned & mentioning)[:config.RELATED_ONE_WAY_POOL]
        one_way = by_lift(mentioned - mentioning)[:config.RELATED_ONE_WAY_POOL]
        # Step 4: meaning check. Even mutual links can be loose ("Asthma" <-> "Preeclampsia"), so both kinds
        # are checked against the main seed, with a lower bar for mutual links. One-way links are only
        # considered when there are not already k good mutual ones.
        main = max(seeds, key=lambda t: (seeds[t], t))
        candidates = mutual + one_way
        sims = dict(zip(candidates, similarity(main, candidates))) if (similarity and candidates) else {t: 0.0 for t in candidates}
        mutual = [t for t in mutual if not similarity or sims[t] >= config.RELATED_MIN_SIMILARITY_MUTUAL]
        one_way = [t for t in one_way if sims[t] >= config.RELATED_MIN_SIMILARITY] if len(mutual) < k else []

        picked: List[Dict[str, Any]] = []
        for t, kind in [(t, "mutual") for t in mutual] + [(t, "one_way") for t in one_way]:
            # Step 5: keep the list varied: skip "Aneurysms" after "Aneurysm"
            if any(t.rstrip("s") == p["topic"].rstrip("s") for p in picked):
                continue
            via = sorted(s for s in seeds if t in self.out.get(s, set()))
            name = self.display.get(t, t)
            picked.append({"topic": t, "name": name[:1].upper() + name[1:], "link": kind, "via": [self.display.get(s, s) for s in via][:2],
                           "lift": round(float(lift[self._ix[t]]), 3)})
            if len(picked) == k:
                break
        return picked


# Never suggested as related topics:
#  - administrative pages (insurance, Medicare, clinical trials...) are not health topics a patient would explore;
#  - crisis topics (suicide, self-harm) must never appear as a casual link under an answer. Questions about
#    them are handled by the safety guard (backend/rag/safety.py), which shows helpline information instead.
#  - end-of-life topics (hospice, funeral, grief) could alarm someone reading about an everyday condition.
_EXCLUDED = re.compile(r"suicid|self[ -]?harm|talking with your doctor|health insurance|medicare|medicaid|clinical trials?|"
                       r"^safety$|^health$|caregiv|end of life|hospice|funeral|grief", re.I)
