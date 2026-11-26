"""Builds the report figures from the saved evaluation results. Run: .venv/bin/python report/make_figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval" / "model_comparison"
OUT = Path(__file__).parent / "figures"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType fonts embedded, as IEEE PDF eXpress expects


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


# ---- Fig. 1: system pipeline -------------------------------------------------------------
def pipeline():
    fig, ax = plt.subplots(figsize=(7.16, 3.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, title, sub="", dark=False, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", lw=0.9,
                                    ec=INK, fc=INK if dark else "#f4f4f2", ls="--" if dashed else "-"))
        ax.text(x + w / 2, y + h / 2 + (1.4 if sub else 0), title, ha="center", va="center", fontsize=6.9, weight="bold",
                color="white" if dark else INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 1.9, sub, ha="center", va="center", fontsize=6.0, color="#dddddd" if dark else MUTED, linespacing=1.1)

    def arrow(x1, y1, x2, y2, text="", dashed=False):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, ls="--" if dashed else "-", mutation_scale=8))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.1, text, ha="center", fontsize=5.9, color=MUTED)

    # top row: question path
    box(0.5, 30, 10.5, 9, "Question", "English, Hinglish\nor Hindi")
    box(13.5, 30, 11, 9, "Safety guard", "emergency and\nself-harm rules")
    box(27, 30, 14, 9, "Query processing", "NFKC, stemming,\nHinglish lexicon")
    box(43.5, 30, 20, 9, "Hybrid retrieval", "TF-IDF lnc.ltc + MiniLM\nfused with RRF (k = 60)")
    box(66, 30, 14, 9, "Link re-ranking", "PageRank + HITS\non topic graph")
    box(82.5, 30, 17, 9, "Abstain?", "score, dense-similarity\nand coverage gates")
    for a, b in ((11.3, 13.3), (24.8, 26.8), (41.3, 43.3), (63.8, 65.8), (80.3, 82.3)):
        arrow(a, 34.5, b, 34.5)
    # middle row: answer path (right to left)
    box(65, 14, 21, 9, "Grounded generation", "LLM or offline generator;\nevery sentence cites [n]", dark=True)
    box(41.5, 14, 21, 9, "Fact checker", "cosine, key-term coverage,\nexact number check")
    box(19.5, 14, 19.5, 9, "Answer page", "citations, badges, related\ntopics, IR inspector")
    box(0.5, 14, 16.5, 9, "Rocchio feedback", "More like this /\nNot relevant", dashed=True)
    arrow(85, 29.7, 85, 23.4); ax.text(85.6, 26.2, "no", fontsize=6.2, color=MUTED)
    arrow(64.8, 18.5, 62.7, 18.5); arrow(41.3, 18.5, 39.2, 18.5); arrow(19.3, 18.5, 17.2, 18.5)
    ax.annotate("", (50, 29.7), (8.75, 23.4), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, ls="--", mutation_scale=8,
                                                             connectionstyle="arc3,rad=-0.12"))
    ax.text(25, 25.9, "re-search with Rocchio query q\u2032", fontsize=6.0, color=MUTED)
    # bottom row: data and side channels
    box(0.5, 0.5, 31, 8.5, "Private patient memory", "prescription and report PDFs: lines quoted\nverbatim; never sent to an outside service", dashed=True)
    box(34.5, 0.5, 30, 8.5, "Corpus (searched above)", "MedQuAD + curated Q&A: 14,372 Q&As,\n70,154 sentence-window passages")
    box(67.5, 0.5, 32, 8.5, "Wikipedia fallback", "\u201cI don\u2019t know\u201d, then search only after the\nuser approves the exact words", dashed=True)
    arrow(97.5, 29.7, 97.5, 9.6); ax.text(93.2, 26.2, "yes", fontsize=6.2, color=MUTED)
    save(fig, "fig_pipeline")


# ---- Fig. 2: nDCG@10 per language ----------------------------------------------------------
def by_language():
    r = json.loads((EV / "report_metrics.json").read_text())["retrieval"]
    systems = ["BM25", "TF-IDF (lnc.ltc)", "MedCPT", "MiniLM-L6 (base paper [1])", "Multilingual MiniLM", "Hybrid RRF + PageRank/HITS", "BGE-M3 dense+sparse"]
    labels = ["BM25", "TF-IDF (lnc.ltc)", "MedCPT", "MiniLM-L6 (base paper)", "Multilingual MiniLM", "HealthNest hybrid\n+ PageRank/HITS", "BGE-M3 dense+sparse"]
    langs = [("en", "English", BLUE, None), ("hinglish", "Hinglish", ORANGE, "////"), ("hindi", "Hindi", AQUA, "....")]
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    h = 0.26
    for j, (key, name, color, hatch) in enumerate(langs):
        ys = [i + (j - 1) * (h + 0.02) for i in range(len(systems))]  # English drawn on top of each group
        vals = [r[s][key]["nDCG@10"] for s in systems]
        ax.barh(ys, vals, height=h, color=color, edgecolor=SURF, lw=0.6, hatch=hatch, label=name)
    ax.set_yticks(range(len(systems)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.6); ax.set_xlabel("nDCG@10 (lexicon-expanded queries)")
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, fontsize=7, handlelength=1.6)
    for i, s in enumerate(systems):
        ax.text(max(r[s][k]["nDCG@10"] for k, *_ in langs) + 0.01, i, f"mean {r[s]['mean']['nDCG@10']:.3f}",
                va="center", fontsize=6.2, color=MUTED)
    save(fig, "fig_ndcg_language")


# ---- Fig. 3: effect of each IR component -------------------------------------------------
def components():
    link = json.loads((EV / "link_results.json").read_text())
    roc = json.loads((EV / "rocchio_results.json").read_text())
    rows = [("PageRank + HITS\nre-ranking", link["baseline_test"], link["modes"]["both"]["test"], f"p = {link['modes']['both']['p']:.2f}"),
            ("Pseudo-relevance\nfeedback (Rocchio)", roc["prf"]["test_mean"][0], roc["prf"]["test_mean"][1], f"p = {roc['prf']['p']:.2f}"),
            ("Explicit feedback\n(Rocchio, simulated)", roc["explicit"]["all"]["before"], roc["explicit"]["all"]["after"], "p < 0.001")]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    h = 0.36
    for i, (name, before, after, p) in enumerate(rows):
        ax.barh(i - h / 2 - 0.01, before, height=h, color=GRAY, edgecolor=SURF, label="without" if i == 0 else None)
        ax.barh(i + h / 2 + 0.01, after, height=h, color=BLUE, edgecolor=SURF, label="with component" if i == 0 else None)
        ax.text(max(before, after) + 0.008, i, f"{before:.3f} → {after:.3f}\n{p}", va="center", fontsize=6.3, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 0.5); ax.set_xlabel("nDCG@10 (held-out needs; feedback: residual collection)")
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False, fontsize=6.8)
    save(fig, "fig_components")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline(); by_language(); components()
    print("written:", sorted(p.name for p in OUT.iterdir()))


# ---- LaTeX tables generated from the same result files (no hand-copied numbers) -----------
def tables():
    T = Path(__file__).parent / "tables"
    T.mkdir(exist_ok=True)
    m = json.loads((EV / "report_metrics.json").read_text())
    r, ab = m["retrieval"], m["abstention"]
    order = ["BM25", "TF-IDF (lnc.ltc)", "MedCPT", "MiniLM-L6 (base paper [1])", "Multilingual MiniLM",
             "Hybrid RRF (TF-IDF + MiniLM)", "Hybrid RRF + PageRank/HITS", "BGE-M3 dense+sparse"]
    names = {"BM25": "BM25 \\cite{robertson2009}", "TF-IDF (lnc.ltc)": "TF-IDF lnc.ltc", "MedCPT": "MedCPT \\cite{jin2023}",
             "MiniLM-L6 (base paper [1])": "MiniLM-L6 (as in \\cite{fedallah2026})", "Multilingual MiniLM": "Multilingual MiniLM \\cite{reimers2020}",
             "Hybrid RRF (TF-IDF + MiniLM)": "Hybrid RRF \\cite{cormack2009}", "Hybrid RRF + PageRank/HITS": "\\textbf{HealthNest} (RRF + PR/HITS)",
             "BGE-M3 dense+sparse": "BGE-M3 \\cite{chen2024}"}
    cols = ["P@5", "P@10", "R@10", "F1@10", "MRR@10", "nDCG@10"]
    best = {c: max(r[s]["mean"][c] for s in order) for c in cols}
    lines = []
    for s in order:
        cells = [(f"\\textbf{{{r[s]['mean'][c]:.3f}}}" if abs(r[s]["mean"][c] - best[c]) < 1e-9 else f"{r[s]['mean'][c]:.3f}") for c in cols]
        names_s = names[s].replace("[", "\\cite{").replace("]", "}") if False else names[s]
        lines.append(f"{names_s} & " + " & ".join(cells) + " \\\\")
    (T / "retrieval_mean.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for s in ["MiniLM-L6 (base paper [1])", "Multilingual MiniLM", "Hybrid RRF + PageRank/HITS", "BGE-M3 dense+sparse"]:
        row = []
        for li in ("en", "hinglish", "hindi"):
            row += [f"{r[s][li]['P@5']:.3f}", f"{r[s][li]['R@10']:.3f}", f"{r[s][li]['F1@10']:.3f}"]
        lines.append(f"{names[s]} & " + " & ".join(row) + " \\\\")
    (T / "retrieval_language.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for key, label in (("en", "English + off-topic"), ("hinglish", "Hinglish + off-topic"), ("hindi", "Hindi + off-topic"), ("all", "All 140 questions")):
        a = ab[key]
        lines.append(f"{label} & {a['TP']} & {a['FP']} & {a['FN']} & {a['TN']} & {a['accuracy']:.3f} & {a['precision']:.3f} & {a['recall']:.3f} & {a['F1']:.3f} \\\\")
    (T / "abstention.tex").write_text("\n".join(lines) + "\n")
    # names used in the retrieval tables refer to reference numbers in the paper
    print("tables:", sorted(p.name for p in T.iterdir()))


if __name__ == "__main__":
    tables()
