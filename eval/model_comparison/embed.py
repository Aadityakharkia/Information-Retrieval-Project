"""Embeds all indexed passages with the base paper's MiniLM, MedCPT and BGE-M3 -> data/processed/model_comparison/ (git-ignored).
Run: python -m eval.model_comparison.embed [base-minilm] [medcpt] [bge-m3]"""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp

import config
from backend.ir.inverted_index import InvertedIndex
from eval.model_comparison.encoders import BGEM3, BasePaperMiniLM, MedCPT

OUT = config.PROCESSED_DATA_DIR / "model_comparison"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = list(InvertedIndex.load().chunks_map.values())
    for key, cls in (("base-minilm", BasePaperMiniLM), ("medcpt", MedCPT), ("bge-m3", BGEM3)):
        if sys.argv[1:] and key not in sys.argv[1:]:
            continue
        t0 = time.time()
        enc = cls()
        dense, sparse = enc.encode_passages(chunks, log=lambda m: print(f"[{key}] {m}  {time.time() - t0:.0f}s", flush=True))
        assert np.isfinite(dense).all(), "non-finite embeddings"
        np.save(OUT / f"{key}.npy", dense)
        if sparse is not None:
            sp.save_npz(OUT / f"{key}-sparse.npz", sparse)
        (OUT / f"{key}.json").write_text(json.dumps({"passages": len(chunks), "dim": dense.shape[1], "seconds": round(time.time() - t0)}))
        print(f"[{key}] done {dense.shape} in {time.time() - t0:.0f}s", flush=True)
        del enc
