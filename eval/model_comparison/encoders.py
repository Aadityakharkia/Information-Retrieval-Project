"""Query/passage encoders for the model comparison (MiniLM baseline, BGE-M3 dense+sparse, MedCPT)."""
import numpy as np
import scipy.sparse as sp
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _batches(texts, bs):
    """Yield index batches of similar length (longest first) so padding is small."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i] if isinstance(texts[i], str) else texts[i][1]))
    for i in range(0, len(order), bs):
        yield order[i:i + bs]


class MiniLM:
    name = "MiniLM (current)"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import config
        self.m = SentenceTransformer(config.DENSE_MODEL_NAME, device=DEVICE)

    def encode_queries(self, qs):
        return self.m.encode(qs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32), None


class BasePaperMiniLM:
    """all-MiniLM-L6-v2, the embedding model of the base paper (Fedallah & Inik, ISADES 2026)."""
    name = "MiniLM-L6 (base paper)"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

    def encode_queries(self, qs):
        return self.m.encode(qs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32), None

    def encode_passages(self, chunks, log=None):
        x = [f"{c['question']} {c['text']}" for c in chunks]
        return self.m.encode(x, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32), None


class BGEM3:
    """Dense = normalised [CLS]; sparse = ReLU(linear(token)) max-pooled per token id (Chen et al., 2024, sec. 3)."""
    name = "BGE-M3"

    def __init__(self):
        repo = "BAAI/bge-m3"
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.m = AutoModel.from_pretrained(repo, dtype=torch.float16).to(DEVICE).eval()
        self.lin = torch.nn.Linear(1024, 1)
        self.lin.load_state_dict(torch.load(hf_hub_download(repo, "sparse_linear.pt"), map_location="cpu", weights_only=True))
        self.lin = self.lin.to(DEVICE, torch.float16).eval()
        self.special = torch.tensor(self.tok.all_special_ids, device=DEVICE)
        self.V = len(self.tok)

    @torch.no_grad()
    def encode(self, texts, bs=32, max_len=1024, log=None):
        dense = np.zeros((len(texts), 1024), dtype=np.float32)
        R, C, W, done = [], [], [], 0
        for n, idx in enumerate(_batches(texts, bs)):
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
            h = self.m(**enc).last_hidden_state
            dense[idx] = torch.nn.functional.normalize(h[:, 0].float(), dim=-1).cpu().numpy()
            w = torch.relu(self.lin(h)).squeeze(-1).float()
            w = (w * enc["attention_mask"] * ~torch.isin(enc["input_ids"], self.special)).cpu().numpy()
            ids = enc["input_ids"].cpu().numpy()
            r, c = np.nonzero(w > 0)
            R.append(np.asarray(idx)[r]); C.append(ids[r, c]); W.append(w[r, c])
            done += len(idx)
            if log and n % 100 == 0:
                log(f"{done}/{len(texts)}")
        R, C, W = map(np.concatenate, (R, C, W))
        o = np.lexsort((-W, C, R)); R, C, W = R[o], C[o], W[o]          # keep the max weight per (doc, token)
        keep = np.ones(len(R), bool); keep[1:] = (R[1:] != R[:-1]) | (C[1:] != C[:-1])
        return dense, sp.csr_matrix((W[keep], (R[keep], C[keep])), shape=(len(texts), self.V), dtype=np.float32)

    def encode_queries(self, qs):
        return self.encode(qs, bs=16)

    def encode_passages(self, chunks, log=None):
        return self.encode([f"{c['question']} {c['text']}" for c in chunks], log=log)


class MedCPT:
    """Separate query/article encoders, [CLS] pooling, inner product (Jin et al., 2023). Article input = [title, abstract]."""
    name = "MedCPT"

    def __init__(self):
        self.qt = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.q = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(DEVICE).eval()
        self.at = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self.a = None

    @torch.no_grad()
    def encode_queries(self, qs):
        enc = self.qt(qs, truncation=True, padding=True, max_length=64, return_tensors="pt").to(DEVICE)
        return self.q(**enc).last_hidden_state[:, 0].float().cpu().numpy(), None

    @torch.no_grad()
    def encode_passages(self, chunks, bs=64, log=None):
        self.a = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder", dtype=torch.float16).to(DEVICE).eval()
        pairs = [(c["question"], c["text"]) for c in chunks]
        out, done = np.zeros((len(pairs), 768), dtype=np.float32), 0
        for n, idx in enumerate(_batches(pairs, bs)):
            enc = self.at([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEVICE)
            out[idx] = self.a(**enc).last_hidden_state[:, 0].float().cpu().numpy()
            done += len(idx)
            if log and n % 100 == 0:
                log(f"{done}/{len(pairs)}")
        return out, None
