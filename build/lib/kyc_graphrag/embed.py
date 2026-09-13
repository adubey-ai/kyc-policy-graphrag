"""Dense embeddings when sentence-transformers is present; hashed TF-IDF otherwise."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from kyc_graphrag.retrieve_text import tokenize


class Embedder:
    def __init__(self, prefer_dense: bool = True) -> None:
        self.model = None
        self.kind = "tfidf"
        if prefer_dense:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.kind = "minilm"
            except Exception:
                self.model = None
                self.kind = "tfidf"
        self._vocab: dict[str, int] | None = None
        self._idf: np.ndarray | None = None

    def fit(self, corpus: list[str]) -> None:
        if self.model is not None:
            return
        vocab: dict[str, int] = {}
        df: dict[str, int] = {}
        docs = [tokenize(t) for t in corpus]
        for doc in docs:
            for tok in set(doc):
                df[tok] = df.get(tok, 0) + 1
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        n = max(len(docs), 1)
        idf = np.zeros(len(vocab), dtype=np.float32)
        for tok, idx in vocab.items():
            idf[idx] = np.log((n + 1) / (df[tok] + 1)) + 1.0
        self._vocab = vocab
        self._idf = idf

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        if self._vocab is None or self._idf is None:
            self.fit(texts)
        assert self._vocab is not None and self._idf is not None
        out = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            if not toks:
                continue
            for tok in toks:
                j = self._vocab.get(tok)
                if j is None:
                    continue
                out[i, j] += self._idf[j]
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    if matrix.size == 0:
        return []
    scores = matrix @ query
    k = min(k, len(scores))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


@lru_cache(maxsize=1)
def default_embedder() -> Embedder:
    return Embedder(prefer_dense=True)
