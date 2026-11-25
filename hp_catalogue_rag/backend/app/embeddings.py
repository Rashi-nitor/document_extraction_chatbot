"""Optional dense (semantic) embeddings.

Retrieval works without any of this (keyword BM25 only). Dense vectors add
"meaning" matching, e.g. 'light laptop for travelling' -> 'weight 1.3 kg'.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("rag.embeddings")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    name = "none"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


def _normalise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype="float32")
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.clip(n, 1e-9, None)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts):
        return _normalise(self.model.encode(texts, batch_size=32, show_progress_bar=False))

    def embed_query(self, text):
        return _normalise(self.model.encode([self.query_prefix + text], show_progress_bar=False))[0]


class NvidiaEmbedder(Embedder):
    """NVIDIA embeddings use an OpenAI-compatible endpoint and passage/query input types."""

    def __init__(self, model: str, api_key: str, base_url: str = "", batch: int = 32):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=120)
        self.model, self.batch = model, batch
        self.name = f"nvidia:{model}"

    def _embed(self, texts, kind: str):
        extra = {"extra_body": {"input_type": kind, "truncate": "END"}}
        out = []
        for i in range(0, len(texts), self.batch):
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + self.batch],
                                                 **extra)
            out += [d.embedding for d in resp.data]
        return _normalise(np.array(out))

    def embed_documents(self, texts):
        return self._embed(texts, "passage")

    def embed_query(self, text):
        return self._embed([text], "query")[0]


def get_embedder(settings) -> Embedder | None:
    """Return an embedder or None (= keyword-only retrieval). Never raises."""
    provider = settings.embedding_provider
    if provider in ("bm25", "none", "off"):
        return None
    try:
        if provider == "nvidia":
            if not settings.nvidia_api_key:
                raise RuntimeError("NVIDIA_API_KEY is not set")
            return NvidiaEmbedder(settings.nvidia_embedding_model, settings.nvidia_api_key,
                                  settings.nvidia_base_url)
        if provider in ("sentence-transformers", "auto"):
            return SentenceTransformerEmbedder(settings.embedding_model)
        log.warning("unknown EMBEDDING_PROVIDER=%s, using keyword search only", provider)
    except Exception as e:  # missing package, no internet for the model download, ...
        level = logging.INFO if provider == "auto" else logging.WARNING
        log.log(level, "dense embeddings unavailable (%s: %s) - using keyword search only",
                type(e).__name__, str(e)[:160])
    return None
