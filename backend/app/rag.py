"""RAG service: retrieve -> build a grounded prompt -> generate -> cite."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from .config import Settings
from .embeddings import get_embedder
from .index import Hit, HybridIndex
from .llm import LLM, get_llm

log = logging.getLogger("rag.service")

SYSTEM_PROMPT = """You are the HP Catalogue Assistant. You answer questions about the HP PC \
Catalogue 2014 (notebooks, tablets, desktops and all-in-ones, workstations, displays, thin \
clients, retail solutions, accessories and HP Care Pack services).

Rules:
1. Use ONLY the numbered context passages below. If they do not contain the answer, say you could \
not find it in the catalogue and suggest what to ask instead. Never invent specs, SKUs or prices.
2. Cite the passages you used like [1] or [2][3] right after the claim.
3. Copy part numbers / SKUs (e.g. UK734E) and model names exactly as written.
4. A Care Pack table lists products by family (e.g. "ProBook 4xx"); the "Applies to" line says \
which models belong to that family. Use the term (3-year, 4-year ...) and service level the user asked for.
5. For comparisons use a short markdown table. Keep answers concise and factual.
6. The catalogue is from 2014: it has no prices, and products may have changed since."""


def format_context(hits: list[Hit], max_chars: int) -> tuple[str, list[Hit]]:
    parts, used, total = [], [], 0
    for n, h in enumerate(hits, 1):
        c = h.chunk
        label = f"[{n}] ({c.doc_type} | {c.category} | catalogue page {c.page})"
        block = f"{label}\n{c.text}"
        if used and total + len(block) > max_chars:
            break
        parts.append(block)
        used.append(h)
        total += len(block)
    return "\n\n".join(parts), used


def source_dict(n: int, h: Hit, snippet_chars: int = 420) -> dict:
    c = h.chunk
    text = re.sub(r"\s+", " ", c.text)
    return {"n": n, "id": c.id, "title": c.title, "doc_type": c.doc_type, "category": c.category,
            "page": c.page, "product": c.product, "score": round(float(h.score), 4),
            "matched_by": h.how, "snippet": text[:snippet_chars] + ("…" if len(text) > snippet_chars else ""),
            "text": c.text}


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm: LLM = get_llm(settings)
        self.embedder = get_embedder(settings)
        self.index: HybridIndex | None = None
        self.load_or_build()

    # ----------------------------------------------------------------- index lifecycle
    def load_or_build(self, force: bool = False) -> dict:
        s = self.settings
        from .index import source_hash

        current = source_hash(s.data_dir)
        if not force:
            idx = HybridIndex.load(s.index_dir, self.embedder)
            want = self.embedder.name if self.embedder else "none"
            if idx and idx.meta.get("source_hash") == current and idx.meta.get("embedder") == want:
                self.index = idx
                log.info("loaded index: %s", idx.meta)
                return idx.meta
        if not Path(s.data_dir).exists() or not (Path(s.data_dir) / "products.csv").exists():
            log.error("no extracted data in %s - run scripts/run_pipeline.py first", s.data_dir)
            self.index = HybridIndex([], None, None, {"n_chunks": 0, "embedder": "none"})
            return self.index.meta
        log.info("building index from %s ...", s.data_dir)
        self.index = HybridIndex.build(s.data_dir, self.embedder)
        self.index.save(s.index_dir)
        log.info("index built: %s", self.index.meta)
        return self.index.meta

    # ----------------------------------------------------------------- retrieval
    def _search_query(self, question: str, history: list[dict]) -> str:
        """Short follow-ups ('and the 5-year one?') borrow words from the previous user turn."""
        if history and len(question.split()) <= 6:
            prev = [m["content"] for m in history if m["role"] == "user"]
            if prev:
                return f"{prev[-1]} {question}"
        return question

    def retrieve(self, question: str, history: list[dict] | None = None, top_k: int | None = None,
                 categories=None, doc_types=None) -> list[Hit]:
        assert self.index is not None
        q = self._search_query(question, history or [])
        return self.index.search(q, k=top_k or self.settings.top_k, categories=categories,
                                 doc_types=doc_types)

    # ----------------------------------------------------------------- generation
    def _messages(self, question: str, history: list[dict], context: str) -> list[dict]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-6:]
                if m["role"] in ("user", "assistant") and m["content"]]
        # APIs require the first turn to be a user turn
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        # merge accidental consecutive same-role turns
        merged: list[dict] = []
        for m in msgs:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += "\n" + m["content"]
            else:
                merged.append(dict(m))
        if merged and merged[-1]["role"] == "user":
            merged.pop()
        merged.append({"role": "user",
                       "content": f"Context passages:\n\n{context}\n\nQuestion: {question}"})
        return merged

    def answer(self, question: str, history=None, top_k=None, categories=None, doc_types=None) -> dict:
        hits = self.retrieve(question, history, top_k, categories, doc_types)
        context, used = format_context(hits, self.settings.max_context_chars)
        sources = [source_dict(n, h) for n, h in enumerate(used, 1)]
        if not used:
            return {"answer": "I could not find anything about that in the catalogue.",
                    "sources": [], "llm": self.llm.name}
        msgs = self._messages(question, history or [], context)
        text = self.llm.complete(SYSTEM_PROMPT + "\n", msgs)
        return {"answer": text, "sources": sources, "llm": self.llm.name}

    def stream_answer(self, question: str, history=None, top_k=None, categories=None,
                      doc_types=None) -> Iterator[dict]:
        """Yields {'type': 'sources'|'token'|'error'|'done', ...} events."""
        hits = self.retrieve(question, history, top_k, categories, doc_types)
        context, used = format_context(hits, self.settings.max_context_chars)
        yield {"type": "sources", "sources": [source_dict(n, h) for n, h in enumerate(used, 1)],
               "llm": self.llm.name}
        if not used:
            yield {"type": "token", "text": "I could not find anything about that in the catalogue."}
            yield {"type": "done"}
            return
        try:
            for piece in self.llm.stream(SYSTEM_PROMPT + "\n", self._messages(question, history or [], context)):
                yield {"type": "token", "text": piece}
        except Exception as e:  # network / auth / quota errors must reach the UI
            log.exception("LLM streaming failed")
            yield {"type": "error", "message": f"{type(e).__name__}: {str(e)[:300]}"}
        yield {"type": "done"}
