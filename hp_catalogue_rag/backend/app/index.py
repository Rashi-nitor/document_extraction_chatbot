"""Hybrid retrieval index: BM25 keywords + optional dense vectors + product-name lookup.

Concept: think of a librarian with three ways to find a book:
  1. the card catalogue (BM25: exact words such as SKU 'UK734E' or 'G2'),
  2. a "this feels similar" sense (dense vectors: meaning, synonyms),
  3. remembering exact titles (name lookup: 'ProBook 450 G2' -> its pages).
Results of 1 and 2 are merged with Reciprocal Rank Fusion; 3 is pinned on top.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .documents import Chunk, build_chunks

log = logging.getLogger("rag.index")

STOPWORDS = frozenset("""a an and are as at be by for from has have how i in is it its of on or
that the their this to was what which who will with can do does you your me my tell about
show give list please""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")

# questions that need *all* models of a category at once -> pull in the comparison tables
AGGREGATE_RE = re.compile(
    r"\b(most|least|lightest|heaviest|largest|biggest|smallest|highest|lowest|fastest|best|"
    r"maximum|minimum|max|min|compare|comparison|which|list|all|every|under|over|below|above|"
    r"less than|more than|at least|how many|cheapest|thinnest|brightest)\b", re.I)


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(text.lower().replace("’", "'"))
    return [t for t in toks if t not in STOPWORDS]


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype="float32")
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype="float32")
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    how: str  # 'name' | 'bm25' | 'dense' | 'bm25+dense'


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None = None,
                 embedder=None, meta: dict | None = None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.meta = meta or {}
        # chunk text is prefixed with its title so names are always searchable
        self._bm25 = BM25([tokenize(f"{c.title} {c.product} {c.category} {c.text}") for c in chunks])
        self._by_product: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            if c.product:
                self._by_product.setdefault(c.product, []).append(i)
        # name keys: full name and the name without the leading 'HP '
        self._name_keys: list[tuple[str, str]] = []
        for p in self._by_product:
            full = norm_name(p)
            short = norm_name(re.sub(r"^HP\s+", "", p))
            self._name_keys.append((full, p))
            if short != full and len(short) >= 6:
                self._name_keys.append((short, p))
        self._name_keys.sort(key=lambda kv: -len(kv[0]))  # longest match first

    # ------------------------------------------------------------------ build / persist
    @classmethod
    def build(cls, data_dir: Path, embedder=None) -> "HybridIndex":
        chunks = build_chunks(data_dir)
        emb = None
        if embedder is not None and chunks:
            try:
                emb = embedder.embed_documents([f"{c.title}\n{c.text}" for c in chunks])
            except Exception as e:
                log.warning("dense embedding request failed (%s: %s) - using keyword search only",
                            type(e).__name__, str(e)[:160])
                embedder = None
        meta = {"n_chunks": len(chunks), "embedder": embedder.name if embedder else "none",
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_hash": source_hash(data_dir)}
        return cls(chunks, emb, embedder, meta)

    def save(self, index_dir: Path):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8")
        (index_dir / "meta.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
        emb_path = index_dir / "embeddings.npy"
        if self.embeddings is not None:
            np.save(emb_path, self.embeddings)
        elif emb_path.exists():
            emb_path.unlink()

    @classmethod
    def load(cls, index_dir: Path, embedder=None) -> "HybridIndex | None":
        index_dir = Path(index_dir)
        try:
            chunks = [Chunk(**d) for d in json.loads((index_dir / "chunks.json").read_text("utf-8"))]
            meta = json.loads((index_dir / "meta.json").read_text("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None
        emb = None
        if (index_dir / "embeddings.npy").exists():
            emb = np.load(index_dir / "embeddings.npy")
            if len(emb) != len(chunks):
                return None
        return cls(chunks, emb, embedder, meta)

    # ------------------------------------------------------------------ search
    def categories(self) -> dict[str, int]:
        return dict(Counter(c.category for c in self.chunks if c.category).most_common())

    def doc_types(self) -> dict[str, int]:
        return dict(Counter(c.doc_type for c in self.chunks))

    def _allowed(self, categories, doc_types) -> np.ndarray:
        ok = np.ones(len(self.chunks), dtype=bool)
        if categories:
            cats = {c.lower() for c in categories}
            ok &= np.array([c.category.lower() in cats for c in self.chunks])
        if doc_types:
            dts = {d.lower() for d in doc_types}
            ok &= np.array([c.doc_type.lower() in dts for c in self.chunks])
        return ok

    def mentioned_products(self, query: str) -> list[str]:
        q = norm_name(query)
        found, covered = [], q
        for key, product in self._name_keys:
            if key in covered and product not in found:
                found.append(product)
                covered = covered.replace(key, "#")  # do not let 'probook450g2' also match 'probook450'
        return found

    def search(self, query: str, k: int = 6, categories=None, doc_types=None,
               use_name_lookup: bool = True) -> list[Hit]:
        if not self.chunks:
            return []
        allowed = self._allowed(categories, doc_types)
        n = len(self.chunks)

        bm = self._bm25.scores(tokenize(query))
        bm = np.where(allowed, bm, -1.0)
        bm_rank = [i for i in np.argsort(-bm)[: max(k * 4, 20)] if bm[i] > 0]

        dense_rank: list[int] = []
        if self.embeddings is not None and self.embedder is not None:
            qv = self.embedder.embed_query(query)
            sims = np.where(allowed, self.embeddings @ qv, -2.0)
            dense_rank = [int(i) for i in np.argsort(-sims)[: max(k * 4, 20)] if sims[i] > -1.5]

        fused: dict[int, float] = {}
        how: dict[int, set] = {}
        for name, ranking in (("bm25", bm_rank), ("dense", dense_rank)):
            for r, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + r)
                how.setdefault(int(i), set()).add(name)

        hits: list[Hit] = []
        pinned: list[int] = []
        if use_name_lookup:
            for product in self.mentioned_products(query):
                for i in self._by_product[product]:
                    if allowed[i] and i not in pinned and self.chunks[i].doc_type in ("overview", "specs"):
                        pinned.append(i)
            # with several named products, give each one a fair share of the slots
            pinned = pinned[: max(k, 4)]
        for i in pinned:
            hits.append(Hit(self.chunks[i], 1.0 + fused.get(i, 0.0), "name"))

        # aggregate question -> best 2 comparison tables (ranked by the same fused score)
        agg: list[int] = []
        if (use_name_lookup and AGGREGATE_RE.search(query) and not doc_types
                and not (len(pinned) >= 4)):
            comp = [i for i, c in enumerate(self.chunks) if c.doc_type == "comparison" and allowed[i]]
            comp.sort(key=lambda i: -(float(bm[i]) / (float(bm.max()) + 1e-9)
                                      + 0.5 * (fused.get(i, 0.0) * 60)))
            agg = [i for i in comp[:2] if bm[i] > 0 or fused.get(i)]
            for i in agg:
                hits.append(Hit(self.chunks[i], 0.9, "comparison"))
        pinned = pinned + agg

        for i, s in sorted(fused.items(), key=lambda kv: -kv[1]):
            if i in pinned:
                continue
            hits.append(Hit(self.chunks[i], s, "+".join(sorted(how[i]))))
        return hits[:k] if not pinned else hits[: max(k, len(pinned))]


def source_hash(data_dir: Path) -> str:
    h = hashlib.sha1()
    for name in ("products.csv", "spec_tables_long.csv", "care_packs_long.csv", "toc.csv",
                 "care_pack_recommended.csv", "features_and_captions.csv", "pages.jsonl",
                 "products_master.csv"):
        p = Path(data_dir) / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]
