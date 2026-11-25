"""Run: pytest -q   (from the project root)"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.documents import build_chunks, family_products
from backend.app.index import HybridIndex
from backend.app.llm import LLM

DATA = get_settings().data_dir


@pytest.fixture(scope="module")
def chunks():
    return build_chunks(DATA)


@pytest.fixture(scope="module")
def index():
    return HybridIndex.build(DATA, None)


# --------------------------------------------------------------------- extraction output
def test_every_sku_is_in_a_chunk(chunks):
    care = pd.read_csv(DATA / "care_packs_long.csv")
    in_chunks = {s for c in chunks if c.doc_type == "care_pack" for s in c.extra["skus"]}
    assert set(care["sku"]) == in_chunks


def test_qa_report_has_no_failures():
    report = (DATA / "qa_report.md").read_text(encoding="utf-8")
    assert "**NO**" not in report


def test_chunk_sizes_and_ids(chunks):
    assert len({c.id for c in chunks}) == len(chunks)
    assert max(len(c.text) for c in chunks) <= 2000
    assert {c.doc_type for c in chunks} >= {"overview", "specs", "care_pack", "service"}


def test_categories_come_from_toc(chunks):
    # page 24 is printed with the header 'HP tablets' but belongs to desktops
    assert {c.category for c in chunks if c.page == 24} == {"HP desktops & all-in-ones"}


def test_family_mapping():
    names = ["HP ProBook 430 G2", "HP ProBook 450 G2", "HP ProBook 640 G1", "HP ElitePad 1000 G2"]
    assert family_products("ProBook 4xx (s+t series)", names) == ["HP ProBook 430 G2", "HP ProBook 450 G2"]
    assert family_products("ElitePad", names) == ["HP ElitePad 1000 G2"]
    assert family_products('Standard up to 22"', names) == []


# --------------------------------------------------------------------- retrieval
def test_exact_sku_lookup(index):
    care = pd.read_csv(DATA / "care_packs_long.csv")
    for sku in care["sku"].unique():
        hits = index.search(sku, k=6)
        assert any(sku in h.chunk.extra.get("skus", []) for h in hits), sku


def test_every_spec_product_is_found_by_name(index):
    spec = pd.read_csv(DATA / "spec_tables_long.csv")
    for product in spec["product"].unique():
        hits = index.search(f"{product} specifications", k=4, doc_types=["specs"])
        assert hits and hits[0].chunk.product == product, product


def test_care_pack_question_finds_right_table(index):
    hits = index.search("3-year next business day care pack SKU for the ProBook 450 G2", k=6)
    care = [h for h in hits if h.chunk.doc_type == "care_pack"]
    assert care, "no care pack chunk retrieved"
    top = care[0].chunk
    assert top.extra["family"].startswith("ProBook 4xx")
    assert "3-year: SKU UK703E" in top.text


def test_two_products_are_both_pinned(index):
    hits = index.search("compare ProBook 450 G2 and ProBook 640 G1", k=6)
    products = {h.chunk.product for h in hits if h.how == "name"}
    assert products == {"HP ProBook 450 G2", "HP ProBook 640 G1"}


def test_aggregate_questions_get_comparison_tables(index):
    hits = index.search("Which workstation supports the most memory?", k=6)
    top = [h for h in hits if h.how == "comparison"][0].chunk
    assert top.category == "HP workstations" and "Upgradeable to 512 GB" in top.text
    hits = index.search("Which HP notebooks weigh less than 1.5 kg?", k=6)
    text = " ".join(h.chunk.text for h in hits if h.chunk.doc_type == "comparison")
    assert "HP EliteBook 820 G1" in text and "1.33" in text


def test_category_filter(index):
    hits = index.search("battery", k=8, categories=["HP workstations"])
    assert hits and all(h.chunk.category == "HP workstations" for h in hits)


# --------------------------------------------------------------------- API (with a fake LLM)
class FakeLLM(LLM):
    name = "fake"

    def __init__(self, fail=False):
        self.fail, self.seen = fail, None

    def stream(self, system, messages):
        self.seen = (system, messages)
        yield "The SKU is UK703E "
        if self.fail:
            raise RuntimeError("quota exceeded")
        yield "[3]."


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bm25")
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


def test_health_and_categories(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["index"]["n_chunks"] > 300
    assert "HP notebooks" in client.get("/catalogue/categories").json()["categories"]


def test_chat_validation(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422


def test_chat_with_llm_builds_grounded_prompt(client):
    from backend.app.main import app

    fake = FakeLLM()
    app.state.rag.llm = fake
    history = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "about ProBook 450 G2"},
               {"role": "assistant", "content": "ok"}]
    r = client.post("/chat", json={"question": "and the 3-year care pack?", "history": history}).json()
    assert r["answer"].startswith("The SKU is UK703E") and r["llm"] == "fake"
    system, messages = fake.seen
    assert "ONLY the numbered context" in system
    assert messages[0]["role"] == "user"  # leading assistant turn dropped
    roles = [m["role"] for m in messages]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles  # strictly alternating
    assert messages[-1]["role"] == "user" and "Context passages" in messages[-1]["content"]
    # the short follow-up borrowed "ProBook 450 G2" from the previous user turn
    assert "HP ProBook 450 G2" in messages[-1]["content"]
    assert "ProBook 4xx" in messages[-1]["content"]


def test_stream_events_and_error_event(client):
    from backend.app.main import app

    def events(fail):
        app.state.rag.llm = FakeLLM(fail=fail)
        with client.stream("POST", "/chat/stream", json={"question": "ProBook 450 G2 care pack"}) as s:
            return [json.loads(line) for line in s.iter_lines() if line]

    ok = events(False)
    assert [e["type"] for e in ok][0] == "sources" and ok[-1]["type"] == "done"
    assert "".join(e["text"] for e in ok if e["type"] == "token") == "The SKU is UK703E [3]."
    bad = events(True)
    assert [e["type"] for e in bad][-2:] == ["error", "done"]


def test_rebuild(client):
    assert client.post("/index/rebuild").json()["n_chunks"] > 300


# --------------------------------------------------------------------- dense / hybrid path
class HashEmbedder:
    """Tiny deterministic bag-of-words embedder: exercises the dense code path offline."""
    name = "fake-hash"
    DIM = 256

    def _vec(self, text):
        import numpy as np
        from backend.app.index import tokenize

        v = np.zeros(self.DIM, dtype="float32")
        for t in tokenize(text):
            v[hash(t) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        import numpy as np
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def test_hybrid_index_roundtrip(tmp_path):
    emb = HashEmbedder()
    idx = HybridIndex.build(DATA, emb)
    assert idx.embeddings is not None and idx.embeddings.shape[0] == len(idx.chunks)
    idx.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, emb)
    assert loaded and len(loaded.chunks) == len(idx.chunks)
    hits = loaded.search("which notebook has a spill-resistant keyboard and touch screen", k=6)
    assert any("dense" in h.how for h in hits)
    # exact identifiers still work in hybrid mode
    assert any("UK703E" in h.chunk.extra.get("skus", []) for h in loaded.search("UK703E", k=6))
    # a keyword-only index must not be mistaken for a dense one
    plain = HybridIndex.build(DATA, None)
    plain.save(tmp_path / "plain")
    assert HybridIndex.load(tmp_path / "plain", None).embeddings is None
