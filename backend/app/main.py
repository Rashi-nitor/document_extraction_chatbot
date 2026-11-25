"""FastAPI backend for the HP catalogue RAG chatbot.

Run from the project root:
    uvicorn backend.app.main:app --reload --port 8000
Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .rag import RAGService, source_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag.api")


# ------------------------------------------------------------------ schemas
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, description="previous turns, oldest first")
    top_k: int | None = Field(None, ge=1, le=20)
    categories: list[str] | None = Field(None, description="restrict to catalogue categories")
    doc_types: list[str] | None = Field(None, description="overview|specs|care_pack|service|feature|notes")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(8, ge=1, le=30)
    categories: list[str] | None = None
    doc_types: list[str] | None = None


# ------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAGService(get_settings())
    yield


app = FastAPI(title="HP Catalogue RAG API", version="1.0.0", lifespan=lifespan)
_origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins or ["*"], allow_methods=["*"],
                   allow_headers=["*"])


def rag() -> RAGService:
    return app.state.rag


def _history(req: ChatRequest) -> list[dict]:
    return [t.model_dump() for t in req.history]


@app.get("/health")
def health():
    r = rag()
    return {"status": "ok" if r.index and r.index.chunks else "no_data",
            "llm": r.llm.name, "generative": r.llm.generative,
            "embedder": r.index.meta.get("embedder") if r.index else "none",
            "index": r.index.meta if r.index else {}}


@app.get("/catalogue/categories")
def categories():
    idx = rag().index
    return {"categories": idx.categories() if idx else {}, "doc_types": idx.doc_types() if idx else {}}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return rag().answer(req.question, _history(req), req.top_k, req.categories, req.doc_types)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("chat failed")
        raise HTTPException(502, f"Generation failed: {type(e).__name__}: {str(e)[:300]}")


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Newline-delimited JSON events: sources -> token* -> done (or error)."""
    def gen():
        for ev in rag().stream_answer(req.question, _history(req), req.top_k, req.categories,
                                      req.doc_types):
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/search")
def search(req: SearchRequest):
    """Retrieval only (no LLM) - useful to debug what the bot can see."""
    hits = rag().index.search(req.query, k=req.top_k, categories=req.categories,
                              doc_types=req.doc_types)
    return {"results": [source_dict(n, h, 600) for n, h in enumerate(hits, 1)]}


@app.post("/index/rebuild")
def rebuild():
    """Re-read output/*.csv and rebuild the index (after re-running the extraction)."""
    return rag().load_or_build(force=True)
