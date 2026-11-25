"""NVIDIA (OpenAI-compatible) provider, tested against a local fake of the NVIDIA endpoints."""
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient

CALLS = {"chat": [], "emb": []}


class FakeNvidia(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers["content-length"])
        body = json.loads(self.rfile.read(n))
        auth = self.headers.get("authorization")
        if self.path.endswith("/embeddings"):
            CALLS["emb"].append((auth, body))
            from backend.app.index import tokenize

            data = []
            for i, t in enumerate(body["input"]):
                v = np.zeros(64)
                for tok in tokenize(t):
                    v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1
                data.append({"object": "embedding", "index": i, "embedding": v.tolist()})
            self._json({"object": "list", "data": data, "model": body["model"],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        elif self.path.endswith("/chat/completions"):
            CALLS["chat"].append((auth, body))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for piece in ["The SKU ", "is UK703E ", "[3]."]:
                ev = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                      "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self._json({"error": "not found"}, 404)


@pytest.fixture()
def nvidia_client(tmp_path, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), FakeNvidia)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    CALLS["chat"].clear(), CALLS["emb"].clear()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("NVIDIA_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    monkeypatch.setenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "nvidia")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    from backend.app.main import app

    with TestClient(app) as c:
        yield c
    srv.shutdown()


def test_nvidia_llm_and_embeddings_wire_format(nvidia_client):
    h = nvidia_client.get("/health").json()
    assert h["llm"] == "nvidia:meta/llama-3.3-70b-instruct" and h["generative"]
    assert h["embedder"] == "nvidia:nvidia/nv-embedqa-e5-v5"

    # index build: passages, batched <= 32, correct model + auth header
    emb = CALLS["emb"]
    assert emb and all(a == "Bearer nvapi-test" for a, _ in emb)
    assert all(b["input_type"] == "passage" and b["truncate"] == "END" and len(b["input"]) <= 32
               and b["model"] == "nvidia/nv-embedqa-e5-v5" for _, b in emb)

    # chat: query embedding + streamed generation
    with nvidia_client.stream("POST", "/chat/stream",
                              json={"question": "3-year next business day SKU for ProBook 450 G2"}) as s:
        events = [json.loads(line) for line in s.iter_lines() if line]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "The SKU is UK703E [3]."
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"
    assert any("dense" in x["matched_by"] for x in events[0]["sources"])
    assert CALLS["emb"][-1][1]["input_type"] == "query"
    auth, body = CALLS["chat"][-1]
    assert auth == "Bearer nvapi-test" and body["stream"] is True
    assert body["model"] == "meta/llama-3.3-70b-instruct"
    assert body["messages"][0]["role"] == "system" and "ONLY the numbered context" in body["messages"][0]["content"]
    assert body["messages"][-1]["role"] == "user" and "UK703E" in body["messages"][-1]["content"]

    # non-streaming endpoint too
    r = nvidia_client.post("/chat", json={"question": "weight of HP 250 G3"}).json()
    assert r["answer"] == "The SKU is UK703E [3]." and r["llm"].startswith("nvidia:")


def test_missing_key_fails_fast(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    from backend.app.config import get_settings
    from backend.app.llm import get_llm

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY is not set"):
        get_llm(get_settings())
