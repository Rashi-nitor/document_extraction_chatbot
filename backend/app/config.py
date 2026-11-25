"""Central configuration. Everything can be overridden with environment
variables or a `.env` file in the project root (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default


@dataclass
class Settings:
    # ---- data ---------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "output"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", str(ROOT / "data" / "index"))))

    # ---- retrieval ------------------------------------------------------------
    # auto = sentence-transformers if installed (and loadable), else keyword-only (BM25)
    # bm25 | sentence-transformers | openai | nvidia | auto
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "auto").lower())
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    openai_embedding_model: str = field(default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))
    max_context_chars: int = field(default_factory=lambda: int(_env("MAX_CONTEXT_CHARS", "14000")))

    # ---- generation -------------------------------------------------------------
    nvidia_api_key: str = field(default_factory=lambda: _env("NVIDIA_API_KEY"))
    nvidia_model: str = field(default_factory=lambda: _env("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"))
    nvidia_base_url: str = field(default_factory=lambda: _env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    nvidia_embedding_model: str = field(default_factory=lambda: _env("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0.1")))
    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "1200")))

    # ---- server -------------------------------------------------------------------
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))


def get_settings() -> Settings:
    return Settings()
