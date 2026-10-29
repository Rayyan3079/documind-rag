"""Configuration, pricing table and logging setup.

All settings can be overridden with environment variables (a ``.env`` file in
the working directory is loaded automatically).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-opus-5-5"

# USD per 1M tokens (input, output). Source: Anthropic pricing page, October 2026.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for DocuMind."""

    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")
    collection: str = "documind"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    retrieval_mode: str = "hybrid"
    model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_MODEL
    effort: str = "medium"
    max_tokens: int = 16000


def get_settings() -> Settings:
    """Build :class:`Settings` from environment variables."""
    env = os.environ.get
    model = env("DOCUMIND_MODEL", DEFAULT_MODEL)
    return Settings(
        docs_dir=Path(env("DOCUMIND_DOCS_DIR", "data/docs")),
        index_dir=Path(env("DOCUMIND_INDEX_DIR", "data/index")),
        embedding_model=env("DOCUMIND_EMBEDDING_MODEL", Settings.embedding_model),
        chunk_size=int(env("DOCUMIND_CHUNK_SIZE", "800")),
        chunk_overlap=int(env("DOCUMIND_CHUNK_OVERLAP", "150")),
        top_k=int(env("DOCUMIND_TOP_K", "5")),
        retrieval_mode=env("DOCUMIND_RETRIEVAL_MODE", "hybrid"),
        model=model,
        judge_model=env("DOCUMIND_JUDGE_MODEL", model),
        effort=env("DOCUMIND_EFFORT", "medium"),
    )


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once (level from ``DOCUMIND_LOG_LEVEL``, default INFO)."""
    level = level or os.environ.get("DOCUMIND_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Third-party libraries are chatty at INFO.
    for noisy in ("httpx", "httpx2", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
