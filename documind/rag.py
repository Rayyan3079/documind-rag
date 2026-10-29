"""High-level entry point that wires ingestion, retrieval and answering together.

The CLI, the FastAPI service and the Streamlit app all go through
:class:`DocuMind`, so they behave identically.
"""

from __future__ import annotations

from functools import cached_property

import anthropic

from documind.answer import Answer, generate_answer
from documind.config import Settings, get_settings
from documind.ingest import IngestReport, ingest
from documind.retrieve import RetrievedChunk, Retriever
from documind.store import Embedder, SentenceTransformerEmbedder, VectorStore


class DocuMind:
    """RAG engine. Components can be injected (tests use fakes)."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or SentenceTransformerEmbedder(self.settings.embedding_model)
        self.store = VectorStore(self.settings.index_dir, self.settings.collection)
        self.retriever = Retriever(self.store, self.embedder)
        self._client = client

    @cached_property
    def client(self) -> anthropic.Anthropic:
        # Created on first use so ingestion and retrieval work without an API key.
        return self._client or anthropic.Anthropic()

    def ingest(self) -> IngestReport:
        return ingest(self.settings, self.store, self.embedder)

    def retrieve(self, question: str, k: int | None = None, mode: str | None = None) -> list[RetrievedChunk]:
        return self.retriever.retrieve(
            question, k or self.settings.top_k, mode or self.settings.retrieval_mode
        )

    def ask(self, question: str, k: int | None = None, mode: str | None = None) -> Answer:
        chunks = self.retrieve(question, k, mode)
        return generate_answer(
            question,
            chunks,
            client=self.client,
            model=self.settings.model,
            effort=self.settings.effort,
            max_tokens=self.settings.max_tokens,
        )
