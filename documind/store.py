"""Embedding model wrapper and ChromaDB-backed vector store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns texts into dense vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Local, free embeddings via ``sentence-transformers`` (loaded lazily)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()


@dataclass
class StoredChunk:
    """A chunk as stored in (and returned from) the vector store."""

    chunk_id: str
    text: str
    source: str
    page: int
    chunk_index: int


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection (cosine distance)."""

    def __init__(self, index_dir: Path, collection: str = "documind") -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(index_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[StoredChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
                for c in chunks
            ],
        )

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def query(self, embedding: list[float], k: int) -> list[tuple[StoredChunk, float]]:
        """Return the ``k`` nearest chunks with cosine similarity scores."""
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append((_to_chunk(cid, doc, meta), 1.0 - float(dist)))
        return out

    def all_chunks(self) -> list[StoredChunk]:
        """Return every stored chunk (used to build the BM25 index)."""
        res = self._collection.get(include=["documents", "metadatas"])
        return [
            _to_chunk(cid, doc, meta)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]


def _to_chunk(chunk_id: str, text: str, meta: dict[str, Any]) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        text=text,
        source=str(meta["source"]),
        page=int(meta["page"]),
        chunk_index=int(meta["chunk_index"]),
    )
