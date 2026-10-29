"""Retrieval: semantic top-k search, BM25 keyword search, and a hybrid of both.

The hybrid mode merges the two ranked lists with Reciprocal Rank Fusion
(RRF): ``score(d) = sum(1 / (RRF_K + rank_i(d)))``. RRF only uses ranks, so
there is no need to calibrate cosine similarities against BM25 scores.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

from rank_bm25 import BM25Okapi

from documind.store import Embedder, StoredChunk, VectorStore

logger = logging.getLogger(__name__)

MODES = ("semantic", "bm25", "hybrid")
RRF_K = 60
# How many candidates each retriever contributes before fusion.
CANDIDATES = 20

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; keeps identifiers like ``800-63b`` or ``gpio25`` intact."""
    return _TOKEN.findall(text.lower())


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its score and rank (1 = best)."""

    chunk_id: str
    text: str
    source: str
    page: int
    score: float
    rank: int

    def to_dict(self) -> dict:
        return asdict(self)


class Retriever:
    """Retrieves chunks from a :class:`VectorStore` in one of :data:`MODES`."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[StoredChunk] = []

    def retrieve(self, query: str, k: int = 5, mode: str = "semantic") -> list[RetrievedChunk]:
        """Return the top ``k`` chunks for ``query``."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if self.store.count() == 0:
            logger.warning("Index is empty; run ingestion first")
            return []
        if mode == "semantic":
            ranked = self._semantic(query, k)
        elif mode == "bm25":
            ranked = self._keyword(query, k)
        else:
            ranked = self._hybrid(query, k)
        return [
            RetrievedChunk(c.chunk_id, c.text, c.source, c.page, round(score, 4), rank)
            for rank, (c, score) in enumerate(ranked, start=1)
        ]

    def _semantic(self, query: str, k: int) -> list[tuple[StoredChunk, float]]:
        return self.store.query(self.embedder.embed([query])[0], k)

    def _keyword(self, query: str, k: int) -> list[tuple[StoredChunk, float]]:
        bm25 = self._get_bm25()
        scores = bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._bm25_chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def _hybrid(self, query: str, k: int) -> list[tuple[StoredChunk, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, StoredChunk] = {}
        for ranked in (self._semantic(query, CANDIDATES), self._keyword(query, CANDIDATES)):
            for rank, (chunk, _) in enumerate(ranked, start=1):
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
                by_id[chunk.chunk_id] = chunk
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(by_id[cid], fused[cid]) for cid in best]

    def _get_bm25(self) -> BM25Okapi:
        # Rebuild lazily whenever the index size changed (new ingestion).
        if self._bm25 is None or len(self._bm25_chunks) != self.store.count():
            self._bm25_chunks = self.store.all_chunks()
            self._bm25 = BM25Okapi([tokenize(c.text) for c in self._bm25_chunks])
            logger.info("Built BM25 index over %d chunks", len(self._bm25_chunks))
        return self._bm25
