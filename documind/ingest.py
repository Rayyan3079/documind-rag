"""Ingestion pipeline: PDFs -> cleaned page text -> chunks -> embeddings -> ChromaDB.

Re-running ingestion is idempotent: a SHA-256 content hash per file is kept in
``<index_dir>/manifest.json`` and unchanged files are skipped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from documind.config import Settings
from documind.store import Embedder, StoredChunk, VectorStore

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


@dataclass
class IngestReport:
    """Summary of one ingestion run."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks_written: int = 0


def clean_text(text: str) -> str:
    """Normalise PDF-extracted text.

    Joins words hyphenated across line breaks, drops control characters and
    collapses runs of whitespace into single spaces.
    """
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split ``text`` into chunks of at most ``size`` characters.

    Consecutive chunks share roughly ``overlap`` characters. Chunk ends are
    moved back to a sentence end or a space when one exists in the last 25%
    of the window, so words are not cut in half.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            min_cut = int(size * 0.75)
            cut = max(window.rfind(". ", min_cut), window.rfind("? ", min_cut))
            if cut != -1:
                end = start + cut + 1
            else:
                space = window.rfind(" ", min_cut)
                if space != -1:
                    end = start + space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # Start the next chunk on a word boundary.
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return ``(page_number, cleaned_text)`` for each page (1-based numbering)."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((i, text))
    return pages


def file_hash(path: Path) -> str:
    """SHA-256 of the file contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def build_chunks(path: Path, size: int, overlap: int) -> list[StoredChunk]:
    """Load one PDF and split every page into chunks with metadata.

    Chunks never span two pages, so every chunk has exactly one page number to cite.
    """
    chunks: list[StoredChunk] = []
    for page_no, text in load_pdf(path):
        for i, piece in enumerate(chunk_text(text, size, overlap)):
            chunks.append(
                StoredChunk(
                    chunk_id=f"{path.name}::p{page_no}::c{i}",
                    text=piece,
                    source=path.name,
                    page=page_no,
                    chunk_index=i,
                )
            )
    return chunks


def _load_manifest(index_dir: Path) -> dict[str, str]:
    path = index_dir / MANIFEST_NAME
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(index_dir: Path, manifest: dict[str, str]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def ingest(settings: Settings, store: VectorStore, embedder: Embedder) -> IngestReport:
    """Index every PDF in ``settings.docs_dir``, skipping files whose hash is unchanged.

    Files that disappeared from the folder are removed from the index.
    """
    report = IngestReport()
    manifest = _load_manifest(settings.index_dir)
    pdfs = sorted(settings.docs_dir.glob("*.pdf"))
    present = {p.name for p in pdfs}

    for name in sorted(set(manifest) - present):
        logger.info("Removing %s (file no longer present)", name)
        store.delete_source(name)
        del manifest[name]
        report.removed.append(name)

    for pdf in pdfs:
        digest = file_hash(pdf)
        previous = manifest.get(pdf.name)
        if previous == digest:
            logger.info("Skipping %s (unchanged)", pdf.name)
            report.skipped.append(pdf.name)
            continue
        if previous is not None:
            store.delete_source(pdf.name)
        chunks = build_chunks(pdf, settings.chunk_size, settings.chunk_overlap)
        logger.info("Embedding %d chunks from %s", len(chunks), pdf.name)
        if chunks:
            store.add(chunks, embedder.embed([c.text for c in chunks]))
        manifest[pdf.name] = digest
        # Save after each file so an interrupted run does not redo finished files.
        _save_manifest(settings.index_dir, manifest)
        (report.updated if previous else report.added).append(pdf.name)
        report.chunks_written += len(chunks)

    _save_manifest(settings.index_dir, manifest)
    return report
