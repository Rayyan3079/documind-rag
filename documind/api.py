"""REST API. Run with ``uvicorn documind.api:app`` and open http://localhost:8000/docs."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from documind.config import setup_logging
from documind.rag import DocuMind
from documind.retrieve import MODES

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuMind API",
    description="Ask questions about technical PDF manuals; answers cite file and page.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_engine() -> DocuMind:
    """Shared engine instance (overridden in tests)."""
    return DocuMind()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How many GPIO pins does the Pico expose?"])
    k: int | None = Field(None, ge=1, le=20, description="Chunks to retrieve.")
    mode: str | None = Field(None, description=f"One of {MODES}.")


class SourceOut(BaseModel):
    source: str
    page: int


class ChunkOut(BaseModel):
    chunk_id: str
    source: str
    page: int
    score: float
    rank: int
    text: str


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    chunks: list[ChunkOut]
    model: str
    usage: UsageOut
    latency_ms: float
    cost_usd: float
    not_found: bool


class IngestResponse(BaseModel):
    added: list[str]
    updated: list[str]
    skipped: list[str]
    removed: list[str]
    chunks_written: int
    total_chunks: int


@app.get("/health")
def health(engine: DocuMind = Depends(get_engine)) -> dict:
    return {"status": "ok", "chunks": engine.store.count()}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    files: list[UploadFile] | None = File(None, description="Optional PDFs to add first."),
    engine: DocuMind = Depends(get_engine),
) -> IngestResponse:
    """Save any uploaded PDFs to the docs folder, then (re-)index the folder."""
    docs_dir = engine.settings.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    for upload in files or []:
        name = Path(upload.filename or "").name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Not a PDF: {upload.filename!r}")
        (docs_dir / name).write_bytes(await upload.read())
        logger.info("Saved upload %s", name)
    report = engine.ingest()
    return IngestResponse(**report.__dict__, total_chunks=engine.store.count())


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest, engine: DocuMind = Depends(get_engine)) -> AskResponse:
    """Retrieve relevant passages and answer with citations."""
    if req.mode is not None and req.mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {MODES}")
    result = engine.ask(req.question, k=req.k, mode=req.mode)
    return AskResponse(**result.to_dict())
