"""Answer generation with Claude, citation parsing and cost accounting."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from documind.config import PRICING_PER_MTOK
from documind.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

NOT_FOUND = "I could not find this in the documents"

SYSTEM_PROMPT = f"""You answer questions about technical documents using ONLY the \
numbered context passages provided in the user message.

Rules:
- Use only facts stated in the context. Do not use outside knowledge, even if you \
know the answer.
- After every sentence that uses a fact, cite its source as [file, page N], copying \
the file name and page number exactly as shown in the passage header, e.g. \
[manual.pdf, page 12]. Cite several sources as [a.pdf, page 1] [b.pdf, page 4].
- If the context does not contain the answer, reply exactly: "{NOT_FOUND}." and \
nothing else.
- If the context only partly answers the question, answer that part and say what \
is missing.
- Be concise: a short paragraph or a few bullet points."""

# Server-side refusal fallback is supported on these models (not on Haiku 5.5).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_CITATION = re.compile(r"\[([^\[\]]+?\.pdf)\s*,\s*(?:page|p\.?|pg\.?)\s*(\d+)\]", re.I)


@dataclass
class Source:
    """A cited ``(file, page)`` pair."""

    source: str
    page: int


@dataclass
class Answer:
    """Result of one question: text, citations, retrieved context and usage."""

    question: str
    answer: str
    sources: list[Source]
    chunks: list[RetrievedChunk]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    stop_reason: str | None = None
    not_found: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.__dict__ for s in self.sources],
            "chunks": [c.to_dict() for c in self.chunks],
            "model": self.model,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "stop_reason": self.stop_reason,
            "not_found": self.not_found,
        }


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered passages with file/page headers."""
    return "\n\n".join(
        f"[{i}] (file: {c.source}, page {c.page})\n{c.text}" for i, c in enumerate(chunks, 1)
    )


def parse_citations(text: str) -> list[Source]:
    """Extract unique ``[file.pdf, page N]`` citations in order of first appearance."""
    seen: set[tuple[str, int]] = set()
    out: list[Source] = []
    for name, page in _CITATION.findall(text):
        key = (name.strip(), int(page))
        if key not in seen:
            seen.add(key)
            out.append(Source(*key))
    return out


def is_not_found(text: str) -> bool:
    """True if the model said the answer is not in the documents."""
    return NOT_FOUND.lower() in text.lower()


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD cost from the pricing table (0.0 for unknown models)."""
    prices = PRICING_PER_MTOK.get(model)
    if prices is None:
        logger.warning("No pricing known for %s; reporting cost 0", model)
        return 0.0
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def create_message(client: anthropic.Anthropic, model: str, **params: Any) -> Any:
    """Call the Messages API, opting into server-side refusal fallback where supported."""
    if model in _FALLBACK_MODELS:
        return client.beta.messages.create(
            model=model, betas=[_FALLBACK_BETA], fallbacks="default", **params
        )
    return client.messages.create(model=model, **params)


def response_text(response: Any) -> str:
    """Concatenate the text blocks of a Messages API response."""
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    client: anthropic.Anthropic,
    model: str,
    effort: str = "medium",
    max_tokens: int = 16000,
) -> Answer:
    """Ask Claude to answer ``question`` from ``chunks`` and measure latency/cost."""
    if not chunks:
        return Answer(question, f"{NOT_FOUND}.", [], [], model, not_found=True)

    user = (
        f"Context passages:\n\n{build_context(chunks)}\n\n"
        f"Question: {question}"
    )
    start = time.perf_counter()
    response = create_message(
        client,
        model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": user}],
    )
    latency_ms = (time.perf_counter() - start) * 1000

    served_by = getattr(response, "model", None) or model
    if response.stop_reason == "refusal":
        text = "The model declined to answer this question."
    else:
        text = response_text(response).strip()
    usage = response.usage
    answer = Answer(
        question=question,
        answer=text,
        sources=parse_citations(text),
        chunks=chunks,
        model=served_by,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_ms=latency_ms,
        cost_usd=estimate_cost(served_by, usage.input_tokens, usage.output_tokens),
        stop_reason=response.stop_reason,
        not_found=is_not_found(text),
    )
    logger.info(
        "Answered in %.0f ms (%d in / %d out tokens, $%.4f)",
        latency_ms, usage.input_tokens, usage.output_tokens, answer.cost_usd,
    )
    return answer
