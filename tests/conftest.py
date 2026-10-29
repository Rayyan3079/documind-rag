"""Shared fixtures: tiny generated PDFs, a deterministic fake embedder and a mocked Claude client."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from documind.config import Settings
from documind.rag import DocuMind
from documind.retrieve import tokenize


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal valid PDF with one line of text per page (no extra dependencies)."""
    objects: list[bytes] = []
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]
    font_id = 3 + 2 * n
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 10 Tf 20 700 Td ({safe}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


class FakeEmbedder:
    """Hashed bag-of-words vectors: similar wording gives similar vectors."""

    dim = 128

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            v = [0.0] * self.dim
            for tok in tokenize(text):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


def fake_response(text: str, input_tokens: int = 1200, output_tokens: int = 80, model: str = "claude-opus-5-5"):
    """Object shaped like an Anthropic Messages API response."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
        model=model,
    )


def mock_client(text: str = "The board exposes 26 GPIO pins [manual.pdf, page 1].") -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = fake_response(text)
    client.beta.messages.create.return_value = fake_response(text)
    return client


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    make_pdf(
        d / "manual.pdf",
        [
            "The board exposes 26 multi-function GPIO pins and three of them are ADC capable.",
            "The input voltage range on VSYS is 1.8 V to 5.5 V for normal operation.",
            "To enter bootloader mode hold the BOOTSEL button while connecting USB.",
        ],
    )
    make_pdf(
        d / "guide.pdf",
        ["Passwords must be at least 15 characters long when used as a single factor."],
    )
    return d


@pytest.fixture
def settings(tmp_path: Path, docs_dir: Path) -> Settings:
    return Settings(docs_dir=docs_dir, index_dir=tmp_path / "index", chunk_size=200, chunk_overlap=40)


@pytest.fixture
def engine(settings: Settings) -> DocuMind:
    return DocuMind(settings=settings, embedder=FakeEmbedder(), client=mock_client())
