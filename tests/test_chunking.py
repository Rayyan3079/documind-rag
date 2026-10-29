import pytest

from documind.ingest import chunk_text, clean_text


def test_clean_text_joins_hyphenation_and_whitespace():
    assert clean_text("micro-\ncontroller  board\n\n with\tpins") == "microcontroller board with pins"


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # Every word appears in at least one chunk, and no word is cut in half.
    joined = " ".join(chunks).split()
    assert set(joined) == set(text.split())


def test_consecutive_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(text, size=120, overlap=40)
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-1] in b.split(), "expected the last word of a chunk to reappear in the next"


def test_prefers_sentence_boundaries():
    text = ("This is a sentence about pins. " * 10).strip()
    chunks = chunk_text(text, size=200, overlap=40)
    assert all(c.endswith(".") for c in chunks[:-1])


def test_short_and_empty_text():
    assert chunk_text("short text", size=800, overlap=150) == ["short text"]
    assert chunk_text("   ", size=800, overlap=150) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", size=size, overlap=overlap)
