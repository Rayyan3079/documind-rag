import pytest

from documind.retrieve import MODES, RetrievedChunk, tokenize


def test_tokenize_keeps_identifiers():
    assert tokenize("See SP 800-63B, GPIO25 and 3.3 V.") == ["see", "sp", "800-63b", "gpio25", "and", "3.3", "v"]


@pytest.mark.parametrize("mode", MODES)
def test_retrieval_output_shape(engine, mode):
    engine.ingest()
    results = engine.retrieve("How long must passwords be?", k=2, mode=mode)
    assert 1 <= len(results) <= 2
    for rank, r in enumerate(results, start=1):
        assert isinstance(r, RetrievedChunk)
        assert r.rank == rank
        assert r.source.endswith(".pdf") and r.page >= 1 and r.text
        assert isinstance(r.score, float)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert set(results[0].to_dict()) == {"chunk_id", "text", "source", "page", "score", "rank"}


@pytest.mark.parametrize("mode", MODES)
def test_retrieval_finds_relevant_page(engine, mode):
    engine.ingest()
    top = engine.retrieve("hold the BOOTSEL button to enter bootloader mode", k=1, mode=mode)[0]
    assert (top.source, top.page) == ("manual.pdf", 3)


def test_empty_index_returns_nothing(engine):
    assert engine.retrieve("anything", k=3, mode="hybrid") == []


def test_invalid_mode(engine):
    engine.ingest()
    with pytest.raises(ValueError):
        engine.retrieve("x", mode="magic")
