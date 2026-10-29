from documind.ingest import build_chunks
from tests.conftest import make_pdf


def test_chunks_carry_metadata(docs_dir):
    chunks = build_chunks(docs_dir / "manual.pdf", size=200, overlap=40)
    assert {c.page for c in chunks} == {1, 2, 3}
    assert all(c.source == "manual.pdf" for c in chunks)
    assert chunks[0].chunk_id == "manual.pdf::p1::c0"
    assert "26 multi-function GPIO" in chunks[0].text


def test_ingest_is_idempotent(engine):
    first = engine.ingest()
    assert sorted(first.added) == ["guide.pdf", "manual.pdf"]
    count = engine.store.count()
    calls = engine.embedder.calls

    second = engine.ingest()
    assert second.added == [] and second.chunks_written == 0
    assert sorted(second.skipped) == ["guide.pdf", "manual.pdf"]
    assert engine.store.count() == count
    assert engine.embedder.calls == calls, "unchanged files must not be re-embedded"


def test_changed_file_is_reindexed(engine, docs_dir):
    engine.ingest()
    make_pdf(docs_dir / "guide.pdf", ["Updated text.", "A second page."])
    report = engine.ingest()
    assert report.updated == ["guide.pdf"]
    assert report.skipped == ["manual.pdf"]
    pages = {c.page for c in engine.store.all_chunks() if c.source == "guide.pdf"}
    assert pages == {1, 2}


def test_removed_file_is_dropped(engine, docs_dir):
    engine.ingest()
    (docs_dir / "guide.pdf").unlink()
    report = engine.ingest()
    assert report.removed == ["guide.pdf"]
    assert all(c.source != "guide.pdf" for c in engine.store.all_chunks())
