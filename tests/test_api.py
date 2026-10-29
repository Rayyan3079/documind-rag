import pytest
from fastapi.testclient import TestClient

from documind.api import app, get_engine
from tests.conftest import make_pdf


@pytest.fixture
def client(engine):
    app.dependency_overrides[get_engine] = lambda: engine
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "chunks": 0}


def test_ingest_then_ask(client):
    r = client.post("/ingest")
    assert r.status_code == 200
    body = r.json()
    assert sorted(body["added"]) == ["guide.pdf", "manual.pdf"]
    assert body["total_chunks"] > 0

    r = client.post("/ask", json={"question": "How many GPIO pins?", "k": 3, "mode": "hybrid"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("The board exposes 26 GPIO pins")
    assert body["sources"] == [{"source": "manual.pdf", "page": 1}]
    assert 1 <= len(body["chunks"]) <= 3
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert body["latency_ms"] >= 0 and body["cost_usd"] > 0


def test_ingest_with_upload(client, tmp_path, engine):
    pdf = make_pdf(tmp_path / "extra.pdf", ["An uploaded page."])
    with pdf.open("rb") as f:
        r = client.post("/ingest", files=[("files", ("extra.pdf", f, "application/pdf"))])
    assert r.status_code == 200
    assert "extra.pdf" in r.json()["added"]
    assert (engine.settings.docs_dir / "extra.pdf").exists()


def test_rejects_non_pdf_upload(client):
    r = client.post("/ingest", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 400


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x", "mode": "magic"}, {}])
def test_ask_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/ask" in client.get("/openapi.json").json()["paths"]
