from documind.answer import (
    NOT_FOUND,
    Source,
    build_context,
    estimate_cost,
    generate_answer,
    is_not_found,
    parse_citations,
)
from documind.retrieve import RetrievedChunk
from tests.conftest import fake_response, mock_client

CHUNKS = [
    RetrievedChunk("manual.pdf::p1::c0", "26 GPIO pins.", "manual.pdf", 1, 0.9, 1),
    RetrievedChunk("guide.pdf::p4::c0", "15 characters.", "guide.pdf", 4, 0.8, 2),
]


def test_parse_citations_variants_and_dedup():
    text = (
        "It has 26 pins [manual.pdf, page 1]. Passwords need 15 chars [guide.pdf, p. 4] "
        "[NIST.SP.800-63B-4.pdf, page 25]. Again [manual.pdf, page 1] and [guide.pdf, pg 4]."
    )
    assert parse_citations(text) == [
        Source("manual.pdf", 1),
        Source("guide.pdf", 4),
        Source("NIST.SP.800-63B-4.pdf", 25),
    ]


def test_parse_citations_ignores_non_citations():
    assert parse_citations("See [1] and [the manual] and [notes.txt, page 2].") == []


def test_not_found_detection():
    assert is_not_found(f"{NOT_FOUND}.")
    assert not is_not_found("The answer is 42 [a.pdf, page 1].")


def test_build_context_has_headers():
    ctx = build_context(CHUNKS)
    assert "[1] (file: manual.pdf, page 1)" in ctx
    assert "[2] (file: guide.pdf, page 4)" in ctx


def test_estimate_cost():
    # Opus 5.5: $4 / $20 per million tokens.
    assert estimate_cost("claude-opus-5-5", 1_000_000, 0) == 4.0
    assert estimate_cost("claude-opus-5-5", 1000, 500) == (1000 * 4 + 500 * 20) / 1e6
    assert estimate_cost("unknown-model", 1000, 1000) == 0.0


def test_generate_answer_uses_context_and_reports_usage():
    client = mock_client("It has 26 GPIO pins [manual.pdf, page 1].")
    ans = generate_answer("How many GPIO pins?", CHUNKS, client=client, model="claude-opus-5-5")
    assert ans.sources == [Source("manual.pdf", 1)]
    assert ans.input_tokens == 1200 and ans.output_tokens == 80
    assert ans.cost_usd > 0 and ans.latency_ms >= 0
    assert not ans.not_found

    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert "(file: guide.pdf, page 4)" in kwargs["messages"][0]["content"]
    assert "ONLY" in kwargs["system"]


def test_haiku_uses_plain_messages_endpoint():
    client = mock_client()
    client.messages.create.return_value = fake_response("x", model="claude-haiku-5-5")
    ans = generate_answer("q", CHUNKS, client=client, model="claude-haiku-5-5")
    client.messages.create.assert_called_once()
    client.beta.messages.create.assert_not_called()
    assert ans.model == "claude-haiku-5-5"


def test_no_chunks_means_not_found_without_api_call():
    client = mock_client()
    ans = generate_answer("q", [], client=client, model="claude-opus-5-5")
    assert ans.not_found
    client.beta.messages.create.assert_not_called()
