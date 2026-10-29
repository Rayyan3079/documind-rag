"""Command-line interface: ``python -m documind ingest`` / ``python -m documind ask "..."``."""

from __future__ import annotations

import json

import typer

from documind.config import setup_logging
from documind.rag import DocuMind
from documind.retrieve import MODES

app = typer.Typer(add_completion=False, help="DocuMind: ask questions about your PDF manuals.")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    setup_logging("DEBUG" if verbose else None)


@app.command("ingest")
def ingest_cmd() -> None:
    """Index all PDFs in the docs folder (unchanged files are skipped)."""
    report = DocuMind().ingest()
    typer.echo(
        f"added={len(report.added)} updated={len(report.updated)} "
        f"skipped={len(report.skipped)} removed={len(report.removed)} "
        f"chunks_written={report.chunks_written}"
    )


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Your question."),
    k: int = typer.Option(None, "--k", help="Number of chunks to retrieve."),
    mode: str = typer.Option(None, "--mode", help=f"Retrieval mode: {', '.join(MODES)}."),
    show_chunks: bool = typer.Option(False, "--show-chunks", help="Print retrieved passages."),
    as_json: bool = typer.Option(False, "--json", help="Print the full result as JSON."),
) -> None:
    """Answer a question from the indexed documents, with citations."""
    result = DocuMind().ask(question, k=k, mode=mode)
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return
    typer.echo(result.answer)
    typer.echo("")
    if result.sources:
        typer.echo("Sources: " + "; ".join(f"{s.source} p.{s.page}" for s in result.sources))
    typer.echo(
        f"[{result.model} | {result.latency_ms:.0f} ms | "
        f"{result.input_tokens} in / {result.output_tokens} out tokens | ${result.cost_usd:.4f}]"
    )
    if show_chunks:
        for c in result.chunks:
            typer.echo(f"\n--- #{c.rank} {c.source} p.{c.page} (score {c.score}) ---\n{c.text}")
