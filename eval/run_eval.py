"""Evaluate DocuMind on eval/questions.jsonl.

Measures:
  * Retrieval  - hit rate@k and MRR per retrieval mode (is an evidence page among the
                 retrieved chunks?). Free: no API calls.
  * Answers    - LLM-as-judge verdict (correct / partially_correct / wrong) against the
                 expected answer, correct-refusal rate on unanswerable questions,
                 citation hit rate, latency p50/p95 and total cost.

Usage:
  python eval/run_eval.py                  # full run (needs ANTHROPIC_API_KEY)
  python eval/run_eval.py --retrieval-only # free, no API key needed
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from documind.answer import create_message, estimate_cost, response_text  # noqa: E402
from documind.config import setup_logging  # noqa: E402
from documind.rag import DocuMind  # noqa: E402
from documind.retrieve import MODES, RetrievedChunk  # noqa: E402

logger = logging.getLogger("eval")

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"

JUDGE_SYSTEM = """You grade answers produced by a document question-answering system.
Compare the SYSTEM ANSWER with the REFERENCE ANSWER for the QUESTION.

- correct: contains all key facts of the reference and nothing that contradicts it.
- partially_correct: some key facts are right, but something important is missing or wrong.
- wrong: the key facts are missing or contradicted, or the system said it could not find the answer.

Ignore style, length, and citation formatting. Extra correct detail is fine."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "wrong"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def load_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_hit_rank(chunks: list[RetrievedChunk], evidence: list[dict[str, Any]]) -> int | None:
    """Rank (1-based) of the first chunk whose (file, page) is an evidence page."""
    wanted = {(e["source"], e["page"]) for e in evidence}
    for c in chunks:
        if (c.source, c.page) in wanted:
            return c.rank
    return None


def retrieval_metrics(engine: DocuMind, questions: list[dict], k: int) -> dict[str, dict]:
    """Hit rate@k and MRR@k for every retrieval mode, over answerable questions."""
    answerable = [q for q in questions if q["answerable"]]
    out: dict[str, dict] = {}
    for mode in MODES:
        ranks, misses = [], []
        for q in answerable:
            rank = first_hit_rank(engine.retrieve(q["question"], k=k, mode=mode), q["evidence"])
            ranks.append(rank)
            if rank is None:
                misses.append(q["id"])
        n = len(ranks)
        out[mode] = {
            "k": k,
            "n": n,
            "hit_rate": sum(r is not None for r in ranks) / n,
            "hit_rate@1": sum(r == 1 for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
            "misses": misses,
            "ranks": dict(zip((q["id"] for q in answerable), ranks)),
        }
    return out


def judge(engine: DocuMind, question: str, expected: str, answer: str) -> tuple[dict, float]:
    """Ask the judge model for a verdict; returns (verdict dict, cost in USD)."""
    model = engine.settings.judge_model
    response = create_message(
        engine.client,
        model,
        max_tokens=2000,
        system=JUDGE_SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{
            "role": "user",
            "content": f"QUESTION: {question}\n\nREFERENCE ANSWER: {expected}\n\nSYSTEM ANSWER: {answer}",
        }],
    )
    cost = estimate_cost(response.model or model, response.usage.input_tokens, response.usage.output_tokens)
    return json.loads(response_text(response)), cost


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[idx]


def answer_eval(engine: DocuMind, questions: list[dict], k: int, mode: str) -> tuple[list[dict], dict]:
    rows = []
    answer_cost = judge_cost = 0.0
    for i, q in enumerate(questions, 1):
        logger.info("[%d/%d] %s", i, len(questions), q["id"])
        result = engine.ask(q["question"], k=k, mode=mode)
        answer_cost += result.cost_usd
        row = {
            "id": q["id"],
            "type": q["type"],
            "answerable": q["answerable"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "answer": result.answer,
            "cited": [s.__dict__ for s in result.sources],
            "retrieved": [(c.source, c.page) for c in result.chunks],
            "latency_ms": round(result.latency_ms, 1),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": round(result.cost_usd, 6),
            "not_found": result.not_found,
        }
        if q["answerable"]:
            if result.not_found:
                row["verdict"], row["reason"] = "wrong", "false refusal (said not found)"
            else:
                verdict, cost = judge(engine, q["question"], q["expected_answer"], result.answer)
                judge_cost += cost
                row["verdict"], row["reason"] = verdict["verdict"], verdict["reason"]
            wanted = {(e["source"], e["page"]) for e in q["evidence"]}
            row["citation_hit"] = any((s["source"], s["page"]) in wanted for s in row["cited"])
        else:
            row["verdict"] = "correct_refusal" if result.not_found else "hallucinated"
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] > 0]
    count = lambda v: sum(r["verdict"] == v for r in ans)  # noqa: E731
    summary = {
        "mode": mode,
        "k": k,
        "model": engine.settings.model,
        "judge_model": engine.settings.judge_model,
        "effort": engine.settings.effort,
        "n_answerable": len(ans),
        "n_unanswerable": len(neg),
        "correct": count("correct"),
        "partially_correct": count("partially_correct"),
        "wrong": count("wrong"),
        "accuracy": count("correct") / len(ans),
        "accuracy_incl_partial": (count("correct") + count("partially_correct")) / len(ans),
        "false_refusals": sum(r["not_found"] for r in ans),
        "citation_hit_rate": sum(r["citation_hit"] for r in ans) / len(ans),
        "correct_refusal_rate": sum(r["verdict"] == "correct_refusal" for r in neg) / len(neg),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_mean_ms": statistics.mean(latencies),
        "answer_cost_usd": answer_cost,
        "judge_cost_usd": judge_cost,
        "total_cost_usd": answer_cost + judge_cost,
        "avg_input_tokens": statistics.mean(r["input_tokens"] for r in rows),
        "avg_output_tokens": statistics.mean(r["output_tokens"] for r in rows),
    }
    return rows, summary


def render_markdown(retrieval: dict, answers: dict | None, rows: list[dict] | None, meta: dict) -> str:
    lines = [
        "# DocuMind evaluation results",
        "",
        f"Run: {meta['timestamp']} · corpus: {meta['n_chunks']} chunks from {meta['n_docs']} PDFs · "
        f"chunk size {meta['chunk_size']} / overlap {meta['chunk_overlap']} · "
        f"embeddings `{meta['embedding_model']}`",
        "",
        f"## Retrieval (answerable questions, k={next(iter(retrieval.values()))['k']})",
        "",
        "| Mode | Hit rate@k | Hit rate@1 | MRR | Misses |",
        "|---|---|---|---|---|",
    ]
    for mode, m in retrieval.items():
        lines.append(
            f"| {mode} | {m['hit_rate']:.0%} | {m['hit_rate@1']:.0%} | {m['mrr']:.3f} | "
            f"{', '.join(m['misses']) or '-'} |"
        )
    if answers:
        a = answers
        lines += [
            "",
            f"## Answers (`{a['model']}`, effort `{a['effort']}`, {a['mode']} retrieval, k={a['k']}; "
            f"judge `{a['judge_model']}`)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Correct | {a['correct']}/{a['n_answerable']} ({a['accuracy']:.0%}) |",
            f"| Correct or partially correct | {a['correct'] + a['partially_correct']}/{a['n_answerable']} "
            f"({a['accuracy_incl_partial']:.0%}) |",
            f"| Wrong | {a['wrong']} (of which false refusals: {a['false_refusals']}) |",
            f"| Answer cites an evidence page | {a['citation_hit_rate']:.0%} |",
            f"| Correct refusals on unanswerable questions | "
            f"{round(a['correct_refusal_rate'] * a['n_unanswerable'])}/{a['n_unanswerable']} "
            f"({a['correct_refusal_rate']:.0%}) |",
            f"| Latency p50 / p95 | {a['latency_p50_ms'] / 1000:.1f} s / {a['latency_p95_ms'] / 1000:.1f} s |",
            f"| Avg tokens per answer (in / out) | {a['avg_input_tokens']:.0f} / {a['avg_output_tokens']:.0f} |",
            f"| Cost: answers / judge / total | ${a['answer_cost_usd']:.3f} / ${a['judge_cost_usd']:.3f} / "
            f"${a['total_cost_usd']:.3f} |",
            "",
            "## Per-question verdicts",
            "",
            "| ID | Type | Verdict | Cited evidence page | Latency |",
            "|---|---|---|---|---|",
        ]
        for r in rows or []:
            cite = "-" if not r["answerable"] else ("yes" if r["citation_hit"] else "no")
            lines.append(f"| {r['id']} | {r['type']} | {r['verdict']} | {cite} | {r['latency_ms'] / 1000:.1f} s |")
        not_correct = [r for r in rows or [] if r["verdict"] not in ("correct", "correct_refusal")]
        if not_correct:
            lines += ["", "## Not fully correct", ""]
            for r in not_correct:
                lines += [
                    f"### {r['id']}: {r['question']}",
                    f"- Expected: {r['expected_answer']}",
                    f"- Got: {r['answer']}",
                    f"- Judge: {r['verdict']} - {r.get('reason', '')}",
                    f"- Retrieved: {', '.join(f'{s} p.{p}' for s, p in r['retrieved'])}",
                    "",
                ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=MODES, default=None, help="Retrieval mode for the answer eval.")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM calls (free).")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke test).")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    setup_logging()
    engine = DocuMind()
    report = engine.ingest()
    logger.info("Index ready: %d chunks (%d files skipped as unchanged)", engine.store.count(), len(report.skipped))

    questions = load_questions()[: args.limit]
    mode = args.mode or engine.settings.retrieval_mode
    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_chunks": engine.store.count(),
        "n_docs": len(list(engine.settings.docs_dir.glob("*.pdf"))),
        "chunk_size": engine.settings.chunk_size,
        "chunk_overlap": engine.settings.chunk_overlap,
        "embedding_model": engine.settings.embedding_model,
    }

    t0 = time.perf_counter()
    retrieval = retrieval_metrics(engine, questions, args.k)
    rows, answers = (None, None)
    if not args.retrieval_only:
        rows, answers = answer_eval(engine, questions, args.k, mode)
    meta["wall_time_s"] = round(time.perf_counter() - t0, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "retrieval.json").write_text(json.dumps(retrieval, indent=2))
    if rows is not None:
        with (args.out / "answers.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (args.out / "summary.json").write_text(json.dumps({"meta": meta, "retrieval": retrieval, "answers": answers}, indent=2))
    md = render_markdown(retrieval, answers, rows, meta)
    (args.out / "summary.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
