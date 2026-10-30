# DocuMind design notes

This document explains the main design decisions: how PDFs are chunked, how hybrid
retrieval works, how retrieval is scored (hit rate and MRR), and how answers are graded
by an LLM judge. The measured numbers live in [`eval/results/summary.md`](../eval/results/summary.md).

## 1. Ingestion and chunking

```
PDF -> pypdf text per page -> clean_text() -> chunk_text() -> MiniLM embeddings -> ChromaDB
```

**Text extraction and cleaning** (`documind/ingest.py`). Each page is extracted with
`pypdf` and normalised: words hyphenated across a line break are re-joined, soft hyphens
and control characters are removed, and runs of whitespace collapse to one space.

**Chunks never cross a page boundary.** Every chunk is cut from a single page, so it
carries exactly one `(file, page)` pair. That makes citations exact: when the model
cites `[pico-datasheet.pdf, page 8]`, page 8 is where the passage really is. The cost
is that a sentence running over a page break gets split, which I accepted for clean
citations.

**Fixed-size windows with overlap.** Chunks are at most 800 characters with about 150
characters of overlap (both configurable). Character windows are simple, predictable
and model-independent. 800 characters is roughly 150-200 tokens, small enough that
`all-MiniLM-L6-v2` (which truncates at 256 word pieces) sees the whole chunk, and
large enough to hold a table row with its heading.

**Boundary snapping.** Instead of cutting at exactly 800 characters, the end of a chunk
moves back to the last sentence end (`. ` or `? `) in the final 25% of the window, or
failing that to the last space. The next chunk starts on a word boundary. This avoids
half-words, which hurt both BM25 (broken tokens) and embeddings.

**Idempotent re-ingestion.** A SHA-256 hash of each PDF is stored in
`data/index/manifest.json`. Unchanged files are skipped, changed files have their old
chunks deleted before re-indexing, and files removed from the folder are removed from
the index. The manifest is saved after every file, so an interrupted run resumes
where it stopped.

## 2. Retrieval: semantic, BM25 and hybrid (RRF)

`documind/retrieve.py` supports three modes.

| Mode | How it scores | Good at | Weak at |
|---|---|---|---|
| `semantic` | cosine similarity of MiniLM embeddings (ChromaDB) | paraphrases, conceptual questions | exact identifiers, numbers, part names |
| `bm25` | Okapi BM25 over lower-cased tokens | exact terms like `VSYS`, `GPIO25`, `AAL2` | synonyms and rephrasing |
| `hybrid` | Reciprocal Rank Fusion of both lists | both | slightly more compute |

**Tokenisation for BM25.** The tokenizer keeps dotted and hyphenated identifiers
together (`800-63b`, `gpio25`, `3.3`), because technical manuals are full of them and
splitting them destroys the exact-match signal BM25 is good at.

**Reciprocal Rank Fusion.** Each retriever returns its top 20 candidates. Every chunk
gets the score

```
RRF(d) = sum over retrievers r of  1 / (60 + rank_r(d))
```

and the top `k` (default 5) by fused score are returned. Why RRF instead of a weighted
sum of scores:

- Cosine similarities (roughly 0-1) and BM25 scores (unbounded, query-dependent) are on
  different scales. Combining them would need normalisation and a tuned weight.
- RRF only uses **ranks**, so there is nothing to calibrate.
- A chunk ranked well by *both* retrievers beats one ranked first by only one of them,
  which is the behaviour you want when the two signals agree.
- The constant 60 (from Cormack et al., 2009) flattens the curve so that rank 1 vs rank
  2 matters, but no single retriever dominates.

## 3. Answer generation

The top-k chunks are sent to Claude as numbered passages with `(file: ..., page N)`
headers. The system prompt requires the model to:

- use only facts from the passages, not outside knowledge;
- cite each fact as `[file.pdf, page N]`;
- reply exactly `I could not find this in the documents.` when the answer is not there.

Citations are parsed back out of the answer with a regex, so the UI and API can show
and check them. The exact "not found" sentence makes refusals machine-detectable. Every
answer records latency, input/output tokens and an estimated cost from a pricing table.

## 4. Evaluation

`eval/questions.jsonl` holds 26 hand-written questions over three PDFs: 22 answerable
questions with a reference answer and the evidence page(s), and 4 whose answer is not in
the corpus. `eval/run_eval.py` measures two things separately, so a bad answer can be
traced to retrieval or to generation.

### Retrieval metrics (free, no API calls)

For each answerable question, a retrieved chunk is a **hit** if its `(file, page)` is
one of the evidence pages.

- **Hit rate@k**: share of questions with at least one hit in the top k.
- **Hit rate@1**: share where the very first chunk is a hit.
- **MRR (Mean Reciprocal Rank)**: average of `1 / rank of the first hit` (0 if there is
  no hit in the top k).

MRR rewards putting the right page *first*: hit at rank 1 scores 1.0, rank 2 scores 0.5,
rank 5 scores 0.2. Two retrievers with the same hit rate@5 can differ a lot in MRR, and
MRR matters because the model reads the context in order and the top passages carry
most weight.

These are computed for all three modes on the same questions, which is how hybrid was
chosen as the default.

### Answer metrics (LLM-as-judge)

Each question is answered with the configured model and retrieval mode. Then:

- **Answerable questions** go to a judge model (Claude, low effort, structured JSON
  output) with the question, the reference answer and the system's answer. It returns
  `correct`, `partially_correct` or `wrong` plus a one-line reason. The rubric tells it to
  ignore style, length and citation format, and to accept extra correct detail. If the
  system said "not found", the question is marked `wrong` (a false refusal) without
  calling the judge.
- **Unanswerable questions** are not judged: the answer is a `correct_refusal` if the
  system said "not found" and `hallucinated` otherwise.
- **Citation hit rate**: share of answerable questions where the answer cites at least
  one evidence page. This checks the citations point at the right place, not only that
  the text is right.
- Latency p50/p95, average tokens and total cost (answers and judge separately).

**Judge limitations.** The judge is the same model family as the answerer, so it may
share its blind spots. It sees only the reference answer, not the source pages, so it
cannot reward a correct fact the reference left out. With 22 answerable questions one
question is about 4.5 percentage points, so small differences between runs are noise.
Every verdict and its reason is written to `eval/results/answers.jsonl` and the
non-correct ones are listed in `summary.md`, so they can be checked by hand.
