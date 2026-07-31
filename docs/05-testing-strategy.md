# 05 — Testing strategy (and how it worked)

## Why we test

A RAG app can fail in two places: **retrieval** (did search find the right
chunks?) and **generation** (did the LLM write a good, grounded answer?).
We also need the interfaces to run. So the tests cover the whole stack, from
the data to the dashboard.

## What we test, and how

### 1. Data / ingestion

**What:** the chunked dataset loads and every chunk has the fields the
pipeline needs (`content`, `source`, `type`, `timestamp`/`file_path`).

**How:** `ingest.py` builds `data/documents.json`; the notebook and `rag.py`
open it and print the counts.

**Pass:** 3,740 chunks — 477 transcript (3 lectures) + 3,263 code.

### 2. Retrieval (search quality)

**What:** does each search strategy find the right chunks for a question?

**How (manual):**
```bash
python search.py "RoPE positional embeddings"      # hybrid + rerank, no LLM
```
**How (measured):** `python eval.py` — 10 ground-truth questions, Hit Rate
and MRR across TF-IDF, vector, hybrid, hybrid + rerank.

**Pass:** hybrid matches the best MRR (0.90) and Hit Rate stays ≥ 0.90, so
the app uses hybrid search.

### 3. Answers (generation quality)

**What:** does the full pipeline return a grounded answer?

**How (manual):**
```bash
python ask.py "What is the Chinchilla scaling law?"
```
**How (measured):** `python eval_rag.py` — LLM-as-a-Judge classifies each of
the 10 answers as RELEVANT / PARTLY_RELEVANT / NON_RELEVANT.

**Pass:** majority RELEVANT, few NON_RELEVANT, every answer has cited sources.

### 4. Random questions (variety / robustness)

**What:** the app should work on questions it was never tuned for, and it
should say "not in the context" instead of inventing an answer.

**How:** `python scripts/random_test.py` runs 12 varied questions (optimizer,
positional encoding, cross-entropy loss, attention, softmax, tokenization,
memory usage, backward pass, ...). It records the answer, sources, latency,
and token usage for each question in `results/random-questions.jsonl`.

**Pass:** every question returns 5 sources; answers either cite the context
(`[n]` with timestamps/files) or explicitly say the context does not contain
the answer (no hallucination).

### 5. Interfaces

**What:** the chat app and the dashboard start and serve the UI.

**How:** launch each with `streamlit run` and check the server responds
(HTTP 200) on its port.

**Pass:** both return HTTP 200.

### 6. Notebook

**What:** the teaching notebook runs top to bottom and saves its outputs.

**How:**
```bash
jupyter notebook notebooks/cs336-rag-test.ipynb
# or execute headlessly:
python -m nbconvert --to notebook --execute --inplace notebooks/cs336-rag-test.ipynb
```
**Pass:** executes with no error cells; the final RAG cell shows a live,
cited answer.

## Results from this run (2026-07-31, DeepSeek V4 Flash)

### Retrieval (`python eval.py`)

| Strategy | Hit Rate | MRR | Latency |
|----------|----------|-----|---------|
| TF-IDF (BM25) | 1.00 | 0.80 | 0.03s |
| Vector | 0.90 | 0.90 | 0.03s |
| **Hybrid** | **0.90** | **0.90** | **0.06s** |
| Hybrid + Rerank | 0.90 | 0.90 | 0.80s |

### Answers (`python eval_rag.py`)

| Relevance | Count | Share |
|-----------|-------|-------|
| RELEVANT | 8 | 80% |
| PARTLY_RELEVANT | 1 | 10% |
| NON_RELEVANT | 1 | 10% |

### Random questions (`python scripts/random_test.py`)

12 questions ran through the full pipeline. Every question returned 5 cited
sources. 6 questions were answered directly from the course content, and 6
were correctly refused with "the context does not contain..." (those topics
— e.g. cross-entropy loss, data splitting — are not covered by the 3
lectures + assignment-1 code in the dataset). Average latency ~7s per
question after the first (model warm-up) call; ~1,240 tokens per answer.

Sample cited answers:

> "Attention is quadratic in sequence length: as stated in [2]
> (timestamp=1825.8s), 'attention is n squared' where n is sequence length..."

> "Out-of-vocabulary words are handled by training the tokenizer on raw text
> so that rare sequences are broken up into smaller units rather than using
> an unk token [1] (timestamp=4347.6s)."

### Interfaces

- Chat app (`streamlit run app.py`) — HTTP 200 on port 8501.
- Dashboard (`streamlit run dashboard.py`) — HTTP 200 on port 8502.

### Notebook

`notebooks/cs336-rag-test.ipynb` executed end-to-end; all cells ran, the
retrieval table rendered, and the RAG cell produced a cited BPE answer.

## What the results tell us

- Hybrid search is the right default (best MRR at ~60ms).
- The RAG prompt keeps answers grounded: 80% fully relevant, and the model
  refuses when the context lacks the answer rather than guessing.
- Retrieval and answer quality are measured *before* trusting the UI, so the
  Streamlit app is built on tested behavior.

See also [03 — Evaluation](03-evaluation.md) for the metrics and
[04 — Student guide](04-student-guide.md) for how to run everything.
