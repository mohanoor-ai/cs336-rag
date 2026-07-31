# 03 — Evaluation: what we test, how, and why

## The question evaluation answers

A RAG app can fail in two very different places:

1. **Retrieval** — did the search find the right chunks?
2. **Answer quality** — even with the right chunks, did the LLM write a good
   answer?

If retrieval is bad, the LLM cannot answer well no matter how good it is.
So this project evaluates **retrieval first**, then the **final answers**.
The evaluation also decides **which search strategy the app should use**.

## Test criteria (from the course project spec)

The DataTalks course specifies evaluation criteria for the project itself.
Here is what we test and where it is covered:

| Requirement | What it tests | Where |
|-------------|---------------|-------|
| Problem description | Is the problem clear? | `README.md`, `docs/01-project-story.md` |
| Retrieval flow | Is a knowledge base + LLM used? | `rag.py`, `minsearch.py` |
| Retrieval evaluation | Are multiple strategies compared and the best used? | `eval.py` |
| LLM evaluation | Is the final output evaluated? | `eval_rag.py` |
| Interface | Can a user interact with it? | `app.py` (UI) + CLIs |
| Ingestion | Is data ingested? | `ingest.py` |
| Monitoring | Feedback + dashboard with 5+ charts | `app.py` logs, `dashboard.py` |
| Containerization | Can it run in containers? | `Dockerfile`, `docker-compose.yaml` |
| Reproducibility | Can someone run it? | pinned `requirements.txt`, `.env.example` |
| Best practices | Hybrid search, reranking, query rewriting | all three implemented |

## Retrieval evaluation (`eval.py`)

### Ground truth

`evaluation/eval_questions.json` holds 10 questions about CS336. Each one
lists the **expected sources** that should be retrieved — for example:

```json
{
  "question": "How does byte pair encoding (BPE) tokenization work?",
  "expected_sources": [
    {"type": "transcript", "video_id": "JuoVZkPBiKk"},
    {"type": "code", "file_path": "tests/test_tokenizer.py"}
  ]
}
```

A retrieved chunk is **relevant** if it matches one of those sources
(`is_relevant()` in `eval.py`).

### Metrics (course, module 4)

- **Hit Rate** — fraction of questions where at least one relevant chunk
  appears in the top-k.
- **MRR** (Mean Reciprocal Rank) — `1 / rank` of the first relevant chunk,
  averaged. It rewards strategies that rank the right chunk higher.

### Strategies compared

```python
"TF-IDF":        idx.search(...)
"Vector":        idx.vector_search(...)
"Hybrid":        idx.hybrid_search(...)
"Hybrid+Rerank": rerank(hybrid(...))
```

### Results (run 2026-07-31)

| Strategy | Hit Rate | MRR | Latency |
|----------|----------|-----|---------|
| TF-IDF (BM25) | 1.00 | 0.80 | 0.02s |
| Vector | 0.90 | 0.90 | 0.03s |
| **Hybrid** | **0.90** | **0.90** | **0.05s** |
| Hybrid + Rerank | 0.90 | 0.90 | 0.78s |

**What this tells us:**

- TF-IDF found a relevant chunk for every question (Hit Rate 1.00) but its
  first relevant result was often ranked lower (MRR 0.80).
- Vector and hybrid both rank relevant chunks first more often (MRR 0.90).
- Hybrid gets the same MRR as vector while keeping keyword coverage — so
  **the app uses hybrid search**.
- Reranking does not change the metric here (the answer is already at the
  top) but costs ~15x latency. It is kept on by default because it visibly
  improves which chunk is ranked #1 on real questions.

## RAG evaluation (`eval_rag.py`)

### What it tests

Retrieval metrics do not measure the final answer. `eval_rag.py` runs the
full pipeline on the same 10 questions and asks the LLM to grade each answer
with an **LLM-as-a-Judge** (course, module 4):

```
You are an expert evaluator for a RAG system. Classify the relevance of the
generated answer to the question as NON_RELEVANT / PARTLY_RELEVANT / RELEVANT.
```

It saves `question, answer, relevance, explanation` rows to
`results/rag-eval.csv` and prints the distribution.

### Results (run 2026-07-31, DeepSeek V4 Flash)

| Relevance | Count | Share |
|-----------|-------|-------|
| RELEVANT | 8 | 80% |
| PARTLY_RELEVANT | 1 | 10% |
| NON_RELEVANT | 1 | 10% |

80% of the answers are fully relevant to the question. The judge prompt,
the answers, and the explanations are all saved in `results/rag-eval.csv`
for review.

### What we're trying to achieve with this

- Detect when the LLM "hallucinates" — answers that sound confident but do
  not match the question.
- Decide whether a more expensive model is worth it (course project example
  compares two models this way).
- Have a number to report in the README instead of "it seems to work".

## How to run the evaluation

```bash
python eval.py        # retrieval: Hit Rate + MRR, no API key needed
python eval_rag.py    # answers: LLM-as-a-Judge, needs OPENCODE_API_KEY
make eval
make eval-rag
```

The same walkthrough, with explanations, is in
`notebooks/cs336-rag-test.ipynb`.
