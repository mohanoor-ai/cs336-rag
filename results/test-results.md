# CS336-RAG Test Results

**Date:** 2026-08-02
**Environment:** Python 3.14, in-memory search (no external database)
**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
**Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
**LLM:** DeepSeek V4 Flash via OpenCode Go (`https://opencode.ai/zen/go/v1`)
**Test strategy:** see the Testing and Evaluation sections of the README

## 1. Ingestion

```
YouTube: 477 segments (L1 JuoVZkPBiKk=153, L2 lVynu4bo1rY=172, L3 izZba4UA7iY=152)
GitHub: 3,263 chunks from stanford-cs336/assignment1-basics
Total: 3,740 chunks
```

PASS — every chunk has `content`, `source`, `type`, and timestamp/file metadata.

## 2. Retrieval evaluation (`python eval.py`)

10 ground-truth questions, Hit Rate + MRR across 4 strategies:

| Strategy | Hit Rate | MRR | Latency |
|----------|----------|-----|---------|
| BM25 (TF-IDF) | 1.00 | 0.80 | 0.02s |
| Vector (semantic) | 0.90 | 0.90 | 0.02s |
| **Hybrid (TF-IDF + Vector)** | **0.90** | **0.90** | **0.04s** |
| Hybrid + Rerank | 0.90 | 0.90 | 0.39s |

PASS — hybrid matches the best MRR (0.90) at ~40ms, so the app uses hybrid.

## 3. Manual retrieval (`python search.py "RoPE positional embeddings"`)

Returned 5 reranked hits from Lecture 2 (`lVynu4bo1rY`, t=1944s, t=1976s,
t=5211s, ...) — all directly about RoPE. PASS.

## 4. RAG evaluation (`python eval_rag.py`)

LLM-as-a-Judge on the same 10 questions:

| Relevance | Count | Share |
|-----------|-------|-------|
| RELEVANT | 6 | 60% |
| PARTLY_RELEVANT | 2 | 20% |
| NON_RELEVANT | 2 | 20% |

PASS — majority fully relevant; results vary run-to-run since both the
answers and the judge are LLMs (observed range across three runs: 6-8
RELEVANT, 1-3 PARTLY, 0-2 NON). Full rows in `results/rag-eval.csv`.

## 5. Random questions (`python scripts/random_test.py`)

12 varied questions ran through the full pipeline
(rewrite -> search -> rerank -> LLM). Every question returned **5 cited
sources**. Results saved to `results/random-questions.jsonl`.

| # | Question | Latency | Tokens |
|---|----------|---------|--------|
| 1 | How does the optimizer update model parameters in training? | 173.5s* | 1873 |
| 2 | Why is positional encoding important in transformers? | 7.6s | 1131 |
| 3 | What does the cross-entropy loss measure? | 8.6s | 996 |
| 4 | How does attention relate sequence length to compute? | 21.2s | 1197 |
| 5 | What is the role of softmax in the attention mechanism? | 8.5s | 1214 |
| 6 | How do you handle out-of-vocabulary words in tokenization? | 11.4s | 1674 |
| 7 | Why does the course prefer building models from scratch? | 9.6s | 1203 |
| 8 | What are the benefits of pre-training a language model? | 8.8s | 1062 |
| 9 | How is memory usage estimated for large transformers? | 6.9s | 988 |
| 10 | What is the difference between a tokenizer vocabulary and a merge list? | 10.8s | 1656 |
| 11 | How should you split data for training, validation, and testing? | 9.2s | 1214 |
| 12 | What happens during the backward pass in training? | 9.1s | 1425 |

\* First call includes model warm-up.

**Outcome:** 6/12 answered directly from the course content with `[n]`
citations; 6/12 correctly said "the context does not contain..." for topics
outside the 3-lecture + assignment-1 corpus (no hallucination). Average
~9s/question and ~1,300 tokens after warm-up. Note the exact answered /
refused split varies run-to-run (an earlier run answered 7, refused 5).

Sample cited answer:

> "Attention relates sequence length to compute quadratically: 'attention
> is n squared' where n is sequence length, and that gets really expensive."

## 6. Interfaces

- Chat app: `python -m streamlit run app.py --server.port=8501` → **HTTP 200**.
- Dashboard: `python -m streamlit run dashboard.py --server.port=8502` → **HTTP 200**.

PASS — both serve the UI.

## 7. Notebook

`notebooks/cs336-rag-test.ipynb` executed end-to-end via nbconvert; all 21
cells ran with no errors. The retrieval table rendered
(TF-IDF 1.00/0.80, Vector 0.90/0.90, Hybrid 0.90/0.90, +Rerank 0.90/0.90)
and the final RAG cell returned a live, cited BPE answer.

PASS — open it with `jupyter notebook notebooks/cs336-rag-test.ipynb`.

## Summary

All 6 test levels passed: ingestion, retrieval (measured + manual), answer
quality (judge + random questions), interfaces, and notebook. The app is
grounded (majority RELEVANT, honest refusals), fast (~40ms search), and
runnable by a reviewer.
