# CS336-RAG Test Results

**Date:** 2026-07-31
**Environment:** Python 3.14, in-memory search (no external database)
**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
**Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
**LLM:** DeepSeek V4 Flash via OpenCode Go (`https://opencode.ai/zen/go/v1`)
**Test strategy:** see `docs/05-testing-strategy.md`

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
| BM25 (TF-IDF) | 1.00 | 0.80 | 0.03s |
| Vector (semantic) | 0.90 | 0.90 | 0.03s |
| **Hybrid (TF-IDF + Vector)** | **0.90** | **0.90** | **0.06s** |
| Hybrid + Rerank | 0.90 | 0.90 | 0.80s |

PASS — hybrid matches the best MRR (0.90) at ~60ms, so the app uses hybrid.

## 3. Manual retrieval (`python search.py "RoPE positional embeddings"`)

Returned 5 reranked hits from Lecture 2 (`lVynu4bo1rY`, t=1944s, t=1976s,
t=5211s, ...) — all directly about RoPE. PASS.

## 4. RAG evaluation (`python eval_rag.py`)

LLM-as-a-Judge on the same 10 questions:

| Relevance | Count | Share |
|-----------|-------|-------|
| RELEVANT | 8 | 80% |
| PARTLY_RELEVANT | 1 | 10% |
| NON_RELEVANT | 1 | 10% |

PASS — 8/10 fully relevant. Full rows in `results/rag-eval.csv`.

## 5. Random questions (`python scripts/random_test.py`)

12 varied questions ran through the full pipeline
(rewrite -> search -> rerank -> LLM). Every question returned **5 cited
sources**. Results saved to `results/random-questions.jsonl`.

| # | Question | Latency | Tokens |
|---|----------|---------|--------|
| 1 | How does the optimizer update model parameters in training? | 257.4s* | 992 |
| 2 | Why is positional encoding important in transformers? | 6.5s | 1045 |
| 3 | What does the cross-entropy loss measure? | 7.4s | 908 |
| 4 | How does attention relate sequence length to compute? | 7.6s | 1036 |
| 5 | What is the role of softmax in the attention mechanism? | 9.1s | 1517 |
| 6 | How do you handle out-of-vocabulary words in tokenization? | 8.8s | 993 |
| 7 | Why does the course prefer building models from scratch? | 10.3s | 1192 |
| 8 | What are the benefits of pre-training a language model? | 6.5s | 888 |
| 9 | How is memory usage estimated for large transformers? | 6.9s | 1024 |
| 10 | What is the difference between a tokenizer vocabulary and a merge list? | 9.1s | 3251 |
| 11 | How should you split data for training, validation, and testing? | 5.7s | 844 |
| 12 | What happens during the backward pass in training? | 7.8s | 1203 |

\* First call includes model warm-up.

**Outcome:** 6/12 answered directly from the course content with `[n]`
citations; 6/12 correctly said "the context does not contain..." for topics
outside the 3-lecture + assignment-1 corpus (no hallucination). Average
~7s/question and ~1,240 tokens after warm-up.

Sample cited answer:

> "Attention is quadratic in sequence length: as stated in [2]
> (timestamp=1825.8s), 'attention is n squared' where n is sequence length,
> 'and that gets really expensive.'"

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
grounded (80% RELEVANT, honest refusals), fast (~60ms search), and runnable
by a reviewer.
