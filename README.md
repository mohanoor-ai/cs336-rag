# CS336-RAG

**Mohammed Sheikh-Noor**
**Course: DataTalks LLM Zoomcamp 2026**

A RAG application that answers questions about Stanford's CS336 course
(Language Modeling from Scratch). Searches lecture transcripts and code
from the course GitHub repo using an in-memory search index — no external
database required.

## Documentation

| Doc | Contents |
|-----|----------|
| [01 — Project story](docs/01-project-story.md) | How the project was created, what it does and achieves |
| [02 — How it works](docs/02-how-it-works.md) | The pipeline stage by stage (ingest → index → search → rerank → generate → monitor) |
| [03 — Evaluation](docs/03-evaluation.md) | Test criteria, what we test, metrics, results |
| [04 — Student guide](docs/04-student-guide.md) | Repo map, key ideas, how to run, troubleshooting |
| [05 — Testing strategy](docs/05-testing-strategy.md) | What we test, how, and the results |

---

## Problem

Stanford CS336 has 19 lectures on YouTube and code on GitHub. Finding
specific topics means scrubbing through long videos or reading a lot
of code. This app lets you ask questions in natural language and get
answers with citations to the exact video timestamp or file.

**Course:** [Stanford CS336](https://cs336.stanford.edu/) (Spring 2026)
**Playlist:** https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV
**GitHub:** https://github.com/stanford-cs336/lectures

---

## Dataset

| Source | Content |
|--------|---------|
| YouTube lectures 1-3 | Overview & Tokenization (Percy), PyTorch & Resource Accounting (Percy), Architectures & Hyperparameters (Tatsu) — 477 timestamped segments |
| GitHub: stanford-cs336/assignment1-basics | Student assignment code — BPE tokenizer, transformer implementation, training loop (3,263 chunks) |

**Total: 3,740 chunks**

---

## How it works

Ask a question → LLM rewrites it → Hybrid search (TF-IDF + vector embeddings → RRF fusion) → Cross-encoder reranking → LLM generates cited answer

### Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 384-dim vectors, runs locally, no API cost |
| Vector search | scikit-learn cosine similarity | Simple, no external service needed |
| Keyword search | TF-IDF (scikit-learn) | In-memory, fast for small datasets |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) | Combines keyword + vector rankings |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Improves precision on top results |
| LLM | DeepSeek V4 Flash (OpenCode Go) | Fast, OpenAI-compatible, low cost |
| UI | Streamlit | Single framework for chat + dashboard |
| Data store | In-memory (JSON + numpy) | No external database — just pip install (Docker optional) |

---

## Quickstart

```bash
# 1. Set up your API key
cp .env.example .env
# Edit .env and add your OPENCODE_API_KEY

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch and index data
python ingest.py

# 4. Start the chat UI
streamlit run app.py

# 5. Monitoring dashboard
streamlit run dashboard.py

# Or with Docker Compose (app + dashboard)
docker compose up -d
```

---

## Testing

```bash
# Search without LLM
python search.py "How does BPE tokenization work?"

# Full RAG pipeline
python ask.py "What is the Chinchilla scaling law?"

# Retrieval evaluation (Hit Rate + MRR)
python eval.py

# RAG evaluation (LLM-as-a-Judge) — needs API key
python eval_rag.py

# Random questions (12 varied, records latency/tokens/sources)
python scripts/random_test.py

# Interactive notebook
jupyter notebook notebooks/cs336-rag-test.ipynb
```

See [docs/05-testing-strategy.md](docs/05-testing-strategy.md) for the full
testing strategy and the recorded results.

---

## Evaluation Results

Retrieval quality measured with **Hit Rate** and **MRR** on 10 ground-truth
questions (evaluation/eval_questions.json):

| Strategy | Hit Rate | MRR | Latency |
|----------|----------|-----|---------|
| BM25 (TF-IDF) | 1.00 | 0.80 | 0.02s |
| Vector (semantic) | 0.90 | 0.90 | 0.03s |
| **Hybrid (TF-IDF + Vector)** | **0.90** | **0.90** | **0.05s** |
| Hybrid + Rerank | 0.90 | 0.90 | 0.78s |

**Hybrid search wins**: it matches the vector strategy's MRR while staying
keyword-aware (the BM25 route found relevant chunks for every question).
Reranking trades ~15x latency for cleaner top results. Answer quality is
scored separately with LLM-as-a-Judge via `python eval_rag.py`
(RELEVANT / PARTLY_RELEVANT / NON_RELEVANT): **8/10 answers were RELEVANT**,
1 PARTLY_RELEVANT, 1 NON_RELEVANT (results/rag-eval.csv).

---

## Project Structure

```
cs336-rag/
├── app.py              # Streamlit chat UI (logs conversations + feedback)
├── rag.py              # RAG flow: rewrite → search → prompt → LLM → judge
├── ask.py              # RAG pipeline CLI
├── search.py           # Hybrid search CLI (no LLM)
├── eval.py             # Retrieval evaluation (Hit Rate + MRR)
├── eval_rag.py         # RAG evaluation (LLM-as-a-Judge)
├── ingest.py           # Data ingestion (YouTube + GitHub)
├── minsearch.py        # In-memory search index (TF-IDF + vector + hybrid)
├── config.py           # Settings
├── dashboard.py        # Monitoring dashboard (reads real logs)
├── notebooks/cs336-rag-test.ipynb  # Test notebook
├── evaluation/eval_questions.json   # 10 evaluation questions
├── data/               # Ingested documents + JSONL logs
├── images/             # Dashboard screenshot
├── requirements.txt    # Pinned dependencies
├── Dockerfile          # Container image
├── docker-compose.yaml # Run app + dashboard
├── .env.example        # Config template
└── .gitignore
```

---

## Monitoring

Dashboard at `streamlit run dashboard.py` — 8 charts tracking query
volume, latency, feedback, query types, token usage, top queries,
and strategy performance. The chat app logs every conversation to
`data/conversations.jsonl` (query, answer, latency, tokens, model)
and thumbs feedback to `data/feedback.jsonl`; the dashboard reads
these real logs, falling back to synthetic demo data when empty.

![Dashboard](images/dashboard1.jpg)

---

## Stack choices and reasons

| Choice | Reason |
|--------|--------|
| **In-memory search (`minsearch`) instead of a vector database** | 3,740 chunks fit in memory; search is cosine similarity over numpy matrices. A vector database is needed for larger datasets, not here. |
| **Hybrid search (TF-IDF + vector + RRF) instead of one method** | TF-IDF matches exact terms, vector matches meaning; RRF combines both. Measured best MRR (0.90) in `eval.py` at ~60ms. |
| **Cross-encoder reranking** | Scores each (query, chunk) pair together, so the best chunk ranks first. ~0.7s on only the top ~10 candidates. |
| **sentence-transformers (MiniLM + ms-marco cross-encoder)** | Runs locally, no GPU, no API cost. 384-dim vectors are enough for this corpus. |
| **DeepSeek V4 Flash via OpenCode Go** | OpenAI-compatible `chat/completions`, low cost, no extra SDK — just `requests`. |
| **Streamlit for chat and dashboard** | One framework for UI and monitoring (allowed by the course); no front-end build step. |
| **JSONL logs instead of PostgreSQL** | Postgres is needed for large-scale, multi-user logging; this single-user course app does not require it. JSONL files record conversations and feedback for the dashboard. |
| **Plain Python ingestion script** | The dataset is small and changes rarely. No orchestration tool needed. |
| **Docker (optional)** | Provided for portability; the app also runs with plain `pip install` — Docker not required. |

Evaluation criteria are documented in [docs/03-evaluation.md](docs/03-evaluation.md).
