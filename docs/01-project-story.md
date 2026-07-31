# 01 — Project story: how CS336-RAG was created

## Why this project

I'm taking the DataTalks **LLM Zoomcamp 2026**. The capstone is to build an
end-to-end RAG (or agent) app: pick a dataset, ingest it, build a retrieval
flow, evaluate it, add an interface, and monitor it.

I was also following Stanford's **CS336 — Language Modeling from Scratch**
(Spring 2026). The course has 19 lectures on YouTube and a lot of code on
GitHub, but the content is hard to search: you scrub through hours of video
or read hundreds of files to find one topic.

So the two interests came together: **build a RAG assistant for CS336**
that answers questions in natural language and points to the exact video
timestamp or code file. That is what CS336-RAG is.

## What the project does

You ask a question, for example:

> "How does BPE tokenization work?"

The app returns an answer written by an LLM, **with citations**:

> BPE starts with each byte as a token and iteratively merges the most
> frequent adjacent pair of tokens... [1]
>
> [1] Lecture 1, at 1:01:38 — https://youtube.com/watch?v=JuoVZkPBiKk&t=3698s

Every claim links back to a lecture timestamp or a file in the repo, so the
answer can be verified.

## What it achieves

- **Answers questions about CS336** from 3,740 chunks of lecture transcripts
  and course code.
- **Citations you can click** — timestamps for lectures, file paths for code.
- **Runs anywhere with `pip install`** — no database, no Docker required.
  Everything is in memory.
- **Shows the whole RAG lifecycle**: ingestion, hybrid search, reranking,
  LLM generation, evaluation, UI, and monitoring.

## How it was created (the steps I followed)

### 1. Followed the course modules, then picked a dataset

The course says: pick a corpus you care about. I used:

- **YouTube transcripts** (3 lectures, 477 timestamped segments) via
  `youtube-transcript-api`
- **GitHub code** (the `assignment1-basics` repo, 3,263 chunks) via the
  GitHub API

### 2. Ingestion first (`ingest.py`)

I wrote `ingest.py` to download the transcripts and the repo, split the text
into ~512-word chunks with overlap, and save everything to
`data/documents.json`. Each chunk keeps its `source` URL, so citations work
later.

### 3. Built the search index from scratch (`minsearch.py`)

Following module 1, I built an in-memory `Index` with TF-IDF (scikit-learn).
It needs no database — just a matrix and cosine similarity. Later (module 2)
I added embeddings so the index also does semantic search.

### 4. Hybrid search + reranking (module 6)

Neither keyword nor vector search is perfect:

- TF-IDF matches exact words (good for code: "bpe", "flops").
- Vector search matches meaning (good for transcripts where the same idea is
  phrased differently).

So I combined both with **Reciprocal Rank Fusion (RRF)** in
`minsearch.hybrid_search`, and added a **cross-encoder reranker** on top.

### 5. Evaluation (module 4)

I wrote `eval.py` to measure retrieval with **Hit Rate and MRR** on 10
ground-truth questions, comparing all four strategies (TF-IDF, vector,
hybrid, hybrid + rerank). This told me which strategy the app should use.

I later added `eval_rag.py`, which scores final answers with an
**LLM-as-a-Judge** (`RELEVANT / PARTLY_RELEVANT / NON_RELEVANT`).

### 6. The LLM part (`rag.py`)

`rag.py` ties it together: optional query rewriting → hybrid search →
rerank → build a prompt with citations → call the LLM → return the answer.
I use **DeepSeek V4 Flash** through **OpenCode Go**, an OpenAI-compatible
endpoint that is fast, reliable, and low cost. The LLM call is a simple
`POST /chat/completions` with the API key from `.env`, so no extra SDK is
needed.

### 7. Interface and monitoring (modules 5 and 7)

- **`app.py`** — a Streamlit chat UI. It logs every conversation and thumbs
  up/down feedback to JSONL files.
- **`dashboard.py`** — a Streamlit dashboard that reads those logs and shows
  charts (queries, latency, feedback, token usage...).

### 8. Cleanup and polish

- Removed duplicated pipeline code (the app and the CLI now share `rag.py`).
- Added `Dockerfile` + `docker-compose.yaml` for containerization.
- Pinned all dependencies in `requirements.txt`.

## What I learned along the way

1. **Search results must merge on a unique document id.** Every chunk of a
   file shares the same `source` URL, so merging search results by `source`
   would keep only one chunk per file. Merging on a unique id (`_id`) keeps
   every chunk reachable.
2. **The right metric matters.** Hit Rate and MRR (from the course) tell you
   whether the right *document* was found, which keyword overlap could not.
3. **Keep the LLM call simple.** An OpenAI-compatible `chat/completions` call
   with a key from `.env` is easy to understand, swap, and debug.
4. **Log real usage.** The chat app writes every conversation and the
   dashboard reads those logs, so the charts always reflect real usage.

## The final structure

```
cs336-rag/
├── app.py              # Streamlit chat UI (logs conversations + feedback)
├── rag.py              # RAG flow: rewrite -> search -> prompt -> LLM -> judge
├── ask.py              # RAG pipeline CLI
├── search.py           # Hybrid search CLI (no LLM)
├── eval.py             # Retrieval evaluation (Hit Rate + MRR)
├── eval_rag.py         # RAG evaluation (LLM-as-a-Judge)
├── ingest.py           # Data ingestion (YouTube + GitHub)
├── minsearch.py        # In-memory search index (TF-IDF + vector + hybrid)
├── config.py           # Settings
├── dashboard.py        # Monitoring dashboard (reads real logs)
├── notebooks/cs336-rag-test.ipynb  # Executable walkthrough of the pipeline
├── evaluation/eval_questions.json  # 10 evaluation questions
├── data/               # Ingested documents + JSONL logs
├── Dockerfile          # Container image
├── docker-compose.yaml # Run app + dashboard
├── requirements.txt    # Pinned dependencies
└── .env.example        # Config template
```

See [02 — How it works](02-how-it-works.md) for what each stage does, and
[04 — Student guide](04-student-guide.md) for how to run and extend it.
