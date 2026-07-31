# 02 — How it works: the pipeline stage by stage

This page explains every stage of the project: what it looks like, what it
does, and what each stage is trying to achieve. The pipeline is:

```
ingest -> index -> search -> rerank -> generate -> monitor
 (1)      (2)      (3)       (4)        (5)         (6)
```

A question from the user flows through stages 3→5; stages 1, 2 and 6 are
"behind the scenes".

---

## Stage 1 — Ingestion (`ingest.py`)

**What we're trying to achieve:** turn raw, hard-to-search sources (videos
and a git repo) into a flat list of searchable text chunks, each with enough
metadata to cite later.

**What it looks like:**

```python
# fetch YouTube transcripts -> 30-second segments
# fetch GitHub files (only .py, .md, .txt, ...) -> raw text
# chunk_text() splits each piece of text into ~512-word chunks (128 overlap)
# every chunk becomes: {"content", "source", "type", "timestamp", ...}
# all chunks saved to data/documents.json
```

**Why chunk?** An LLM prompt and a search index both work with small pieces
of text. A whole lecture is too long to retrieve and too long to stuff into
a prompt. Overlapping chunks mean a topic near a boundary is not lost.

**Result:** 3,740 chunks (477 transcript + 3,263 code). Each has a clickable
`source` — a YouTube URL with `&t=<seconds>s`, or a GitHub file URL.

---

## Stage 2 — Indexing (`minsearch.py`)

**What we're trying to achieve:** make search fast by pre-computing two
numeric representations of every chunk, once, instead of comparing text on
every query.

**What it looks like:**

```python
idx = Index(text_fields=["content"], keyword_fields=["type"])
idx.fit(docs)                      # builds the TF-IDF matrix
idx.fit_embeddings(embeddings)     # stores precomputed vectors
```

Two matrices are built:

| Matrix | What it stores | Used by |
|--------|----------------|---------|
| TF-IDF | How important each word is in each chunk | `search()` — keyword search |
| Embeddings | A 384-dim vector per chunk (MiniLM) | `vector_search()` — semantic search |

Both are just numpy arrays, so search is cosine similarity. There is no
database.

---

## Stage 3 — Search (hybrid, RRF)

**What we're trying to achieve:** find the best candidate chunks for a
question. Because no single method is perfect, we run two and fuse the
rankings.

**The three search functions in `minsearch.py`:**

1. `search(query)` — **TF-IDF keyword search.** Cosine similarity between
   the query vector and the chunk matrix. Good for exact terms ("bpe").
2. `vector_search(query_vec)` — **semantic search.** Cosine similarity in
   embedding space. Good for meaning ("byte pair encoding" ↔ "BPE").
3. `hybrid_search(query, query_vec)` — **both, fused with RRF.**

**Reciprocal Rank Fusion (RRF):** each chunk gets `1 / (60 + rank)` from each
list. A chunk ranked high in either list scores well; a chunk ranked high in
**both** lists gets a boost. This is why hybrid usually beats each method
alone.

`search()` and `vector_search()` also accept `filter_dict` (e.g.
`{"type": "code"}`) for exact-match filtering on keyword fields.

**What we're trying to achieve at the end:** a shortlist of ~10 candidates
that contains the right chunks even if the query is sloppy.

---

## Stage 4 — Reranking (`rag.py` / cross-encoder)

**What we're trying to achieve:** put the most relevant chunk first.

Hybrid search scores each chunk independently of the query. A
**cross-encoder** scores the (query, chunk) pair *together*, which is much
more precise. In `rag.py`:

```python
hits = index.hybrid_search(query, qv, TOP_K)          # ~10 candidates
scores = reranker.predict([(query, h["content"]) for h in hits])
hits = top RERANK_TOP_K by score                      # keep best 5
```

**Trade-off:** reranking adds a few hundred milliseconds (the cross-encoder
is a neural model). We only run it on the ~10 candidates, not all 3,740.

---

## Stage 5 — Generation (`rag.py`)

**What we're trying to achieve:** turn retrieved chunks into a correct,
grounded, citable answer.

The `rag()` function in `rag.py` does four things:

1. **Rewrite the query (optional).** Ask the LLM to expand the question with
   synonyms ("BPE tokenization" → "byte pair encoding, merges, vocabulary")
   so retrieval works better.
2. **Search + rerank** (stages 3–4).
3. **Build the prompt** with `build_prompt()`. Each chunk becomes a numbered
   block with its metadata:

   ```
   [1] (timestamp=3698.0s | video=JuoVZkPBiKk)
   BPE starts with each byte as a token and iteratively merges...
   ```
4. **Call the LLM** with `llm()`. The prompt says: answer using ONLY the
   context, cite sources with [1], [2], and say so if the context does not
   contain the answer.

The result dict includes the answer, the hits, the response time, and token
usage — which the app logs for monitoring.

**What we're trying to achieve:** answers that are grounded in the course
material and verifiable through citations, instead of a model guessing.

---

## Stage 6 — Interface and monitoring (`app.py`, `dashboard.py`)

**What the Streamlit chat app (`app.py`) is supposed to do and how:**

- Show a chat box where you type a question.
- Sidebar toggles: turn query rewriting and reranking on/off, choose how
  many chunks to retrieve (top-k).
- On each question it calls `rag()`, shows the answer, and lists the sources
  in an expander — each source is a clickable link.
- **Logs** every conversation to `data/conversations.jsonl` (query, answer,
  latency, tokens, model) and thumbs up/down feedback to
  `data/feedback.jsonl`.

**What the dashboard (`dashboard.py`) does and how:**

- Reads `conversations.jsonl` and `feedback.jsonl`.
- Shows 5 metric cards + charts: query volume over time, latency
  distribution, feedback distribution, query type breakdown, tokens vs
  latency, top queries, and a strategy comparison table.
- If there is no data yet, it generates synthetic demo data so the charts
  are never empty.

**What we're trying to achieve:** see how the app is used in the real world
— how often, how slow, what users ask, and whether they like the answers.

---

## How the pieces share code

`app.py`, `ask.py` (CLI) and `search.py` (debug CLI) all call functions in
`rag.py`. The model loading is cached, so the first call loads the models
and every later call reuses them. There is no duplicated pipeline code —
that keeps the project small enough to read in one sitting.
