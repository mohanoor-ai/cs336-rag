# 04 — Student guide: what you should know

This page is written for a student (or a reviewer) who wants to understand
the project quickly: what each file does, how it maps to the course, how to
run it, and the key ideas behind it.

## Repo map — every file explained

| File | What it does | Course module |
|------|--------------|---------------|
| `config.py` | All settings in one place: data paths, model names, top-k values, video ids. | everywhere |
| `ingest.py` | Downloads YouTube transcripts + GitHub files, chunks them, saves `data/documents.json`. | 3 / 7 |
| `minsearch.py` | The in-memory search index: TF-IDF, vector search, hybrid (RRF), filters. | 1, 2, 6 |
| `rag.py` | The shared RAG flow: `search()`, `build_prompt()`, `llm()`, `rewrite_query()`, `evaluate_relevance()`, `rag()`. | 1, 4, 6, 7 |
| `ask.py` | CLI: `python ask.py "question"` → full RAG answer + sources. | 1 |
| `search.py` | CLI: `python search.py "question"` → search results only, no LLM. Handy for debugging retrieval. | 1, 2 |
| `eval.py` | Retrieval evaluation: Hit Rate + MRR across 4 strategies. | 4 |
| `eval_rag.py` | Answer evaluation: LLM-as-a-Judge. | 4 |
| `app.py` | Streamlit chat UI. Logs conversations + feedback. | 5 |
| `dashboard.py` | Streamlit monitoring dashboard. Reads the logs. | 5 |
| `evaluation/eval_questions.json` | The 10 ground-truth questions. | 4 |
| `notebooks/cs336-rag-test.ipynb` | Executable step-by-step walkthrough of the pipeline. | all |
| `Makefile` | Shortcuts: `make install`, `make ingest`, `make run`, `make eval`... | — |
| `Dockerfile`, `docker-compose.yaml` | Containerization. | 7 |

## Key ideas you should understand

**1. Chunking.** Long content is cut into ~512-word pieces with overlap.
Retrieval and prompts both work on small pieces, so chunking is the first
step of any RAG system.

**2. TF-IDF vs embeddings.** TF-IDF scores word importance and matches exact
words. Embeddings map text to vectors and match meaning. One is not "better"
— they complement each other.

**3. Hybrid search + RRF.** Run both searches and merge the two ranked lists
with Reciprocal Rank Fusion: each chunk scores `1/(60+rank)` per list, so
being high in either list helps and high in both helps most.

**4. Reranking.** A cross-encoder scores each (query, chunk) pair together.
It is more accurate than the earlier stages, so it is applied to the top ~10
candidates only.

**5. Hit Rate and MRR.** Hit Rate = did any relevant chunk show up? MRR = how
high was the first relevant chunk ranked? These tell you if retrieval works
*before* you spend money on LLM calls.

**6. LLM-as-a-Judge.** A second LLM call grades the answer
(`RELEVANT / PARTLY_RELEVANT / NON_RELEVANT`). It is a cheap, rough way to
evaluate answer quality without hand-writing reference answers.

**7. The RAG prompt.** The answer is only as good as the prompt + context.
The prompt says: use ONLY the context, cite `[1]`, `[2]`, and admit when the
context does not contain the answer.

## How to run everything

```bash
# 1. set up the API key
cp .env.example .env            # choose a provider and add its key
#   LLM_PROVIDER=opencode  -> set OPENCODE_API_KEY (DeepSeek V4 Flash)
#   LLM_PROVIDER=openai    -> set OPENAI_API_KEY (GPT-4o-mini)

# 2. install
pip install -r requirements.txt

# 3. build the dataset (downloads transcripts + repo)
python ingest.py

# 4. try things
python search.py "How does BPE tokenization work?"   # retrieval only
python ask.py   "What is the Chinchilla scaling law?" # full RAG

# 5. evaluate
python eval.py                                       # Hit Rate + MRR
python eval_rag.py                                   # LLM-as-a-Judge

# 6. run the UI + dashboard
streamlit run app.py
streamlit run dashboard.py

# 7. containerized
docker compose up -d
```

Or use the Makefile: `make install`, `make ingest`, `make search q="..."`,
`make ask q="..."`, `make eval`, `make eval-rag`, `make run`, `make dashboard`.

## What the notebook demonstrates

`notebooks/cs336-rag-test.ipynb` runs the whole pipeline step by step:
load the data → build the index → try TF-IDF, vector, hybrid, rerank →
measure Hit Rate / MRR on the 10 questions → run a full RAG answer. It is
meant to be executed top to bottom. Run it with:

```bash
jupyter notebook notebooks/cs336-rag-test.ipynb
```

## Troubleshooting

- **`Index ready` takes a while the first time** — the sentence-transformers
  and cross-encoder models download on first use, then stay cached.
- **`[Error: No API key]`** — the key for your chosen provider is missing
  in `.env` (`OPENCODE_API_KEY` for `LLM_PROVIDER=opencode`,
  `OPENAI_API_KEY` for `LLM_PROVIDER=openai`).
- **`429 Too Many Requests`** — you hit a rate limit; the code retries
  automatically with backoff, so the request usually goes through.
- **`Run ingest.py first`** — `data/documents.json` is missing; run
  `python ingest.py`.
- **Dashboard shows "synthetic demo data"** — that is by design when there
  are no real logs yet. Ask questions in the chat app, then reload the
  dashboard.

## How to extend it (if you keep going)

- **More data:** add video ids to `CS336_VIDEO_IDS` and repos to
  `CS336_REPOS` in `config.py`, re-run `ingest.py`.
- **Better evaluation:** make the ground truth chunk-level (exact chunk ids)
  instead of video/file level; add a second model to `eval_rag.py` and
  compare.
- **Different LLM:** set `LLM_PROVIDER` in `.env` to `opencode` (DeepSeek V4
  Flash) or `openai` (GPT-4o-mini) and add the matching key; or override
  `LLM_MODEL`.
- **Real monitoring:** point the dashboard at a database instead of JSONL
  files, and add alerting.

## Reproducibility notes

- All dependency versions are pinned in `requirements.txt`.
- `.env.example` documents the required variable for each provider
  (`OPENCODE_API_KEY` or `OPENAI_API_KEY`, selected by `LLM_PROVIDER`).
- `ingest.py` rebuilds the dataset from public sources, so `data/` does not
  need to be committed.
- The evaluation numbers in `README.md` and `docs/03-evaluation.md` were
  produced by `python eval.py` and `python eval_rag.py` on this dataset, and
  can be reproduced by running those commands.
