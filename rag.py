"""
RAG flow: rewrite -> search -> build prompt -> LLM -> built-in judge.

Simple functions, in the style of the course project example:
search(), build_prompt(), llm(), rag(), evaluate_relevance().
"""

import ast
import json
from time import sleep, time

import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (DATA_DIR, EMBED_MODEL, LLM_API_KEY, LLM_BASE_URL,
                    LLM_MODEL, RERANKER_MODEL, RERANK_TOP_K, TOP_K)
from minsearch import Index

PROMPT_TEMPLATE = """You are a CS336 TA. Answer the QUESTION using ONLY the CONTEXT.
Cite sources with [1], [2], etc. If the context includes timestamps, mention them.
If the context does not contain the answer, say so.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

EVAL_PROMPT_TEMPLATE = """You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as 'NON_RELEVANT', 'PARTLY_RELEVANT', or 'RELEVANT'.

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()

_index = None
_embed_model = None
_reranker = None


def load_index():
    """Load documents and build the in-memory index (with embeddings)."""
    with open(DATA_DIR / "documents.json", encoding="utf-8") as f:
        docs = json.load(f)
    idx = Index(text_fields=["content"], keyword_fields=["type"])
    idx.fit(docs)
    embed_model = SentenceTransformer(EMBED_MODEL)
    texts = [d["content"] for d in docs]
    idx.fit_embeddings(embed_model.encode(texts))
    return idx, embed_model


def get_index():
    global _index, _embed_model
    if _index is None:
        _index, _embed_model = load_index()
    return _index, _embed_model


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def chat(messages, model=LLM_MODEL, temperature=0.3, max_tokens=800, max_retries=3):
    """Call the LLM (OpenAI-compatible). Returns (answer, token_stats).

    max_retries handles rate limits (429): waits and retries with growing
    backoff. Set max_retries=1 to fail fast.
    """
    if not LLM_API_KEY:
        return "[Error: No API key]", {}
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "opencode/1.0",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(max_retries):
        resp = requests.post(f"{LLM_BASE_URL}/chat/completions",
                             json=payload, headers=headers, timeout=60)
        if resp.status_code == 429 and attempt < max_retries - 1:
            sleep(30 * (attempt + 1))
            continue
        resp.raise_for_status()
        data = resp.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
        if content:
            usage = data.get("usage", {})
            return content, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        if attempt < max_retries - 1:
            sleep(5 * (attempt + 1))  # empty answer: wait and retry
    raise RuntimeError("LLM returned an empty answer")


def rewrite_query(query):
    """Expand the query with synonyms for better retrieval."""
    if not LLM_API_KEY:
        return query
    try:
        rewritten, _ = chat(
            [{"role": "user", "content": f"Rewrite for better search. Add synonyms. Return ONLY the rewritten query.\n\nQuery: {query}"}],
            temperature=0.2, max_tokens=200, max_retries=1,
        )
        return rewritten if rewritten else query
    except Exception:
        return query


def search(query, num_results=TOP_K, rerank=True, rerank_k=RERANK_TOP_K):
    """Hybrid search with optional cross-encoder reranking."""
    index, embed_model = get_index()
    qv = embed_model.encode(query)
    hits = index.hybrid_search(query, qv, num_results)
    if rerank and hits:
        pairs = [(query, h["content"]) for h in hits]
        scores = get_reranker().predict(pairs)
        for h, s in zip(hits, scores):
            h["_rerank_score"] = float(s)
        hits = sorted(hits, key=lambda x: x["_rerank_score"], reverse=True)[:rerank_k]
    return hits


def build_prompt(query, hits):
    context = ""
    for i, hit in enumerate(hits, 1):
        meta = []
        if hit.get("timestamp") is not None:
            meta.append(f"timestamp={hit['timestamp']:.1f}s")
        if hit.get("video_id"):
            meta.append(f"video={hit['video_id']}")
        if hit.get("file_path"):
            meta.append(f"file={hit['file_path']}")
        context += f"[{i}] ({' | '.join(meta)})\n{hit['content']}\n\n"
    return PROMPT_TEMPLATE.format(question=query, context=context.strip())


def llm(prompt, model=LLM_MODEL, max_retries=3):
    """Call the LLM with a plain text prompt. Returns (answer, token_stats)."""
    return chat([{"role": "user", "content": prompt}], model=model,
                temperature=0.3, max_tokens=800, max_retries=max_retries)


def parse_json(text):
    """Best-effort JSON parsing: strips markdown fences, falls back to Python literals."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def evaluate_relevance(question, answer):
    """LLM-as-a-judge: classify the answer as RELEVANT / PARTLY_RELEVANT / NON_RELEVANT."""
    prompt = EVAL_PROMPT_TEMPLATE.format(question=question, answer=answer)
    evaluation, _ = llm(prompt)
    parsed = parse_json(evaluation)
    if parsed is None:
        return {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
    return {
        "Relevance": parsed.get("Relevance", "UNKNOWN"),
        "Explanation": parsed.get("Explanation", ""),
    }


def rag(query, rewrite=True, rerank=True, model=LLM_MODEL,
        num_results=TOP_K, rerank_k=RERANK_TOP_K):
    """Full RAG flow. Returns a dict with answer, hits, timing, and token usage."""
    t0 = time()
    q = rewrite_query(query) if rewrite else query
    hits = search(q, num_results=num_results, rerank=rerank, rerank_k=rerank_k)
    prompt = build_prompt(query, hits)
    answer, token_stats = llm(prompt, model=model)
    return {
        "query": query,
        "rewritten_query": q if rewrite else None,
        "hits": hits,
        "answer": answer,
        "model_used": model,
        "response_time": time() - t0,
        **token_stats,
    }
