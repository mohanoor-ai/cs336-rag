"""
Retrieval evaluation: Hit Rate and MRR for different search strategies.
Uses ground truth questions from evaluation/eval_questions.json.

Course style: the evaluate() function checks whether a relevant document
appears in the top-k results, and how high it is ranked.
"""

import json
import time

from sentence_transformers import SentenceTransformer, CrossEncoder

from config import DATA_DIR, EMBED_MODEL, RERANKER_MODEL, RERANK_TOP_K, TOP_K
from minsearch import Index


def load_data():
    with open(DATA_DIR / "documents.json", encoding="utf-8") as f:
        docs = json.load(f)
    with open(DATA_DIR.parent / "evaluation" / "eval_questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    return docs, questions


def build_index(docs, embed_model):
    idx = Index(text_fields=["content"], keyword_fields=["type"])
    idx.fit(docs)
    texts = [d["content"] for d in docs]
    idx.fit_embeddings(embed_model.encode(texts))
    return idx


def is_relevant(hit, question):
    """A hit is relevant if it matches one of the expected sources in the ground truth."""
    for src in question.get("expected_sources", []):
        if hit["type"] != src.get("type"):
            continue
        if src.get("video_id") and hit.get("video_id") != src.get("video_id"):
            continue
        if src.get("file_path") and hit.get("file_path") != src.get("file_path"):
            continue
        return True
    return False


def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt += 1
    return cnt / len(relevance_total)


def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank]:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate(ground_truth, search_function):
    relevance_total = []
    latencies = []
    for q in ground_truth:
        start = time.time()
        hits = search_function(q["question"])
        latencies.append(time.time() - start)
        relevance_total.append([is_relevant(h, q) for h in hits])
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
        "avg_latency": sum(latencies) / len(latencies),
    }


def rerank_hits(query, hits, reranker, k=RERANK_TOP_K):
    if not hits:
        return []
    pairs = [(query, h["content"]) for h in hits]
    scores = reranker.predict(pairs)
    for h, s in zip(hits, scores):
        h["_rerank_score"] = float(s)
    return sorted(hits, key=lambda x: x["_rerank_score"], reverse=True)[:k]


def main():
    docs, questions = load_data()
    print(f"Loaded {len(questions)} questions, {len(docs)} docs")

    embed_model = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    idx = build_index(docs, embed_model)

    strategies = {
        "bm25": lambda q: idx.search(q, num_results=TOP_K),
        "vector": lambda q: idx.vector_search(embed_model.encode(q), num_results=TOP_K),
        "hybrid": lambda q: idx.hybrid_search(q, embed_model.encode(q), TOP_K),
        "hybrid+rerank": lambda q: rerank_hits(
            q, idx.hybrid_search(q, embed_model.encode(q), TOP_K), reranker),
    }

    results = {name: evaluate(questions, fn) for name, fn in strategies.items()}

    print(f"\n{'Strategy':<20} {'Hit Rate':<10} {'MRR':<8} {'Latency':<10}")
    print("-" * 48)
    for name in strategies:
        r = results[name]
        print(f"{name:<20} {r['hit_rate']:<10.3f} {r['mrr']:<8.3f} {r['avg_latency']:<10.3f}s")


if __name__ == "__main__":
    main()
