"""
Random question testing for the RAG pipeline.

Runs a set of varied questions through the full pipeline (rewrite ->
search -> rerank -> LLM) and records the answers, sources, latency,
and token usage for every question.

Results are appended to results/random-questions.jsonl.

Usage:
    python scripts/random_test.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import rag

QUESTIONS = [
    "How does the optimizer update model parameters in training?",
    "Why is positional encoding important in transformers?",
    "What does the cross-entropy loss measure?",
    "How does attention relate sequence length to compute?",
    "What is the role of softmax in the attention mechanism?",
    "How do you handle out-of-vocabulary words in tokenization?",
    "Why does the course prefer building models from scratch?",
    "What are the benefits of pre-training a language model?",
    "How is memory usage estimated for large transformers?",
    "What is the difference between a tokenizer vocabulary and a merge list?",
    "How should you split data for training, validation, and testing?",
    "What happens during the backward pass in training?",
]

OUT = Path(__file__).resolve().parent.parent / "results" / "random-questions.jsonl"


def main():
    results = []
    print(f"{'#':<3} {'Latency':<9} {'Tokens':<7} {'Hits':<5} Question")
    print("-" * 95)
    for i, q in enumerate(QUESTIONS, 1):
        r = rag(q)
        tokens = r.get("total_tokens", 0)
        print(f"{i:<3} {r['response_time']:<9.2f}s {tokens:<7} {len(r['hits']):<5} {q}")
        results.append({
            "timestamp": datetime.now().isoformat(),
            "question": q,
            "rewritten_query": r.get("rewritten_query"),
            "answer": r["answer"],
            "response_time": round(r["response_time"], 3),
            "total_tokens": tokens,
            "model_used": r["model_used"],
            "sources": [h["source"] for h in r["hits"]],
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    avg_latency = sum(x["response_time"] for x in results) / len(results)
    avg_tokens = sum(x["total_tokens"] for x in results) / len(results)
    print(f"\n{len(results)} questions | avg latency {avg_latency:.2f}s | avg tokens {avg_tokens:.0f}")
    print(f"Saved results to {OUT}")


if __name__ == "__main__":
    main()
