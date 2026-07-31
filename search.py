"""
CLI for testing hybrid search without an LLM.
"""

import argparse

from config import TOP_K
from rag import search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    hits = search(args.query, num_results=TOP_K, rerank=not args.no_rerank)
    print(f"Results: {len(hits)}")

    for i, hit in enumerate(hits, 1):
        score = hit.get("_rerank_score", hit.get("_score", 0))
        print(f"--- {i} (score: {score:.3f}) ---")
        print(f"Source: {hit['source']}")
        if hit.get("timestamp"):
            print(f"Time: {hit['timestamp']:.1f}s")
        print(f"{hit['content'][:200]}...\n")


if __name__ == "__main__":
    main()
