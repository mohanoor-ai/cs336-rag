"""
RAG pipeline CLI: query rewriting -> hybrid search -> reranking -> answer.
"""

import argparse

from rag import rag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    result = rag(args.query, rewrite=not args.no_rewrite, rerank=not args.no_rerank)
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for i, hit in enumerate(result["hits"], 1):
        print(f"  [{i}] {hit['source']}")


if __name__ == "__main__":
    main()
