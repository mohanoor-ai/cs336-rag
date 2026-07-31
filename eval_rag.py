"""
RAG evaluation with LLM-as-a-Judge.
Runs the RAG pipeline on the ground truth questions and classifies
each answer as RELEVANT / PARTLY_RELEVANT / NON_RELEVANT.

Needs an OPENCODE_API_KEY. Saves results to results/rag-eval.csv.
"""

import csv
import json
from time import sleep

from config import DATA_DIR
from rag import EVAL_PROMPT_TEMPLATE, llm, parse_json, rag


def main():
    with open(DATA_DIR.parent / "evaluation" / "eval_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        question = q["question"]
        result = rag(question)
        evaluation = llm(EVAL_PROMPT_TEMPLATE.format(
            question=question, answer=result["answer"]))[0]
        parsed = parse_json(evaluation)
        if parsed is None:
            relevance, explanation = "UNKNOWN", evaluation
        else:
            relevance = parsed.get("Relevance", "UNKNOWN")
            explanation = parsed.get("Explanation", "")
        results.append({
            "question": question,
            "answer": result["answer"],
            "relevance": relevance,
            "explanation": explanation,
        })
        print(f"{relevance:<18} {question}")
        sleep(2)  # stay under the free-tier rate limit

    total = len(results)
    counts = {}
    for r in results:
        counts[r["relevance"]] = counts.get(r["relevance"], 0) + 1
    print("\nDistribution:")
    for k, v in counts.items():
        print(f"  {k}: {v} ({v / total:.0%})")

    out = DATA_DIR.parent / "results" / "rag-eval.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "relevance", "explanation"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
