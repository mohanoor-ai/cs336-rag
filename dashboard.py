"""
Monitoring dashboard for CS336-RAG.

Reads real conversations and user feedback logged by app.py
(data/conversations.jsonl and data/feedback.jsonl).
When there is no data yet, generates synthetic data for the demo.

Charts: 5 metric cards, query volume, latency distribution,
feedback distribution, query type breakdown, token usage vs latency,
top queries, and a retrieval strategy comparison table.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CS336-RAG Dashboard", page_icon="📊", layout="wide")

CONVERSATIONS_FILE = Path("data") / "conversations.jsonl"
FEEDBACK_FILE = Path("data") / "feedback.jsonl"

st.title("CS336-RAG Monitoring Dashboard")
st.markdown("Real-time metrics, user feedback, and retrieval performance.")


def read_jsonl(path):
    records = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def classify_query(query):
    query = query.lower()
    code_words = ["code", "function", "implement", "python", "class", "loop", "kernel", "bpe"]
    concept_words = ["token", "loss", "attention", "transformer", "model", "scaling",
                     "embed", "flops", "chinchilla"]
    if any(w in query for w in code_words):
        return "Code"
    if any(w in query for w in concept_words):
        return "Concepts"
    return "Other"


def load_data():
    """Real data from JSONL logs; synthetic demo data when there's nothing yet."""
    conversations = read_jsonl(CONVERSATIONS_FILE)
    feedback = read_jsonl(FEEDBACK_FILE)

    synthetic = not conversations
    if synthetic:
        now = datetime.now()
        sample_queries = [
            "How does BPE tokenization work?",
            "FlashAttention memory usage",
            "KV cache implementation",
            "What is the Chinchilla scaling law?",
            "Adam optimizer beta values",
            "RoPE positional embeddings",
        ]
        for i in range(24):
            t = now - timedelta(hours=23 - i)
            conversations.append({
                "timestamp": t.isoformat(),
                "query": random.choice(sample_queries),
                "answer": "",
                "response_time": round(random.uniform(0.3, 3.0), 2),
                "total_tokens": random.randint(200, 2000),
                "num_hits": random.randint(3, 10),
            })

    df = pd.DataFrame(conversations)
    df["time"] = pd.to_datetime(df["timestamp"])
    df["type"] = df["query"].apply(classify_query)
    return df, conversations, feedback, synthetic


df, conversations, feedback, synthetic = load_data()

if synthetic:
    st.info("No real data yet — showing synthetic demo data. Ask questions in the chat app to see real metrics.")

# ---- Row 1: Key Metrics (5 metric cards) ----
fb_counts = {"positive": 0, "neutral": 0, "negative": 0}
for fb in feedback:
    score = fb.get("feedback", 0)
    if score == 1:
        fb_counts["positive"] += 1
    elif score == -1:
        fb_counts["negative"] += 1
    else:
        fb_counts["neutral"] += 1
total_feedback = len(feedback)
avg_score = ((fb_counts["positive"] - fb_counts["negative"]) / max(total_feedback, 1))

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Queries", len(conversations))
col2.metric("Avg Latency", f"{df['response_time'].mean():.2f}s")
col3.metric("Avg Feedback Score", f"{avg_score:.2f}" if total_feedback else "N/A")
col4.metric("Total Feedback", total_feedback)
col5.metric("Avg Tokens / Query", f"{df['total_tokens'].mean():.0f}")

st.markdown("---")

# ---- Row 2: Query Volume & Latency Distribution ----
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Query Volume over Time")
    st.line_chart(df.set_index("time")["query"].resample("h").count())

with row1_col2:
    st.subheader("Latency Distribution")
    hist, edges = np.histogram(df["response_time"], bins=8)
    hist_df = pd.DataFrame({"latency_bucket": [f"{edges[i]:.1f}-{edges[i+1]:.1f}s" for i in range(len(hist))],
                            "count": hist})
    st.bar_chart(hist_df.set_index("latency_bucket"))

# ---- Row 3: Feedback & Query Types ----
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Feedback Distribution")
    fb_df = pd.DataFrame({
        "sentiment": ["Negative", "Neutral", "Positive"],
        "count": [fb_counts["negative"], fb_counts["neutral"], fb_counts["positive"]],
    })
    st.bar_chart(fb_df.set_index("sentiment"))

with row2_col2:
    st.subheader("Query Type Breakdown")
    types_df = df.groupby("type").size().rename("count")
    st.bar_chart(types_df)

# ---- Row 4: Token Usage & Top Queries ----
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.subheader("Tokens vs Latency")
    st.line_chart(df.set_index("time")[["total_tokens", "response_time"]])

with row3_col2:
    st.subheader("Top Queries")
    top_qs = df["query"].value_counts().head(10).rename_axis("Query").rename("Count")
    st.dataframe(top_qs, use_container_width=True)

# ---- Row 5: Retrieval Strategy Performance ----
st.markdown("---")
st.subheader("Retrieval Strategy Performance")
st.dataframe(pd.DataFrame({
    "Strategy": ["TF-IDF", "Vector", "Hybrid", "Hybrid + Rerank"],
    "Hit Rate": [1.00, 0.90, 0.90, 0.90],
    "MRR": [0.80, 0.90, 0.90, 0.90],
    "Latency (s)": [0.02, 0.03, 0.05, 0.78],
}), use_container_width=True)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
           f"Conversations: {CONVERSATIONS_FILE} | Feedback: {FEEDBACK_FILE}")
