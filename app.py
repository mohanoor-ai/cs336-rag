"""
Streamlit chat UI for CS336-RAG.
Logs conversations and user feedback to JSONL files (see dashboard.py).
"""

import json
from datetime import datetime

import streamlit as st

from config import DATA_DIR, RERANK_TOP_K, TOP_K
from rag import rag

FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"
CONVERSATIONS_FILE = DATA_DIR / "conversations.jsonl"

st.set_page_config(page_title="CS336-RAG", page_icon="🎓", layout="wide")
st.title("CS336-RAG: Language Modeling from Scratch Assistant")
st.markdown("Ask about Stanford CS336 lectures and code.")


def log_line(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


try:
    with open(DATA_DIR / "documents.json", encoding="utf-8") as f:
        doc_count = len(json.load(f))
except FileNotFoundError:
    doc_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    use_rewrite = st.toggle("Query Rewriting", value=True)
    use_rerank = st.toggle("Reranking", value=True)
    top_k = st.slider("Top-K", 5, 50, TOP_K)
    rerank_k = st.slider("Rerank Top-K", 3, 20, RERANK_TOP_K)
    st.markdown(f"**Index:** {doc_count} chunks")

    st.markdown("---")
    st.markdown("**Feedback**")
    fb_score = st.feedback("thumbs", key="global_feedback")
    if fb_score is not None and fb_score != st.session_state.get("last_feedback"):
        st.session_state["last_feedback"] = fb_score
        log_line(FEEDBACK_FILE, {
            "timestamp": datetime.now().isoformat(),
            "feedback": fb_score,
            "query": st.session_state.get("last_query", ""),
        })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}]** [{src['source']}]({src['source']})")
                    if src.get("timestamp"):
                        st.caption(f"Timestamp: {src['timestamp']:.1f}s")

if prompt := st.chat_input("Ask about CS336..."):
    st.session_state["last_query"] = prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            result = rag(prompt, rewrite=use_rewrite, rerank=use_rerank,
                         num_results=top_k, rerank_k=rerank_k)
        st.markdown(result["answer"])
        sources = [{"source": h["source"], "timestamp": h.get("timestamp"),
                    "type": h.get("type")} for h in result["hits"]]
        with st.expander("Sources"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**[{i}]** [{src['source']}]({src['source']})")
                if src.get("timestamp"):
                    st.caption(f"Timestamp: {src['timestamp']:.1f}s")
        st.session_state.messages.append({"role": "assistant",
                                          "content": result["answer"], "sources": sources})

    log_line(CONVERSATIONS_FILE, {
        "timestamp": datetime.now().isoformat(),
        "query": prompt,
        "rewritten_query": result.get("rewritten_query"),
        "answer": result["answer"],
        "response_time": result["response_time"],
        "model_used": result["model_used"],
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "total_tokens": result.get("total_tokens"),
        "num_hits": len(result["hits"]),
    })
