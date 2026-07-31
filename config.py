"""
Configuration for CS336-RAG. All settings in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# LLM (OpenCode Go — DeepSeek V4 Flash, OpenAI-compatible API)
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/go/v1")

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Search
TOP_K = int(os.getenv("TOP_K", "10"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# Data sources
CS336_VIDEO_IDS = [
    "JuoVZkPBiKk",  # L1: Overview, Tokenization (Percy)
    "lVynu4bo1rY",  # L2: PyTorch, Resource Accounting (Percy)
    "izZba4UA7iY",  # L3: Architectures, Hyperparameters (Tatsu)
]

CS336_REPOS = [
    "stanford-cs336/assignment1-basics",
]

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
