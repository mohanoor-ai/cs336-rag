"""
Fetches YouTube transcripts and GitHub code, chunks them, and saves to JSON.
Run this before using the RAG pipeline.
"""

import json
import os

import requests
from youtube_transcript_api import YouTubeTranscriptApi

from config import DATA_DIR, CS336_VIDEO_IDS, CS336_REPOS, CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += size - overlap
    return chunks


def fetch_youtube_transcripts(video_ids):
    docs = []
    for vid in video_ids:
        try:
            transcript = YouTubeTranscriptApi().fetch(vid).to_raw_data()
            segments = []
            current_text = []
            current_start = transcript[0]["start"]
            for entry in transcript:
                current_text.append(entry["text"])
                if entry["start"] - current_start > 30:
                    segments.append({
                        "text": " ".join(current_text),
                        "start": current_start,
                        "video_id": vid,
                    })
                    current_text = []
                    current_start = entry["start"]
            if current_text:
                segments.append({
                    "text": " ".join(current_text),
                    "start": current_start,
                    "video_id": vid,
                })
            for seg in segments:
                for chunk in chunk_text(seg["text"]):
                    docs.append({
                        "content": chunk,
                        "source": f"https://youtube.com/watch?v={vid}&t={int(seg['start'])}s",
                        "timestamp": seg["start"],
                        "type": "transcript",
                        "video_id": vid,
                    })
            print(f"YouTube: {len(segments)} segments for {vid}")
        except Exception as e:
            print(f"YouTube error for {vid}: {e}")
    return docs


def fetch_github_repo(repo, branch="main"):
    docs = []
    url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    headers = {}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        tree = resp.json().get("tree", [])
        exts = (".py", ".md", ".txt", ".sh", ".yaml", ".yml", ".json", ".cpp", ".c", ".h")
        for item in tree:
            if item["type"] == "blob" and item["path"].endswith(exts):
                raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{item['path']}"
                try:
                    content = requests.get(raw_url, headers=headers, timeout=10).text
                    for chunk in chunk_text(content):
                        docs.append({
                            "content": chunk,
                            "source": f"https://github.com/{repo}/blob/{branch}/{item['path']}",
                            "timestamp": None,
                            "type": "code",
                            "file_path": item["path"],
                        })
                except Exception as e:
                    print(f"GitHub error: {item['path']}: {e}")
        print(f"GitHub: {len(docs)} chunks from {repo}")
    except Exception as e:
        print(f"GitHub error: {repo}: {e}")
    return docs


def main():
    all_docs = []
    all_docs.extend(fetch_youtube_transcripts(CS336_VIDEO_IDS))
    for repo in CS336_REPOS:
        all_docs.extend(fetch_github_repo(repo))

    out_path = DATA_DIR / "documents.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_docs)} chunks to {out_path}")


if __name__ == "__main__":
    main()
