"""
In-memory search index with TF-IDF keyword search and vector similarity search.

Hybrid search combines both using Reciprocal Rank Fusion (RRF).
No external database needed.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Index:
    def __init__(self, text_fields=[], keyword_fields=[]):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.docs = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.embeddings = None

    def fit(self, docs):
        self.docs = docs
        texts = [doc.get("content", "") for doc in docs]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self

    def fit_embeddings(self, embeddings):
        self.embeddings = embeddings

    def _filter_mask(self, filter_dict):
        """Boolean mask over docs for exact-match filters, e.g. {'type': 'transcript'}."""
        if not filter_dict:
            return None
        mask = np.ones(len(self.docs), dtype=bool)
        for field, value in filter_dict.items():
            mask &= np.array([doc.get(field) == value for doc in self.docs])
        return mask

    def _top_results(self, scores, num_results, score_key):
        num_results = min(num_results, len(scores))
        top_idx = np.argpartition(scores, -num_results)[-num_results:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results = []
        for i in top_idx:
            if scores[i] > 0:
                doc = dict(self.docs[i])
                doc["_id"] = int(i)
                doc[score_key] = float(scores[i])
                results.append(doc)
        return results

    def search(self, query, filter_dict=None, boost_dict=None, num_results=10):
        """TF-IDF keyword search. filter_dict does exact matching, boost_dict weights text fields."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        if boost_dict:
            for field, boost in boost_dict.items():
                if field in self.text_fields:
                    scores = scores * boost
        mask = self._filter_mask(filter_dict)
        if mask is not None:
            scores = scores * mask
        return self._top_results(scores, num_results, "_score")

    def vector_search(self, query_vec, filter_dict=None, num_results=10):
        """Vector similarity search over precomputed embeddings."""
        if self.embeddings is None:
            return []
        sim = cosine_similarity([query_vec], self.embeddings).flatten()
        mask = self._filter_mask(filter_dict)
        if mask is not None:
            sim = sim * mask
        return self._top_results(sim, num_results, "_vector_score")

    def hybrid_search(self, query, query_vec, num_results=10, rrf_k=60, filter_dict=None):
        """Combine keyword and vector rankings with Reciprocal Rank Fusion."""
        tfidf_results = self.search(query, filter_dict=filter_dict, num_results=num_results * 2)
        vector_results = self.vector_search(query_vec, filter_dict=filter_dict, num_results=num_results * 2)
        scores = {}
        for rank, doc in enumerate(tfidf_results):
            scores[doc["_id"]] = {"doc": doc, "score": 1.0 / (rrf_k + rank)}
        for rank, doc in enumerate(vector_results):
            if doc["_id"] in scores:
                scores[doc["_id"]]["score"] += 1.0 / (rrf_k + rank)
            else:
                scores[doc["_id"]] = {"doc": doc, "score": 1.0 / (rrf_k + rank)}
        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:num_results]]
