"""Unit tests for vector store — uses mock embeddings (no API calls)."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vector_store import VectorStore, Document


def _make_doc(id_: str, content: str, dim: int = 8) -> Document:
    doc = Document(id=id_, content=content)
    return doc


def _inject_embeddings(store: VectorStore, docs: list[Document], vecs: list[list[float]]) -> None:
    """Bypass API calls by directly injecting embeddings."""
    for doc, vec in zip(docs, vecs):
        doc.embedding = np.array(vec, dtype=np.float32)
        store._docs.append(doc)
    store._rebuild_matrix()


class TestVectorStore:
    def test_empty_store_returns_nothing(self):
        store = VectorStore()
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.similarity_search(q, top_k=3)
        assert results == []

    def test_similarity_search_returns_correct_top_k(self):
        store = VectorStore()
        docs = [_make_doc(str(i), f"doc {i}") for i in range(5)]
        vecs = [
            [1.0, 0.0, 0.0, 0.0],  # identical to query
            [0.0, 1.0, 0.0, 0.0],  # orthogonal
            [0.9, 0.1, 0.0, 0.0],  # very similar
            [0.0, 0.0, 1.0, 0.0],  # orthogonal
            [0.8, 0.2, 0.0, 0.0],  # similar
        ]
        _inject_embeddings(store, docs, vecs)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.similarity_search(query, top_k=3)
        assert len(results) == 3
        # doc 0 should be top result (identical)
        assert results[0][0].id == "0"
        # scores should be descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_cosine_similarity_score_range(self):
        store = VectorStore()
        docs = [_make_doc("a", "doc a"), _make_doc("b", "doc b")]
        _inject_embeddings(store, docs, [[1.0, 0.0], [0.0, 1.0]])
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = store.similarity_search(query, top_k=2)
        for _, score in results:
            assert -1.0 <= score <= 1.0

    def test_top_k_capped_at_store_size(self):
        store = VectorStore()
        docs = [_make_doc(str(i), f"d{i}") for i in range(3)]
        _inject_embeddings(store, docs, [[1, 0], [0, 1], [0.5, 0.5]])
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = store.similarity_search(query, top_k=10)
        assert len(results) == 3

    def test_zero_vector_query_handled(self):
        store = VectorStore()
        doc = _make_doc("x", "some doc")
        _inject_embeddings(store, [doc], [[1.0, 0.0]])
        query = np.array([0.0, 0.0], dtype=np.float32)
        # should not raise, just return a result
        results = store.similarity_search(query, top_k=1)
        assert len(results) == 1
