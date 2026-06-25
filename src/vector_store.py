"""
Lightweight in-memory vector store using numpy cosine similarity.
No external vector DB dependency — keeps the demo portable.
For production, swap _VectorStore internals with ChromaDB / FAISS / Pinecone.
"""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from src.embeddings import embed_texts


@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: np.ndarray | None = None


class VectorStore:
    def __init__(self):
        self._docs: list[Document] = []
        self._matrix: np.ndarray | None = None  # (N, dim)

    def add_documents(self, docs: list[Document]) -> None:
        texts = [d.content for d in docs]
        embeddings = embed_texts(texts)
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb
            self._docs.append(doc)
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        if not self._docs:
            self._matrix = None
            return
        mat = np.vstack([d.embedding for d in self._docs])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._matrix = mat / np.where(norms == 0, 1, norms)

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Document, float]]:
        if self._matrix is None or len(self._docs) == 0:
            return []
        norm = np.linalg.norm(query_embedding)
        q = query_embedding / (norm if norm > 0 else 1)
        scores = self._matrix @ q
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self._docs[i], float(scores[i])) for i in indices]

    def __len__(self) -> int:
        return len(self._docs)


def build_store_from_csv(csv_path: str) -> VectorStore:
    df = pd.read_csv(csv_path)
    docs = [
        Document(
            id=str(row["id"]),
            content=f"Q: {row['question']}\nA: {row['answer']}",
            metadata={
                "question": row["question"],
                "answer": row["answer"],
                "category": row.get("category", "general"),
                "source": row.get("source", ""),
            },
        )
        for _, row in df.iterrows()
    ]
    store = VectorStore()
    store.add_documents(docs)
    return store
