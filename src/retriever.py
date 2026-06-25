from dataclasses import dataclass
from src.vector_store import VectorStore, Document
from src.embeddings import embed_query
from src.config import TOP_K_RESULTS


@dataclass
class RetrievedContext:
    question: str
    documents: list[tuple[Document, float]]  # (doc, score)

    def format_context(self) -> str:
        parts = []
        for i, (doc, score) in enumerate(self.documents, 1):
            parts.append(
                f"[Source {i} | category: {doc.metadata.get('category', 'N/A')} | relevance: {score:.2f}]\n"
                f"{doc.metadata.get('answer', doc.content)}"
            )
        return "\n\n".join(parts)

    def source_questions(self) -> list[str]:
        return [doc.metadata.get("question", "") for doc, _ in self.documents]


class RAGRetriever:
    def __init__(self, store: VectorStore, top_k: int = TOP_K_RESULTS):
        self._store = store
        self._top_k = top_k

    def retrieve(self, query: str) -> RetrievedContext:
        q_emb = embed_query(query)
        results = self._store.similarity_search(q_emb, top_k=self._top_k)
        return RetrievedContext(question=query, documents=results)

    def get_top_score(self, query: str) -> float:
        ctx = self.retrieve(query)
        if not ctx.documents:
            return 0.0
        return ctx.documents[0][1]
