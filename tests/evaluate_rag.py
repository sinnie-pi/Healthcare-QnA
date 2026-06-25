"""
RAG Evaluation Suite — measures faithfulness, answer relevance, and context precision.

Run:  python tests/evaluate_rag.py

Metrics implemented:
  - Faithfulness       : fraction of answer sentences supported by retrieved context
  - Answer Relevance   : cosine similarity between query embedding and answer embedding
  - Context Precision  : fraction of retrieved docs that are topically relevant to the query
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from src.config import OPENAI_API_KEY, DATA_PATH
from src.vector_store import build_store_from_csv
from src.retriever import RAGRetriever
from src.embeddings import embed_query, embed_texts
from src.agents import MedicalQAOrchestrator
from src.llm import chat


# ── Evaluation helpers ────────────────────────────────────────

def score_faithfulness(answer: str, context: str) -> float:
    """Ask LLM to judge what fraction of the answer is grounded in the context."""
    prompt = """You are a faithfulness evaluator.
Given an ANSWER and its CONTEXT, return ONLY a JSON object like: {"score": 0.85, "reason": "..."}
Score 1.0 if every claim in the answer is directly supported by the context.
Score 0.0 if the answer contains claims not found in the context."""

    user_msg = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    raw = chat(prompt, user_msg)
    try:
        return float(json.loads(raw.strip()).get("score", 0.0))
    except Exception:
        return 0.5  # neutral if parsing fails


def score_answer_relevance(query: str, answer: str) -> float:
    """Cosine similarity between query and answer embeddings."""
    vecs = embed_texts([query, answer])
    q_vec = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-9)
    a_vec = vecs[1] / (np.linalg.norm(vecs[1]) + 1e-9)
    return float(np.dot(q_vec, a_vec))


def score_context_precision(query: str, retrieved_docs, threshold: float = 0.3) -> float:
    """Fraction of retrieved docs with cosine similarity > threshold to the query."""
    if not retrieved_docs:
        return 0.0
    q_emb = embed_query(query)
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    relevant = sum(
        1 for _, score in retrieved_docs if score >= threshold
    )
    return relevant / len(retrieved_docs)


# ── Evaluation dataset ────────────────────────────────────────

EVAL_QUERIES = [
    {
        "query": "What are the symptoms of type 2 diabetes?",
        "expected_category": "diabetes",
    },
    {
        "query": "How is hypertension treated with medication?",
        "expected_category": "hypertension",
    },
    {
        "query": "What are warning signs of a heart attack?",
        "expected_category": "cardiology",
    },
    {
        "query": "How is asthma diagnosed and managed?",
        "expected_category": "respiratory",
    },
    {
        "query": "What is the difference between type 1 and type 2 diabetes?",
        "expected_category": "diabetes",
    },
]


def run_evaluation():
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not set — skipping evaluation.")
        return

    print("Building knowledge base...")
    store = build_store_from_csv(DATA_PATH)
    retriever = RAGRetriever(store, top_k=5)
    orchestrator = MedicalQAOrchestrator(retriever)

    results = []
    print(f"\nRunning RAG evaluation on {len(EVAL_QUERIES)} queries...\n")

    for item in EVAL_QUERIES:
        query = item["query"]
        expected_cat = item["expected_category"]
        print(f"  Query: {query[:60]}...")

        resp = orchestrator.run(query)
        context_text = resp.context.format_context()

        faithfulness = score_faithfulness(resp.answer, context_text)
        relevance = score_answer_relevance(query, resp.answer)
        precision = score_context_precision(query, resp.context.documents)

        # Check if top retrieved doc has the expected category
        top_categories = [
            doc.metadata.get("category", "") for doc, _ in resp.context.documents[:3]
        ]
        category_hit = expected_cat in top_categories

        results.append({
            "query": query,
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_precision": precision,
            "category_hit": category_hit,
        })
        print(f"    Faithfulness={faithfulness:.2f}  Relevance={relevance:.2f}  "
              f"Precision={precision:.2f}  CategoryHit={category_hit}")

    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["answer_relevance"] for r in results) / len(results)
    avg_precision = sum(r["context_precision"] for r in results) / len(results)
    category_accuracy = sum(r["category_hit"] for r in results) / len(results)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Avg Faithfulness       : {avg_faithfulness:.3f}")
    print(f"  Avg Answer Relevance   : {avg_relevance:.3f}")
    print(f"  Avg Context Precision  : {avg_precision:.3f}")
    print(f"  Category Hit Rate      : {category_accuracy:.1%}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    run_evaluation()
