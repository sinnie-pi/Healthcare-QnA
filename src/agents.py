"""
4-stage Hybrid RAG + Agentic pipeline:

  User Query
      │
  [Stage 1] QueryClassifierAgent   → classify intent + urgency
      │
  [Stage 2] ContextRetrieverAgent  → RAG retrieval from vector store
      │
  [Stage 3] ResponseGeneratorAgent → draft answer with citations
      │
  [Stage 4] SafetyValidatorAgent   → emergency check + disclaimer
      │
  Final Response
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum

from src.llm import chat
from src.retriever import RAGRetriever, RetrievedContext


class QueryIntent(str, Enum):
    SYMPTOM_INQUIRY = "symptom_inquiry"
    MEDICATION_INFO = "medication_info"
    DIAGNOSIS_SUPPORT = "diagnosis_support"
    TREATMENT_OPTIONS = "treatment_options"
    GENERAL_HEALTH = "general_health"
    EMERGENCY = "emergency"


@dataclass
class ClassificationResult:
    intent: QueryIntent
    urgency: str          # "low" | "medium" | "high" | "emergency"
    refined_query: str    # cleaned/expanded version of original query
    reasoning: str


@dataclass
class AgentResponse:
    query: str
    classification: ClassificationResult
    context: RetrievedContext
    answer: str
    sources: list[str]
    disclaimer: str
    is_emergency: bool = False


# ─────────────────────────────────────────────────────────────
# Stage 1 — Query Classifier
# ─────────────────────────────────────────────────────────────

_CLASSIFIER_SYSTEM = """You are a medical query classifier. Analyse the user's health question and return ONLY valid JSON with this exact structure:
{
  "intent": "<symptom_inquiry|medication_info|diagnosis_support|treatment_options|general_health|emergency>",
  "urgency": "<low|medium|high|emergency>",
  "refined_query": "<cleaned, expanded version of the question>",
  "reasoning": "<one sentence explaining your classification>"
}

Set urgency="emergency" and intent="emergency" only for life-threatening situations (chest pain, difficulty breathing, stroke symptoms, severe bleeding, unconsciousness).
Do not add any text outside the JSON object."""


def classify_query(query: str) -> ClassificationResult:
    raw = chat(_CLASSIFIER_SYSTEM, query)
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        # graceful fallback
        data = {
            "intent": "general_health",
            "urgency": "low",
            "refined_query": query,
            "reasoning": "Could not parse LLM response; defaulting to general health.",
        }
    return ClassificationResult(
        intent=QueryIntent(data.get("intent", "general_health")),
        urgency=data.get("urgency", "low"),
        refined_query=data.get("refined_query", query),
        reasoning=data.get("reasoning", ""),
    )


# ─────────────────────────────────────────────────────────────
# Stage 2 — Context Retriever (RAG)
# ─────────────────────────────────────────────────────────────

class ContextRetrieverAgent:
    def __init__(self, retriever: RAGRetriever):
        self._retriever = retriever

    def run(self, classification: ClassificationResult) -> RetrievedContext:
        # Use the refined query for better retrieval signal
        return self._retriever.retrieve(classification.refined_query)


# ─────────────────────────────────────────────────────────────
# Stage 3 — Response Generator
# ─────────────────────────────────────────────────────────────

_GENERATOR_SYSTEM = """You are a knowledgeable medical information assistant.
You will be given a user question and relevant medical context retrieved from a trusted knowledge base.

Rules:
1. Answer ONLY based on the provided context. Do not fabricate information.
2. Cite sources inline using [Source N] notation.
3. Be clear, accurate, and empathetic.
4. If the context does not contain enough information to answer the question, say so explicitly.
5. Do not provide a personal medical diagnosis. Recommend consulting a healthcare professional.
6. Keep your answer concise — 3 to 6 sentences unless the question requires more detail."""


def generate_response(query: str, context: RetrievedContext, intent: QueryIntent) -> tuple[str, list[str]]:
    context_text = context.format_context()
    user_message = f"""Question: {query}

Retrieved Medical Context:
{context_text}

Query intent: {intent.value}

Please provide a helpful, accurate answer based solely on the context above."""

    answer = chat(_GENERATOR_SYSTEM, user_message)
    sources = context.source_questions()
    return answer, sources


# ─────────────────────────────────────────────────────────────
# Stage 4 — Safety Validator
# ─────────────────────────────────────────────────────────────

_EMERGENCY_DISCLAIMER = (
    "⚠️  EMERGENCY: If you are experiencing a medical emergency, "
    "call emergency services (911/999/112) immediately. Do not delay."
)

_STANDARD_DISCLAIMER = (
    "ℹ️  This information is for educational purposes only and does not constitute "
    "medical advice. Always consult a qualified healthcare professional for diagnosis "
    "and treatment decisions."
)

_EMERGENCY_KEYWORDS = {
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "stroke", "heart attack", "unconscious", "severe bleeding", "overdose",
    "suicidal", "not breathing", "face drooping", "arm weakness",
}


def validate_and_finalize(
    query: str,
    answer: str,
    urgency: str,
    intent: QueryIntent,
) -> tuple[str, str, bool]:
    """Returns (final_answer, disclaimer, is_emergency)."""
    is_emergency = (
        urgency == "emergency"
        or intent == QueryIntent.EMERGENCY
        or any(kw in query.lower() for kw in _EMERGENCY_KEYWORDS)
    )

    disclaimer = _EMERGENCY_DISCLAIMER if is_emergency else _STANDARD_DISCLAIMER

    if is_emergency:
        # Prepend emergency warning before the answer
        final_answer = f"{_EMERGENCY_DISCLAIMER}\n\n{answer}"
    else:
        final_answer = answer

    return final_answer, disclaimer, is_emergency


# ─────────────────────────────────────────────────────────────
# Orchestrator — ties all 4 stages together
# ─────────────────────────────────────────────────────────────

class MedicalQAOrchestrator:
    def __init__(self, retriever: RAGRetriever):
        self._retriever_agent = ContextRetrieverAgent(retriever)

    def run(self, query: str) -> AgentResponse:
        # Stage 1: classify
        classification = classify_query(query)

        # Stage 2: retrieve context
        context = self._retriever_agent.run(classification)

        # Stage 3: generate answer
        answer, sources = generate_response(query, context, classification.intent)

        # Stage 4: safety check + disclaimer
        final_answer, disclaimer, is_emergency = validate_and_finalize(
            query, answer, classification.urgency, classification.intent
        )

        return AgentResponse(
            query=query,
            classification=classification,
            context=context,
            answer=final_answer,
            sources=sources,
            disclaimer=disclaimer,
            is_emergency=is_emergency,
        )
