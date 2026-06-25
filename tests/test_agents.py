"""Unit tests for agent stages — mocks LLM + retriever, no API calls."""
import json
import numpy as np
import pytest
import sys, os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import (
    classify_query,
    generate_response,
    validate_and_finalize,
    QueryIntent,
    ClassificationResult,
    MedicalQAOrchestrator,
)
from src.retriever import RAGRetriever, RetrievedContext
from src.vector_store import Document


# ── helpers ──────────────────────────────────────────────────

def _make_context(question: str = "What is diabetes?") -> RetrievedContext:
    doc = Document(
        id="1",
        content="Q: What is diabetes?\nA: Diabetes is a condition with high blood sugar.",
        metadata={
            "question": "What is diabetes?",
            "answer": "Diabetes is a condition with high blood sugar.",
            "category": "diabetes",
        },
    )
    doc.embedding = np.array([1.0, 0.0], dtype=np.float32)
    return RetrievedContext(question=question, documents=[(doc, 0.95)])


# ── Stage 1: Classifier ───────────────────────────────────────

class TestQueryClassifier:
    def test_valid_classification_json(self):
        mock_response = json.dumps({
            "intent": "symptom_inquiry",
            "urgency": "low",
            "refined_query": "What are the symptoms of diabetes?",
            "reasoning": "User is asking about symptoms.",
        })
        with patch("src.agents.chat", return_value=mock_response):
            result = classify_query("diabetes symptoms?")
        assert result.intent == QueryIntent.SYMPTOM_INQUIRY
        assert result.urgency == "low"
        assert "diabetes" in result.refined_query.lower()

    def test_emergency_classification(self):
        mock_response = json.dumps({
            "intent": "emergency",
            "urgency": "emergency",
            "refined_query": "Chest pain and shortness of breath",
            "reasoning": "Classic heart attack symptoms.",
        })
        with patch("src.agents.chat", return_value=mock_response):
            result = classify_query("I have chest pain and can't breathe")
        assert result.intent == QueryIntent.EMERGENCY
        assert result.urgency == "emergency"

    def test_malformed_json_falls_back_gracefully(self):
        with patch("src.agents.chat", return_value="not valid json at all"):
            result = classify_query("what is hypertension")
        assert result.intent == QueryIntent.GENERAL_HEALTH
        assert result.urgency == "low"
        assert result.refined_query == "what is hypertension"

    def test_medication_intent(self):
        mock_response = json.dumps({
            "intent": "medication_info",
            "urgency": "low",
            "refined_query": "What medication is used for high blood pressure?",
            "reasoning": "Question is about hypertension medication.",
        })
        with patch("src.agents.chat", return_value=mock_response):
            result = classify_query("medication for blood pressure")
        assert result.intent == QueryIntent.MEDICATION_INFO


# ── Stage 3: Response Generator ──────────────────────────────

class TestResponseGenerator:
    def test_generates_answer_with_sources(self):
        ctx = _make_context("What is diabetes?")
        mock_answer = "Diabetes is a condition with elevated blood sugar [Source 1]."
        with patch("src.agents.chat", return_value=mock_answer):
            answer, sources = generate_response(
                "What is diabetes?", ctx, QueryIntent.GENERAL_HEALTH
            )
        assert answer == mock_answer
        assert len(sources) == 1
        assert "diabetes" in sources[0].lower()


# ── Stage 4: Safety Validator ─────────────────────────────────

class TestSafetyValidator:
    def test_non_emergency_gets_standard_disclaimer(self):
        _, disclaimer, is_emergency = validate_and_finalize(
            "what is hypertension",
            "Hypertension is high blood pressure.",
            "low",
            QueryIntent.GENERAL_HEALTH,
        )
        assert not is_emergency
        assert "educational purposes" in disclaimer

    def test_emergency_urgency_flag(self):
        _, disclaimer, is_emergency = validate_and_finalize(
            "I have severe chest pain",
            "Chest pain can indicate a heart attack.",
            "emergency",
            QueryIntent.EMERGENCY,
        )
        assert is_emergency
        assert "EMERGENCY" in disclaimer

    def test_emergency_keyword_detection(self):
        _, _, is_emergency = validate_and_finalize(
            "my father is unconscious",
            "Some answer.",
            "high",
            QueryIntent.SYMPTOM_INQUIRY,
        )
        assert is_emergency

    def test_emergency_prepended_to_answer(self):
        final_answer, _, _ = validate_and_finalize(
            "I have chest pain",
            "Original answer.",
            "emergency",
            QueryIntent.EMERGENCY,
        )
        assert "EMERGENCY" in final_answer
        assert "Original answer" in final_answer


# ── Full Pipeline ─────────────────────────────────────────────

class TestMedicalQAOrchestrator:
    def _make_orchestrator(self):
        store_mock = MagicMock()
        store_mock.similarity_search.return_value = [
            (_make_context().documents[0][0], 0.9)
        ]
        retriever = RAGRetriever.__new__(RAGRetriever)
        retriever._store = store_mock
        retriever._top_k = 3
        return MedicalQAOrchestrator(retriever)

    def test_full_pipeline_returns_agent_response(self):
        orchestrator = self._make_orchestrator()
        classify_mock = json.dumps({
            "intent": "general_health",
            "urgency": "low",
            "refined_query": "What is diabetes?",
            "reasoning": "General health question.",
        })
        answer_mock = "Diabetes is a metabolic disease [Source 1]."
        with patch("src.agents.chat", side_effect=[classify_mock, answer_mock]), \
             patch("src.retriever.embed_query", return_value=np.array([1.0, 0.0])):
            resp = orchestrator.run("What is diabetes?")
        assert resp.query == "What is diabetes?"
        assert resp.answer is not None
        assert not resp.is_emergency

    def test_emergency_query_is_flagged(self):
        orchestrator = self._make_orchestrator()
        classify_mock = json.dumps({
            "intent": "emergency",
            "urgency": "emergency",
            "refined_query": "Chest pain and shortness of breath",
            "reasoning": "Emergency situation.",
        })
        answer_mock = "Seek immediate help."
        with patch("src.agents.chat", side_effect=[classify_mock, answer_mock]), \
             patch("src.retriever.embed_query", return_value=np.array([1.0, 0.0])):
            resp = orchestrator.run("I have chest pain and can't breathe")
        assert resp.is_emergency
        assert "EMERGENCY" in resp.answer
