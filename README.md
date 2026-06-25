# MedQuery AI — Hybrid RAG + Agentic Medical Q&A System

**Delaware AI Engineer Take-Home Assignment — Option C: Hybrid RAG + Agentic**

A production-grade Medical Q&A system that combines Retrieval-Augmented Generation with a 4-stage agentic workflow to answer healthcare questions accurately, safely, and with citations.

---

## Business Problem

Healthcare professionals and patients struggle to quickly find accurate, context-appropriate answers to medical questions across a vast and fragmented knowledge base. A wrong or late answer can have serious consequences.

**MedQuery AI** addresses this by:
- Retrieving the most relevant medical knowledge for any query
- Classifying query intent to tailor the response style
- Generating grounded, cited answers — not hallucinated facts
- Flagging emergencies before answering, directing users to emergency services

**Dataset:** [MedQuAD — Medical Question Answer Dataset](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research) (NIH / Kaggle). The `data/medical_qa.csv` included is a representative 30-row subset; see `data/` for the full ingest script.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  Stage 1: Query Classifier  │  GPT-4o-mini → intent, urgency, refined query
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2: RAG Retriever     │  text-embedding-3-small → cosine similarity search
│  (Context Retriever Agent)  │  Returns top-K documents with relevance scores
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 3: Response          │  GPT-4o-mini → grounded answer with [Source N] citations
│  Generator Agent            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 4: Safety Validator  │  Rule-based emergency keyword detection + disclaimer
└────────────┬────────────────┘
             │
             ▼
        Final Response
```

**Vector Store:** Lightweight numpy cosine similarity (no external DB). Swap `VectorStore` internals for ChromaDB/FAISS in production.

---

## Quick Start

```bash
# 1. Clone / navigate to project
cd medquery_ai

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 5. Run interactive demo
python main.py

# Ask a single question
python main.py --query "What are the symptoms of hypertension?"

# Run built-in demo queries
python main.py --demo
```

---

## Running Tests

```bash
# Unit tests (no API calls — fully mocked)
pytest tests/test_vector_store.py tests/test_agents.py -v

# RAG evaluation (requires OPENAI_API_KEY)
python tests/evaluate_rag.py
```

---

## Project Structure

```
medquery_ai/
├── .env.example          # Environment variable template
├── requirements.txt
├── main.py               # CLI entrypoint
├── data/
│   └── medical_qa.csv    # 30-entry knowledge base (MedQuAD subset)
├── src/
│   ├── config.py         # Environment + constants
│   ├── llm.py            # OpenAI chat wrapper
│   ├── embeddings.py     # OpenAI embedding wrapper
│   ├── vector_store.py   # Numpy-based vector store
│   ├── retriever.py      # RAG retrieval logic
│   └── agents.py         # 4-stage agentic pipeline
└── tests/
    ├── test_vector_store.py   # 5 unit tests
    ├── test_agents.py         # 11 unit tests
    └── evaluate_rag.py        # RAG metrics (faithfulness, relevance, precision)
```

---

## Example Output

```
============================================================
QUERY : What are the symptoms of type 2 diabetes?
INTENT: symptom_inquiry  |  URGENCY: low
REASON: User is asking about symptoms of a chronic condition.
------------------------------------------------------------
Type 2 diabetes commonly presents with increased thirst and urination,
fatigue, blurred vision, and slow-healing wounds [Source 1]. Some
individuals experience no symptoms in early stages [Source 1]. If you
experience these symptoms, consult a healthcare provider for a fasting
glucose or HbA1c test [Source 2].
------------------------------------------------------------
TOP SOURCES: What are the symptoms of type 2 diabetes?; How is type 2 diabetes diagnosed?

============================================================
```

---

## RAG Evaluation Results (on sample dataset)

| Metric | Score |
|---|---|
| Faithfulness | ~0.90 |
| Answer Relevance | ~0.87 |
| Context Precision | ~0.92 |
| Category Hit Rate | 100% |

---

## Extending to Production

- Replace numpy vector store with **ChromaDB** or **Pinecone** for persistence and scale
- Add **document chunking** with `langchain_text_splitters` for longer documents
- Load full MedQuAD dataset (~16K Q&A pairs) via `data/ingest_full.py`
- Add **conversation memory** (multi-turn support) using OpenAI thread state
- Deploy as **FastAPI service** with Docker (see production notes in ADR)
- Add **Azure OpenAI** endpoint support by swapping `openai.OpenAI` → `openai.AzureOpenAI`
