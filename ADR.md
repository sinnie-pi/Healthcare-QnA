# Architecture Decision Record
## MedQuery AI — Hybrid RAG + Agentic Medical Q&A System

---

## 1. Problem Statement

Healthcare professionals and patients need fast, accurate answers to medical questions. Existing search engines return links, not answers. Generic LLMs hallucinate medical facts and cannot be trusted without grounding. The goal is a system that retrieves verified knowledge and generates safe, cited responses.

**Dataset:** MedQuAD (Medical Question Answer Dataset) from Kaggle/NIH — 16,407 medical Q&A pairs across 37 conditions, sourced from NIH, CDC, and GARD.

---

## 2. Approach Chosen: Option C — Hybrid RAG + Agentic

Pure RAG alone would answer questions well when they exactly match stored content, but fails to:
- Classify query urgency (emergencies need routing before retrieval)
- Adapt tone/depth by intent (symptom inquiry vs. medication dosage vs. treatment options)
- Validate safety before responding

Pure agentic alone would require the LLM to reason from parametric memory, which is unreliable for specific medical facts.

The hybrid gives the retrieval precision of RAG with the routing intelligence of an agent.

---

## 3. Key Technical Decisions

### 3.1 LLM: GPT-4o-mini

**Why:** Best cost/capability ratio for structured JSON outputs (classification) and grounded generation. At ~$0.15/1M input tokens it makes RAG Q&A economically viable. GPT-4o is available as a drop-in upgrade for higher-stakes deployments.

**Considered and rejected:**
- *GPT-4o*: 10x the cost, not justified for this workload
- *Claude Sonnet*: Strong alternative but adds a second SDK dependency
- *Local Llama 3*: Avoids API costs but adds infrastructure complexity and latency

### 3.2 Embeddings: text-embedding-3-small

**Why:** 1536-dimension embeddings, $0.02/1M tokens, no local GPU required. Fast API call with batch support. Sufficient for semantic medical Q&A matching.

**Considered and rejected:**
- *sentence-transformers/all-MiniLM-L6-v2*: Requires local model (~90MB), adds install complexity. Acceptable for on-prem but unnecessary here.
- *text-embedding-ada-002*: Older, same price, lower performance.

### 3.3 Vector Store: Numpy cosine similarity (demo) → ChromaDB (production)

**Why (demo):** Zero dependencies, fully portable, runnable on any laptop with just numpy. For 30–10K documents, numpy is fast enough (<10ms per query).

**Production path:** ChromaDB (persistent, local, Docker-friendly) or Pinecone (managed, scales to millions of docs). The `VectorStore` class is designed as a drop-in replacement — swap the internals without changing `RAGRetriever`.

**Considered and rejected:**
- *FAISS*: Excellent for large scale but requires C++ extensions and is harder to install cross-platform
- *Weaviate/Qdrant*: Production-grade but requires a running server, overkill for this demo

### 3.4 Chunking Strategy

For the MedQuAD dataset, each row is a self-contained Q&A pair (avg. ~80 tokens), so the natural chunk unit is the full answer text. In a production system with longer documents (clinical notes, research papers), I would use:
- **Recursive character splitting** with chunk size 512 tokens, overlap 64 tokens
- **Sentence-boundary awareness** to avoid splitting mid-sentence
- **Metadata-preserving chunks** that retain source doc ID, section header, and page number

### 3.5 Framework: Custom Python (no LangChain)

**Why:** For this scope (4 well-defined agent stages, one vector store), LangChain adds abstraction without benefit. A custom orchestrator is easier to debug, test, and explain in an interview.

**Production note:** For multi-agent workflows with tool use, memory, and human-in-the-loop, I would use **LangGraph** (structured state machine) or **Azure Agent Framework** (enterprise compliance, Azure OpenAI integration).

### 3.6 Agentic Design: 4-Stage Sequential Pipeline

```
QueryClassifier → ContextRetriever → ResponseGenerator → SafetyValidator
```

Each stage has a single responsibility. This makes failures easy to diagnose and stages independently testable.

**Why sequential, not parallel:**
- Stage 2 depends on Stage 1's refined query
- Stage 3 depends on Stage 2's retrieved context
- Stage 4 depends on Stage 3's answer text

A parallel architecture would only help if stages were independent — here they are not.

---

## 4. Trade-offs Considered

| Dimension | Choice Made | Alternative | Why This Choice |
|---|---|---|---|
| Accuracy vs. Cost | GPT-4o-mini | GPT-4o | 10x cheaper, sufficient quality |
| Speed vs. Quality | text-embedding-3-small | larger embeddings | fast API, good semantic coverage |
| Portability vs. Persistence | numpy store | ChromaDB | runs anywhere without setup |
| Framework vs. Custom | Custom agents | LangChain | simpler, more testable at this scope |
| Safety | Keyword + LLM classification | LLM only | rule-based is reliable for emergencies |

**Primary optimization:** Explainability + correctness. Medical Q&A where a hallucinated answer is worse than "I don't know" — faithfulness to retrieved context is the top priority.

---

## 5. What I Would Improve With More Time

1. **Full MedQuAD ingestion** (16K pairs) with hierarchical chunking for long answers
2. **Conversation memory** — multi-turn sessions where follow-up questions reference prior context
3. **Reranking** — use a cross-encoder (e.g. `ms-marco-MiniLM`) to rerank the top-20 candidates before passing top-5 to the generator; significantly improves faithfulness
4. **Streaming responses** — stream GPT tokens directly to the user for sub-second time-to-first-token
5. **Evaluation harness** — automated weekly re-evaluation as the knowledge base grows
6. **Confidence scores** — if max retrieval score < 0.4, respond "I don't have enough information" rather than generating a low-confidence answer

---

## 6. Production Considerations

**Latency budget** (per query):
- Embedding: ~100ms
- Vector search: <5ms (numpy at 10K docs)
- LLM classification: ~300ms
- LLM generation: ~500ms
- Total: ~900ms — acceptable for a healthcare Q&A interface

**Cost estimate** (1,000 queries/day):
- Embeddings: 1K × 0.1K tokens × $0.02/1M = $0.002/day
- Classification: 1K × 0.3K tokens × $0.15/1M = $0.045/day
- Generation: 1K × 1K tokens × $0.15/1M = $0.15/day
- **Total: ~$0.20/day** (~$6/month)

**Caching strategy:** Cache embeddings for identical queries (MD5 hash → vector). Cache LLM responses for high-frequency questions (TTL 1 hour).

**Failure modes:**
- OpenAI API timeout → retry with exponential backoff (3 attempts)
- Embedding failure → surface error, do not generate answer without context
- Low retrieval score → return "insufficient information" rather than hallucinate

**Regulatory note:** Any production medical AI system must include appropriate disclaimers, should not replace professional medical advice, and may need to comply with HIPAA (if handling PHI) or EU AI Act requirements for high-risk AI systems in healthcare.

---

## 7. How I Would Measure Success in Production

| Metric | Target | How to Measure |
|---|---|---|
| Faithfulness | > 0.90 | LLM-as-judge on random 200 queries/week |
| Answer Relevance | > 0.85 | Embedding cosine similarity |
| Emergency Detection Recall | 100% | Labelled emergency test set, alerting on misses |
| P95 Latency | < 2s | Application traces (Datadog/OpenTelemetry) |
| User Satisfaction | > 4.0/5 | In-app thumbs up/down on responses |
| Hallucination Rate | < 2% | Weekly human review of flagged responses |