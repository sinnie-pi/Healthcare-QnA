# Demo Documentation — MedQuery AI

## Setup (60 seconds)

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI key
copy .env.example .env        # then edit .env and set OPENAI_API_KEY=sk-...

# Run
python main.py --demo         # 5 built-in sample queries
python main.py                # interactive mode
python main.py --query "What are the symptoms of diabetes?"
```

---

## Example 1 — Symptom Inquiry

**Input:**
```
What are the early symptoms of type 2 diabetes?
```

**System output:**
```
QUERY : What are the early symptoms of type 2 diabetes?
INTENT: symptom_inquiry  |  URGENCY: low
REASON: User is asking about observable symptoms of a chronic condition.
------------------------------------------------------------
Early symptoms of type 2 diabetes include increased thirst and frequent
urination, unexplained fatigue, and blurred vision [Source 1]. Some
individuals experience slow-healing cuts or frequent infections [Source 1].
Importantly, many people have no symptoms in the early stages, making
routine screening important [Source 2].
------------------------------------------------------------
TOP SOURCES: What are the symptoms of type 2 diabetes?;
             How is type 2 diabetes diagnosed?
ℹ️  This information is for educational purposes only and does not
constitute medical advice.
```

---

## Example 2 — Emergency Detection

**Input:**
```
I have chest pain and my left arm hurts
```

**System output:**
```
QUERY : I have chest pain and my left arm hurts
INTENT: emergency  |  URGENCY: emergency
REASON: Classic heart attack symptoms requiring immediate emergency response.
------------------------------------------------------------
⚠️  EMERGENCY: If you are experiencing a medical emergency, call
emergency services (911/999/112) immediately. Do not delay.

Chest pain radiating to the left arm is a classic warning sign of a
heart attack [Source 1]. Other symptoms may include shortness of breath,
cold sweats, and nausea [Source 1]. Women may experience atypical
symptoms such as jaw pain or fatigue without prominent chest pain [Source 1].
------------------------------------------------------------
TOP SOURCES: What are the warning signs of a heart attack?
⚠️  EMERGENCY: Call emergency services immediately.
```

---

## Example 3 — Medication Information

**Input:**
```
What medications are used to treat high blood pressure?
```

**System output:**
```
QUERY : What medications are used to treat high blood pressure?
INTENT: medication_info  |  URGENCY: low
REASON: User is asking about pharmacological treatment options.
------------------------------------------------------------
Common medications for hypertension include ACE inhibitors (lisinopril),
ARBs (losartan), calcium channel blockers (amlodipine), and diuretics
(hydrochlorothiazide) [Source 1]. Beta-blockers such as metoprolol are
also used, and many patients require combination therapy to reach their
blood pressure target [Source 1].
------------------------------------------------------------
TOP SOURCES: What medications are used to treat hypertension?
ℹ️  This information is for educational purposes only.
```

---

## Running Tests

```bash
# Unit tests — no API key needed (all mocked)
pytest tests/test_vector_store.py tests/test_agents.py -v
# Expected: 16 passed

# RAG evaluation — requires OPENAI_API_KEY
python tests/evaluate_rag.py
```

**Test output:**
```
16 passed in 2.01s

EVALUATION SUMMARY
  Avg Faithfulness       : 0.91
  Avg Answer Relevance   : 0.87
  Avg Context Precision  : 0.92
  Category Hit Rate      : 100%
```