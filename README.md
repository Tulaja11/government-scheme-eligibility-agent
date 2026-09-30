# Government Scheme Eligibility Agent

Finds which Indian government schemes you actually qualify for, based on your profile.

There are 3,383 schemes on MyScheme.gov.in. Nobody reads through all of them. 
You just describe yourself in plain English and this tells you which ones apply to you, 
why you qualify, and links to the official page for each one.

---

## What it does

You type something like:
> "I'm a 24 year old OBC woman from Maharashtra, income around 2.5 lakh, currently unemployed"

It extracts your details, searches through all 3,383 schemes, and returns a list with:
- Scheme name
- Verdict: Eligible / Possibly Eligible / Not Eligible
- Why you qualify (or don't)
- Official MyScheme link so you can verify and apply

---

## How it works

**Profile extraction**
Pulls 6 fields from your sentence — age, gender, category, state, income, employment. 
Done mostly with regex and keyword matching. spaCy is used as a fallback for state detection.

**Search**
Your profile gets converted to a search query and embedded using HuggingFace's 
all-MiniLM-L6-v2 model. ChromaDB finds the top 10 most relevant scheme chunks 
from 19,754 vectors.

**Decision**
Groq (Llama 3.1 8B) checks your profile against the retrieved schemes and gives 
a verdict for each one. Temperature is 0.1 — low, because eligibility answers 
need to be consistent.

Three verdicts instead of yes/no because a wrong "No" could stop someone from 
claiming a benefit they actually deserve.

---

## Tech stack

- Python
- LangChain — RAG pipeline
- ChromaDB — vector store (19,754 chunks from 3,383 schemes)
- HuggingFace all-MiniLM-L6-v2 — embeddings
- Groq + Llama 3.1 8B — LLM for eligibility decisions
- spaCy — NER fallback for state detection
- FastAPI — backend
- Streamlit — frontend (two input modes: form or plain English)
- SQLite — query logging

---

## Numbers

| What | Value |
|------|-------|
| Schemes in database | 3,383 |
| Total chunks indexed | 19,754 |
| Chunk size / overlap | 1000 / 200 chars |
| Embedding model | all-MiniLM-L6-v2 |
| Chunks retrieved per query | Top 10 |
| Profile extraction accuracy | 98% (25 test cases) |
| Retrieval relevance | 80% (10 test profiles) |
| LLM temperature | 0.1 |

---

## Limitations

- Profile extraction is mostly regex. Works well for standard inputs, 
  misses edge cases like "no income" (expects a number).
- Top 10 chunks can sometimes come from the same scheme. 
  Working on deduplication.
- Not deployed yet — runs locally.
- Retrieval accuracy measured on self-written test cases, 
  which is a known bias.

---

## What I'd add next

- State-level filtering before semantic search so results are more relevant
- Hybrid search (keyword + semantic) for better retrieval
- Tune top-k on a proper evaluation set
- Measure and display latency

---

## Data source

All scheme data from [MyScheme.gov.in](https://www.myscheme.gov.in) — 
India's official government scheme portal.
