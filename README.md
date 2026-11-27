# BioGuard AI — Autonomous Pharmacovigilance & Trial Intelligence

> **Research & Educational Disclaimer:** Built for exploratory biomedical data retrieval and workflow prototyping. Does **not** provide medical advice, diagnosis, treatment recommendations, or clinical trial eligibility determinations.

## Architecture Overview

```text
Clinical / Research Notes
        │
        ▼
  [Triage Agent]  ──► Structured PatientProfile (Pydantic)
        │
        ▼
[Parallel Evidence Agent]  ──► ThreadPoolExecutor
   ├── NIH RxNorm API (Drug Normalization & RxCUI)
   ├── OpenFDA Drug Event API (Sampled Adverse Events)
   └── ClinicalTrials.gov API v2 (Recruiting Trials)
        │
        ▼
  [Critic Agent]  ──► Sufficient? ──No (1 retry)──► Broadened Condition Search
        │ Yes / Max Retries
        ▼
[Synthesizer & Verifier] ──► Grounded Report + NCT Regex Audit + Signal Score
        │
        ▼
 [DuckDB Audit Ledger] + Streamlit UI (.md / .json / .csv exports)
```

## Key Features
- **Multi-Agent Cyclic Orchestration (LangGraph):** Structured triage extraction, parallel API retrieval, self-correcting critic loop, and grounded report synthesis.
- **Parallel Biomedical Retrieval:** Concurrent queries across NIH RxNorm, OpenFDA, and ClinicalTrials.gov with retry/backoff handling.
- **Deterministic Citation Grounding:** Regex-based `NCT` ID verification against retrieved trial records.
- **Telemetry & Exports:** Run history and DuckDB audit ledger with Markdown, JSON, and CSV export options.

## Live Deployment
- **Live App URL:** *(https://bioguard-saas-yhiwnn43twgrmxxtde7hhs.streamlit.app/)*

## Local Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Run locally:

```bash
streamlit run app.py
```
