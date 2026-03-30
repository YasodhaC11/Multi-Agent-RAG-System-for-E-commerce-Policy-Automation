# Multi-Agent RAG System for E-commerce Policy Automation

## Overview

A graph-based multi-agent system that automates customer support decisions using Retrieval-Augmented Generation (RAG) and validation guardrails. The system processes customer tickets, retrieves relevant policy information, generates structured decisions, and verifies outputs for reliability and explainability.

**Tech Stack:** LangGraph · ChromaDB · OpenAI Embeddings · GPT-4o-mini · Python

---

## Architecture

### Workflow

```
Triage → Retrieve → Generate → Verify → (Retry / End)
```

### Agent Responsibilities

| Agent | Role |
|---|---|
| **Triage** | Classifies issue type, identifies missing info, outputs structured JSON |
| **Retriever** | Fetches top-k relevant policy chunks using semantic embeddings |
| **Generator** | Produces grounded, citation-based structured JSON decisions |
| **Verifier** | Validates citations, decision quality, and evidence sufficiency |
| **Router** | Controls workflow transitions; enables retry loop or safe termination |

---

## Key Features

- Multi-agent architecture with clear separation of responsibilities
- Vector-based semantic search using ChromaDB
- Structured JSON outputs with citation-based reasoning
- Guardrails with validation and retry loop
- Explainable decisions grounded in retrieved policy context

---

## Data Sources

Synthetic e-commerce policy documents covering:

- Return Policy
- Perishable Items
- Incorrect Item Received
- Lost Package

Each document is chunked, embedded, and indexed in ChromaDB. Every chunk includes `document` and `chunk_id` metadata for citation tracking.

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd <repo-folder>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export OPENAI_API_KEY=your_openai_api_key
```

### 4. Run Ingestion

```bash
python ingest.py
```

### 5. Run Retrieval

```bash
python retriever.py
```
### 6. Run System

```bash
python main.py
```

### 7. Run Evaluation

```bash
python evaluate.py
```

---

## Example Input

```json
{
  "ticket_text": "My order arrived late and cookies are melted. I want full refund.",
  "order_context": {
    "item_category": "perishable",
    "order_status": "delivered",
    "seller_policy_override": false
  }
}
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Citation Coverage | % of responses with valid citations |
| Escalation Accuracy | Correct escalation decisions |
| Unsupported Claim Rate | Claims not grounded in retrieved context |

---

## Strengths and Limitations

### Strengths
- Accurate semantic retrieval with citation traceability
- Explainable outputs grounded in policy documents
- Robust pipeline with retry and escalation mechanisms

### Known Limitations
- Over-triggered clarification requests in edge cases
- Occasional citation inconsistencies
- Conservative decision-making on ambiguous tickets

---

## Future Improvements

- Section-based chunking for better retrieval granularity
- Metadata filtering in ChromaDB retrieval
- Re-ranking layer for improved precision
- Stronger output schema validation
- Streamlit UI integration

---

## Project Structure

```
├── ingest.py          # Document chunking and ChromaDB ingestion
├── main.py            # Entry point; runs the multi-agent pipeline
├── evaluate.py        # Evaluation script
├── agents/
│   ├── triage.py
│   ├── retriever.py
│   ├── generator.py
│   └── verifier.py
├── data/              # Policy documents (.txt)
├── requirements.txt
└── README.md
```

---

## Author

**Yasodha**  
GenAI / ML Engineer  
[LinkedIn](#) · [GitHub](#)
