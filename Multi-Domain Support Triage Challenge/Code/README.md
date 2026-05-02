# Support Triage Agent — Code README

## Architecture Overview

```
code/
├── main.py                    # CLI entry point
├── requirements.txt
├── .index/                    # Auto-generated vector index cache (gitignored)
├── agents/
│   ├── triage_agent.py        # Top-level orchestrator: loads corpus, runs graph, writes CSV
│   └── triage_graph.py        # LangGraph state machine: 6-node pipeline
├── retriever/
│   └── vector_retriever.py    # FAISS + OpenAI embeddings semantic retriever
├── data_loader/
│   └── corpus_loader.py       # File loader + chunker for data/ corpus
└── utils/
    ├── config.py              # All config, read from env vars
    ├── models.py              # Typed data models (SupportTicket, TriageResult, etc.)
    ├── safety.py              # Deterministic rule-based risk & injection pre-filter
    └── logger.py              # Logging setup
```

## Design Decisions

### Why LangGraph?
LangGraph models the triage pipeline as an explicit state machine, making the flow auditable and deterministic. Each decision point (safety check → retrieve → classify → generate) is a named node with clear inputs/outputs, which simplifies debugging and extension.

### Why FAISS + text-embedding-3-small?
- FAISS runs entirely in-process (no server needed, reproducible)
- `text-embedding-3-small` offers excellent retrieval quality at low cost
- Index is persisted to `.index/` so it's built only once
- Domain-aware retrieval boosts company-specific chunks (1.2× weight)

### Why GPT-4o for triage/generation?
- Structured JSON classification with zero-shot prompting is reliable
- Strong instruction following for "only use corpus content" constraint
- `temperature=0` ensures deterministic outputs

### Safety-first design
A deterministic rule-based `SafetyChecker` runs before any LLM call to catch:
- Prompt injection / jailbreak attempts (including multilingual)
- Harmful/dangerous requests (delete files, exploit code, etc.)
- High-risk financial/fraud issues that require human handling
- Account permission changes requiring admin action

### Escalation vs. Reply logic
- **Escalate**: fraud, identity theft, billing disputes, security vulnerabilities, platform outages, admin-only operations
- **Reply (invalid)**: out-of-scope, irrelevant, or trivial tickets
- **Reply**: standard how-to questions answerable from the corpus

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY=sk-...
# Or copy .env.example → .env and fill it in
```

## Running

```bash
# From the repo root:
python code/main.py

# Options:
python code/main.py --dry-run          # Test first 3 tickets
python code/main.py --verbose          # Debug logging
python code/main.py --rebuild-index    # Force re-embed corpus

# Custom paths:
python code/main.py --input path/to/tickets.csv --output path/to/output.csv
```

First run will:
1. Load all documents from `data/hackerrank/`, `data/claude/`, `data/visa/`
2. Chunk and embed them (takes 1–3 minutes depending on corpus size)
3. Save the index to `code/.index/` for all subsequent runs

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key |

Never hardcode secrets. All keys are read from env.

## Output

Results are written to `support_tickets/output.csv` with columns:

| Column | Description |
|---|---|
| `issue` | Original ticket issue |
| `subject` | Original subject |
| `company` | Original company |
| `status` | `replied` or `escalated` |
| `product_area` | Support domain/category |
| `response` | User-facing response grounded in corpus |
| `justification` | Decision rationale |
| `request_type` | `product_issue`, `feature_request`, `bug`, or `invalid` |

## Reproducibility

- `temperature=0` on all LLM calls
- Pinned dependency versions in `requirements.txt`
- Vector index is seeded deterministically by corpus content
- `random.seed(42)` applied at startup (in `main.py`)
