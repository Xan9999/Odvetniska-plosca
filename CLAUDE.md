# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**"Prvi odziv naprednega odvetnika"** — An AI-powered legal intake tool for Jadek & Pensa (Slovenia's largest law firm). The tool transforms incoming client emails into two simultaneous outputs:

- **Result A**: Draft client email response (ready for attorney review/send in 2–3 minutes)
- **Result B**: Internal compliance dashboard (conflict of interest, AML/KYC, deadlines, team availability)

The core problem: attorneys waste up to 1 day/month on empty acknowledgement emails, missing the opportunity to be first with a substantive response (79% of clients choose the first firm to respond meaningfully).

## Key Requirements

### Result A — Client Email Draft
- Legal field classification (IP, GDPR, M&A, Labor, Tax, Corporate, etc.)
- 3–5 sentence problem summary with professional legal terminology
- 2–3 recommended attorneys (by specialization + current workload)
- Reference to relevant firm experience from knowledge base
- Targeted follow-up questions for scope clarification
- Proposed next step with timeline

### Result B — Compliance Dashboard
- **Conflict of interest**: Entity extraction → compare against client database (✅ none / ⚠️ risk / 🔴 conflict). Conflict must **block** sending the email until manual review.
- **AML/KYC**: Assess if ZPPDFT-2 obligations apply; classify risk (low/medium/high); generate required documentation checklist
- **Deadline detection**: Identify statutory/contractual deadlines (e.g., GDPR 72-hour notification) with countdown
- **Team availability**: Traffic light status (green/orange/red) for proposed attorneys

## Test Scenarios

Five test cases in `navodilo.txt` cover the main edge cases:
1. **Deeptech startup** — IP/licensing + conflict trigger (German partner is existing client)
2. **E-commerce** — GDPR breach + urgent 72h deadline already running
3. **Foreign subsidiary** — Labor dispute, bilingual SLO/ENG, parent company conflict check
4. **German M&A** — Due diligence + mandatory AML/KYC (former client as acquisition target)
5. **UAE holding** — High-risk AML (opaque ownership, cash, pressure) → tool should suggest rejection

## Data

- **`Podatki - Contacts & Workload Database.xlsx`** — Attorney specializations, client contacts, case history, workload/availability data. This is the primary knowledge base for conflict checks and team assignment.

## Tech Stack & Commands

**Backend:** FastAPI (`main.py`) + Python 3.10+
**Frontend:** Vanilla HTML + Tailwind CSS CDN + vanilla JS (`static/`)
**LLM:** OpenAI GPT-4o via `openai` SDK
**Fuzzy match:** `rapidfuzz`
**Storage:** JSON file `db/emails.json` (swap for real DB later)

```bash
# Setup
cp .env.example .env        # add OPENAI_API_KEY
pip install -r requirements.txt

# Run
uvicorn main:app --reload   # http://localhost:8000

# Test emails (paste content into the UI form)
sample_emails/email_ip_konflikt.txt    # triggers conflict (TechVision d.o.o.)
sample_emails/email_gdpr_nujno.txt     # triggers urgent deadline
```

## Architecture Guidance

The system requires these distinct components wired together:

1. **Email parser** — Extract entities (persons, companies, jurisdictions, amounts, dates) from incoming email; handle SLO/ENG mixed language
2. **Legal classifier** — Multi-label classification into practice areas; generate concise summary
3. **Knowledge retrieval** — Semantic search over firm experience/case history for relevant references
4. **Conflict checker** — Entity matching against client database (fuzzy match on company names, principals)
5. **AML engine** — Rule-based risk scoring: jurisdiction flags, ownership transparency, transaction type, ZPPDFT-2 triggers
6. **Team selector** — Score attorneys by: specialization match → availability → experience depth
7. **Email drafter** — LLM generation using structured context from above components
8. **Dashboard renderer** — Present compliance results with visual indicators

## Compliance Context

- **ZOdv** (Attorney Act) — governs conflict of interest obligations
- **ZPPDFT-2** — Slovenian AML law; triggers mandatory KYC for certain transaction types (M&A, asset management, foreign holdings)
- **GDPR** — data handling constraints on client information in AI processing
- Conflict detection must be a hard gate — no email leaves without conflict clearance

## Evaluation Priorities (higher → lower points)

1. Legal classification + summarization
2. Team assignment with reasoning
3. Follow-up question generation
4. Conflict of interest checking (with blocking mechanism)
5. AML/KYC risk assessment with ZPPDFT-2 checklist
6. *(Bonus)* Experience reference linking, multilingual/attachment support, deadline countdown, fee estimation, auto case-file creation
