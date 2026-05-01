# Muddy Reporter — Investigative Note Generator (Prototype)

Generates a **forensic, skeptical, "Muddy Waters-style"** investigative research note for any US-listed company (including ADRs and foreign private issuers) directly from public SEC EDGAR filings, with a built-in **verification layer** that separates:

- **Sourced facts** (backed by verbatim quotes / citations)
- **Analytical inferences** (your reasoning over multiple facts)
- **Open questions** (require further diligence)
- **Speculation flags** (explicitly labeled, never presented as fact)

This is a **case-study prototype**. It does **not** claim fraud detection and is **not** investment advice.

---

## Highlights

- **Generic system, not a single-company script.** Input any US ticker or CIK; the pipeline auto-detects whether the issuer files US-domestic forms (10-K / 10-Q / 8-K) or foreign-issuer forms (20-F / 6-K / F-1).
- **Dual analysis layers.** Deterministic XBRL-driven anomaly detection runs *before* the LLM and feeds the model with quantitative red flags (so the narrative is grounded in real numbers, not vibes).
- **Free LLM by default.** Auto-routes to **Google Gemini 2.5 Pro** (free tier via AI Studio) → OpenAI fallback → keyword heuristic so the prototype runs end-to-end even without any API key.
- **Citation-first.** Every "fact" / "inference" must include a verbatim excerpt; findings without citations are auto-demoted to "open questions".
- **Editorial-style web app.** Custom dark UI inspired by short-seller research publications. Single-page app with live SSE progress streaming, three views (landing → loading → report), Plotly charts, sticky table of contents, expandable citations, and one-click PDF export.

---

## Quickstart

### 1. Install

```bash
cd muddy-reporter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# edit .env and set:
#   GEMINI_API_KEY=...        ← free key from https://aistudio.google.com/apikey
#   SEC_USER_AGENT=Your Name your.email@example.com
```

If you skip the LLM key, the system will still run (XBRL anomalies + keyword-based fallback) so you can see the architecture working before wiring up a key.

### 3. Run the web app

```bash
python web.py
```

Then open `http://localhost:8000` in your browser. Type a ticker → hit **Investigate** → watch the live progress stream → explore the report. No build step, no Node, single Python service.

### 4. Or run from the CLI

```bash
python run.py --ticker LKNCY --max-filings 5 --out outputs/LKNCY.html
python run.py --ticker AAPL  --max-filings 4 --out outputs/AAPL.html
```

Outputs land in `outputs/<TICKER>.html` (full report) and `outputs/<TICKER>.json` (structured data). Open the HTML in a browser and use the built-in **Print / Save as PDF** button for a PDF.

---

## Demo tickers

| Ticker | Why it's interesting |
|---|---|
| **LKNCY** | Luckin Coffee — the original Muddy Waters short. The system surfaces real XBRL anomalies (NI vs CFO gap; AR > revenue growth) on the post-restructuring filings. |
| **AAPL** | Control company — system reports **zero** financial anomalies, demonstrating it doesn't fabricate red flags on clean balance sheets. |
| **NKLA** | Nikola — well-known short target (Hindenburg, 2020). Good test for accounting / disclosure red flags. |
| **GOTU** | Gaotu (formerly GSX) — historic short-seller target; tests governance / related-party detection. |

---

## What the system actually does

```
ticker / CIK
   │
   ▼
[ SEC EDGAR ] ─► resolve CIK, list recent filings, auto-pick 10-K/10-Q/8-K vs 20-F/6-K/F-1
   │
   ▼
[ XBRL company-facts ] ─► time-series of revenue, AR, CFO, NI, inventory, goodwill, equity
   │
   ▼
[ Deterministic anomaly engine ]    rule-based, no LLM
   • AR growing > Revenue              (channel stuffing)
   • Net income running ahead of CFO   (accruals quality)
   • Sudden gross-margin jump          (cost reclassification)
   • Inventory growth > revenue        (demand softness)
   • Goodwill > 50% of equity          (impairment risk)
   │
   ▼
[ Per-filing extractor (LLM) ]  text chunks + financial context → candidate findings (JSON)
   │
   ▼
[ Verification layer ]
   • drop / demote findings without verbatim citations
   • categorize: fact | inference | question | speculation
   • assign confidence: low | medium | high
   │
   ▼
[ Synthesizer (LLM) ] ─► Muddy-Waters-style structured note
   │
   ▼
HTML report + JSON schema + Streamlit UI
```

---

## Output schema

Every report is also persisted as JSON for downstream use. Top-level fields:

```json
{
  "ticker": "LKNCY",
  "company_name": "Luckin Coffee Inc.",
  "snapshot": { "business_description": "...", "where_it_operates": "...", ... },
  "core_thesis": "...",
  "red_flags": [ { "title", "category", "label", "confidence",
                   "claim_or_observation", "why_it_matters",
                   "counterpoints_or_alt_explanations", "open_questions",
                   "citations": [ { "doc_id", "url", "excerpt" } ] } ],
  "management_claims_vs_counterpoints": [ { "claim", "source_excerpt", "counterpoint", "confidence" } ],
  "concerns_by_category": { "accounting": [...], "governance": [...], ... },
  "open_questions": [...],
  "conclusion": "...",
  "limitations": [...],
  "financial_anomalies": [ { "title", "severity", "metric", "description", ... } ],
  "financial_table": [ { "year", "revenue", "net_income", ... } ],
  "provider_info": { "provider": "gemini", "model": "gemini-2.5-pro" }
}
```

---

## Verification layer (core design principle)

Every finding carries:

- **`label`**: `fact` | `inference` | `question` | `speculation`
- **`confidence`**: `low` | `medium` | `high`
- **`citations`**: array of `{ doc_id, url, excerpt }` with **verbatim** quoted excerpts

Hard rules baked into the prompts:

1. No "fraud" / "scam" claims unless explicitly stated by source or regulator.
2. Every `fact` / `inference` must cite a verbatim excerpt; otherwise it's auto-downgraded to `question`.
3. The model must use the `doc_id` and `url` exactly as supplied — it can't invent sources.
4. Boilerplate risk-factor language is to be ignored, not flagged.
5. Quantitative anomalies are computed deterministically (no LLM in the loop) and passed in as ground truth.

---

## Repo layout

```
muddy-reporter/
├── web.py                          ← FastAPI server (frontend + API + SSE)
├── run.py                          ← CLI shim (no editable install needed)
├── frontend/
│   └── index.html                  ← Single-page app (Tailwind + Alpine + Plotly via CDN)
├── requirements.txt
├── .env.example
├── README.md
├── DOCUMENTATION.md                ← build doc (tools, prompts, workflow, limits)
├── REFLECTION.md                   ← what worked, what didn't, what's next
├── outputs/                        ← generated HTML + JSON reports
├── cache/                          ← cached SEC responses (filings + XBRL)
└── src/muddy_reporter/
    ├── pipeline.py                 ← orchestration
    ├── sec_edgar.py                ← EDGAR client (filings)
    ├── financials.py               ← XBRL client + deterministic anomaly engine
    ├── llm.py                      ← Gemini → OpenAI → heuristic gateway
    ├── prompts.py                  ← system prompt + JSON schemas + builders
    ├── models.py                   ← Pydantic schemas (Report / Finding / Citation)
    ├── text_extract.py             ← HTML → text + chunking
    ├── render.py                   ← Jinja2 HTML renderer (printable export)
    ├── cli.py                      ← argparse entry point
    └── templates/report.html.j2    ← printable HTML report (with PDF button)
```

## Web app architecture

```
Browser (single-page Alpine app)
   │   GET  /                          → frontend/index.html
   │   GET  /api/health                → which LLM is wired up
   │   GET  /api/reports               → list of cached runs
   │   GET  /api/reports/{ticker}      → cached JSON
   │   GET  /api/reports/{ticker}/html → printable HTML
   │   POST /api/generate              → kick off pipeline → returns job_id
   │   GET  /api/stream/{job_id}       → Server-Sent Events stream
   ▼
FastAPI (web.py)
   • Background thread runs generate_report() with a progress callback
   • Callback pushes events into an asyncio queue per job
   • SSE endpoint drains the queue → "progress" / "complete" / "error" events
   ▼
Pipeline (src/muddy_reporter/pipeline.py)  ← unchanged from CLI mode
```

---

## Limitations & legal notes

- **Source scope.** Only SEC EDGAR filings + XBRL. No earnings transcripts, no news, no social media, no expert calls. Findings can miss context that a real analyst would have.
- **Hallucination risk.** Even with verbatim-citation guardrails, the LLM can mis-attribute quotes or over-interpret boilerplate. **Every finding requires human review before being relied on.**
- **No fraud assertions.** This system frames concerns as red flags / hypotheses / open questions. It does not — and should not be used to — accuse any company of wrongdoing.
- **Not investment advice.** Nothing here constitutes a recommendation to buy or sell any security.
- **Reputational / legal risk.** Publishing automated bearish content about a public company carries defamation and market-manipulation risk. The output is intended as an internal research scaffold, not as a publishable note.
