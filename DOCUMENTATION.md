# Build Documentation — Muddy Reporter

## 1. Tools used

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.10+ | EDGAR + XBRL ergonomics, mature LLM SDKs, Streamlit. |
| LLM (default) | **Google Gemini 2.5 Pro** via `google-genai` | Free tier (AI Studio); 2M-token context window comfortably fits a full 10-K / 20-F + multiple 10-Q / 6-K filings without RAG; native JSON output mode. |
| LLM (fallback) | OpenAI (`gpt-4.1-mini`) | Drop-in fallback; same JSON-mode contract. |
| LLM (last resort) | Keyword heuristic | Lets the prototype run end-to-end with no API key, so reviewers can see the architecture and the deterministic anomaly engine even without quotas. |
| Filings | SEC EDGAR (`data.sec.gov`, `www.sec.gov/Archives/edgar/data`) | Authoritative, free, cacheable. |
| Financials | SEC XBRL company-facts API | Tagged financial concepts; works for both US-GAAP and IFRS-tagged foreign issuers. |
| Models | `pydantic` v2 | Strict schemas for Reports, Findings, Citations. |
| HTML parsing | `beautifulsoup4` + `lxml` | Tolerant of EDGAR's messy HTML. |
| Templating | `jinja2` | Self-contained printable HTML export. |
| Web server | `fastapi` + `uvicorn` + `sse-starlette` | Async API with Server-Sent Events for live progress streaming. |
| Frontend | Tailwind CSS + Alpine.js + Plotly.js (all via CDN) | Editorial dark UI; no Node, no build step, no second deployable. |

## 2. Workflow / system diagram

```
┌────────────────────────────────────────────────────────────┐
│ Browser — Single-page app (frontend/index.html)            │
│  • Landing : ticker input, preset cases, recent runs       │
│  • Loading : SSE-streamed progress + activity log          │
│  • Report  : sticky TOC + flag cards + Plotly charts       │
└────────────────┬───────────────────────────────────────────┘
                 │  POST /api/generate { ticker }
                 │  ◄── { job_id }
                 │  GET  /api/stream/{job_id}  (Server-Sent Events)
                 ▼
┌────────────────────────────────────────────────────────────┐
│ web.py — FastAPI                                           │
│  • Spawns a background thread per job                      │
│  • Pushes progress events into an asyncio queue            │
│  • SSE endpoint drains the queue → progress/complete/error │
└────────────────┬───────────────────────────────────────────┘
                 │  generate_report(ticker, ..., progress=cb)
                 ▼
┌────────────────────────────────────────────────────────────┐
│ pipeline.py                                                │
│  1. ticker_to_cik() ────► sec_edgar.py                     │
│  2. list_recent_filings() — auto-detects domestic/foreign  │
│  3. download_filing_primary_doc() (cached on disk)         │
│  4. build_financial_snapshot() ────► financials.py         │
│       └─► deterministic anomaly checks (no LLM)            │
│  5. _extract_findings_for_doc() per filing  ──► llm.py     │
│       │   └─► Gemini → OpenAI → heuristic                  │
│       │   └─► verification: drop facts w/o citations       │
│  6. _build_report() — synthesis prompt → final JSON        │
│  7. render_html() ────► templates/report.html.j2           │
└────────────────────────────────────────────────────────────┘
```

## 3. Data sources & rationale

| Source | Purpose | Why this and not something else |
|---|---|---|
| SEC EDGAR submissions feed | List recent filings per CIK | Authoritative; free; clean JSON. |
| EDGAR Archives (10-K / 10-Q / 8-K / 20-F / 6-K / F-1) | Primary text for narrative red flags | The legally responsible disclosures. Anything else is secondary. |
| SEC XBRL company-facts | Time-series of tagged financials | Already structured (no PDF parsing); supports both US-GAAP and IFRS taxonomies; consistent across filers. |
| (Future) Earnings call transcripts, news, short interest | Cross-validation, claim/counter-claim mining | Not in v1 — adds licensing/scraping cost without changing core architecture. |

We deliberately *exclude* social-media data, opinion pieces, and unverified short-seller reports from the input layer to keep the system grounded in primary public disclosures.

## 4. Key prompts and prompt iterations

### System prompt (current)
- Forensic / Muddy Waters voice.
- Hard rules: no "fraud" without explicit source support, every fact/inference must cite verbatim, separate fact / inference / question / speculation, no invented numbers, ignore boilerplate.
- Foreign issuers + multi-doc instructions.

### Per-chunk extractor prompt
- Pinned to one document at a time (`Document ID`, `Filing type`, `Source URL`).
- Inserts a precomputed list of XBRL anomalies as **Quantitative Context**, so the model can corroborate or challenge the narrative against numbers.
- Demands verbatim citations using the supplied `doc_id` / `url`.

### Synthesis prompt
- Takes deduped per-doc findings + XBRL anomaly summary.
- Asks for the **best 3–5 red flags** (consolidated), claim-vs-counterpoint pairs, and a single sharpest core thesis.
- Forbids overreach; thin findings must become open questions.

### Iteration log
| Iteration | Change | Why |
|---|---|---|
| v0 (baseline) | Single prompt, no schema, free-form text | Output was unstructured, hard to render, no citations. |
| v1 | JSON schema + per-finding citation field | Made verification possible. |
| v2 | Strict citation rule (verbatim or downgrade) | LLM was paraphrasing source language; broke audit trail. |
| v3 | Added quantitative-context block from XBRL | LLM was missing obvious numeric anomalies the deterministic engine catches. Now it grounds narrative in real numbers. |
| v4 | Auto-detect foreign-issuer forms (20-F / 6-K / F-1) | Original code only looked for 10-K / 10-Q / 8-K, so it found nothing on Luckin or other ADRs. |
| v5 | Deduplication of similar findings before synthesis | Synthesis prompt was being drowned in near-duplicate keyword hits across chunks. |

## 5. How the system separates fact / inference / allegation / open question

| Label | Required evidence | Allowed in report |
|---|---|---|
| `fact` | A verbatim excerpt from the supplied source text. Without it, automatically demoted. | Yes — bolded as observation. |
| `inference` | At least one supporting citation, plus reasoning shown in `claim_or_observation`. | Yes — flagged as inference. |
| `question` | None required — but should be a *specific* question, not a vague concern. | Yes — preferred when evidence is thin. |
| `speculation` | Must be explicitly labeled. Used very rarely. | Yes — but visually flagged as speculation in the UI (red badge). |

The pipeline's `_extract_findings_for_doc` enforces this at code level: any `fact` or `inference` without a non-empty citation `excerpt` is silently rewritten to `question`. The LLM cannot promote weak evidence past the gate.

## 6. How the system handles confidence, citations, and quality checks

- **Confidence**: `low | medium | high`, requested from the LLM and constrained by the cited evidence. The UI badges them by color.
- **Citations**: every excerpt is rendered as a clickable link (back to the EDGAR primary URL) plus the verbatim quoted text. This makes manual verification a one-click operation.
- **Deterministic backstop**: financial anomalies have severity (`high | medium | low`) computed from the magnitude of the deviation (e.g., AR / revenue growth gap). These do *not* depend on the LLM — even the heuristic mode shows them.
- **Negative control**: AAPL (Apple) returns **zero** financial anomalies in our default thresholds, demonstrating the system isn't inventing concerns on clean balance sheets.

## 7. Challenges & limitations

| Challenge | Mitigation | Residual risk |
|---|---|---|
| LLM hallucination of quotes | Force verbatim-only citations; auto-demote findings without them. | Model may still misquote a real passage. Human review required. |
| Long filings (20-F often >150 pages) | Chunk to 12K chars, cap at 6 chunks per filing, rely on Gemini's 2M context to keep cross-chunk awareness via the synthesis pass. | Some details still get truncated. |
| Foreign issuers using IFRS taxonomy | Concept map searches both `us-gaap` and `ifrs-full`; many ADRs (incl. Luckin) use US-GAAP anyway. | Some IFRS-only filers may have sparse data. |
| Pre-restatement / pre-IPO data missing from XBRL | Flagged in `limitations`. | Cannot detect anomalies in years not tagged. |
| SEC rate-limit / etiquette | All HTTP cached on disk; polite `User-Agent`; 120 ms delay between cold fetches. | Heavy parallel use could still trip rate limits. |
| Defamation / market-manipulation risk if output is published | "Hypothesis"-only framing in prompts; explicit "not investment advice" in headers; built-in `limitations` section. | Output is **for internal research only**, not for publication. |

## 8. What the AI did well vs what required human intervention

**AI did well:**
- Picking up domain-specific phrasing in MD&A and risk-factor sections that maps cleanly to forensic categories (governance, related parties, capital structure, etc.).
- Drafting compact "claim vs counterpoint" pairs once both halves were available.
- Following a strict JSON schema across heterogeneous filings.
- Generating diligence questions that read like a real analyst's first round of management Q&A.

**Required human (or deterministic) intervention:**
- **Quantitative anomalies.** The LLM consistently *under-detected* numeric anomalies when reading prose alone. We had to compute these deterministically from XBRL and *feed them in* before the LLM read the filings.
- **Defamation control.** Without explicit guardrails, the LLM tended to escalate language ("manipulation", "scheme") on weak evidence.
- **Foreign issuer detection.** Required a small heuristic (auto-pick form types based on what the issuer actually files).
- **Deduplication.** The LLM happily produced near-duplicate findings across overlapping document chunks; we de-dup on `(title, claim_or_observation)` prefixes before synthesis.
