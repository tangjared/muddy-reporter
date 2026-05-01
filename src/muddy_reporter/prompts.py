"""Prompts and JSON schema hints for the investigative analyst pipeline.

Design goals:
- Force a skeptical, forensic stance — but with explicit anti-hallucination guardrails.
- Every non-trivial claim must be tied to a citation excerpt that exists in the input.
- The model is instructed to *downgrade* findings to questions when evidence is thin.
- Quantitative anomalies (computed deterministically from XBRL data) are passed in
  as context so the LLM grounds its narrative in real numbers, not vibes.
"""


SYSTEM_INVESTIGATIVE_ANALYST = """\
You are a forensic, skeptical investigative analyst writing a research note in the style of a Muddy Waters / Hindenburg-type short-seller report.

Your job is to find disclosure gaps, contradictions, accounting oddities, governance weaknesses, and inconsistencies — and to frame them precisely.

NON-NEGOTIABLE RULES:
1. NEVER claim "fraud", "scam", "manipulation" or similar unless the source filing or a regulator explicitly says so. Otherwise frame as "red flag", "inconsistency", "weak disclosure", or "open question".
2. EVERY finding must be tied to a citation excerpt that you can quote VERBATIM from the supplied text. If you cannot quote it verbatim, the finding does not exist — drop it or convert it to an open question with no citations.
3. Distinguish strictly between: fact (directly supported by an exact quote), inference (your reasoning over multiple facts), question (you would need more information), and speculation (must be explicitly labeled and rare).
4. NEVER invent numbers, names, dates, or counterparties that are not in the source text.
5. Prefer concrete, specific findings over generic risk-factor commentary. Boilerplate ("we operate in a competitive market") is not a red flag.
6. When a quantitative anomaly is supplied (e.g., AR growing faster than revenue), explain *what specific disclosure or wording in the filing* corroborates or fails to address it.
7. Avoid defamatory language. Stick to neutral forensic phrasing.

You will be shown excerpts from public filings (SEC EDGAR) and may also be shown a precomputed list of financial anomalies. Treat these as the only allowed sources.
"""


FINDINGS_SCHEMA_HINT = """\
{
  "findings": [
    {
      "title": "concise, specific title (avoid vague keywords)",
      "category": "accounting|governance|disclosure|operations|capital_structure|related_parties|regulatory_legal|other",
      "label": "fact|inference|question|speculation",
      "confidence": "low|medium|high",
      "claim_or_observation": "what specifically you noticed, with the relevant numbers or names",
      "why_it_matters": "the bearish/cautionary implication (1-2 sentences)",
      "counterpoints_or_alt_explanations": ["benign explanations a defender of the company would offer"],
      "open_questions": ["specific questions you'd want management to answer"],
      "citations": [
        {"doc_id": "DOC_ID exactly as provided", "url": "SOURCE_URL exactly as provided", "excerpt": "VERBATIM quote from the excerpt"}
      ]
    }
  ]
}
"""


REPORT_SCHEMA_HINT = """\
{
  "company_name": "string",
  "snapshot": {
    "business_description": "1-2 sentence plain-English description of what the company actually does",
    "where_it_operates": "geographic / market scope",
    "segments_or_revenue_model": "how it makes money",
    "recent_corporate_actions": "M&A, restatements, auditor changes, offerings, leadership changes (cite filings)"
  },
  "core_thesis": "2-4 sentences. The single most concerning pattern, framed as a hypothesis worth investigating — not as a verdict.",
  "management_claims_vs_counterpoints": [
    {
      "claim": "what management/the filing asserts (paraphrase)",
      "source_excerpt": "VERBATIM supporting quote",
      "counterpoint": "the inconsistency, missing context, or contradicting evidence (cite numbers)",
      "confidence": "low|medium|high"
    }
  ],
  "concerns_by_category": {
    "accounting": ["short bullet"],
    "governance": ["short bullet"],
    "disclosure": ["short bullet"],
    "operations": ["short bullet"],
    "capital_structure": ["short bullet"],
    "related_parties": ["short bullet"],
    "regulatory_legal": ["short bullet"],
    "other": ["short bullet"]
  },
  "open_questions": ["pointed, specific questions for management or for further diligence"],
  "conclusion": "2-4 sentences. Cautionary framing only. Do NOT recommend a trade. Reiterate verification needs.",
  "limitations": ["honest limitations of this automated analysis"]
}
"""


def build_extractor_user_prompt(*, doc_id: str, filing_type: str, primary_url: str,
                                chunk_idx: int, total_chunks: int, excerpt: str,
                                financial_context: str | None = None) -> str:
    """Per-chunk extraction prompt. Keeps the model anchored to one document at a time."""
    fin_block = ""
    if financial_context:
        fin_block = (
            "\nQUANTITATIVE CONTEXT (precomputed from XBRL — use to ground or challenge the text):\n"
            f"{financial_context}\n"
        )
    return f"""\
Document ID: {doc_id}
Filing type: {filing_type}
Source URL: {primary_url}
{fin_block}
Task:
From this excerpt only, identify potential red flags / inconsistencies / weak disclosures / suspicious wording.
Be skeptical but precise. Prefer concrete, citable findings over generic concerns.
If evidence is thin, output an "open question" instead of a "fact" or "inference".

Strict requirements:
- Every "fact" or "inference" finding MUST include at least one citation with a VERBATIM excerpt from the text below.
- Use doc_id and url EXACTLY as shown above.
- Do NOT introduce numbers, names, or events that are not in the excerpt below.
- Skip boilerplate risk-factor language unless it materially changed vs. typical filings.

Excerpt (chunk {chunk_idx}/{total_chunks}):
\"\"\"
{excerpt}
\"\"\"
"""


def build_synthesis_user_prompt(*, ticker: str, company_name: str | None,
                                sources_brief: str, findings_brief: str,
                                financial_brief: str | None) -> str:
    """Final-report synthesis prompt. Consolidates per-doc findings into a Muddy Waters-style note."""
    fin_block = ""
    if financial_brief:
        fin_block = (
            "\nQUANTITATIVE FINDINGS (deterministic, computed from XBRL):\n"
            f"{financial_brief}\n"
            "Treat these as already-verified numerical facts. Use them to anchor your narrative.\n"
        )
    return f"""\
Ticker: {ticker}
Company name (if known): {company_name}

You have extracted candidate findings from public SEC filings. Synthesize them into a structured investigative note in the voice of a forensic short-seller research firm (Muddy Waters / Hindenburg style) — but governed by these rules:

- Do NOT overreach. Convert weak evidence into open questions, not conclusions.
- Do NOT invent facts, numbers, or counterparties.
- Consolidate the candidate findings into the BEST 3-5 red flags (deduplicate, merge similar items, drop boilerplate).
- For "management_claims_vs_counterpoints": pull the management/issuer claim from the filing wording and pair it with the most concrete contradicting observation you have.
- Keep the tone professional, forensic, and cautious — never defamatory.
- The "core_thesis" is the single sharpest concern, not a laundry list.

Sources analyzed:
{sources_brief}
{fin_block}
Candidate findings (may contain duplicates; consolidate):
{findings_brief}
"""
