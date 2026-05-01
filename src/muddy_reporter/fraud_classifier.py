"""Few-shot in-context fraud classifier.

This is the system's "ML layer" — but instead of training an XGBoost on
Compustat + AAER labels (8-10 hours of data wrangling) we use few-shot
in-context learning with DeepSeek V4 Pro (1M context, MoE 1.6T params).

The model is anchored to a curated catalog of landmark cases so its output
is calibrated rather than free-form:

    Confirmed fraud:    Enron, Wirecard, Luckin, Nikola, WorldCom
    Clean controls:     Apple, Costco, Procter & Gamble, Microsoft

For each new case we feed the model:
    - The 27-dim feature signature (Beneish components, Piotroski tests,
      Altman Z, our XBRL anomalies, the LLM-extracted red-flag counts)
    - The few-shot library above
    - Strict JSON output schema

Output:
    {
      "fraud_probability": 0.0-1.0,
      "verdict": "clean | watch | suspicious | likely_fraud",
      "confidence": "low | medium | high",
      "reasoning": "...",
      "similar_to": ["Wirecard (2020)"],
      "differentiators": "...",
      "key_red_flags": [...],
      "mitigating_factors": [...],
      "model": "deepseek-v4-pro"   // populated by gateway
    }

Compared to a trained classifier this approach trades a bit of latency
(~3s per call) for: zero training data wrangling, full natural-language
reasoning explaining each decision, and easy iteration (just edit the
catalog to teach new patterns).
"""

from __future__ import annotations

import json
from typing import Any

from .llm import chat_json, provider_info


SYSTEM_FORENSIC_CLASSIFIER = """\
You are a forensic accounting expert with two decades of experience analyzing financial \
misrepresentation. You think like a Muddy Waters / Hindenburg analyst: skeptical, \
quantitative, and obsessed with the gap between what management says and what the \
numbers show.

You will be given a curated library of landmark cases (confirmed frauds and clean \
controls) plus a new company's feature signature. Your task is to estimate the \
probability that the new company exhibits financial misrepresentation patterns and to \
explain *why* in the language of forensic accounting.

Hard rules:
- NEVER claim the company *is* committing fraud. Output is a probability + similarity \
  to known cases, not a verdict.
- Treat the few-shot library as your only source of truth for "what fraud looks like".
- If the signature is ambiguous, output a moderate probability (0.3-0.5) and explain \
  the uncertainty.
- Control companies (Apple, Costco, P&G, MSFT) should always score < 0.20 because \
  the catalog explicitly tells you they're clean.
- If the input data is too sparse (most fields null), set confidence="low" and \
  probability ≈ 0.5 with a "data insufficient" reasoning.
"""


# Few-shot library. Each case has a short fingerprint that mirrors the feature
# vector we'll send for new companies. Numbers are illustrative based on
# publicly available analyses; they're calibration guidance for the model, not
# claims of fact.
FEW_SHOT_LIBRARY = """\
=== CONFIRMED FRAUD CASES ===

CASE 1 — Enron Corp (filed bankruptcy Dec 2001)
  Industry: Energy / commodity trading
  Beneish M-Score (1997-2000): consistently ~0.81 to 1.20 (well above -1.78 threshold)
  Piotroski F: 3-4 / 9 (weak in last 2 years)
  Altman Z': declining year over year
  Pattern: aggressive use of mark-to-market accounting on illiquid contracts; off-balance-sheet
    Special Purpose Entities (Raptors, JEDI) hide debt and inflate earnings; revenue grew 800%
    in 4 years while operating cash flow stagnated; auditor (Andersen) deeply conflicted via
    consulting fees.
  Forensic signature: Net income running well ahead of CFO; "related-party" disclosures buried
    in footnotes; unexplained jumps in goodwill / intangibles; receivables and "other assets"
    growing faster than sales.
  Outcome: ~$74B market-cap evaporated; CEO + CFO criminally convicted; Sarbanes-Oxley enacted.

CASE 2 — Wirecard AG (collapsed June 2020)
  Industry: German payments processing
  Beneish M-Score: ~0.85 (manipulator territory) for years before collapse
  Piotroski F: 4 / 9
  Altman Z': flattering, masked by phantom assets
  Pattern: €1.9B in claimed escrow cash deposits in Philippines / Singapore did not exist;
    auditor (EY) failed to verify cash balances for years; pattern surfaced by FT journalists
    via on-site checks of partner offices that were empty rooms.
  Forensic signature: Cash on balance sheet > industry norm with no clear use; "third-party
    acquiring" revenue from undisclosed counterparties; rapid M&A roll-up obscuring organic
    trends; auditor change considered but not executed.
  Outcome: Insolvency filed 25 June 2020; CEO Markus Braun arrested.

CASE 3 — Luckin Coffee (LK / LKNCY, exposed Jan 2020)
  Industry: Chinese coffee retail chain
  Beneish: limited data (foreign issuer, sparse XBRL tagging)
  Pattern: $310M in fabricated sales receipts via fake transactions; Muddy Waters obtained
    11,000+ hours of store-level video and 25,843 receipts showing actual same-store sales
    were 40-80% below reported.
  Forensic signature: AR growing materially faster than revenue (channel stuffing);
    NI > CFO gap; explosive growth claims unmatched by physical store operations;
    related-party deals with the chairman's family; auditor change after IPO.
  Outcome: Stock fell 80% in 2 weeks; SEC fined $180M; delisted from NASDAQ; later relisted on OTC.

CASE 4 — Nikola Corp (NKLA, exposed Sep 2020)
  Industry: EV / hydrogen fuel-cell trucks
  Pattern: Hindenburg revealed Nikola One truck "rolling downhill" demo (no actual drive train),
    no patents on claimed in-house technology, fake hydrogen station, founder Trevor Milton
    repeatedly misrepresented partnerships and capabilities.
  Forensic signature: Persistent operating losses funded by reservations / pre-orders
    that never converted; revenue from "promised contracts" rather than deliveries;
    related-party funding (founder's family); R&D capitalization unusually high vs cash R&D
    spend; no PP&E base for claimed manufacturing capacity.
  Outcome: Founder convicted of fraud in 2022; stock down >90% from peak.

CASE 5 — WorldCom (filed bankruptcy July 2002)
  Industry: US long-distance telecom
  Beneish M: well above -1.78 in 2000-2001
  Pattern: $3.8B in operating expenses (line costs) capitalized as fixed assets to inflate
    EBITDA over 5 quarters in 2001-2002; whistleblower in internal audit (Cynthia Cooper)
    discovered the fraud.
  Forensic signature: Capex jump unexplained by operating activity; gross-margin smoothness
    suspicious in declining industry; large goodwill from acquisitions masking organic
    deterioration; "non-recurring" charges used to mask trend.

=== CLEAN CONTROL CASES (low fraud probability) ===

CASE 6 — Apple Inc (AAPL, current 2024-2025)
  Beneish M: not reliably computable (mature tech with limited PP&E base)
  Piotroski F: 7-8 / 9 (consistently strong)
  Altman Z': artificially LOW (~1.2) due to massive buybacks reducing equity to negative
    territory; this is OPERATIONAL EFFICIENCY, NOT DISTRESS.
  Forensic signature: CFO consistently exceeds NI; transparent disclosure; Big-4 auditor
    (E&Y) for 30+ years with no material auditor opinion changes; receivables grow in line
    with revenue; gross-margin stable.
  Verdict: CLEAN. Low Altman is a known false-positive for buyback-heavy mature companies.

CASE 7 — Costco Wholesale (COST, current)
  Piotroski F: 7-8 / 9
  Forensic signature: AR < 1 day-of-sales (cash business); inventory turnover stable for 20
    years; same-store sales reported with consistent definition; predictable mid-single-digit
    growth; consistent gross margin within ±50 bps band.
  Verdict: CLEAN. The "boring" financial profile is itself a credibility signal.

CASE 8 — Procter & Gamble (PG, current)
  Piotroski F: 7 / 9
  Forensic signature: Mature consumer staples; mid-single-digit revenue growth; CFO > NI
    every year for 20+ years; transparent segment reporting; no goodwill/equity ratio
    concerns; large but stable retained earnings base.
  Verdict: CLEAN.

CASE 9 — Microsoft (MSFT, current)
  Piotroski F: 8 / 9
  Forensic signature: Strong CFO conversion; recurring software revenue with high retention;
    transparent stock-based-compensation disclosure; no auditor changes; minimal off-balance
    structures.
  Verdict: CLEAN.

=== CALIBRATION PRINCIPLES ===

1. Red flags COMPOUND. A single anomaly (e.g., AR > Revenue by 20%) is "needs investigation",
   not "fraud". But AR > Revenue + Beneish > -1.78 + Piotroski < 4 + auditor change is HIGH probability.

2. Foreign issuers (20-F filers) often have sparse XBRL tagging. Missing PP&E / SGA /
   depreciation should LOWER confidence, NOT raise probability.

3. Mature buyback companies (Apple, ExxonMobil) often show low Altman Z due to negative book
   equity. This is NOT distress — DO NOT flag these as fraud risk.

4. Cash-heavy retail (Costco, Walmart) naturally show very low AR. Normal, not suspicious.

5. Tech companies often have minimal PP&E and intangible-heavy balance sheets. Beneish AQI
   may be wonky for them; do not over-weight.

6. Survivorship: many companies look healthy on paper for years before fraud is discovered.
   "Clean signature" alone is not exoneration; the absence of red flags is necessary but
   not sufficient. State this in your reasoning when probability is low.
"""


CLASSIFIER_SCHEMA = """\
{
  "fraud_probability": 0.0,
  "verdict": "clean | watch | suspicious | likely_fraud",
  "confidence": "low | medium | high",
  "reasoning": "3-5 sentence forensic explanation in the language of an analyst note",
  "similar_to": ["case names from the catalog above; empty list if no clear match"],
  "differentiators": "what is different from the most similar case",
  "key_red_flags": ["specific signals that drive the probability up"],
  "mitigating_factors": ["specific signals that drive the probability down"]
}
"""


def _format_features(*, ticker: str, company_name: str | None, financial_brief: dict | None,
                     llm_findings_summary: dict, risk_grade: dict | None) -> str:
    """Render the input feature signature in a stable, machine-readable form."""
    fs = (financial_brief or {}).get("forensic_scores") or {}
    anomalies = (financial_brief or {}).get("anomalies") or []
    table = (financial_brief or {}).get("annual_table") or []

    # Latest-2-year deltas — these are the most diagnostic.
    last_two = ""
    if len(table) >= 2:
        a, b = table[-2], table[-1]
        def yoy(k):
            v0, v1 = a.get(k), b.get(k)
            if v0 and v1 and v0 != 0:
                return f"{((v1 - v0) / abs(v0)) * 100:+.0f}%"
            return "n/a"
        last_two = (
            f"  YoY revenue: {yoy('revenue')}\n"
            f"  YoY AR:      {yoy('accounts_receivable')}\n"
            f"  YoY NI:      {yoy('net_income')}\n"
            f"  YoY CFO:     {yoy('operating_cash_flow')}\n"
            f"  YoY inventory: {yoy('inventory')}\n"
        )

    anomaly_lines = "\n".join(
        f"  - [{a.get('severity','').upper()}] {a.get('title','')}"
        for a in anomalies[:8]
    ) or "  (none above thresholds)"

    return f"""\
=== NEW CASE TO CLASSIFY ===

Ticker: {ticker}
Company: {company_name or 'unknown'}

ACADEMIC SCORES:
  Beneish M-Score: {fs.get('beneish_m')} ({fs.get('beneish_signal')})
  Piotroski F-Score: {fs.get('piotroski_f')}/9 ({fs.get('piotroski_strength')})
  Altman Z'-Score: {fs.get('altman_z')} ({fs.get('altman_zone')})
  Risk Grade (composite): {risk_grade.get('grade') if risk_grade else 'n/a'} ({risk_grade.get('score') if risk_grade else 'n/a'}/100)

LATEST YEAR-OVER-YEAR DELTAS:
{last_two or '  (insufficient annual data points)'}
QUANTITATIVE ANOMALIES (from XBRL, deterministic):
{anomaly_lines}

LLM-EXTRACTED RED FLAGS (deduped, by label):
  fact:        {llm_findings_summary.get('fact', 0)}
  inference:   {llm_findings_summary.get('inference', 0)}
  question:    {llm_findings_summary.get('question', 0)}
  speculation: {llm_findings_summary.get('speculation', 0)}
"""


def classify_fraud_likelihood(
    *,
    ticker: str,
    company_name: str | None,
    financial_brief: dict | None,
    llm_findings: list,
    risk_grade: dict | None,
) -> dict[str, Any]:
    """Run the few-shot classifier and return its structured verdict.

    Specifically prefers DeepSeek V4 Pro for its 1M context + reasoning depth,
    but falls back to whatever provider is configured (so the classifier
    still runs in heuristic mode and just returns a low-confidence stub).
    """
    label_counts: dict[str, int] = {}
    for f in llm_findings or []:
        lbl = getattr(f, "label", None) or "question"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    features = _format_features(
        ticker=ticker,
        company_name=company_name,
        financial_brief=financial_brief,
        llm_findings_summary=label_counts,
        risk_grade=risk_grade,
    )

    user = (
        FEW_SHOT_LIBRARY
        + "\n\n"
        + features
        + "\n\nNow output your classification as a JSON object matching the schema."
    )

    try:
        out = chat_json(
            system=SYSTEM_FORENSIC_CLASSIFIER,
            user=user,
            schema_hint=CLASSIFIER_SCHEMA,
            temperature=0.15,
            prefer="deepseek",
        )
    except Exception as e:
        out = {
            "fraud_probability": 0.5,
            "verdict": "watch",
            "confidence": "low",
            "reasoning": f"Classifier call failed: {e}",
            "similar_to": [],
            "differentiators": "",
            "key_red_flags": [],
            "mitigating_factors": [],
        }

    # Normalize numeric type / clamp range / fill defaults so the UI never breaks.
    try:
        p = float(out.get("fraud_probability") or 0.5)
    except (TypeError, ValueError):
        p = 0.5
    out["fraud_probability"] = max(0.0, min(1.0, round(p, 3)))
    out.setdefault("verdict", "watch")
    out.setdefault("confidence", "medium")
    out.setdefault("reasoning", "")
    out.setdefault("similar_to", [])
    out.setdefault("differentiators", "")
    out.setdefault("key_red_flags", [])
    out.setdefault("mitigating_factors", [])
    _ensure_classifier_narrative(out)
    out["model"] = provider_info()  # provenance: which model produced this verdict

    return out


def _stringify_llm_field(val: Any) -> str:
    """Coerce list/dict-shaped LLM output to plain text for the UI."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        return "\n".join(str(x).strip() for x in val if str(x).strip()).strip()
    if isinstance(val, dict):
        for k in ("summary", "text", "explanation", "analysis", "reasoning"):
            if val.get(k):
                return str(val[k]).strip()
        try:
            return json.dumps(val, ensure_ascii=False, indent=2)[:4000]
        except Exception:
            return str(val).strip()
    return str(val).strip()


def _ensure_classifier_narrative(out: dict[str, Any]) -> None:
    """Guarantee non-empty reasoning for the SPA (models often omit or nest it)."""
    reasoning = _stringify_llm_field(out.get("reasoning"))
    diff = _stringify_llm_field(out.get("differentiators"))
    if not reasoning and diff:
        reasoning = diff

    if not reasoning:
        parts: list[str] = []
        kr = out.get("key_red_flags") or []
        if kr:
            parts.append(
                "Signals weighing on the score: " + "; ".join(str(x) for x in kr[:8] if x)
            )
        mit = out.get("mitigating_factors") or []
        if mit:
            parts.append(
                "Offsetting / benign explanations to weigh: " + "; ".join(str(x) for x in mit[:6] if x)
            )
        sim = out.get("similar_to") or []
        if sim:
            parts.append(
                "Closest analogues in the internal case library: "
                + ", ".join(str(x) for x in sim[:5] if x)
                + "."
            )
        reasoning = " ".join(p for p in parts if p).strip()

    if not reasoning:
        p = float(out.get("fraud_probability") or 0.5)
        v = out.get("verdict") or "watch"
        reasoning = (
            f"The few-shot classifier calibrated a {p * 100:.0f}% misrepresentation-risk score ({v}) "
            "but did not return a narrative. Treat this as incomplete output: rely on deterministic "
            "anomalies, cited red flags, and manual filing review."
        )
    out["reasoning"] = reasoning
    out["differentiators"] = diff

    st: list[str] = []
    for x in out.get("similar_to") or []:
        s = str(x).strip()
        if s:
            st.append(s)
    out["similar_to"] = st

    def _norm_list(key: str) -> None:
        acc: list[str] = []
        for x in out.get(key) or []:
            s = str(x).strip()
            if s:
                acc.append(s)
        out[key] = acc

    _norm_list("key_red_flags")
    _norm_list("mitigating_factors")
