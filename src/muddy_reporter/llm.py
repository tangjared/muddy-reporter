"""LLM gateway with Gemini-first, OpenAI-fallback, heuristic-last strategy.

Selection rules (auto):
1. If GEMINI_API_KEY (or GOOGLE_API_KEY) is set → use Google Gemini
   (free tier via AI Studio; large context window; native JSON mode).
2. Else if OPENAI_API_KEY is set → use OpenAI.
3. Else → keyword-based heuristic so the prototype still runs end-to-end.

You can force a provider with LLM_PROVIDER=gemini|openai|heuristic.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


def _provider() -> str:
    forced = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if forced in {"gemini", "openai", "heuristic"}:
        return forced
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "heuristic"


def chat_json(*, system: str, user: str, schema_hint: str, temperature: float = 0.2) -> dict[str, Any]:
    """Return a JSON object from the configured LLM.

    All providers are wrapped so callers always get a plain dict. Failures
    degrade gracefully to the heuristic so the pipeline never crashes.
    """
    provider = _provider()
    try:
        if provider == "gemini":
            return _gemini_json(system=system, user=user, schema_hint=schema_hint, temperature=temperature)
        if provider == "openai":
            return _openai_json(system=system, user=user, schema_hint=schema_hint, temperature=temperature)
    except Exception as e:
        # Soft-fail to heuristic so the report still gets produced.
        return _heuristic_json(user=user, schema_hint=schema_hint, error=str(e))
    return _heuristic_json(user=user, schema_hint=schema_hint)


def provider_info() -> dict[str, str]:
    """Expose what's actually wired up (for the UI status badge)."""
    p = _provider()
    if p == "gemini":
        return {"provider": "gemini", "model": os.getenv("GEMINI_MODEL") or "gemini-2.5-pro"}
    if p == "openai":
        return {"provider": "openai", "model": os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"}
    return {"provider": "heuristic", "model": "keyword-fallback"}


# ---------------------------------------------------------------------------
# Gemini (google-genai unified SDK)
# ---------------------------------------------------------------------------


def _gemini_json(*, system: str, user: str, schema_hint: str, temperature: float) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL") or "gemini-2.5-pro"

    client = genai.Client(api_key=api_key)

    prompt = (
        f"{user}\n\n"
        "Return ONLY a single JSON object that matches this informal schema:\n"
        f"{schema_hint}\n"
        "Do not include markdown fences or commentary."
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            max_output_tokens=8192,
        ),
    )
    text = (resp.text or "").strip()
    return _safe_load_json(text)


# ---------------------------------------------------------------------------
# OpenAI fallback
# ---------------------------------------------------------------------------


def _openai_json(*, system: str, user: str, schema_hint: str, temperature: float) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"

    prompt = (
        f"{user}\n\n"
        "Return ONLY valid JSON.\n"
        f"JSON schema (informal):\n{schema_hint}\n"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    return _safe_load_json(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_load_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        # Strip common ```json fences.
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to recover by locating the outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# ---------------------------------------------------------------------------
# Heuristic last-resort fallback (no API key needed)
# ---------------------------------------------------------------------------


_DOC_ID_RE = re.compile(r"Document ID:\s*(.+)")
_URL_RE = re.compile(r"Source URL:\s*(.+)")
_EXCERPT_RE = re.compile(r"Excerpt .*?:\n([\s\S]+)$")


def _heuristic_json(*, user: str, schema_hint: str, error: str | None = None) -> dict[str, Any]:
    doc_id = None
    url = None
    m = _DOC_ID_RE.search(user)
    if m:
        doc_id = m.group(1).strip()
    m = _URL_RE.search(user)
    if m:
        url = m.group(1).strip()
    m = _EXCERPT_RE.search(user)
    excerpt = (m.group(1).strip() if m else user[-4000:]).strip()

    def cite(snippet: str) -> dict[str, str]:
        return {"doc_id": doc_id or "UNKNOWN", "url": url or "", "excerpt": snippet.strip()[:900]}

    lines = [ln.strip() for ln in excerpt.splitlines() if ln.strip()]
    joined = "\n".join(lines)

    triggers = [
        ("going concern", "disclosure", "question", "low"),
        ("material weakness", "accounting", "question", "medium"),
        ("restatement", "accounting", "question", "medium"),
        ("related party", "related_parties", "question", "medium"),
        ("non-gaap", "disclosure", "question", "low"),
        ("impair", "accounting", "question", "low"),
        ("covenant", "capital_structure", "question", "medium"),
        ("liquidity", "capital_structure", "question", "low"),
        ("internal control", "governance", "question", "low"),
        ("regulatory", "regulatory_legal", "question", "low"),
        ("litigation", "regulatory_legal", "question", "low"),
    ]

    findings = []
    for kw, cat, label, conf in triggers:
        if kw in joined.lower():
            hit = next((ln for ln in lines if kw in ln.lower()), "") or joined[:400]
            findings.append(
                {
                    "title": f"Keyword signal: {kw}",
                    "category": cat,
                    "label": label,
                    "confidence": conf,
                    "claim_or_observation": (
                        f"The filing excerpt includes language related to '{kw}', which may warrant follow-up diligence."
                    ),
                    "why_it_matters": (
                        "Such disclosures sometimes correlate with elevated financial, governance, or "
                        "operational risk (needs verification in full context)."
                    ),
                    "counterpoints_or_alt_explanations": [
                        "Keyword hits can be boilerplate risk-factor language rather than a specific adverse event.",
                        "Full document context may reduce or eliminate the concern.",
                    ],
                    "open_questions": [
                        f"What is the precise context around the '{kw}' language (section, scope, magnitude, timeframe)?",
                        "Has this disclosure changed materially vs prior filings?",
                    ],
                    "citations": [cite(hit)],
                }
            )

    if not findings:
        seed = lines[0] if lines else excerpt[:400]
        findings = [
            {
                "title": "Insufficient evidence in excerpt (needs review)",
                "category": "other",
                "label": "question",
                "confidence": "low",
                "claim_or_observation": (
                    "This excerpt alone does not contain a clear contradiction or red flag; further document review is required."
                ),
                "why_it_matters": "A skeptical workflow should avoid overreach and convert weak signals into diligence questions.",
                "counterpoints_or_alt_explanations": [],
                "open_questions": [
                    "Which metrics, KPIs, or accounting policies appear most judgment-heavy in the full filing?",
                    "Are there sudden changes in definitions, segmentation, or disclosure detail year-over-year?",
                ],
                "citations": [cite(seed)],
            }
        ]

    if '"core_thesis"' in schema_hint and '"snapshot"' in schema_hint:
        note = (
            "This prototype ran without an LLM API key (heuristic mode); treat outputs as placeholders only."
            if not error
            else f"LLM call failed and pipeline degraded to heuristic mode. Last error: {error}"
        )
        return {
            "company_name": None,
            "snapshot": {
                "business_description": "Unknown (LLM disabled; heuristic mode).",
                "where_it_operates": "Unknown (LLM disabled; heuristic mode).",
                "segments_or_revenue_model": "Unknown (LLM disabled; heuristic mode).",
                "recent_corporate_actions": "Unknown (LLM disabled; heuristic mode).",
            },
            "core_thesis": "Preliminary caution: automated heuristic scan found potential disclosure keywords; requires human verification.",
            "management_claims_vs_counterpoints": [],
            "concerns_by_category": {
                "accounting": [],
                "governance": [],
                "disclosure": [],
                "operations": [],
                "capital_structure": [],
                "related_parties": [],
                "regulatory_legal": [],
                "other": [],
            },
            "open_questions": [
                "Run with a real LLM key (Gemini or OpenAI) to extract richer contradictions, claim-vs-evidence comparisons, and cross-document inconsistencies.",
            ],
            "conclusion": note,
            "limitations": [
                "LLM disabled: generated content is heuristic and low-signal.",
                "Only SEC filings are analyzed by default.",
                "All points require human verification.",
            ],
        }

    return {"findings": findings}
