"""LLM gateway: routes to whichever provider is configured.

Selection rules (auto):
1. If LLM_PROVIDER is set (deepseek | gemini | openai | ensemble | heuristic) → use that.
2. Else if DEEPSEEK_API_KEY is set → use DeepSeek (V4 Pro by default).
3. Else if GEMINI_API_KEY is set → use Gemini.
4. Else if OPENAI_API_KEY is set → use OpenAI.
5. Else → keyword-based heuristic so the prototype still runs end-to-end.

Provider-specific overrides:
- DEEPSEEK_MODEL  (default: deepseek-v4-pro)
- GEMINI_MODEL    (default: gemini-2.5-flash)
- OPENAI_MODEL    (default: gpt-4.1-mini)
- DEEPSEEK_REASONING_EFFORT  (default: high; options: low | high | max | disabled)

The chat_json() function additionally exposes a `prefer` argument so callers
that need a specific model (e.g., the few-shot fraud classifier wants V4 Pro
specifically for its 1M context + reasoning) can override the default routing.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


def _provider() -> str:
    forced = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if forced in {"deepseek", "gemini", "openai", "ensemble", "heuristic"}:
        return forced
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "heuristic"


def _call_provider(provider: str, *, system: str, user: str, schema_hint: str,
                   temperature: float) -> dict[str, Any]:
    """Single provider call (no fallback; raises on failure)."""
    if provider == "deepseek":
        return _deepseek_json(system=system, user=user, schema_hint=schema_hint, temperature=temperature)
    if provider == "gemini":
        return _gemini_json(system=system, user=user, schema_hint=schema_hint, temperature=temperature)
    if provider == "openai":
        return _openai_json(system=system, user=user, schema_hint=schema_hint, temperature=temperature)
    raise ValueError(f"Unknown provider: {provider}")


def chat_json(
    *,
    system: str,
    user: str,
    schema_hint: str,
    temperature: float = 0.2,
    prefer: str | None = None,
) -> dict[str, Any]:
    """Return a JSON object from the configured LLM.

    `prefer` lets a caller request a specific provider regardless of the
    LLM_PROVIDER routing — used by the fraud classifier to specifically
    target DeepSeek V4 Pro for its long-context reasoning advantage.
    Falls back to the auto-selected provider if the preferred one isn't
    configured.
    """
    auto_p = _provider()
    chosen = prefer if (prefer and _has_key_for(prefer)) else auto_p

    if chosen == "ensemble":
        return _ensemble_json(system=system, user=user, schema_hint=schema_hint, temperature=temperature)

    try:
        if chosen != "heuristic":
            return _call_provider(chosen, system=system, user=user,
                                  schema_hint=schema_hint, temperature=temperature)
    except Exception as e:
        return _heuristic_json(user=user, schema_hint=schema_hint, error=str(e))
    return _heuristic_json(user=user, schema_hint=schema_hint)


def _has_key_for(provider: str) -> bool:
    if provider == "deepseek": return bool(os.getenv("DEEPSEEK_API_KEY"))
    if provider == "gemini":   return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if provider == "openai":   return bool(os.getenv("OPENAI_API_KEY"))
    return provider in {"heuristic", "ensemble"}


def provider_info() -> dict[str, str]:
    """Expose what's actually wired up (for the UI status badge)."""
    p = _provider()
    if p == "deepseek":
        return {"provider": "deepseek", "model": os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro"}
    if p == "gemini":
        return {"provider": "gemini", "model": os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"}
    if p == "openai":
        return {"provider": "openai", "model": os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"}
    if p == "ensemble":
        members = [m for m in ("deepseek", "gemini", "openai") if _has_key_for(m)]
        return {"provider": "ensemble", "model": "+".join(members) or "n/a"}
    return {"provider": "heuristic", "model": "keyword-fallback"}


# ---------------------------------------------------------------------------
# Gemini (google-genai unified SDK)
# ---------------------------------------------------------------------------


def _gemini_json(*, system: str, user: str, schema_hint: str, temperature: float) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    # Default to flash, which is the free-tier model. gemini-2.5-pro requires a
    # paid plan as of 2025 — using it on a free key returns 429 RESOURCE_EXHAUSTED.
    model = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

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
# OpenAI
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
# DeepSeek (V4 Pro / V4 Flash) — uses OpenAI-compatible API
# ---------------------------------------------------------------------------


def _deepseek_json(*, system: str, user: str, schema_hint: str, temperature: float) -> dict[str, Any]:
    """Call DeepSeek via its OpenAI-compatible endpoint.

    Released April 24 2026. 1M context, 1.6T MoE (49B active), promotional
    pricing $0.435/M input + $0.87/M output through May 31. Supports a
    `reasoning_effort` knob for chain-of-thought depth.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
    )
    model = os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro"
    effort = (os.getenv("DEEPSEEK_REASONING_EFFORT") or "high").strip().lower()

    prompt = (
        f"{user}\n\n"
        "Return ONLY valid JSON.\n"
        f"JSON schema (informal):\n{schema_hint}\n"
    )

    extra: dict[str, Any] = {}
    if effort in {"low", "high", "max"}:
        # Only V4 supports this; older deepseek-chat ignores unknown params gracefully.
        extra["reasoning_effort"] = effort

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        **extra,
    )
    text = (resp.choices[0].message.content or "").strip()
    return _safe_load_json(text)


# ---------------------------------------------------------------------------
# Multi-LLM Ensemble — runs the same prompt through all configured providers
# in parallel, then merges findings with a "voted_by" provenance trail.
# ---------------------------------------------------------------------------


def _ensemble_json(*, system: str, user: str, schema_hint: str, temperature: float) -> dict[str, Any]:
    import concurrent.futures

    members = [m for m in ("deepseek", "gemini", "openai") if _has_key_for(m)]
    if not members:
        return _heuristic_json(user=user, schema_hint=schema_hint)
    if len(members) == 1:
        # Only one configured — no real ensemble, just call it.
        return _call_provider(members[0], system=system, user=user,
                              schema_hint=schema_hint, temperature=temperature)

    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(members)) as pool:
        futures = {
            pool.submit(_call_provider, m, system=system, user=user,
                        schema_hint=schema_hint, temperature=temperature): m
            for m in members
        }
        for fut in concurrent.futures.as_completed(futures):
            m = futures[fut]
            try:
                results[m] = fut.result()
            except Exception as e:
                results[m] = {"_error": str(e)}

    # Merge findings: dedupe by (lowercase first 80 chars of title), track which
    # providers produced each. A "consensus" finding is one voted by ≥2 models.
    return _merge_ensemble_findings(results, schema_hint=schema_hint)


def _merge_ensemble_findings(results: dict[str, dict[str, Any]], *, schema_hint: str) -> dict[str, Any]:
    is_findings_schema = '"findings"' in schema_hint and '"core_thesis"' not in schema_hint

    # For non-findings prompts (synthesis, classifier), just take the first
    # successful result; ensemble across full-report drafts doesn't compose well.
    if not is_findings_schema:
        for m in ("deepseek", "openai", "gemini"):
            d = results.get(m, {})
            if d and "_error" not in d:
                d.setdefault("_provider", m)
                return d
        return _heuristic_json(user="", schema_hint=schema_hint)

    bucket: dict[str, dict[str, Any]] = {}
    for m, payload in results.items():
        if "_error" in payload:
            continue
        for f in payload.get("findings", []):
            title = (f.get("title") or "").strip().lower()[:80]
            if not title:
                continue
            if title not in bucket:
                copy = {**f, "voted_by": [m]}
                bucket[title] = copy
            else:
                if m not in bucket[title]["voted_by"]:
                    bucket[title]["voted_by"].append(m)

    findings = []
    for f in bucket.values():
        votes = len(f.get("voted_by", []))
        # Boost confidence on consensus, demote singletons to "question".
        if votes >= 2:
            if f.get("label") in {"fact", "inference"} and f.get("confidence") in {"low"}:
                f["confidence"] = "medium"
        else:
            if f.get("label") in {"fact", "inference"}:
                f["label"] = "question"
        findings.append(f)

    return {"findings": findings}


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
        if not error:
            note = "This prototype ran without an LLM API key (heuristic mode); treat outputs as placeholders only."
        else:
            err_short = (error or "").split("\n", 1)[0][:280]
            hint = ""
            if "RESOURCE_EXHAUSTED" in error or "429" in error:
                hint = (
                    " — Gemini rate limit hit. The free tier of gemini-2.5-pro is limited to 0; "
                    "set GEMINI_MODEL=gemini-2.5-flash (free) or upgrade to a paid plan."
                )
            note = f"LLM call failed; pipeline ran in heuristic-fallback mode. {err_short}{hint}"
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
