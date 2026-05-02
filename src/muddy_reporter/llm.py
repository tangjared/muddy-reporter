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


def _retry_provider_call(provider: str, *, system: str, user: str, schema_hint: str,
                         temperature: float, retries: int = 2) -> dict[str, Any]:
    """Per-provider retry with short exponential backoff for transient failures.

    Triggered specifically by `httpx.RemoteProtocolError`, `APIConnectionError`,
    `APITimeoutError`, and 5xx responses — DeepSeek's edge nodes intermittently
    drop long-running connections mid-response.
    """
    import time as _t

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _call_provider(
                provider, system=system, user=user,
                schema_hint=schema_hint, temperature=temperature,
            )
        except Exception as e:  # noqa: BLE001
            last_exc = e
            msg = str(e).lower()
            transient = any(s in msg for s in (
                "connection error", "remoteprotocolerror", "server disconnected",
                "timeout", "timed out", "502", "503", "504", "internal server error",
                "rate limit", "429",
            ))
            if attempt >= retries or not transient:
                break
            _t.sleep(1.5 * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


def _fallback_chain(prefer: str | None) -> list[str]:
    """Decide what providers to try in order, preserving the caller's preference.

    Order: caller preference (if keyed) → auto-selected → other keyed providers
    → heuristic. Each provider only appears once in the list.
    """
    chain: list[str] = []

    def add(p: str) -> None:
        if p and p not in chain and (p == "heuristic" or _has_key_for(p)):
            chain.append(p)

    if prefer:
        add(prefer)
    auto_p = _provider()
    add(auto_p)
    for p in ("deepseek", "gemini", "openai"):
        add(p)
    chain.append("heuristic")  # always last
    return chain


def chat_json(
    *,
    system: str,
    user: str,
    schema_hint: str,
    temperature: float = 0.2,
    prefer: str | None = None,
) -> dict[str, Any]:
    """Return a JSON object from the configured LLM, with provider fallback.

    Routing:
    1. Build a fallback chain that puts `prefer` first (when keyed), then the
       auto-selected provider, then any other keyed providers, then heuristic.
    2. Each provider gets retried on transient connection failures.
    3. If a provider returns a parseable result, use it. Else move to the next.
    4. Last resort: heuristic. Annotate the heuristic output with the *real*
       upstream error so the UI surfaces something actionable.
    """
    auto_p = _provider()

    if (prefer or auto_p) == "ensemble":
        return _ensemble_json(system=system, user=user, schema_hint=schema_hint, temperature=temperature)

    chain = _fallback_chain(prefer)
    last_error: str | None = None
    for provider in chain:
        if provider == "heuristic":
            return _heuristic_json(user=user, schema_hint=schema_hint, error=last_error)
        try:
            return _retry_provider_call(
                provider, system=system, user=user,
                schema_hint=schema_hint, temperature=temperature, retries=2,
            )
        except Exception as e:  # noqa: BLE001
            last_error = f"[{provider}] {type(e).__name__}: {str(e)[:240]}"
            print(f"[llm] {last_error} — falling back to next provider", flush=True)
            continue
    # Defensive: if we exit the loop without returning, fall to heuristic.
    return _heuristic_json(user=user, schema_hint=schema_hint, error=last_error)


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


# Free-tier Gemini caps at 20 requests / minute on gemini-2.5-flash. We add a
# tiny global throttle + transparent retry on 429s so a normal report (12-18
# chunks + synthesis + classifier) stays inside quota without surfacing errors.
_LAST_GEMINI_CALL_TS: float = 0.0


def _gemini_json(*, system: str, user: str, schema_hint: str, temperature: float) -> dict[str, Any]:
    import time as _time

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

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        max_output_tokens=8192,
    )

    # ~3.2s gap = 18-19 RPM, comfortably under the 20-RPM free-tier ceiling.
    min_gap = float(os.getenv("GEMINI_MIN_GAP_S") or "3.2")

    last_err: Exception | None = None
    for attempt in range(3):
        global _LAST_GEMINI_CALL_TS
        now = _time.monotonic()
        wait = max(0.0, min_gap - (now - _LAST_GEMINI_CALL_TS))
        if wait:
            _time.sleep(wait)
        _LAST_GEMINI_CALL_TS = _time.monotonic()

        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            return _safe_load_json((resp.text or "").strip())
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                # Honour the server's retry suggestion if it provided one.
                m = re.search(r"retry in ([\d.]+)s", msg)
                backoff = float(m.group(1)) + 0.5 if m else (4.0 + 4.0 * attempt)
                _time.sleep(min(backoff, 30.0))
                continue
            raise

    raise last_err if last_err else RuntimeError("Gemini call failed without an error")


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
    `reasoning_effort` knob for chain-of-thought depth (disabled by default
    here because it adds ~100s of latency per call).
    """
    import httpx
    from openai import OpenAI

    # Hard 600s ceiling. DeepSeek V4 with reasoning_effort=high routinely takes
    # 90-180s on a 9k-char prompt; the cheap edge nodes also intermittently
    # close idle TCP after ~60s, which would manifest as RemoteProtocolError.
    # We mitigate by streaming + a generous timeout.
    timeout_s = float(os.getenv("DEEPSEEK_TIMEOUT_S") or "600")
    http_client = httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=20.0),
        # Keep-alive headers help the upstream LB hold the connection open
        # while the model is reasoning silently.
        headers={"Connection": "keep-alive"},
    )
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        timeout=timeout_s,
        max_retries=0,
        http_client=http_client,
    )
    model = os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro"
    # Default DISABLED — reasoning_effort=high adds 60-150s latency per call,
    # which dramatically increases the chance of mid-flight disconnects on a
    # 10-call report. Override with low/high/max via env if you specifically
    # want deeper CoT for a one-off batch.
    effort = (os.getenv("DEEPSEEK_REASONING_EFFORT") or "disabled").strip().lower()

    prompt = (
        f"{user}\n\n"
        "Return ONLY valid JSON.\n"
        f"JSON schema (informal):\n{schema_hint}\n"
    )

    extra: dict[str, Any] = {}
    if effort in {"low", "high", "max"}:
        extra["reasoning_effort"] = effort

    max_out = int(os.getenv("DEEPSEEK_MAX_TOKENS") or "8192")
    use_stream = (os.getenv("DEEPSEEK_STREAM") or "1").strip() not in {"0", "false", "no"}

    if not use_stream:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_out,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **extra,
        )
        text = (resp.choices[0].message.content or "").strip()
    else:
        # Streaming keeps the TCP connection warm — DeepSeek's edge nodes drop
        # silent connections after ~60s, which is the root of the
        # `RemoteProtocolError: Server disconnected` we saw on synthesis.
        chunks: list[str] = []
        stream = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_out,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **extra,
        )
        for ev in stream:
            try:
                delta = ev.choices[0].delta.content or ""
            except (IndexError, AttributeError):
                delta = ""
            if delta:
                chunks.append(delta)
        text = ("".join(chunks)).strip()

    if not text:
        raise RuntimeError(
            "DeepSeek returned empty content (response was likely truncated; "
            f"raise DEEPSEEK_MAX_TOKENS, currently {max_out})"
        )
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
_FILING_TYPE_RE = re.compile(r"Filing type:\s*(.+)")


_HEURISTIC_TRIGGERS: list[tuple[str, str, str, str, str, str]] = [
    # (keyword regex, title, category, label, confidence, why_it_matters)
    (r"going\s+concern", "Going-concern language present in this filing",
     "disclosure", "question", "medium",
     "Going-concern wording is one of the strongest auditor-level signals of solvency stress and demands a focused review of liquidity, debt walls, and management remediation plans."),
    (r"material\s+weakness", "Material weakness in internal controls disclosed",
     "governance", "inference", "medium",
     "A material weakness in ICFR raises the probability of mis-stated financials and indicates the audit committee or auditor lost confidence in some part of the close process."),
    (r"\brestated\s+(prior|previous|consolidated|financial|annual)|restatement\s+of\s+(financial|prior)|revised\s+previously\s+(reported|issued)\s+financial",
     "Prior-period financial restatement / revision flagged",
     "accounting", "inference", "medium",
     "Restatements imply previously reported numbers were wrong; the size and direction of the change usually drives whether bull or bear gets to keep the original story."),
    (r"related[\s-]?part(y|ies)", "Related-party transactions disclosed",
     "related_parties", "question", "medium",
     "Related-party deals can move economics off the audited statements; the specific party, board approval, and pricing methodology are the diligence priorities."),
    (r"non[-\s]?gaap|adjust(ed)?\s+ebitda", "Heavy reliance on non-GAAP / adjusted metrics",
     "disclosure", "question", "low",
     "When management leans on non-GAAP measures, reconcile their adjustments line by line; aggressive add-backs (SBC, restructuring, lease) often hide deteriorating GAAP economics."),
    (r"impair(ment)?", "Impairment / write-down disclosed",
     "accounting", "question", "low",
     "Impairments tell you where prior optimism (acquisition price, capex, intangibles) didn’t pan out; cluster vs. peers and prior years to spot a serial pattern."),
    (r"covenant", "Debt covenant language requires review",
     "capital_structure", "question", "medium",
     "A breached or close-to-breach covenant compresses optionality and can force asset sales, equity raises, or restructuring on terms unfavourable to existing holders."),
    (r"liquidity\s+(risk|concerns?)|insufficient\s+(cash|liquidity)",
     "Liquidity-related risk language flagged",
     "capital_structure", "question", "medium",
     "Liquidity language combined with debt walls and tight cash is the classic stress trigger; map the next 18-24 months of obligations against on-balance-sheet cash."),
    (r"internal\s+control(s)?", "Internal-control disclosures present",
     "governance", "question", "low",
     "Companies disclose internal-control content even in clean years; what to read for is *changes* in scope, deficiencies, and remediation status year over year."),
    (r"regulator(y|s)\s+(action|inquir|investigat)|sec\s+(enforcement|investigation|subpoena)",
     "Active or recent regulatory action disclosed",
     "regulatory_legal", "inference", "medium",
     "Open enforcement actions are existential; quantify potential fines, disgorgement, and restrictions and check whether the disclosure has expanded in subsequent filings."),
    (r"litigation|class\s+action|settl(ement|ed)\s+(claims?|lawsuit)",
     "Litigation / class-action exposure disclosed",
     "regulatory_legal", "question", "low",
     "Material lawsuits should be pressure-tested for reserve adequacy and disclosed loss contingency framing (probable / reasonably possible / remote)."),
    (r"auditor\s+(change|resign|dismiss)|change\s+in\s+(certifying\s+)?accountant",
     "Auditor change disclosed",
     "governance", "inference", "high",
     "Auditor changes — especially involuntary ones — are one of the highest base-rate forensic signals; check the 8-K item 4.01 for resignation language and any disagreements."),
    (r"channel\s+stuff|days\s+sales\s+outstanding|\bDSO\b",
     "Channel-stuffing / receivables collection language",
     "accounting", "question", "medium",
     "Aggressive end-of-period sell-in lifts revenue without lifting cash; pair this excerpt with the AR-vs-revenue chart in the Anomalies section."),
    (r"reserve\s+release|change\s+in\s+(estimate|reserve)|loss\s+reserve",
     "Reserve / estimate change disclosed",
     "accounting", "question", "low",
     "Reserve releases and estimate changes are the cleanest way to manage earnings; check whether the timing aligns with covenant tests, executive comp targets, or deal completion."),
    (r"customer\s+concentration|single\s+customer|major\s+customer",
     "Customer-concentration risk disclosed",
     "operations", "question", "low",
     "Concentration with a single customer transforms idiosyncratic counterparty risk into existential risk; pull historical revenue mix to confirm whether the dependence is rising."),
    (r"supply\s+chain|raw\s+materials\s+shortage|key\s+supplier",
     "Supply-chain dependency or stress disclosed",
     "operations", "question", "low",
     "Supply-side disclosures bear on gross margin durability and inventory write-down risk."),
    (r"short[\s-]?seller|short[\s-]?selling\s+report|hindenburg|muddy\s+waters|investigative\s+(short|report)|whistleblower\s+(complaint|allegation)",
     "Short-seller / activist allegations referenced",
     "disclosure", "question", "low",
     "Tracking a company’s own response to a short report tells you which allegations management thinks they need to neutralize publicly."),
    (r"discontin(ued|uation)|wind[\s-]?down|exit(ed|ing)\s+(business|segment)",
     "Discontinued operations / segment exit disclosed",
     "operations", "inference", "low",
     "Segment exits often trigger one-time charges and re-baseline guidance; verify that the continuing-ops trend is itself attractive once the exit is removed."),
]


def _heuristic_findings_for_doc(*, doc_id: str, url: str, filing_type: str | None,
                                excerpt: str) -> list[dict[str, Any]]:
    """Pattern-based finding extractor for use when no LLM key is configured."""
    if not excerpt.strip():
        return []
    lines = [ln.strip() for ln in excerpt.splitlines() if ln.strip()]
    joined = "\n".join(lines)
    lower = joined.lower()

    def cite(snippet: str) -> dict[str, str]:
        snippet = snippet.strip()
        if len(snippet) > 900:
            snippet = snippet[:897].rstrip() + "…"
        return {"doc_id": doc_id or "UNKNOWN", "url": url or "", "excerpt": snippet}

    seen_titles: set[str] = set()
    findings: list[dict[str, Any]] = []

    for pattern, title, cat, label, conf, why in _HEURISTIC_TRIGGERS:
        m = re.search(pattern, lower)
        if not m:
            continue
        if title in seen_titles:
            continue
        # Pick the line containing the match for citation context.
        idx = lower.find(m.group(0))
        line = ""
        if idx != -1:
            # Walk to nearest line break window.
            start = max(0, idx - 200)
            end = min(len(joined), idx + 300)
            line = joined[start:end].strip()
        if not line:
            line = next((ln for ln in lines if re.search(pattern, ln.lower())), "")
        if not line:
            continue
        keyword = m.group(0)
        findings.append(
            {
                "title": title,
                "category": cat,
                "label": label,
                "confidence": conf,
                "claim_or_observation": (
                    f"In the {filing_type or 'filing'} excerpt, language matching '{keyword}' is present: "
                    f"\"{line[:280]}\"."
                ),
                "why_it_matters": why,
                "counterpoints_or_alt_explanations": [
                    "Risk-factor and MD&A boilerplate often contains these words even in healthy companies — context decides severity.",
                    "Compare the precise wording to the same section in the prior filing year; an unchanged paragraph is usually neutral.",
                ],
                "open_questions": [
                    f"Does the surrounding section quantify the impact of '{keyword}' (dollars, units, customers, time horizon)?",
                    "Has the disclosure expanded, contracted, or changed tone vs the previous filing of the same form?",
                ],
                "citations": [cite(line)],
            }
        )
        seen_titles.add(title)

    if findings:
        return findings[:5]

    seed = lines[0] if lines else excerpt[:400]
    return [
        {
            "title": "No salient red-flag keywords in this excerpt",
            "category": "other",
            "label": "question",
            "confidence": "low",
            "claim_or_observation": (
                "Pattern-based scan did not surface a high-priority signal in this chunk; the LLM-disabled prototype mode "
                "is conservative and converts weak evidence into diligence questions rather than findings."
            ),
            "why_it_matters": (
                "Even a clean excerpt is informative — the absence of stress language in this section narrows where to "
                "concentrate time during a manual read."
            ),
            "counterpoints_or_alt_explanations": [
                "The pattern library is not exhaustive; subtle disclosures (e.g., footnote-only related-party deals) can be missed.",
            ],
            "open_questions": [
                "Which sections of this filing changed most vs the prior year (use the SEC redline tool if available)?",
                "Are there judgment-heavy accounting policies (revenue recognition, capitalized R&D, lease classification) worth focusing on?",
            ],
            "citations": [cite(seed)],
        }
    ]


def _heuristic_json(*, user: str, schema_hint: str, error: str | None = None) -> dict[str, Any]:
    doc_id = None
    url = None
    filing_type = None
    m = _DOC_ID_RE.search(user)
    if m:
        doc_id = m.group(1).strip()
    m = _URL_RE.search(user)
    if m:
        url = m.group(1).strip()
    m = _FILING_TYPE_RE.search(user)
    if m:
        filing_type = m.group(1).strip()
    m = _EXCERPT_RE.search(user)
    excerpt = (m.group(1).strip() if m else user[-4000:]).strip()

    findings = _heuristic_findings_for_doc(
        doc_id=doc_id or "UNKNOWN",
        url=url or "",
        filing_type=filing_type,
        excerpt=excerpt,
    )

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
