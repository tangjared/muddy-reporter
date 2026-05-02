"""End-to-end pipeline.

Stages:
1. Resolve ticker → CIK and pull recent filings (auto-detects domestic vs foreign forms).
2. Pull XBRL company-facts and run deterministic financial-anomaly checks.
3. For each filing: extract text → chunk → ask the LLM for candidate red flags
   (with the financial context attached so the model can ground its narrative).
4. Synthesize a structured Muddy Waters-style report from all candidate findings.
5. Render HTML + JSON; return a PipelineResult.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .financials import build_financial_snapshot, compute_risk_grade, snapshot_to_brief
from .fraud_classifier import classify_fraud_likelihood
from .llm import chat_json, provider_info
from .ml_scorer import ensemble as ml_ensemble, score as ml_score
from .models import Citation, Finding, PipelineResult, Report
from .prompts import (
    FINDINGS_SCHEMA_HINT,
    REPORT_SCHEMA_HINT,
    SYSTEM_INVESTIGATIVE_ANALYST,
    build_extractor_user_prompt,
    build_synthesis_user_prompt,
)
from .render import render_html, write_html
from .sec_edgar import (
    DEFAULT_ALL_FORMS,
    DEFAULT_DOMESTIC_FORMS,
    DEFAULT_FOREIGN_FORMS,
    download_filing_primary_doc,
    fetch_submissions,
    list_recent_filings,
    ticker_to_cik,
)
from .text_extract import chunk_text, read_doc_text


ProgressFn = Callable[[str, float], None] | None


def _progress(cb: ProgressFn, message: str, fraction: float) -> None:
    if cb:
        try:
            cb(message, max(0.0, min(1.0, fraction)))
        except Exception:
            pass


def _format_financial_context(brief: dict | None) -> str | None:
    if not brief:
        return None
    anomalies = brief.get("anomalies") or []
    if not anomalies:
        return None
    lines = []
    for a in anomalies[:6]:
        lines.append(
            f"- [{a.get('severity','?').upper()}] {a.get('title','')} — {a.get('description','')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Defensive normalization of LLM output. LLMs occasionally invent enum values
# ("inconsistency", "red flag", etc.) that fail strict Pydantic validation.
# We map common variants to legal values so a single weird finding never kills
# the whole pipeline.
# ---------------------------------------------------------------------------

_VALID_LABELS = {"fact", "inference", "question", "speculation"}
_VALID_CONFIDENCE = {"low", "medium", "high"}
_VALID_CATEGORIES = {
    "accounting", "governance", "disclosure", "operations",
    "capital_structure", "related_parties", "regulatory_legal", "other",
}

_LABEL_ALIASES = {
    "inconsistency": "inference",
    "contradiction": "inference",
    "red flag": "inference",
    "red_flag": "inference",
    "concern": "question",
    "warning": "question",
    "allegation": "speculation",
    "claim": "inference",
    "observation": "inference",
    "hypothesis": "inference",
    "open question": "question",
    "open_question": "question",
}

_CATEGORY_ALIASES = {
    "financial": "accounting",
    "auditing": "accounting",
    "audit": "accounting",
    "management": "governance",
    "board": "governance",
    "transparency": "disclosure",
    "reporting": "disclosure",
    "business": "operations",
    "operational": "operations",
    "debt": "capital_structure",
    "leverage": "capital_structure",
    "related party": "related_parties",
    "related-party": "related_parties",
    "legal": "regulatory_legal",
    "regulatory": "regulatory_legal",
    "compliance": "regulatory_legal",
}


def _normalize_label(raw: str | None) -> str:
    s = (raw or "question").strip().lower()
    if s in _VALID_LABELS:
        return s
    if s in _LABEL_ALIASES:
        return _LABEL_ALIASES[s]
    return "question"  # safest conservative default


def _normalize_category(raw: str | None) -> str:
    s = (raw or "other").strip().lower().replace(" ", "_").replace("-", "_")
    if s in _VALID_CATEGORIES:
        return s
    if s in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[s]
    return "other"


def _normalize_confidence(raw: str | None) -> str:
    s = (raw or "low").strip().lower()
    if s in _VALID_CONFIDENCE:
        return s
    aliases = {"weak": "low", "tentative": "low", "moderate": "medium", "strong": "high", "definite": "high"}
    return aliases.get(s, "low")


def _dedupe_question_lines(items: list[str], *, max_items: int = 28) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        q = (raw or "").strip()
        if len(q) < 8:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= max_items:
            break
    return out


def _collect_open_questions(
    payload_questions: list | None,
    findings: list[Finding],
    financial_brief: dict | None,
    classifier: dict | None,
) -> list[str]:
    collected: list[str] = []
    if payload_questions:
        collected.extend(str(x) for x in payload_questions if x)

    for f in findings:
        for q in f.open_questions or []:
            collected.append(str(q))
        if f.label == "question" and f.claim_or_observation and not (f.open_questions or []):
            collected.append(f"{f.title}: {f.claim_or_observation.strip()[:420]}")

    for a in ((financial_brief or {}).get("anomalies") or [])[:12]:
        title = (a.get("title") or "").strip()
        if title:
            collected.append(
                f"[Quant / XBRL] Invite management to explain: {title} "
                "(driver, accounting policy, and where it is disclosed)."
            )

    for flag in (classifier or {}).get("key_red_flags") or []:
        s = str(flag).strip()
        if s:
            collected.append(
                f"[Risk signal] What primary-source evidence would confirm or refute: {s}?"
            )

    return _dedupe_question_lines(collected)


_REQUIRED_CATEGORIES = ("accounting", "governance", "disclosure", "operations")
_OPTIONAL_CATEGORIES = ("capital_structure", "related_parties", "regulatory_legal", "other")


_HEURISTIC_SNAPSHOT_MARKERS = ("LLM disabled", "Unknown (LLM", "Unknown (not extracted")


_NOISE_EXCERPT_RE = re.compile(r"(lkncy:|us-gaap:|dei:|xmlns|xbrl|\bMember\b|\bAxis\b|\bDomain\b)", re.IGNORECASE)


def _excerpt_quality(excerpt: str) -> int:
    """Returns 0 = unusable, 1 = low (digits / tags), 2 = OK, 3 = good."""
    if not excerpt:
        return 0
    s = excerpt.strip()
    if len(s) < 20:
        return 0
    digits = sum(1 for c in s if c.isdigit())
    letters = sum(1 for c in s if c.isalpha())
    if letters == 0:
        return 0
    if digits / max(1, letters) > 0.6:
        return 1
    if _NOISE_EXCERPT_RE.search(s):
        return 1
    if " " not in s:  # single-token blob — usually XBRL tag
        return 1
    return 3 if len(s) >= 80 else 2


_LOW_VALUE_TITLE_PREFIXES = (
    "no salient red-flag",
    "insufficient evidence",
    "keyword signal",
)


def _post_filter_findings(findings: list[Finding]) -> list[Finding]:
    """Re-rank and prune low-value findings before synthesis.

    1. Drop findings whose only citation has unusable / XBRL-noise excerpts.
    2. Push the "no salient" / "insufficient evidence" placeholders to the end.
    3. Cap placeholder findings at 1 across the entire report.
    4. Always return at least one finding so the report renders eight sections.
    """
    cleaned: list[Finding] = []
    for f in findings:
        usable_citations = [c for c in (f.citations or []) if _excerpt_quality(c.excerpt) >= 2]
        if not usable_citations and f.citations:
            if f.title.lower().startswith(_LOW_VALUE_TITLE_PREFIXES):
                continue
        f = Finding(
            **{**f.model_dump(), "citations": usable_citations or f.citations[:1]}
        )
        cleaned.append(f)

    placeholder_seen = 0
    bucket_strong: list[Finding] = []
    bucket_weak: list[Finding] = []
    for f in cleaned:
        if f.title.lower().startswith(_LOW_VALUE_TITLE_PREFIXES):
            if placeholder_seen >= 1:
                continue
            placeholder_seen += 1
            bucket_weak.append(f)
        else:
            bucket_strong.append(f)

    out = bucket_strong + bucket_weak
    if out:
        return out

    # All findings were filtered. Fall back to the original highest-quality one
    # (if any) so the downstream report still has something to render. This is
    # important for clean control companies (e.g., AAPL / MSFT) where pattern
    # heuristics rightly produce nothing.
    if findings:
        ranked = sorted(
            findings,
            key=lambda f: max(
                (_excerpt_quality(c.excerpt) for c in f.citations),
                default=0,
            ),
            reverse=True,
        )
        return ranked[:1]
    return []


def _looks_heuristic(snapshot: dict | None) -> bool:
    if not isinstance(snapshot, dict):
        return True
    text = " ".join(str(v) for v in snapshot.values() if v)
    return not text or any(m in text for m in _HEURISTIC_SNAPSHOT_MARKERS)


def _snapshot_from_submissions(submissions: dict | None) -> dict[str, str]:
    """Build a credible Company Snapshot from cached SEC EDGAR metadata.

    Used both as a fallback when the synthesis LLM is heuristic / disabled and
    as a sanity floor so the user never sees "Unknown (LLM disabled)" rows.
    """
    s = submissions or {}
    name = (s.get("name") or "").strip()
    sic_desc = (s.get("sicDescription") or "").strip()
    sic = (s.get("sic") or "").strip()
    state = (s.get("stateOfIncorporation") or "").strip()
    fye = (s.get("fiscalYearEnd") or "").strip()
    category = (s.get("category") or "").strip()
    exchanges = ", ".join(s.get("exchanges") or [])
    tickers = ", ".join(s.get("tickers") or [])
    addresses = s.get("addresses") or {}
    business_addr = addresses.get("business") or {}
    city = (business_addr.get("city") or "").strip()
    state_addr = (business_addr.get("stateOrCountryDescription") or
                  business_addr.get("stateOrCountry") or "").strip()
    foreign = bool(business_addr.get("isForeignLocation"))
    former = s.get("formerNames") or []
    fye_human = ""
    if fye and len(fye) == 4:
        try:
            fye_human = f"{fye[:2]}-{fye[2:]}"
        except Exception:
            fye_human = fye

    business = "Industry / business description not available from SEC submissions."
    if sic_desc:
        business = (
            f"{name or 'The issuer'} files with the SEC under SIC {sic} ("
            f"{sic_desc}). The line below is sourced from EDGAR submissions metadata, "
            "not management language; treat it as a starting point and replace with the "
            "Item 1 Business description from the latest 10-K / 20-F."
        )

    where = "Geographic scope not yet extracted from filings."
    if city or state_addr or exchanges:
        loc_bits = ", ".join([b for b in [city, state_addr] if b])
        where_parts = []
        if loc_bits:
            where_parts.append(f"Headquartered in {loc_bits}")
        if exchanges:
            where_parts.append(f"listed on {exchanges}")
        if tickers:
            where_parts.append(f"under ticker(s) {tickers}")
        if state:
            where_parts.append(f"incorporated in {state}")
        if foreign:
            where_parts.append("classified as a foreign private issuer (20-F filer)")
        where = "; ".join(where_parts) + "."

    revenue_model = (
        "Revenue model not yet extracted; pull the Segment Information footnote and the "
        "MD&A revenue disaggregation from the latest annual report."
    )
    if category:
        revenue_model = (
            f"SEC filer category: {category}. Detailed segment / product mix should be "
            "lifted from the issuer's annual filing — this prototype's heuristic mode does not "
            "parse segment tables when an LLM key is unavailable."
        )

    actions_parts: list[str] = []
    if former:
        names = "; ".join(
            f"{fn.get('name')} ({(fn.get('from') or '')[:10]} → {(fn.get('to') or '')[:10]})"
            for fn in former[:2]
        )
        actions_parts.append(f"Former corporate name(s): {names}")
    if fye_human:
        actions_parts.append(f"Fiscal year end: {fye_human}")
    if state:
        actions_parts.append(f"State of incorporation: {state}")
    actions = (
        "; ".join(actions_parts)
        if actions_parts
        else "Recent corporate actions not yet extracted — see 8-K (or 6-K for foreign issuers) feed."
    )

    return {
        "business_description": business,
        "where_it_operates": where,
        "segments_or_revenue_model": revenue_model,
        "recent_corporate_actions": actions,
    }


def _synthesize_thesis_when_missing(
    *,
    company_name: str | None,
    findings: list[Finding],
    anomalies: list[dict] | None,
    classifier: dict | None,
    risk_grade: dict | None,
) -> str:
    """Deterministic 2-3 sentence thesis from real findings (used when LLM disabled)."""
    name = (company_name or "The issuer").strip() or "The issuer"
    n_facts = sum(1 for f in findings if f.label == "fact")
    n_inf = sum(1 for f in findings if f.label == "inference")
    n_anoms = len(anomalies or [])
    grade = (risk_grade or {}).get("grade")
    p = (classifier or {}).get("fraud_probability")
    similar = (classifier or {}).get("similar_to") or []

    top_titles = [f.title for f in findings[:3]]

    head = (
        f"{name} surfaces {len(findings)} candidate concerns from this filing pass "
        f"({n_facts} cited fact(s) plus {n_inf} inference(s)) "
        f"and {n_anoms} deterministic XBRL anomal{'y' if n_anoms == 1 else 'ies'}."
    )
    middle = ""
    if top_titles:
        middle = " The sharpest threads worth pressure-testing first: " + "; ".join(top_titles) + "."

    middle += " "
    if grade:
        middle += f"Composite risk grade is {grade}."
    if isinstance(p, (int, float)):
        middle += f" Few-shot misrepresentation probability is {p * 100:.0f}%."
    if similar:
        middle += " Internal pattern-match flags resemblance to: " + ", ".join(similar[:3]) + "."

    tail = (
        " This is a hypothesis to verify with primary documents, not a verdict; "
        "every claim below is a starting point for human review."
    )
    return (head + middle.strip() + tail).strip()


def _synthesize_claims_from_findings(
    findings: list[Finding],
    *,
    max_rows: int = 6,
) -> list[dict[str, str]]:
    """Build a deterministic Claims-vs-Counterpoints table from cited findings.

    Pairs the cited filing excerpt (the issuer's own words) with the analyst's
    why-it-matters (the counterpoint). Lets requirement #5 always render even
    when the LLM didn't produce a side-by-side block.
    """
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for f in findings:
        if not f.citations:
            continue
        excerpt = (f.citations[0].excerpt or "").strip()
        if not excerpt:
            continue
        key = (f.title.lower(), excerpt[:80])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "claim": (
                    f"Issuer-side language from {f.citations[0].doc_id}: "
                    f"\"{excerpt[:300]}{'…' if len(excerpt) > 300 else ''}\""
                ),
                "source_excerpt": excerpt[:600],
                "counterpoint": (
                    (f.why_it_matters or f.claim_or_observation or "Requires further diligence.")[:500]
                ),
                "confidence": f.confidence,
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def _ensure_concerns_by_category(
    raw: dict | None,
    findings: list[Finding],
    anomalies: list[dict] | None,
) -> dict[str, list[str]]:
    """Always render the four required category buckets even if the LLM omits them.

    Falls back to grouping our extracted Findings by category, then to a
    self-explanatory "no items surfaced — see other sections" placeholder
    so the assignment requirement is visible regardless of LLM output.
    """
    base: dict[str, list[str]] = {k: [] for k in _REQUIRED_CATEGORIES + _OPTIONAL_CATEGORIES}
    if isinstance(raw, dict):
        for k, v in raw.items():
            key = str(k).strip().lower().replace(" ", "_")
            if key in base and isinstance(v, list):
                base[key] = [str(x).strip() for x in v if str(x).strip()]

    grouped: dict[str, list[str]] = {k: [] for k in base}
    for f in findings:
        cat = f.category if f.category in base else "other"
        bullet = f.title.strip()
        if f.claim_or_observation:
            bullet = f"{bullet} — {f.claim_or_observation.strip()}"
        if len(bullet) > 220:
            bullet = bullet[:217].rstrip() + "…"
        grouped[cat].append(bullet)

    for cat in base:
        if not base[cat] and grouped.get(cat):
            base[cat] = grouped[cat][:6]

    if anomalies:
        acc_extra = []
        for a in anomalies[:6]:
            t = (a.get("title") or "").strip()
            sev = (a.get("severity") or "").upper()
            if t:
                acc_extra.append(f"[{sev}] {t}" if sev else t)
        if acc_extra and len(base["accounting"]) < 6:
            base["accounting"] = (base["accounting"] + acc_extra)[:6]

    for cat in _REQUIRED_CATEGORIES:
        if not base[cat]:
            base[cat] = [
                "No specific items surfaced from this filing pass — see Top red flags and Open questions for related themes."
            ]

    return {k: v for k, v in base.items() if v}


def _build_evidence_index(findings: list[Finding], sources_brief: list[dict]) -> list[dict]:
    """Per-source evidence summary so requirement #4 (evidence tied to docs) is explicit."""
    by_doc: dict[str, dict] = {
        s["doc_id"]: {
            "doc_id": s["doc_id"],
            "filing_type": s.get("filing_type"),
            "filing_date": s.get("filing_date"),
            "url": s.get("url"),
            "citation_count": 0,
            "finding_titles": [],
            "sample_excerpts": [],
        }
        for s in sources_brief
    }
    for f in findings:
        seen_for_finding: set[str] = set()
        for c in f.citations or []:
            entry = by_doc.get(c.doc_id)
            if entry is None:
                continue
            entry["citation_count"] += 1
            if f.title not in entry["finding_titles"]:
                entry["finding_titles"].append(f.title)
            if (
                c.excerpt
                and c.doc_id not in seen_for_finding
                and len(entry["sample_excerpts"]) < 3
            ):
                excerpt = c.excerpt.strip()
                if len(excerpt) > 320:
                    excerpt = excerpt[:317].rstrip() + "…"
                entry["sample_excerpts"].append(
                    {"finding": f.title, "label": f.label, "excerpt": excerpt}
                )
                seen_for_finding.add(c.doc_id)

    ordered = sorted(
        by_doc.values(),
        key=lambda d: (-d["citation_count"], d.get("filing_date") or ""),
    )
    return ordered


def _build_evidence_summary(findings: list[Finding], sources_brief: list[dict]) -> list[dict]:
    """Flat per-citation evidence list — requirement #4: evidence tied to source documents.

    One row per (finding × citation) so a reader can scan a single table and see
    exactly which excerpt in which filing supports each alleged red flag. This
    is what the assignment calls an "evidence summary tied to source documents"
    and what a forensic analyst expects to be able to fact-check against.

    Rows are ordered by (label severity, confidence, finding title) so the
    strongest evidence (a `fact` cited in a 10-K with `high` confidence) bubbles
    to the top of the list.
    """
    src_lookup = {s["doc_id"]: s for s in sources_brief}
    label_rank = {"fact": 0, "inference": 1, "question": 2, "speculation": 3}
    conf_rank = {"high": 0, "medium": 1, "low": 2}

    rows: list[dict] = []
    for f in findings:
        cits = f.citations or []
        if not cits:
            # Surface findings that lack a citation so the reader sees the gap
            # rather than having the finding silently disappear from the table.
            rows.append({
                "finding_title": f.title,
                "label": f.label,
                "confidence": f.confidence,
                "category": f.category,
                "claim_or_observation": f.claim_or_observation or "",
                "why_it_matters": f.why_it_matters or "",
                "doc_id": None,
                "filing_type": None,
                "filing_date": None,
                "url": None,
                "excerpt": None,
                "missing_citation": True,
            })
            continue
        for c in cits:
            src = src_lookup.get(c.doc_id, {})
            excerpt = (c.excerpt or "").strip()
            if len(excerpt) > 360:
                excerpt = excerpt[:357].rstrip() + "…"
            rows.append({
                "finding_title": f.title,
                "label": f.label,
                "confidence": f.confidence,
                "category": f.category,
                "claim_or_observation": (f.claim_or_observation or "")[:280],
                "why_it_matters": (f.why_it_matters or "")[:280],
                "doc_id": c.doc_id,
                "filing_type": src.get("filing_type"),
                "filing_date": src.get("filing_date"),
                "url": c.url or src.get("url"),
                "excerpt": excerpt,
                "missing_citation": False,
            })

    rows.sort(
        key=lambda r: (
            label_rank.get(r["label"], 9),
            conf_rank.get(r["confidence"], 9),
            r["finding_title"] or "",
        )
    )
    return rows


_HEURISTIC_CONCLUSION_MARKERS = (
    "without an LLM API key",
    "heuristic-fallback mode",
    "heuristic mode",
    "LLM call failed",
    "LLM disabled",
)


def _ensure_conclusion(
    raw: str | None,
    *,
    ticker: str,
    company_name: str | None,
    core_thesis: str,
    n_flags: int,
    n_questions: int,
    grade: str | None,
) -> str:
    t = (raw or "").strip()
    looks_heuristic = any(m in t for m in _HEURISTIC_CONCLUSION_MARKERS)
    if len(t) >= 80 and not looks_heuristic:
        return t
    name = (company_name or "").strip() or ticker
    parts = [
        f"This diligence note summarizes skeptical themes from public SEC filings for {name} ({ticker}). ",
        "It is not an allegation of fraud, not a recommendation, and requires independent verification. ",
    ]
    ct = (core_thesis or "").strip()
    if ct:
        parts.append(ct if ct.endswith(".") else ct + ".")
        parts.append(" ")
    parts.append(
        f"The body lists {n_flags} assessed red flags and {n_questions} explicit follow-ups for management or counsel. "
        "Readers should open each cited filing, confirm the excerpt, and triangulate with footnotes and data tables."
    )
    if grade:
        parts.append(f" Composite risk grade from this prototype: {grade}.")
    return "".join(parts).strip()


def _extract_findings_for_doc(
    doc, text: str, *, financial_context: str | None
) -> list[Finding]:
    # Smaller chunks → smaller prompt → faster + more reliable on slow relays.
    chunks = chunk_text(text, max_chars=int(os.getenv("MUDDY_CHUNK_CHARS") or "8000"))
    all_findings: list[Finding] = []

    # Cap chunks-per-doc to keep wall-clock predictable. Override via env var.
    chunk_cap = int(os.getenv("MUDDY_MAX_CHUNKS_PER_DOC") or "2")
    for idx, ch in enumerate(chunks[:chunk_cap]):
        user = build_extractor_user_prompt(
            doc_id=doc.doc_id,
            filing_type=doc.filing_type,
            primary_url=doc.primary_url,
            chunk_idx=idx + 1,
            total_chunks=min(len(chunks), 6),
            excerpt=ch,
            financial_context=financial_context,
        )

        payload = chat_json(
            system=SYSTEM_INVESTIGATIVE_ANALYST,
            user=user,
            schema_hint=FINDINGS_SCHEMA_HINT,
            temperature=0.2,
        )
        for f in payload.get("findings", []):
            try:
                citations = [
                    Citation(
                        doc_id=ci.get("doc_id", doc.doc_id),
                        url=ci.get("url", doc.primary_url),
                        excerpt=(ci.get("excerpt") or "").strip(),
                    )
                    for ci in (f.get("citations") or [])
                    if (ci.get("excerpt") or "").strip()
                ]
                label = _normalize_label(f.get("label"))
                # Anti-hallucination guard: facts/inferences without a citation
                # are demoted to questions.
                if label in {"fact", "inference"} and not citations:
                    label = "question"

                all_findings.append(
                    Finding(
                        title=(f.get("title") or "Finding").strip()[:300],
                        category=_normalize_category(f.get("category")),
                        label=label,
                        confidence=_normalize_confidence(f.get("confidence")),
                        claim_or_observation=(f.get("claim_or_observation") or "").strip(),
                        why_it_matters=(f.get("why_it_matters") or "").strip(),
                        counterpoints_or_alt_explanations=[
                            x.strip()
                            for x in (f.get("counterpoints_or_alt_explanations") or [])
                            if x.strip()
                        ],
                        open_questions=[
                            x.strip() for x in (f.get("open_questions") or []) if x.strip()
                        ],
                        citations=citations,
                    )
                )
            except Exception as e:  # noqa: BLE001
                # One bad finding shouldn't kill the whole pipeline. Skip it.
                print(f"[pipeline] skipping malformed finding from {doc.doc_id}: {e}")
                continue
    return all_findings


def _build_report(
    *,
    ticker: str,
    company_name: str | None,
    sources,
    findings: list[Finding],
    financial_brief: dict | None,
    fin_snapshot=None,
    submissions: dict | None = None,
) -> Report:
    findings_brief = [
        {
            "title": f.title,
            "category": f.category,
            "label": f.label,
            "confidence": f.confidence,
            "claim_or_observation": f.claim_or_observation,
            "why_it_matters": f.why_it_matters,
            "citations": [c.model_dump() for c in f.citations[:2]],
        }
        for f in findings[:18]
    ]

    src_brief = [
        {
            "doc_id": s.doc_id,
            "filing_type": s.filing_type,
            "filing_date": str(s.filing_date) if s.filing_date else None,
            "url": s.primary_url,
        }
        for s in sources
    ]

    # 1. Composite risk grade (deterministic, no LLM).
    forensic_obj = (fin_snapshot.forensic_scores if fin_snapshot is not None else None)
    anomaly_objs = (fin_snapshot.anomalies if fin_snapshot is not None else [])
    risk = compute_risk_grade(
        anomalies=anomaly_objs,
        red_flags=findings,
        forensic=forensic_obj,
    )
    risk_dict = {"score": risk.score, "grade": risk.grade, "breakdown": risk.breakdown}

    # 2. Few-shot in-context fraud classifier (uses DeepSeek V4 Pro by default).
    #    Runs BEFORE synthesis so the synthesis LLM can reference its verdict.
    classifier_result: dict = {}
    try:
        classifier_result = classify_fraud_likelihood(
            ticker=ticker,
            company_name=company_name,
            financial_brief=financial_brief,
            llm_findings=findings,
            risk_grade=risk_dict,
        )
    except Exception as e:
        classifier_result = {
            "fraud_probability": 0.5,
            "verdict": "watch",
            "confidence": "low",
            "reasoning": f"Classifier disabled / errored: {e}",
            "similar_to": [],
            "differentiators": "",
            "key_red_flags": [],
            "mitigating_factors": [],
        }

    # 2b. Deterministic ML scorer (logistic regression on engineered features).
    #     Runs alongside the LLM classifier so the report has *two independent*
    #     misrepresentation-risk estimates that can be cross-checked.
    label_counts: dict[str, int] = {}
    for f in findings:
        lbl = getattr(f, "label", None) or "question"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    ml_result_obj = ml_score(financial_brief=financial_brief, llm_findings_summary=label_counts)
    ml_result_dict = {
        "probability": ml_result_obj.probability,
        "verdict": ml_result_obj.verdict,
        "confidence": ml_result_obj.confidence,
        "model": ml_result_obj.model,
        "coverage": ml_result_obj.coverage,
        "feature_contributions": ml_result_obj.feature_contributions,
        "notes": ml_result_obj.notes,
    }
    fraud_ensemble = ml_ensemble(classifier_result, ml_result_obj)

    # 3. Synthesis — combine financial brief + classifier verdict into prompt.
    fin_brief_str = None
    if financial_brief:
        payload_fin = {
            "entity": financial_brief.get("entity"),
            "anomalies": financial_brief.get("anomalies", []),
            "forensic_scores": financial_brief.get("forensic_scores"),
        }
        if classifier_result and classifier_result.get("fraud_probability") is not None:
            payload_fin["ml_classifier_verdict"] = {
                "probability": classifier_result.get("fraud_probability"),
                "verdict": classifier_result.get("verdict"),
                "similar_to": classifier_result.get("similar_to"),
                "key_red_flags": classifier_result.get("key_red_flags"),
            }
        payload_fin["deterministic_ml_score"] = {
            "probability": ml_result_dict["probability"],
            "verdict": ml_result_dict["verdict"],
            "confidence": ml_result_dict["confidence"],
            "model": ml_result_dict["model"],
        }
        payload_fin["ensemble_verdict"] = {
            "probability": fraud_ensemble["combined_probability"],
            "verdict": fraud_ensemble["verdict"],
            "agreement": fraud_ensemble["agreement"],
        }
        fin_brief_str = json.dumps(payload_fin, indent=2, default=str)

    user = build_synthesis_user_prompt(
        ticker=ticker,
        company_name=company_name,
        sources_brief=json.dumps(src_brief, indent=2),
        findings_brief=json.dumps(findings_brief, indent=2),
        financial_brief=fin_brief_str,
    )

    payload = chat_json(
        system=SYSTEM_INVESTIGATIVE_ANALYST,
        user=user,
        schema_hint=REPORT_SCHEMA_HINT,
        temperature=0.25,
    )

    merged_open_questions = _collect_open_questions(
        payload.get("open_questions"),
        findings,
        financial_brief,
        classifier_result,
    )
    top_flags = findings[:12]

    # Snapshot: prefer LLM-produced; fall back to EDGAR-derived metadata so the
    # heuristic / no-key path never shows "Unknown (LLM disabled)" rows.
    raw_snapshot = payload.get("snapshot")
    snapshot_meta = _snapshot_from_submissions(submissions)
    snapshot = (
        snapshot_meta if _looks_heuristic(raw_snapshot) else raw_snapshot
    )

    # Core thesis: prefer LLM; fall back to a finding-grounded construction.
    raw_thesis = (payload.get("core_thesis") or "").strip()
    if not raw_thesis or "heuristic" in raw_thesis.lower():
        core_thesis = _synthesize_thesis_when_missing(
            company_name=(payload.get("company_name") or company_name),
            findings=top_flags,
            anomalies=(financial_brief or {}).get("anomalies"),
            classifier=classifier_result,
            risk_grade=risk_dict,
        )
    else:
        core_thesis = raw_thesis

    # Management claims vs counterpoints — derive from cited findings if empty.
    raw_claims = payload.get("management_claims_vs_counterpoints") or []
    if not raw_claims:
        raw_claims = _synthesize_claims_from_findings(top_flags)

    concerns = _ensure_concerns_by_category(
        payload.get("concerns_by_category"),
        findings,
        (financial_brief or {}).get("anomalies"),
    )
    evidence_index = _build_evidence_index(findings, src_brief)
    evidence_summary = _build_evidence_summary(findings, src_brief)
    sources_serialized = [
        {
            "doc_id": s["doc_id"],
            "filing_type": s.get("filing_type"),
            "filing_date": s.get("filing_date"),
            "url": s.get("url"),
        }
        for s in src_brief
    ]

    conclusion = _ensure_conclusion(
        payload.get("conclusion"),
        ticker=ticker,
        company_name=(payload.get("company_name") or company_name),
        core_thesis=core_thesis,
        n_flags=len(top_flags),
        n_questions=len(merged_open_questions),
        grade=risk_dict.get("grade"),
    )

    now = datetime.now(timezone.utc).isoformat()

    return Report(
        ticker=ticker,
        company_name=payload.get("company_name") or company_name,
        generated_at_iso=now,
        snapshot=snapshot,
        core_thesis=core_thesis,
        red_flags=top_flags,
        management_claims_vs_counterpoints=raw_claims,
        concerns_by_category=concerns,
        open_questions=merged_open_questions,
        conclusion=conclusion,
        limitations=payload.get("limitations")
        or [
            "Prototype uses only SEC filings (no transcripts/news unless added).",
            "LLM outputs may be incomplete or incorrect; all points require human verification.",
            "Not investment advice; do not publish without legal review.",
        ],
        financial_anomalies=(financial_brief or {}).get("anomalies", []),
        financial_table=(financial_brief or {}).get("annual_table", []),
        forensic_scores=(financial_brief or {}).get("forensic_scores") or {},
        risk_grade=risk_dict,
        fraud_classifier=classifier_result,
        ml_scorer=ml_result_dict,
        fraud_ensemble=fraud_ensemble,
        provider_info=provider_info(),
        sources_analyzed=sources_serialized,
        evidence_index=evidence_index,
        evidence_summary=evidence_summary,
    )


def _auto_pick_filing_types(submissions: dict, requested: list[str] | None) -> list[str]:
    """If the caller didn't specify, pick a sensible set based on what the issuer files.

    Foreign private issuers file 20-F / 6-K / F-1 instead of 10-K / 10-Q / 8-K.
    """
    if requested:
        return requested
    available = set(submissions.get("filings", {}).get("recent", {}).get("form", []))
    has_domestic = bool(available & set(DEFAULT_DOMESTIC_FORMS))
    has_foreign = bool(available & set(DEFAULT_FOREIGN_FORMS))
    if has_domestic and not has_foreign:
        return DEFAULT_DOMESTIC_FORMS
    if has_foreign and not has_domestic:
        return DEFAULT_FOREIGN_FORMS
    return DEFAULT_ALL_FORMS


def generate_report(
    *,
    ticker: str,
    filing_types: list[str] | None = None,
    max_filings: int = 8,
    out_html_path: str = "outputs/report.html",
    out_json_path: str | None = None,
    cache_dir: str = "cache",
    progress: ProgressFn = None,
) -> PipelineResult:
    ticker = ticker.strip().upper()

    _progress(progress, f"Resolving {ticker} on SEC EDGAR…", 0.02)
    cik10, company_name = ticker_to_cik(ticker, cache_dir=cache_dir)
    submissions = fetch_submissions(cik10, cache_dir=cache_dir)
    if not company_name:
        company_name = submissions.get("name")

    chosen_forms = _auto_pick_filing_types(submissions, filing_types)
    _progress(progress, f"Listing recent filings ({', '.join(chosen_forms)})…", 0.08)
    recent = list_recent_filings(submissions, filing_types=chosen_forms, max_filings=max_filings)

    _progress(progress, f"Downloading {len(recent)} filings…", 0.15)
    sources = []
    for i, row in enumerate(recent):
        sources.append(
            download_filing_primary_doc(
                ticker=ticker,
                cik10=cik10,
                filing_type=row["form"],
                accession=row["accession"],
                filing_date_str=row.get("filingDate"),
                primary_doc=row.get("primaryDocument"),
                cache_dir=cache_dir,
            )
        )
        _progress(
            progress,
            f"Downloaded {row['form']} ({row.get('filingDate')})",
            0.15 + 0.10 * (i + 1) / max(1, len(recent)),
        )

    _progress(progress, "Pulling XBRL financials and detecting anomalies…", 0.27)
    fin_snap = build_financial_snapshot(cik10, cache_dir=cache_dir)
    fin_brief = snapshot_to_brief(fin_snap)
    fin_context = _format_financial_context(fin_brief)

    findings: list[Finding] = []
    for i, doc in enumerate(sources):
        _progress(
            progress,
            f"Analyzing {doc.filing_type} ({doc.filing_date}) for red flags…",
            0.30 + 0.55 * (i / max(1, len(sources))),
        )
        text = read_doc_text(doc.local_path)
        text = text[:240_000]  # bound input to keep latency / cost predictable
        findings.extend(_extract_findings_for_doc(doc, text, financial_context=fin_context))

    findings = _post_filter_findings(findings)

    # Lightweight de-dup so the synthesis prompt isn't drowned in near-duplicates.
    seen = set()
    deduped: list[Finding] = []
    for f in findings:
        key = (f.title.lower()[:80], f.claim_or_observation.lower()[:120])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    findings = deduped

    _progress(progress, "Synthesizing investigative report…", 0.88)
    report = _build_report(
        ticker=ticker,
        company_name=company_name,
        sources=sources,
        findings=findings,
        financial_brief=fin_brief,
        fin_snapshot=fin_snap,
        submissions=submissions,
    )

    _progress(progress, "Rendering HTML…", 0.95)
    template_dir = str(Path(__file__).parent / "templates")
    html = render_html(report, sources, template_dir=template_dir)
    write_html(out_html_path, html)

    if not out_json_path:
        out_json_path = str(Path(out_html_path).with_suffix(".json"))
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json_path).write_text(report.model_dump_json(indent=2), encoding="utf-8")

    _progress(progress, "Done.", 1.0)
    return PipelineResult(
        out_html_path=out_html_path,
        out_json_path=out_json_path,
        sources=sources,
        report=report,
    )
