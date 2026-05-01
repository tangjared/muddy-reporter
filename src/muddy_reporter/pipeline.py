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
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .financials import build_financial_snapshot, snapshot_to_brief
from .llm import chat_json, provider_info
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


def _extract_findings_for_doc(
    doc, text: str, *, financial_context: str | None
) -> list[Finding]:
    chunks = chunk_text(text, max_chars=12000)
    all_findings: list[Finding] = []

    for idx, ch in enumerate(chunks[:6]):  # cap to keep prototype quick
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

    fin_brief_str = None
    if financial_brief:
        fin_brief_str = json.dumps(
            {
                "entity": financial_brief.get("entity"),
                "anomalies": financial_brief.get("anomalies", []),
            },
            indent=2,
        )

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

    now = datetime.now(timezone.utc).isoformat()
    return Report(
        ticker=ticker,
        company_name=payload.get("company_name") or company_name,
        generated_at_iso=now,
        snapshot=payload.get("snapshot")
        or {
            "business_description": "Unknown (not extracted).",
            "where_it_operates": "Unknown (not extracted).",
            "segments_or_revenue_model": "Unknown (not extracted).",
            "recent_corporate_actions": "Unknown (not extracted).",
        },
        core_thesis=payload.get("core_thesis")
        or "This company warrants additional diligence based on the extracted red flags.",
        red_flags=findings[:12],
        management_claims_vs_counterpoints=payload.get("management_claims_vs_counterpoints") or [],
        concerns_by_category=payload.get("concerns_by_category") or {},
        open_questions=payload.get("open_questions") or [],
        conclusion=payload.get("conclusion")
        or "Caution: findings are preliminary and require verification.",
        limitations=payload.get("limitations")
        or [
            "Prototype uses only SEC filings (no transcripts/news unless added).",
            "LLM outputs may be incomplete or incorrect; all points require human verification.",
            "Not investment advice; do not publish without legal review.",
        ],
        financial_anomalies=(financial_brief or {}).get("anomalies", []),
        financial_table=(financial_brief or {}).get("annual_table", []),
        provider_info=provider_info(),
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
