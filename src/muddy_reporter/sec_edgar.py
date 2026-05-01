from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import requests

from .models import SourceDoc


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


# Forms grouped by what they correspond to, so the pipeline can default to a
# sensible list whether the filer is a US domestic or a foreign private issuer.
DEFAULT_DOMESTIC_FORMS = ["10-K", "10-Q", "8-K"]
DEFAULT_FOREIGN_FORMS = ["20-F", "6-K", "F-1", "F-1/A"]
DEFAULT_ALL_FORMS = DEFAULT_DOMESTIC_FORMS + DEFAULT_FOREIGN_FORMS


def _sec_headers() -> dict[str, str]:
    ua = os.getenv("SEC_USER_AGENT") or os.getenv("USER_AGENT")
    if not ua:
        # SEC requires a descriptive User-Agent. We keep a safe default but
        # strongly recommend overriding via env/.env.
        ua = "MuddyReporterPrototype/0.1 (contact: example@example.com)"
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "application/json,text/html,application/xhtml+xml",
        "Connection": "keep-alive",
    }


def _cache_get(url: str, cache_path: Path, sleep_s: float = 0.12) -> bytes:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path.read_bytes()

    # Be polite to SEC.
    time.sleep(sleep_s)
    resp = requests.get(url, headers=_sec_headers(), timeout=60)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return resp.content


def ticker_to_cik(ticker: str, cache_dir: str = "cache") -> tuple[str, str | None]:
    """Resolve a US-listed ticker (incl. ADRs / OTC) to (CIK, company_name).

    Accepts either a ticker symbol or a 10-digit CIK string for ergonomics.
    """
    ticker = ticker.upper().strip()
    # Allow direct CIK input (digits, optionally zero-padded) for delisted tickers.
    if ticker.isdigit():
        return ticker.zfill(10), None

    raw = _cache_get(
        SEC_TICKER_MAP_URL,
        Path(cache_dir) / "sec" / "company_tickers.json",
    )
    data = json.loads(raw.decode("utf-8"))
    for _, row in data.items():
        if str(row.get("ticker", "")).upper() == ticker:
            cik = str(row["cik_str"]).zfill(10)
            return cik, row.get("title")
    raise ValueError(
        f"Ticker not found in SEC mapping: {ticker}. "
        "If the company is delisted or uses a foreign-issuer ticker, try entering its CIK directly."
    )


def fetch_submissions(cik10: str, cache_dir: str = "cache") -> dict[str, Any]:
    url = SEC_SUBMISSIONS_URL.format(cik10=cik10)
    raw = _cache_get(url, Path(cache_dir) / "sec" / f"submissions_{cik10}.json")
    return json.loads(raw.decode("utf-8"))


def _safe_accession(accession: str) -> str:
    # "0000320193-24-000123" -> "000032019324000123"
    return accession.replace("-", "")


def _guess_primary_doc(filing_row: dict[str, Any]) -> str | None:
    # SEC submissions JSON provides primaryDocument in some cases,
    # but recent filings are in arrays. We'll try a few keys.
    for k in ["primaryDocument", "primaryDocDescription"]:
        v = filing_row.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def list_recent_filings(
    submissions: dict[str, Any],
    filing_types: list[str],
    max_filings: int,
) -> list[dict[str, Any]]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    out: list[dict[str, Any]] = []
    for i, form in enumerate(forms):
        if form not in set(filing_types):
            continue
        acc = accession_numbers[i]
        fdate = filing_dates[i] if i < len(filing_dates) else None
        pdoc = primary_docs[i] if i < len(primary_docs) else None
        out.append(
            {
                "form": form,
                "accession": acc,
                "filingDate": fdate,
                "primaryDocument": pdoc,
            }
        )
        if len(out) >= max_filings:
            break
    return out


def build_primary_doc_url(cik10: str, accession: str, primary_doc: str | None) -> str:
    cik_int = str(int(cik10))
    acc_nodash = _safe_accession(accession)
    if not primary_doc:
        # Fallback: common primary doc name patterns.
        primary_doc = "index.html"
    return f"{SEC_ARCHIVES_BASE}/{cik_int}/{acc_nodash}/{primary_doc}"


def download_filing_primary_doc(
    *,
    ticker: str,
    cik10: str,
    filing_type: str,
    accession: str,
    filing_date_str: str | None,
    primary_doc: str | None,
    cache_dir: str = "cache",
) -> SourceDoc:
    url = build_primary_doc_url(cik10, accession, primary_doc)

    cik_int = str(int(cik10))
    acc_nodash = _safe_accession(accession)
    local_dir = Path(cache_dir) / "edgar" / ticker / filing_type / acc_nodash
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / (primary_doc or "index.html")

    raw = _cache_get(url, local_path)
    sha = hashlib.sha256(raw).hexdigest()

    filing_dt = None
    if filing_date_str:
        try:
            y, m, d = filing_date_str.split("-")
            filing_dt = date(int(y), int(m), int(d))
        except Exception:
            filing_dt = None

    doc_id = f"{ticker}:{filing_type}:{acc_nodash}"
    return SourceDoc(
        doc_id=doc_id,
        ticker=ticker,
        cik=cik10,
        filing_type=filing_type,
        filing_date=filing_dt,
        accession=accession,
        primary_url=url,
        local_path=str(local_path),
        sha256=sha,
    )

