"""SEC XBRL company-facts → forensic financial anomaly detection.

We fetch the XBRL "company facts" feed (works for any US-GAAP filer, including
foreign issuers like Luckin), normalize to annual / quarterly time series for a
small set of forensic-relevant concepts, and run cheap rule-based checks that
flag the kinds of patterns short-sellers look for:

- Receivables growing materially faster than revenue (channel stuffing / sham sales)
- Net income running well ahead of operating cash flow (accruals quality)
- Sudden gross-margin expansion (cost capitalization / classification shifts)
- Inventory growth out of step with revenue
- Goodwill / intangibles ballooning vs. equity (acquisitive accounting risk)
- Material restatement / impairment / going-concern keywords from companyconcept tags

Outputs are deterministic and *separate* from the LLM. The LLM later receives
them as quantitative context to ground its narrative red flags in real numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sec_edgar import _cache_get  # reuse polite-cached HTTP


XBRL_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


# Concept fallbacks — first hit wins. Different filers / years tag the same
# economic line under slightly different US-GAAP concept names.
CONCEPT_MAP: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
    ],
    "inventory": ["InventoryNet"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
    ],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "goodwill": ["Goodwill"],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill"],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
}


@dataclass
class FactPoint:
    fy: int
    fp: str  # FY / Q1 / Q2 / Q3 / Q4
    end: str  # ISO date
    value: float
    unit: str
    accn: str  # accession number for traceability


@dataclass
class FinancialAnomaly:
    title: str
    category: str  # accounting | capital_structure | operations | disclosure
    severity: str  # high | medium | low
    metric: str
    value_str: str  # human-readable summary
    description: str
    supporting_points: list[dict] = field(default_factory=list)


@dataclass
class FinancialSnapshot:
    cik: str
    entity: str | None
    series: dict[str, list[FactPoint]] = field(default_factory=dict)
    annual_table: list[dict[str, Any]] = field(default_factory=list)
    anomalies: list[FinancialAnomaly] = field(default_factory=list)


def fetch_company_facts(cik10: str, cache_dir: str = "cache") -> dict[str, Any] | None:
    url = XBRL_COMPANYFACTS_URL.format(cik10=cik10)
    try:
        raw = _cache_get(url, Path(cache_dir) / "sec" / f"companyfacts_{cik10}.json")
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _pick_concept(facts: dict, candidates: list[str]) -> tuple[str, dict] | None:
    us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    ifrs = (facts.get("facts") or {}).get("ifrs-full") or {}
    for c in candidates:
        if c in us_gaap:
            return c, us_gaap[c]
        if c in ifrs:
            return c, ifrs[c]
    return None


def _series_from_concept(node: dict) -> list[FactPoint]:
    """Pick the USD unit if available; convert to FactPoints sorted by end date."""
    units = node.get("units") or {}
    unit_key = next(
        (k for k in ["USD", "USD/shares", "shares"] if k in units),
        next(iter(units), None),
    )
    if not unit_key:
        return []
    out: list[FactPoint] = []
    for r in units[unit_key]:
        end = r.get("end")
        v = r.get("val")
        if end is None or v is None:
            continue
        out.append(
            FactPoint(
                fy=int(r.get("fy") or 0),
                fp=str(r.get("fp") or ""),
                end=str(end),
                value=float(v),
                unit=str(unit_key),
                accn=str(r.get("accn") or ""),
            )
        )
    out.sort(key=lambda p: p.end)
    return out


def _annual_only(points: list[FactPoint]) -> list[FactPoint]:
    """Keep one annual reading per fiscal year. Prefer 'FY' marker; otherwise
    take the latest in-year point as a stand-in (common for foreign issuers)."""
    by_fy: dict[int, FactPoint] = {}
    for p in points:
        if p.fp == "FY" and p.fy:
            by_fy[p.fy] = p
    if by_fy:
        return [by_fy[fy] for fy in sorted(by_fy)]
    # Fallback: bucket by calendar year of the period-end date.
    by_year: dict[int, FactPoint] = {}
    for p in points:
        try:
            yr = int(p.end[:4])
        except Exception:
            continue
        if yr not in by_year or p.end > by_year[yr].end:
            by_year[yr] = p
    return [by_year[y] for y in sorted(by_year)]


def _build_series(facts: dict) -> dict[str, list[FactPoint]]:
    out: dict[str, list[FactPoint]] = {}
    for key, candidates in CONCEPT_MAP.items():
        hit = _pick_concept(facts, candidates)
        if not hit:
            continue
        _, node = hit
        annual = _annual_only(_series_from_concept(node))
        if annual:
            out[key] = annual
    return out


def _annual_table(series: dict[str, list[FactPoint]]) -> list[dict[str, Any]]:
    """Pivot series → list of {fy, fy_end, revenue, ar, ...} rows for the UI."""
    years: set[int] = set()
    for pts in series.values():
        for p in pts:
            try:
                years.add(int(p.end[:4]))
            except Exception:
                continue
    rows = []
    for y in sorted(years)[-7:]:
        row: dict[str, Any] = {"year": y}
        for k, pts in series.items():
            match = next((p for p in pts if p.end.startswith(str(y))), None)
            row[k] = match.value if match else None
            if match:
                row[f"{k}_end"] = match.end
        rows.append(row)
    return rows


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _yoy(prev: float | None, curr: float | None) -> float | None:
    if prev is None or curr is None or prev == 0:
        return None
    return (curr - prev) / abs(prev)


def _detect_anomalies(table: list[dict[str, Any]]) -> list[FinancialAnomaly]:
    out: list[FinancialAnomaly] = []
    if len(table) < 2:
        return out

    for i in range(1, len(table)):
        prev = table[i - 1]
        curr = table[i]
        year = curr["year"]

        rev_prev, rev_curr = prev.get("revenue"), curr.get("revenue")
        ar_prev, ar_curr = prev.get("accounts_receivable"), curr.get("accounts_receivable")
        rev_growth = _yoy(rev_prev, rev_curr)
        ar_growth = _yoy(ar_prev, ar_curr)

        # Receivables outpacing revenue — classic channel-stuffing / sham-sales signal.
        if rev_growth is not None and ar_growth is not None and rev_growth > 0 and ar_growth - rev_growth > 0.20:
            out.append(
                FinancialAnomaly(
                    title=f"Accounts receivable grew {ar_growth*100:.0f}% in FY{year} vs revenue +{rev_growth*100:.0f}%",
                    category="accounting",
                    severity="high" if (ar_growth - rev_growth) > 0.5 else "medium",
                    metric="AR_vs_Revenue",
                    value_str=f"AR Δ {ar_growth*100:.0f}% / Revenue Δ {rev_growth*100:.0f}%",
                    description=(
                        "Receivables growing materially faster than revenue can indicate "
                        "channel stuffing, looser credit terms, or revenue pulled forward; "
                        "warrants review of revenue recognition and DSO trend."
                    ),
                    supporting_points=[
                        {"year": prev["year"], "revenue": rev_prev, "accounts_receivable": ar_prev},
                        {"year": year, "revenue": rev_curr, "accounts_receivable": ar_curr},
                    ],
                )
            )

        # Net income running ahead of operating cash flow — accruals quality.
        ni_curr = curr.get("net_income")
        cfo_curr = curr.get("operating_cash_flow")
        if (
            ni_curr is not None
            and cfo_curr is not None
            and ni_curr > 0
            and (cfo_curr < ni_curr * 0.4 or cfo_curr < 0)
        ):
            gap = ni_curr - cfo_curr
            out.append(
                FinancialAnomaly(
                    title=f"Net income exceeds operating cash flow by ${gap/1e6:,.0f}M in FY{year}",
                    category="accounting",
                    severity="high" if cfo_curr < 0 or cfo_curr < ni_curr * 0.2 else "medium",
                    metric="NI_vs_CFO",
                    value_str=f"NI ${ni_curr/1e6:,.0f}M vs CFO ${cfo_curr/1e6:,.0f}M",
                    description=(
                        "Persistent positive net income alongside weak or negative operating cash flow "
                        "is a classic earnings-quality red flag (accruals, capitalized costs, working "
                        "capital inflation). Compare against peers and inspect the working-capital bridge."
                    ),
                    supporting_points=[{"year": year, "net_income": ni_curr, "operating_cash_flow": cfo_curr}],
                )
            )

        # Sudden gross-margin jump.
        gm_prev = _safe_div(prev.get("gross_profit"), prev.get("revenue"))
        gm_curr = _safe_div(curr.get("gross_profit"), curr.get("revenue"))
        if gm_prev is not None and gm_curr is not None and gm_curr - gm_prev > 0.10:
            out.append(
                FinancialAnomaly(
                    title=f"Gross margin jumped {(gm_curr-gm_prev)*100:.1f} pts in FY{year} ({gm_prev*100:.1f}% → {gm_curr*100:.1f}%)",
                    category="accounting",
                    severity="medium",
                    metric="GrossMargin_jump",
                    value_str=f"{gm_prev*100:.1f}% → {gm_curr*100:.1f}%",
                    description=(
                        "Sharp gross-margin expansion can be legitimate (mix shift, scale) but is also a "
                        "common signature of cost reclassification or capitalization. Investigate accounting policy notes."
                    ),
                    supporting_points=[
                        {"year": prev["year"], "gross_margin": gm_prev},
                        {"year": year, "gross_margin": gm_curr},
                    ],
                )
            )

        # Inventory growing well ahead of revenue.
        inv_prev, inv_curr = prev.get("inventory"), curr.get("inventory")
        inv_growth = _yoy(inv_prev, inv_curr)
        if (
            rev_growth is not None
            and inv_growth is not None
            and rev_growth >= 0
            and inv_growth - rev_growth > 0.30
        ):
            out.append(
                FinancialAnomaly(
                    title=f"Inventory grew {inv_growth*100:.0f}% in FY{year} vs revenue +{rev_growth*100:.0f}%",
                    category="operations",
                    severity="medium",
                    metric="Inventory_vs_Revenue",
                    value_str=f"Inv Δ {inv_growth*100:.0f}% / Rev Δ {rev_growth*100:.0f}%",
                    description=(
                        "Inventory build well ahead of sales can signal demand softness, "
                        "obsolescence risk, or upcoming write-downs."
                    ),
                    supporting_points=[
                        {"year": prev["year"], "revenue": rev_prev, "inventory": inv_prev},
                        {"year": year, "revenue": rev_curr, "inventory": inv_curr},
                    ],
                )
            )

        # Goodwill ballooning vs equity.
        gw_curr = curr.get("goodwill")
        eq_curr = curr.get("stockholders_equity")
        ratio = _safe_div(gw_curr, eq_curr)
        if ratio is not None and ratio > 0.5:
            out.append(
                FinancialAnomaly(
                    title=f"Goodwill is {ratio*100:.0f}% of equity in FY{year}",
                    category="capital_structure",
                    severity="medium" if ratio > 1 else "low",
                    metric="Goodwill_to_Equity",
                    value_str=f"{ratio*100:.0f}%",
                    description=(
                        "High goodwill relative to equity makes the balance sheet sensitive to "
                        "impairment charges and reliant on acquisition synergies materializing."
                    ),
                    supporting_points=[{"year": year, "goodwill": gw_curr, "stockholders_equity": eq_curr}],
                )
            )

    return out


def build_financial_snapshot(cik10: str, cache_dir: str = "cache") -> FinancialSnapshot:
    facts = fetch_company_facts(cik10, cache_dir=cache_dir) or {}
    series = _build_series(facts)
    table = _annual_table(series)
    anomalies = _detect_anomalies(table)
    return FinancialSnapshot(
        cik=cik10,
        entity=facts.get("entityName"),
        series=series,
        annual_table=table,
        anomalies=anomalies,
    )


def snapshot_to_brief(snap: FinancialSnapshot) -> dict[str, Any]:
    """Compact dict for prompting / JSON dump (no dataclasses)."""
    return {
        "entity": snap.entity,
        "annual_table": snap.annual_table,
        "anomalies": [
            {
                "title": a.title,
                "category": a.category,
                "severity": a.severity,
                "metric": a.metric,
                "value_str": a.value_str,
                "description": a.description,
                "supporting_points": a.supporting_points,
            }
            for a in snap.anomalies
        ],
    }
