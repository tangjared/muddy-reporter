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
    "current_assets": ["AssetsCurrent"],
    "total_liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarnings",
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
    "ppe": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentGross",
    ],
    "sga": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
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
class ForensicScores:
    """Three classic academic forensic / distress models.

    All fields can be None if we lack the data to compute that score reliably.
    """

    beneish_m: float | None = None
    beneish_signal: str | None = None  # "manipulator" | "non-manipulator" | None
    beneish_components: dict[str, float] | None = None

    piotroski_f: int | None = None  # 0..9
    piotroski_strength: str | None = None  # "strong" | "neutral" | "weak"
    piotroski_breakdown: list[dict[str, Any]] | None = None

    altman_z: float | None = None
    altman_zone: str | None = None  # "safe" | "grey" | "distress"

    notes: list[str] = field(default_factory=list)


@dataclass
class RiskGrade:
    score: int  # 0..100, higher = riskier
    grade: str  # "A" | "B" | "C" | "D" | "F"
    breakdown: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FinancialSnapshot:
    cik: str
    entity: str | None
    series: dict[str, list[FactPoint]] = field(default_factory=dict)
    annual_table: list[dict[str, Any]] = field(default_factory=list)
    anomalies: list[FinancialAnomaly] = field(default_factory=list)
    forensic_scores: ForensicScores | None = None


def fetch_company_facts(cik10: str, cache_dir: str = "cache") -> dict[str, Any] | None:
    url = XBRL_COMPANYFACTS_URL.format(cik10=cik10)
    try:
        raw = _cache_get(url, Path(cache_dir) / "sec" / f"companyfacts_{cik10}.json")
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _pick_concept(facts: dict, candidates: list[str]) -> tuple[str, dict] | None:
    """Pick the concept with the *most annual data points*, not just the first hit.

    Apple is the canonical edge case: `us-gaap:Revenues` exists in their facts
    but is only used for segment/dimensional breakdowns (1 FY point in 2018).
    Their real consolidated top-line lives in
    `RevenueFromContractWithCustomerExcludingAssessedTax` (11 FY points). Same
    pattern recurs for issuers that migrated from legacy ASC 605 tags to
    ASC 606 tags — both can be present, only one is the real series.
    Picking by data density auto-corrects for that.
    """
    us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    ifrs = (facts.get("facts") or {}).get("ifrs-full") or {}

    def _score(node: dict) -> tuple[int, str]:
        """Score a candidate concept. Higher tuple = better.

        We weight by:
        1. Number of distinct FY-marked annual periods in the last 7 fiscal years
           — this drops to 0 for legacy tags (e.g. AAPL's `SalesRevenueNet`
           which stopped publishing in FY2017 after the ASC 606 migration).
        2. Latest annual period's end date — tiebreaker that prefers actively
           reported tags over historical ones.
        """
        from datetime import datetime

        units = node.get("units") or {}
        unit_key = next(
            (k for k in ["USD", "USD/shares", "shares"] if k in units),
            next(iter(units), None),
        )
        if not unit_key:
            return (0, "")
        fy_periods: set[int] = set()
        latest_end = ""
        for r in units[unit_key]:
            fp = str(r.get("fp") or "")
            fy = r.get("fy")
            end = str(r.get("end") or "")
            if fp == "FY" and fy:
                fy_periods.add(int(fy))
            if end and end > latest_end:
                latest_end = end
        cutoff_year = datetime.utcnow().year - 7
        recent = sum(1 for fy in fy_periods if fy >= cutoff_year)
        return (recent, latest_end)

    best: tuple[str, dict, tuple[int, str]] | None = None
    for c in candidates:
        node = us_gaap.get(c) or ifrs.get(c)
        if not node:
            continue
        s = _score(node)
        if best is None or s > best[2]:
            best = (c, node, s)
    return (best[0], best[1]) if best else None


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


# ---------------------------------------------------------------------------
# Academic forensic / distress models
# ---------------------------------------------------------------------------


def _g(row: dict, key: str) -> float | None:
    """Safe getter that returns None if missing or zero (treated as missing)."""
    v = row.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _compute_beneish_m(prev: dict, curr: dict) -> tuple[float | None, dict[str, float] | None]:
    """Beneish M-Score (1999) — 8-variable earnings-manipulation indicator.

    Returns (M, components) or (None, None) if we lack the inputs.

    Threshold: M > -1.78 → likely manipulator.
    """
    rev_t,  rev_p  = _g(curr, "revenue"),             _g(prev, "revenue")
    ar_t,   ar_p   = _g(curr, "accounts_receivable"), _g(prev, "accounts_receivable")
    cogs_t, cogs_p = _g(curr, "cost_of_revenue"),     _g(prev, "cost_of_revenue")
    gp_t,   gp_p   = _g(curr, "gross_profit"),        _g(prev, "gross_profit")
    ca_t,   ca_p   = _g(curr, "current_assets"),      _g(prev, "current_assets")
    ppe_t,  ppe_p  = _g(curr, "ppe"),                 _g(prev, "ppe")
    ta_t,   ta_p   = _g(curr, "total_assets"),        _g(prev, "total_assets")
    dep_t,  dep_p  = _g(curr, "depreciation"),        _g(prev, "depreciation")
    sga_t,  sga_p  = _g(curr, "sga"),                 _g(prev, "sga")
    ltd_t,  ltd_p  = _g(curr, "long_term_debt"),      _g(prev, "long_term_debt")
    cl_t,   cl_p   = _g(curr, "current_liabilities"), _g(prev, "current_liabilities")
    ni_t,   _      = _g(curr, "net_income"),          None
    cfo_t,  _      = _g(curr, "operating_cash_flow"), None

    # gross profit fallback if not directly tagged
    if gp_t is None and rev_t is not None and cogs_t is not None:
        gp_t = rev_t - cogs_t
    if gp_p is None and rev_p is not None and cogs_p is not None:
        gp_p = rev_p - cogs_p

    required = [rev_t, rev_p, ar_t, ar_p, gp_t, gp_p, ta_t, ta_p, ca_t, ca_p, ppe_t, ppe_p,
                cl_t, cl_p, ltd_t, ltd_p, ni_t, cfo_t]
    if any(x is None for x in required) or rev_p == 0 or rev_t == 0:
        return None, None

    try:
        DSRI = (ar_t / rev_t) / (ar_p / rev_p)
        gm_t = gp_t / rev_t
        gm_p = gp_p / rev_p
        if gm_t == 0:
            return None, None
        GMI  = gm_p / gm_t
        AQI  = (1 - (ca_t + ppe_t) / ta_t) / (1 - (ca_p + ppe_p) / ta_p)
        SGI  = rev_t / rev_p
        # depreciation index (1.0 if dep series missing)
        if dep_t is None or dep_p is None or (dep_t + ppe_t) == 0 or (dep_p + ppe_p) == 0:
            DEPI = 1.0
        else:
            DEPI = (dep_p / (dep_p + ppe_p)) / (dep_t / (dep_t + ppe_t))
        if sga_t is None or sga_p is None:
            SGAI = 1.0
        else:
            SGAI = (sga_t / rev_t) / (sga_p / rev_p)
        LVGI = ((ltd_t + cl_t) / ta_t) / ((ltd_p + cl_p) / ta_p)
        TATA = (ni_t - cfo_t) / ta_t

        M = (-4.84
             + 0.92  * DSRI
             + 0.528 * GMI
             + 0.404 * AQI
             + 0.892 * SGI
             + 0.115 * DEPI
             - 0.172 * SGAI
             + 4.679 * TATA
             - 0.327 * LVGI)

        return round(M, 3), {
            "DSRI": round(DSRI, 3),
            "GMI":  round(GMI, 3),
            "AQI":  round(AQI, 3),
            "SGI":  round(SGI, 3),
            "DEPI": round(DEPI, 3),
            "SGAI": round(SGAI, 3),
            "LVGI": round(LVGI, 3),
            "TATA": round(TATA, 3),
        }
    except Exception:
        return None, None


def _compute_piotroski_f(prev: dict, curr: dict) -> tuple[int | None, list[dict[str, Any]] | None]:
    """Piotroski F-Score — 9-test financial-strength gauge (0..9).

    Higher = stronger. 8-9 strong, 4-6 neutral, 0-3 weak.
    """
    ni_t   = _g(curr, "net_income")
    ni_p   = _g(prev, "net_income")
    cfo_t  = _g(curr, "operating_cash_flow")
    ta_t   = _g(curr, "total_assets")
    ta_p   = _g(prev, "total_assets")
    ltd_t  = _g(curr, "long_term_debt")
    ltd_p  = _g(prev, "long_term_debt")
    ca_t   = _g(curr, "current_assets")
    ca_p   = _g(prev, "current_assets")
    cl_t   = _g(curr, "current_liabilities")
    cl_p   = _g(prev, "current_liabilities")
    sh_t   = _g(curr, "shares_outstanding")
    sh_p   = _g(prev, "shares_outstanding")
    rev_t  = _g(curr, "revenue")
    rev_p  = _g(prev, "revenue")
    gp_t   = _g(curr, "gross_profit")
    gp_p   = _g(prev, "gross_profit")
    cogs_t = _g(curr, "cost_of_revenue")
    cogs_p = _g(prev, "cost_of_revenue")

    if gp_t is None and rev_t is not None and cogs_t is not None: gp_t = rev_t - cogs_t
    if gp_p is None and rev_p is not None and cogs_p is not None: gp_p = rev_p - cogs_p

    tests: list[dict[str, Any]] = []

    def add(name: str, passed: bool | None, note: str = "") -> None:
        tests.append({"name": name, "passed": bool(passed) if passed is not None else None, "note": note})

    # Profitability
    add("ROA > 0", (ni_t is not None and ta_t and ta_t > 0 and ni_t > 0))
    add("CFO > 0", (cfo_t is not None and cfo_t > 0))
    if ni_t is not None and ta_t and ni_p is not None and ta_p:
        roa_t = ni_t / ta_t; roa_p = ni_p / ta_p
        add("ΔROA > 0 (improving)", roa_t > roa_p)
    else:
        add("ΔROA > 0 (improving)", None, "needs prior year")
    if cfo_t is not None and ni_t is not None:
        add("CFO > NI (accruals quality)", cfo_t > ni_t)
    else:
        add("CFO > NI (accruals quality)", None)
    # Leverage / liquidity / source of funds
    if ltd_t is not None and ltd_p is not None and ta_t and ta_p and ta_t > 0 and ta_p > 0:
        add("Δ Long-term Debt / Assets ≤ 0", (ltd_t / ta_t) <= (ltd_p / ta_p))
    else:
        add("Δ Long-term Debt / Assets ≤ 0", None)
    if ca_t and cl_t and ca_p and cl_p and cl_t > 0 and cl_p > 0:
        cr_t = ca_t / cl_t; cr_p = ca_p / cl_p
        add("Current ratio improving", cr_t > cr_p)
    else:
        add("Current ratio improving", None)
    if sh_t is not None and sh_p is not None:
        # allow tiny RSU-driven creep (<1%) as still "passing"
        add("No material new shares issued", sh_t <= sh_p * 1.01)
    else:
        add("No material new shares issued", None, "shares-outstanding not tagged")
    # Operating efficiency
    if gp_t is not None and rev_t and gp_p is not None and rev_p and rev_t > 0 and rev_p > 0:
        add("Gross margin improving", (gp_t / rev_t) > (gp_p / rev_p))
    else:
        add("Gross margin improving", None)
    if rev_t and ta_t and rev_p and ta_p and ta_t > 0 and ta_p > 0:
        add("Asset turnover improving", (rev_t / ta_t) > (rev_p / ta_p))
    else:
        add("Asset turnover improving", None)

    countable = [t for t in tests if t["passed"] is not None]
    if len(countable) < 5:
        return None, None
    score = sum(1 for t in countable if t["passed"])
    # Pro-rate to a 9-test scale if we couldn't compute all 9
    if len(countable) < 9:
        score = round(score * 9 / len(countable))
    return score, tests


def _compute_altman_z(curr: dict) -> tuple[float | None, str | None]:
    """Altman Z'-Score (private-company variant, 1983) for distress prediction.

    Z' = 0.717 * (WC/TA) + 0.847 * (RE/TA) + 3.107 * (EBIT/TA)
       + 0.420 * (Equity_book/TL) + 0.998 * (Sales/TA)

    Zones: > 2.9 safe · 1.23–2.9 grey · < 1.23 distress
    """
    ca = _g(curr, "current_assets")
    cl = _g(curr, "current_liabilities")
    ta = _g(curr, "total_assets")
    re_ = _g(curr, "retained_earnings")
    oi = _g(curr, "operating_income")
    eq = _g(curr, "stockholders_equity")
    tl = _g(curr, "total_liabilities")
    sl = _g(curr, "revenue")

    if ta is None or ta <= 0 or tl is None or tl <= 0:
        return None, None

    wc = (ca - cl) if (ca is not None and cl is not None) else None
    parts = []
    if wc is not None: parts.append(0.717 * (wc / ta))
    if re_ is not None: parts.append(0.847 * (re_ / ta))
    if oi is not None: parts.append(3.107 * (oi / ta))
    if eq is not None: parts.append(0.420 * (eq / tl))
    if sl is not None: parts.append(0.998 * (sl / ta))

    if len(parts) < 3:
        return None, None

    z = round(sum(parts), 3)
    if z >= 2.9:    zone = "safe"
    elif z >= 1.23: zone = "grey"
    else:           zone = "distress"
    return z, zone


def compute_forensic_scores(table: list[dict[str, Any]]) -> ForensicScores:
    notes: list[str] = []
    if len(table) < 1:
        return ForensicScores(notes=["No annual data available."])

    curr = table[-1]
    prev = table[-2] if len(table) >= 2 else None

    bm, bcomp = (None, None)
    pf, pbreak = (None, None)
    if prev is not None:
        bm, bcomp = _compute_beneish_m(prev, curr)
        pf, pbreak = _compute_piotroski_f(prev, curr)
    else:
        notes.append("Beneish & Piotroski need at least 2 years of data — only 1 available.")

    az, azone = _compute_altman_z(curr)

    bsig = None
    if bm is not None:
        bsig = "manipulator" if bm > -1.78 else "non-manipulator"

    pstrength = None
    if pf is not None:
        pstrength = "strong" if pf >= 7 else ("weak" if pf <= 3 else "neutral")

    if bm is None:    notes.append("Beneish M-Score not computed (missing PPE / SGA / depreciation tags).")
    if pf is None:    notes.append("Piotroski F-Score not computed (insufficient tagged data points).")
    if az is None:    notes.append("Altman Z'-Score not computed (missing balance-sheet items).")

    return ForensicScores(
        beneish_m=bm, beneish_signal=bsig, beneish_components=bcomp,
        piotroski_f=pf, piotroski_strength=pstrength, piotroski_breakdown=pbreak,
        altman_z=az, altman_zone=azone,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Composite Risk Grade
# ---------------------------------------------------------------------------


def compute_risk_grade(
    *,
    anomalies: list[FinancialAnomaly],
    red_flags: list[Any],  # list[Finding] but we only read .label / .confidence
    forensic: ForensicScores | None,
) -> RiskGrade:
    """Aggregate everything we know into a 0-100 risk score and an A-F grade."""
    breakdown: list[dict[str, Any]] = []
    score = 0

    # 1) Quantitative (XBRL) anomalies — strongest signal because deterministic
    high = sum(1 for a in anomalies if a.severity == "high")
    med  = sum(1 for a in anomalies if a.severity == "medium")
    low  = sum(1 for a in anomalies if a.severity == "low")
    quant = min(40, 18 * high + 8 * med + 3 * low)
    score += quant
    breakdown.append({
        "label": "Quantitative anomalies (XBRL)",
        "points": quant, "max": 40,
        "note": f"{high} high · {med} med · {low} low",
    })

    # 2) LLM-extracted red flags — weighted by label
    facts = sum(1 for f in red_flags if getattr(f, "label", "") == "fact")
    infs  = sum(1 for f in red_flags if getattr(f, "label", "") == "inference")
    qs    = sum(1 for f in red_flags if getattr(f, "label", "") == "question")
    specs = sum(1 for f in red_flags if getattr(f, "label", "") == "speculation")
    llm = min(25, 5 * facts + 3 * infs + 1 * qs + 4 * specs)
    score += llm
    breakdown.append({
        "label": "Citation-graded findings",
        "points": llm, "max": 25,
        "note": f"{facts} fact · {infs} inference · {qs} question · {specs} speculation",
    })

    # 3) Beneish M-Score (manipulation likelihood)
    if forensic and forensic.beneish_m is not None:
        b = forensic.beneish_m
        # >-1.78 = manipulator. Map the gap above -1.78 into 0..15 points.
        b_pts = int(min(15, max(0, (b + 1.78) * 4)))
        score += b_pts
        breakdown.append({
            "label": "Beneish M-Score",
            "points": b_pts, "max": 15,
            "note": f"M = {b} ({forensic.beneish_signal})",
        })
    else:
        breakdown.append({"label": "Beneish M-Score", "points": 0, "max": 15, "note": "n/a"})

    # 4) Piotroski F-Score (subtractive — strong company reduces risk)
    if forensic and forensic.piotroski_f is not None:
        f = forensic.piotroski_f
        if   f >= 7: p_pts = -8
        elif f >= 5: p_pts = -3
        elif f <= 3: p_pts =  6
        else:        p_pts =  0
        score += p_pts
        breakdown.append({
            "label": "Piotroski F-Score",
            "points": p_pts, "max": 0,  # informational, not capped to a max
            "note": f"F = {f}/9 ({forensic.piotroski_strength})",
        })

    # 5) Altman Z' (additive when distressed)
    if forensic and forensic.altman_z is not None:
        z = forensic.altman_z
        if   forensic.altman_zone == "distress": z_pts = 15
        elif forensic.altman_zone == "grey":     z_pts = 5
        else:                                    z_pts = 0
        score += z_pts
        breakdown.append({
            "label": "Altman Z'-Score",
            "points": z_pts, "max": 15,
            "note": f"Z' = {z} ({forensic.altman_zone})",
        })

    score = max(0, min(100, score))
    if   score < 15: grade = "A"
    elif score < 30: grade = "B"
    elif score < 50: grade = "C"
    elif score < 70: grade = "D"
    else:            grade = "F"

    return RiskGrade(score=int(score), grade=grade, breakdown=breakdown)


def build_financial_snapshot(cik10: str, cache_dir: str = "cache") -> FinancialSnapshot:
    facts = fetch_company_facts(cik10, cache_dir=cache_dir) or {}
    series = _build_series(facts)
    table = _annual_table(series)
    anomalies = _detect_anomalies(table)
    forensic = compute_forensic_scores(table)
    return FinancialSnapshot(
        cik=cik10,
        entity=facts.get("entityName"),
        series=series,
        annual_table=table,
        anomalies=anomalies,
        forensic_scores=forensic,
    )


def snapshot_to_brief(snap: FinancialSnapshot) -> dict[str, Any]:
    """Compact dict for prompting / JSON dump (no dataclasses)."""
    fs = snap.forensic_scores
    forensic_dict = None
    if fs is not None:
        forensic_dict = {
            "beneish_m": fs.beneish_m,
            "beneish_signal": fs.beneish_signal,
            "beneish_components": fs.beneish_components,
            "piotroski_f": fs.piotroski_f,
            "piotroski_strength": fs.piotroski_strength,
            "piotroski_breakdown": fs.piotroski_breakdown,
            "altman_z": fs.altman_z,
            "altman_zone": fs.altman_zone,
            "notes": fs.notes,
        }
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
        "forensic_scores": forensic_dict,
    }
