from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .pipeline import generate_report


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Generate a Muddy Waters-style skeptical note from SEC filings."
    )
    ap.add_argument("--ticker", required=True, help="Ticker symbol or CIK (e.g. LKNCY, NKLA, 1767582).")
    ap.add_argument("--out", default=None, help="Output HTML path (default: outputs/<TICKER>.html).")
    ap.add_argument("--max-filings", type=int, default=8, help="Max number of filings to pull.")
    ap.add_argument(
        "--filing-types",
        default="auto",
        help="Comma-separated filing types, or 'auto' to detect domestic vs foreign issuer.",
    )
    args = ap.parse_args()

    ticker = args.ticker.strip().upper()
    out = args.out or f"outputs/{ticker}.html"
    raw = (args.filing_types or "").strip().lower()
    if raw in {"", "auto"}:
        filing_types = None
    else:
        filing_types = [x.strip() for x in args.filing_types.split(",") if x.strip()]

    def _print(msg: str, frac: float) -> None:
        print(f"[{int(frac*100):3d}%] {msg}")

    res = generate_report(
        ticker=ticker,
        filing_types=filing_types,
        max_filings=args.max_filings,
        out_html_path=out,
        progress=_print,
    )
    print(res.out_html_path)


if __name__ == "__main__":
    main()

