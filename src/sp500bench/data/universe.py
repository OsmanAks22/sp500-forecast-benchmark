"""The frozen 50-ticker universe.

Selection rule
--------------
Five names per GICS sector, all with continuous listed history back to
2005-01-01 so that no ticker enters the panel mid-sample.

Survivorship bias - stated plainly
----------------------------------
This universe is drawn from *current* index membership. Firms that were in the
S&P 500 during the sample and later failed, merged or were delisted are absent.
The bias inflates realised returns and, more subtly, removes exactly the
tail-risk episodes a forecasting model most needs to handle. Correcting it
requires point-in-time constituent data (CRSP, Compustat), which is licensed and
outside this project's zero-cost constraint.

Do not bury this. It belongs in the Limitations section, and a viva panel will
ask about it.
"""

from __future__ import annotations

UNIVERSE: dict[str, list[str]] = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "ORCL", "CSCO"],
    "Health Care":            ["JNJ", "PFE", "MRK", "ABT", "UNH"],
    "Financials":             ["JPM", "BAC", "WFC", "GS", "AXP"],
    "Consumer Discretionary": ["AMZN", "HD", "MCD", "NKE", "LOW"],
    "Consumer Staples":       ["PG", "KO", "PEP", "WMT", "COST"],
    "Industrials":            ["HON", "UNP", "CAT", "GE", "LMT"],
    "Energy":                 ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Communication Services": ["GOOGL", "DIS", "CMCSA", "T", "VZ"],
    "Utilities":              ["NEE", "DUK", "SO", "D", "AEP"],
    "Materials & Real Estate":["APD", "SHW", "ECL", "NEM", "AMT"],
}

TICKERS: list[str] = [t for names in UNIVERSE.values() for t in names]
SECTOR_OF: dict[str, str] = {
    t: sector for sector, names in UNIVERSE.items() for t in names
}

assert len(TICKERS) == 50, f"universe must hold exactly 50 tickers, found {len(TICKERS)}"
assert len(set(TICKERS)) == 50, "duplicate ticker in universe"
