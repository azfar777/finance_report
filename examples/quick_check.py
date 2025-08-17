"""Quick manual check for KPI computation."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from finance.kpis import compute_kpis


def main() -> None:
    data = compute_kpis("AAPL")
    fields = [
        ("rev_cagr_3y", data["growth"].get("rev_cagr_3y")),
        ("op_margin_ttm", data["profit"].get("op_margin_ttm")),
        ("roic_ttm", data["quality"].get("roic_ttm")),
        ("net_debt_ebitda", data["leverage"].get("net_debt_ebitda")),
        ("current_ratio", data["liquidity"].get("current_ratio")),
        ("dividend_yield_ttm", data["shareholder"].get("dividend_yield_ttm")),
        ("total_return_3y", data["performance"].get("total_return_3y")),
    ]
    for name, val in fields:
        print(f"{name:20s}: {val}")


if __name__ == "__main__":
    main()
