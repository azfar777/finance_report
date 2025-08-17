"""Deterministic KPI calculations for financial analysis.

This module loads prices and financial statements via :class:`FinanceLoader`
then computes a bundle of key performance indicators. All calculations are
pure Python and tolerate missing data by returning ``None``.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import pandas as pd

from .loader import FinanceLoader

# ---------------------------------------------------------------------------
# Helpers

def _cagr(series: pd.Series, years: int) -> Optional[float]:
    """Return the compound annual growth rate for ``series`` over ``years``.

    The series must contain at least ``years + 1`` non-null values ordered by
    time. Returns ``None`` if the calculation is not possible.
    """
    series = series.dropna().sort_index()
    if len(series) < years + 1:
        return None
    start = series.iloc[-(years + 1)]
    end = series.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def _latest(data, key: Optional[str] = None) -> Optional[float]:
    """Return the latest non-null scalar from ``data``.

    ``data`` may be a Series or DataFrame. If ``key`` is provided for a
    DataFrame, that column is used.
    """
    try:
        if isinstance(data, pd.DataFrame):
            if key is None or key not in data.columns:
                return None
            series = data[key]
        else:
            series = data
        val = series.dropna().iloc[-1]
        return float(val)
    except Exception:
        return None


def _ttm_from_quarterlies(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    """Return trailing-twelve-month sums for selected columns."""
    out: Dict[str, float] = {}
    if df is None or df.empty:
        return out
    df = df.sort_index().tail(4)
    for col in cols:
        if col in df.columns:
            val = df[col].dropna().sum()
            if pd.notna(val):
                out[col] = float(val)
    return out


def _close_col(prices: pd.DataFrame) -> Optional[str]:
    """Return the column name for close prices, adjusted if available."""
    for col in ("Adj Close", "Close"):
        if col in prices.columns:
            return col
    return None


def _dividend_yield_ttm(prices: pd.DataFrame) -> Optional[float]:
    """Compute trailing 12 month dividend yield from ``prices`` if possible."""
    if "Dividends" not in prices.columns:
        return None
    last = prices.index[-1]
    start = last - pd.Timedelta(days=365)
    divs = prices["Dividends"][prices.index >= start].sum()
    col = _close_col(prices)
    if col is None:
        return None
    close = prices[col].iloc[-1]
    if close and close != 0:
        return float(divs) / float(close)
    return None


def _total_return(prices: pd.DataFrame, years: int) -> Optional[float]:
    """Simple total return for ``years`` using close prices and dividends."""
    if prices.empty:
        return None
    col = _close_col(prices)
    if col is None:
        return None
    adj = prices[col].dropna().sort_index()
    end_date = adj.index[-1]
    start_date = end_date - pd.DateOffset(years=years)
    adj_since = adj[adj.index >= start_date]
    if adj_since.empty:
        return None
    start_price = adj_since.iloc[0]
    end_price = adj.iloc[-1]
    if start_price == 0:
        return None
    divs = 0.0
    if "Dividends" in prices.columns:
        divs = prices["Dividends"][prices.index >= adj_since.index[0]].sum()
    return float(end_price + divs) / float(start_price) - 1


# ---------------------------------------------------------------------------
# KPI computation

def compute_kpis(ticker: str, period_years: int = 5, tax_rate_fallback: float = 0.21) -> Dict:
    """Compute a bundle of KPIs for ``ticker``.

    Parameters
    ----------
    ticker:
        Ticker symbol, e.g. ``"AAPL"`` or ``"SU.TO"``.
    period_years:
        Number of years of price history to download.
    tax_rate_fallback:
        Used when the effective tax rate cannot be derived from the income
        statement.
    """

    loader = FinanceLoader()
    prices = loader.load_prices(ticker, period_years)
    stmts = loader.load_statements(ticker)
    notes: List[str] = []

    as_of = pd.Timestamp.utcnow().date().isoformat()
    if not prices.empty:
        as_of = prices.index[-1].tz_convert("UTC").date().isoformat()

    income_a = stmts["income_annual"].sort_index()
    balance_a = stmts["balance_annual"].sort_index()
    cash_a = stmts["cashflow_annual"].sort_index()
    income_q = stmts["income_quarterly"]
    cash_q = stmts["cashflow_quarterly"]

    ttm_income = _ttm_from_quarterlies(
        income_q,
        [
            "total_revenue",
            "gross_profit",
            "operating_income",
            "net_income",
            "ebit",
            "ebitda",
            "income_tax_expense",
            "pre_tax_income",
            "interest_expense",
        ],
    )
    ttm_cash = _ttm_from_quarterlies(cash_q, ["operating_cash_flow", "capital_expenditures"])

    # ---------------------------- Growth
    rev_series = income_a.get("total_revenue", pd.Series(dtype=float))
    eps_series = income_a.get("diluted_eps")
    if eps_series is None or eps_series.dropna().empty:
        ni = income_a.get("net_income")
        shares = income_a.get("diluted_shares_outstanding") or income_a.get("diluted_average_shares")
        if ni is not None and shares is not None:
            eps_series = ni / shares
        else:
            eps_series = pd.Series(dtype=float)

    growth = {
        "rev_cagr_3y": _cagr(rev_series, 3),
        "rev_cagr_5y": _cagr(rev_series, 5),
        "eps_cagr_3y": _cagr(eps_series, 3),
        "eps_cagr_5y": _cagr(eps_series, 5),
    }

    # ---------------------------- Profitability
    revenue = ttm_income.get("total_revenue") or _latest(income_a, "total_revenue")
    gross_profit = ttm_income.get("gross_profit") or _latest(income_a, "gross_profit")
    operating_income = ttm_income.get("operating_income") or _latest(income_a, "operating_income")
    net_income = ttm_income.get("net_income") or _latest(income_a, "net_income")

    ocf = ttm_cash.get("operating_cash_flow") or _latest(cash_a, "operating_cash_flow")
    capex = ttm_cash.get("capital_expenditures") or _latest(cash_a, "capital_expenditures")
    fcf = ocf - capex if ocf is not None and capex is not None else None

    def _margin(num, den):
        return (num / den) if (num is not None and den not in (None, 0)) else None

    profit = {
        "gross_margin_ttm": _margin(gross_profit, revenue),
        "op_margin_ttm": _margin(operating_income, revenue),
        "net_margin_ttm": _margin(net_income, revenue),
        "fcf_margin_ttm": _margin(fcf, revenue),
    }

    # ---------------------------- Quality & Efficiency
    tax_exp = ttm_income.get("income_tax_expense") or _latest(income_a, "income_tax_expense")
    pre_tax = ttm_income.get("pre_tax_income") or _latest(income_a, "pre_tax_income")
    if tax_exp is not None and pre_tax not in (None, 0):
        tax_rate = tax_exp / pre_tax
    else:
        tax_rate = tax_rate_fallback
        notes.append("tax_rate_fallback_used")

    nopat = operating_income * (1 - tax_rate) if operating_income is not None else None

    debt = _latest(balance_a, "total_debt")
    if debt is None:
        lt = _latest(balance_a, "long_term_debt")
        st = _latest(balance_a, "short_long_term_debt")
        if lt is not None or st is not None:
            debt = (lt or 0) + (st or 0)
    equity = _latest(balance_a, "total_stockholder_equity") or _latest(balance_a, "total_equity")
    cash = _latest(balance_a, "cash_and_cash_equivalents") or _latest(balance_a, "cash")
    invested_capital: Optional[float] = None
    if debt is not None and equity is not None and cash is not None:
        invested_capital = debt + equity - cash
    else:
        assets = _latest(balance_a, "total_assets")
        curr_liab = _latest(balance_a, "total_current_liabilities") or _latest(balance_a, "current_liabilities")
        if assets is not None and curr_liab is not None and cash is not None:
            invested_capital = assets - curr_liab - cash

    roic = (nopat / invested_capital) if (nopat is not None and invested_capital not in (None, 0)) else None
    roe = _margin(net_income, equity)
    fcf_over_ni = _margin(fcf, net_income)

    quality = {
        "roic_ttm": roic,
        "roe_ttm": roe,
        "fcf_over_ni_ttm": fcf_over_ni,
    }

    # ---------------------------- Leverage & Solvency
    net_debt = (debt - cash) if (debt is not None and cash is not None) else None
    ebitda = ttm_income.get("ebitda") or _latest(income_a, "ebitda")
    ndebt_ebitda = _margin(net_debt, ebitda) if net_debt is not None else None

    ebit = ttm_income.get("ebit") or _latest(income_a, "ebit") or operating_income
    interest = ttm_income.get("interest_expense") or _latest(income_a, "interest_expense")
    interest_cov = (ebit / abs(interest)) if (ebit is not None and interest not in (None, 0)) else None

    leverage = {
        "net_debt": net_debt,
        "net_debt_ebitda": ndebt_ebitda,
        "interest_coverage": interest_cov,
    }

    # ---------------------------- Liquidity
    curr_assets = _latest(balance_a, "total_current_assets") or _latest(balance_a, "current_assets")
    curr_liab = _latest(balance_a, "total_current_liabilities") or _latest(balance_a, "current_liabilities")
    inventory = _latest(balance_a, "inventory")
    current_ratio = _margin(curr_assets, curr_liab)
    quick_ratio = _margin((curr_assets - inventory) if (curr_assets is not None and inventory is not None) else None, curr_liab)

    liquidity = {
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
    }

    # ---------------------------- Shareholder returns
    dividend_yield = _dividend_yield_ttm(prices) if not prices.empty else None
    shares_col = None
    for col in ["diluted_shares_outstanding", "diluted_average_shares"]:
        if col in income_a.columns:
            shares_col = col
            break
    buyback_yield = None
    if shares_col:
        shares = income_a[shares_col].dropna().sort_index()
        if len(shares) >= 2:
            last = shares.iloc[-1]
            prev = shares.iloc[-2]
            if prev != 0:
                buyback_yield = -(last - prev) / prev
    shareholder = {
        "dividend_yield_ttm": dividend_yield,
        "buyback_yield": buyback_yield,
    }

    # ---------------------------- Performance
    performance = {
        "total_return_1y": _total_return(prices, 1),
        "total_return_3y": _total_return(prices, 3),
        "total_return_5y": _total_return(prices, 5),
    }

    return {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "growth": growth,
        "profit": profit,
        "quality": quality,
        "leverage": leverage,
        "liquidity": liquidity,
        "shareholder": shareholder,
        "performance": performance,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# CLI
if __name__ == "__main__":  # pragma: no cover - manual invocation
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--years", type=int, default=5)
    args = ap.parse_args()
    out = compute_kpis(args.ticker, period_years=args.years)
    print(json.dumps(out, indent=2, default=lambda x: None))
