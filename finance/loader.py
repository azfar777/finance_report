"""Data loading utilities for financial analysis.

This module fetches price and statement data from Yahoo Finance using
`yfinance` and caches the raw DataFrames as Parquet files for
reproducibility.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Dict

import pandas as pd
import yfinance as yf

CACHE_DIR = os.getenv("CACHE_DIR", "storage/cache")


def _to_snake(name: str) -> str:
    """Convert CamelCase or spaces to lower snake_case."""
    name = re.sub("[^0-9a-zA-Z]+", "_", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.replace("__", "_").strip("_").lower()


def _to_snake_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with snake_cased columns."""
    return df.rename(columns={c: _to_snake(str(c)) for c in df.columns})


def load_cached_or(fn: Callable[[], pd.DataFrame], path: str) -> pd.DataFrame:
    """Load a DataFrame from *path* or compute with *fn* and cache."""
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    df = fn()
    if df is None:
        df = pd.DataFrame()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
    except Exception as exc:  # pragma: no cover - caching failure shouldn't stop
        print(f"warning: could not cache {path}: {exc}")
    return df


class FinanceLoader:
    """Small abstraction over yfinance with on-disk caching."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------ Prices
    def load_prices(self, ticker: str, period_years: int = 5) -> pd.DataFrame:
        """Load daily OHLCV prices for *ticker*.

        Data are downloaded from Yahoo Finance for the given period and cached
        under ``{cache_dir}/{ticker}/prices.parquet``.
        """

        tdir = os.path.join(self.cache_dir, ticker.upper())
        path = os.path.join(tdir, "prices.parquet")

        def _download() -> pd.DataFrame:
            try:
                df = yf.download(ticker, period=f"{period_years}y", progress=False, actions=True)
            except Exception as exc:  # pragma: no cover - network errors
                print(f"warning: failed to download prices for {ticker}: {exc}")
                return pd.DataFrame()
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize("UTC")
            return df

        df = load_cached_or(_download, path)
        return df

    # -------------------------------------------------------------- Statements
    def load_statements(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Load financial statements for *ticker*.

        Annual and quarterly income statement, balance sheet and cash flow
        tables are fetched and cached individually. Missing tables are returned
        as empty DataFrames.
        """

        tdir = os.path.join(self.cache_dir, ticker.upper())
        os.makedirs(tdir, exist_ok=True)
        tkr = yf.Ticker(ticker)

        def _fetch_attr(attr: str, quarterly: bool) -> pd.DataFrame:
            try:
                data = getattr(tkr, attr)
                df = data.T if hasattr(data, "T") else pd.DataFrame()
            except Exception:
                df = pd.DataFrame()
            if df is None:
                df = pd.DataFrame()
            if not df.empty:
                df = _to_snake_cols(df)
            if df.empty:
                kind = "quarterly" if quarterly else "annual"
                print(f"warning: {ticker} missing {attr} {kind} data")
            return df

        income_annual = load_cached_or(lambda: _fetch_attr("financials", False), os.path.join(tdir, "income_annual.parquet"))
        balance_annual = load_cached_or(lambda: _fetch_attr("balance_sheet", False), os.path.join(tdir, "balance_annual.parquet"))
        cashflow_annual = load_cached_or(lambda: _fetch_attr("cashflow", False), os.path.join(tdir, "cashflow_annual.parquet"))

        income_quarterly = load_cached_or(lambda: _fetch_attr("quarterly_financials", True), os.path.join(tdir, "income_quarterly.parquet"))
        balance_quarterly = load_cached_or(lambda: _fetch_attr("quarterly_balance_sheet", True), os.path.join(tdir, "balance_quarterly.parquet"))
        cashflow_quarterly = load_cached_or(lambda: _fetch_attr("quarterly_cashflow", True), os.path.join(tdir, "cashflow_quarterly.parquet"))

        return {
            "income_annual": income_annual,
            "balance_annual": balance_annual,
            "cashflow_annual": cashflow_annual,
            "income_quarterly": income_quarterly,
            "balance_quarterly": balance_quarterly,
            "cashflow_quarterly": cashflow_quarterly,
        }
