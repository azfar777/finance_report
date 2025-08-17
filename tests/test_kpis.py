"""Lightweight tests for KPI computation."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from finance.kpis import compute_kpis, _dividend_yield_ttm, _total_return


def _check_sections(data: dict) -> None:
    for key in [
        "growth",
        "profit",
        "quality",
        "leverage",
        "liquidity",
        "shareholder",
        "performance",
    ]:
        assert key in data
        assert isinstance(data[key], dict)
        for v in data[key].values():
            assert v is None or isinstance(v, (int, float))


def test_aapl_basic() -> None:
    data = compute_kpis("AAPL")
    _check_sections(data)


def test_canadian_ticker() -> None:
    data = compute_kpis("SU.TO")
    _check_sections(data)


def test_short_period() -> None:
    data = compute_kpis("AAPL", period_years=1)
    _check_sections(data)


def test_dividend_yield_close_only() -> None:
    idx = pd.date_range("2023-01-01", periods=3, freq="M")
    prices = pd.DataFrame({"Close": [100, 101, 102], "Dividends": [0.5, 0.0, 0.0]}, index=idx)
    dy = _dividend_yield_ttm(prices)
    assert dy == pytest.approx(0.5 / 102)


def test_total_return_close_only() -> None:
    idx = pd.to_datetime(["2022-01-01", "2023-01-01"])
    prices = pd.DataFrame({"Close": [100, 110], "Dividends": [0.0, 5.0]}, index=idx)
    tr = _total_return(prices, years=1)
    assert tr == pytest.approx(0.15)
