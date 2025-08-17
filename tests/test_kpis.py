"""Lightweight tests for KPI computation."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from finance.kpis import compute_kpis


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
