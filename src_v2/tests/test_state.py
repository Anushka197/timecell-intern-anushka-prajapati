"""
Tests for core/state.py — state persistence round-trip and missing-file defaults.

Property 9: State Persistence Round-Trip
  Validates: Requirements 13.1, 13.2, 13.3, 8.4
"""

import math
import pathlib
import tempfile

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src_v2.core.models import (
    Asset,
    CompanyWatchlistEntry,
    CorrelationWatchlist,
    Portfolio,
    ReportWatchlist,
    WatchlistEntry,
)
from src_v2.core.state import (
    load_correlation_watchlist,
    load_portfolio,
    load_report_watchlist,
    save_correlation_watchlist,
    save_portfolio,
    save_report_watchlist,
)


# ── Hypothesis strategies ──────────────────────────────────────────────────

_asset_name = st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")))

_asset_strategy = st.builds(
    Asset,
    name=_asset_name,
    allocation_pct=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    crash_pct=st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False),
)

_portfolio_strategy = st.builds(
    Portfolio,
    assets=st.lists(_asset_strategy, min_size=0, max_size=10),
    total_value=st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False),
    monthly_expenses=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)

_watchlist_entry_strategy = st.builds(
    WatchlistEntry,
    ticker=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",))),
    is_crypto=st.booleans(),
    added_at=st.just("2025-01-15T10:30:00"),
)

_correlation_watchlist_strategy = st.builds(
    CorrelationWatchlist,
    entries=st.lists(_watchlist_entry_strategy, min_size=0, max_size=10),
)

_company_entry_strategy = st.builds(
    CompanyWatchlistEntry,
    ticker=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",))),
    company_name=st.text(min_size=1, max_size=50),
    added_at=st.just("2025-01-15T10:30:00"),
    last_indexed_date=st.one_of(st.none(), st.just("2024-11-01")),
)

_report_watchlist_strategy = st.builds(
    ReportWatchlist,
    entries=st.lists(_company_entry_strategy, min_size=0, max_size=10),
)


# ── Helper: float equality with tolerance ─────────────────────────────────

def _floats_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# ── Property 9: State Persistence Round-Trip ──────────────────────────────
# Validates: Requirements 13.1, 13.2, 13.3, 8.4

@given(portfolio=_portfolio_strategy)
@settings(max_examples=100)
def test_portfolio_round_trip(portfolio: Portfolio) -> None:
    """
    **Property 9: State Persistence Round-Trip**
    **Validates: Requirements 13.1, 13.2, 13.3, 8.4**

    save_portfolio followed by load_portfolio must reproduce the original
    Portfolio within floating-point tolerance.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "portfolio.json"
        save_portfolio(portfolio, path)
        loaded = load_portfolio(path)

    assert len(loaded.assets) == len(portfolio.assets)
    for orig, back in zip(portfolio.assets, loaded.assets):
        assert orig.name == back.name
        assert _floats_equal(orig.allocation_pct, back.allocation_pct)
        assert _floats_equal(orig.crash_pct, back.crash_pct)
    assert _floats_equal(loaded.total_value, portfolio.total_value)
    assert _floats_equal(loaded.monthly_expenses, portfolio.monthly_expenses)


@given(watchlist=_correlation_watchlist_strategy)
@settings(max_examples=100)
def test_correlation_watchlist_round_trip(watchlist: CorrelationWatchlist) -> None:
    """
    **Property 9: State Persistence Round-Trip (CorrelationWatchlist)**
    **Validates: Requirements 13.1, 13.2, 13.3, 8.4**
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "corr.json"
        save_correlation_watchlist(watchlist, path)
        loaded = load_correlation_watchlist(path)

    assert len(loaded.entries) == len(watchlist.entries)
    for orig, back in zip(watchlist.entries, loaded.entries):
        assert orig.ticker == back.ticker
        assert orig.is_crypto == back.is_crypto
        assert orig.added_at == back.added_at


@given(watchlist=_report_watchlist_strategy)
@settings(max_examples=100)
def test_report_watchlist_round_trip(watchlist: ReportWatchlist) -> None:
    """
    **Property 9: State Persistence Round-Trip (ReportWatchlist)**
    **Validates: Requirements 13.1, 13.2, 13.3, 8.4**
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "report.json"
        save_report_watchlist(watchlist, path)
        loaded = load_report_watchlist(path)

    assert len(loaded.entries) == len(watchlist.entries)
    for orig, back in zip(watchlist.entries, loaded.entries):
        assert orig.ticker == back.ticker
        assert orig.company_name == back.company_name
        assert orig.added_at == back.added_at
        assert orig.last_indexed_date == back.last_indexed_date


# ── Unit tests: missing-file defaults ─────────────────────────────────────

def test_load_portfolio_missing_file(tmp_path: pathlib.Path) -> None:
    """load_portfolio returns empty Portfolio() when the file does not exist."""
    result = load_portfolio(tmp_path / "nonexistent.json")
    assert result == Portfolio()


def test_load_correlation_watchlist_missing_file(tmp_path: pathlib.Path) -> None:
    """load_correlation_watchlist returns empty CorrelationWatchlist() when file is missing."""
    result = load_correlation_watchlist(tmp_path / "nonexistent.json")
    assert result == CorrelationWatchlist()


def test_load_report_watchlist_missing_file(tmp_path: pathlib.Path) -> None:
    """load_report_watchlist returns empty ReportWatchlist() when file is missing."""
    result = load_report_watchlist(tmp_path / "nonexistent.json")
    assert result == ReportWatchlist()


# ── Unit tests: concrete round-trip examples ──────────────────────────────

def test_portfolio_concrete_round_trip(tmp_path: pathlib.Path) -> None:
    """Concrete example: portfolio with multiple assets survives save/load."""
    p = Portfolio(
        assets=[
            Asset("BTC", 30.0, -80.0),
            Asset("NIFTY50", 40.0, -40.0),
            Asset("GOLD", 20.0, -15.0),
            Asset("CASH", 10.0, 0.0),
        ],
        total_value=10_000_000.0,
        monthly_expenses=80_000.0,
    )
    path = tmp_path / "portfolio.json"
    save_portfolio(p, path)
    loaded = load_portfolio(path)
    assert loaded == p


def test_save_creates_parent_directory(tmp_path: pathlib.Path) -> None:
    """save_portfolio creates intermediate directories if they don't exist."""
    path = tmp_path / "nested" / "dir" / "portfolio.json"
    p = Portfolio(assets=[Asset("BTC", 100.0, -80.0)], total_value=1000.0)
    save_portfolio(p, path)
    assert path.exists()
    loaded = load_portfolio(path)
    assert loaded == p


def test_correlation_watchlist_concrete_round_trip(tmp_path: pathlib.Path) -> None:
    """Concrete example: CorrelationWatchlist with mixed entries survives save/load."""
    wl = CorrelationWatchlist(
        entries=[
            WatchlistEntry("AAPL", False, "2025-01-15T10:30:00"),
            WatchlistEntry("BTC", True, "2025-01-15T10:31:00"),
        ]
    )
    path = tmp_path / "corr.json"
    save_correlation_watchlist(wl, path)
    loaded = load_correlation_watchlist(path)
    assert loaded == wl


def test_report_watchlist_concrete_round_trip(tmp_path: pathlib.Path) -> None:
    """Concrete example: ReportWatchlist with optional field survives save/load."""
    wl = ReportWatchlist(
        entries=[
            CompanyWatchlistEntry("AAPL", "Apple Inc.", "2025-01-15T10:30:00", "2024-11-01"),
            CompanyWatchlistEntry("MSFT", "Microsoft Corporation", "2025-01-15T10:31:00", None),
        ]
    )
    path = tmp_path / "report.json"
    save_report_watchlist(wl, path)
    loaded = load_report_watchlist(path)
    assert loaded == wl
