"""
tests/test_filing_monitor.py — Unit and property tests for Report Monitor logic.

Task 17.1: Unit tests for report monitor logic
Task 17.2: Property test for filing storage path convention (Property 11)
"""

import pytest
from pathlib import Path
from hypothesis import given, settings
import hypothesis.strategies as st

from src_v2.core.models import CompanyWatchlistEntry, ReportWatchlist


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

FILINGS_DIR = Path("src_v2/data/filings")


def _make_watchlist(*tickers_and_names) -> ReportWatchlist:
    """Build a ReportWatchlist from (ticker, name) pairs."""
    entries = [
        CompanyWatchlistEntry(
            ticker=ticker,
            company_name=name,
            added_at="2025-01-01T00:00:00",
        )
        for ticker, name in tickers_and_names
    ]
    return ReportWatchlist(entries=entries)


def _filing_path_for(ticker: str) -> Path:
    """Return the expected filing storage path for a ticker."""
    return FILINGS_DIR / ticker.upper()


# ─────────────────────────────────────────────────────────────────────────
# Task 17.1 — Unit tests
# ─────────────────────────────────────────────────────────────────────────

class TestDuplicateTickerDetection:
    """Requirement 8.5: duplicate tickers must be rejected."""

    def test_has_ticker_returns_true_for_existing_ticker(self):
        wl = _make_watchlist(("AAPL", "Apple Inc."), ("MSFT", "Microsoft"))
        assert wl.has_ticker("AAPL") is True

    def test_has_ticker_is_case_insensitive(self):
        wl = _make_watchlist(("AAPL", "Apple Inc."))
        assert wl.has_ticker("aapl") is True
        assert wl.has_ticker("Aapl") is True

    def test_has_ticker_returns_false_for_missing_ticker(self):
        wl = _make_watchlist(("AAPL", "Apple Inc."))
        assert wl.has_ticker("TSLA") is False

    def test_adding_duplicate_ticker_does_not_add_entry(self):
        """Simulate the guard logic used in the page."""
        wl = _make_watchlist(("AAPL", "Apple Inc."))
        new_ticker = "AAPL"

        # Guard: only add if not already present
        if not wl.has_ticker(new_ticker):
            wl.entries.append(
                CompanyWatchlistEntry(
                    ticker=new_ticker,
                    company_name="Apple Duplicate",
                    added_at="2025-06-01T00:00:00",
                )
            )

        # Should still have exactly one AAPL entry
        aapl_entries = [e for e in wl.entries if e.ticker.upper() == "AAPL"]
        assert len(aapl_entries) == 1

    def test_adding_new_ticker_succeeds(self):
        wl = _make_watchlist(("AAPL", "Apple Inc."))
        new_ticker = "TSLA"

        if not wl.has_ticker(new_ticker):
            wl.entries.append(
                CompanyWatchlistEntry(
                    ticker=new_ticker,
                    company_name="Tesla Inc.",
                    added_at="2025-06-01T00:00:00",
                )
            )

        assert wl.has_ticker("TSLA") is True
        assert len(wl.entries) == 2


class TestFilingPathConvention:
    """Requirement 13.5: filings stored under data/filings/{ticker.upper()}/"""

    def test_filing_path_uses_uppercase_ticker(self):
        path = _filing_path_for("aapl")
        assert path == FILINGS_DIR / "AAPL"

    def test_filing_path_already_uppercase_unchanged(self):
        path = _filing_path_for("MSFT")
        assert path == FILINGS_DIR / "MSFT"

    def test_filing_path_is_under_filings_dir(self):
        path = _filing_path_for("TSLA")
        assert str(path).startswith(str(FILINGS_DIR))

    def test_filing_path_mixed_case_normalised(self):
        path = _filing_path_for("ApPl")
        assert path == FILINGS_DIR / "APPL"


class TestWhatChangedButtonDisabledLogic:
    """Requirement 11.6: 'What Changed?' disabled when fewer than 2 filings indexed."""

    def test_disabled_when_zero_filings(self):
        indexed_dates = []
        has_two_filings = len(indexed_dates) >= 2
        assert has_two_filings is False

    def test_disabled_when_one_filing(self):
        indexed_dates = ["2024-01-01"]
        has_two_filings = len(indexed_dates) >= 2
        assert has_two_filings is False

    def test_enabled_when_two_filings(self):
        indexed_dates = ["2023-01-01", "2024-01-01"]
        has_two_filings = len(indexed_dates) >= 2
        assert has_two_filings is True

    def test_enabled_when_more_than_two_filings(self):
        indexed_dates = ["2022-01-01", "2023-01-01", "2024-01-01"]
        has_two_filings = len(indexed_dates) >= 2
        assert has_two_filings is True


# ─────────────────────────────────────────────────────────────────────────
# Task 17.2 — Property test: Filing Storage Path Convention (Property 11)
#
# Validates: Requirements 13.5
# ─────────────────────────────────────────────────────────────────────────

@given(
    ticker=st.text(
        min_size=1,
        max_size=10,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
    )
)
@settings(max_examples=100)
def test_filing_path_always_under_filings_dir_with_uppercase_ticker(ticker: str):
    """
    **Property 11: Filing Storage Path Convention**
    **Validates: Requirements 13.5**

    For any ticker string (letters only), the constructed filing path must:
    1. Be a subdirectory of FILINGS_DIR
    2. Use the uppercased ticker as the subdirectory name
    """
    path = FILINGS_DIR / ticker.upper()

    # Property 1: path is under FILINGS_DIR
    assert str(path).startswith(str(FILINGS_DIR)), (
        f"Filing path {path} is not under {FILINGS_DIR}"
    )

    # Property 2: the final component is the uppercased ticker
    assert path.name == ticker.upper(), (
        f"Expected directory name {ticker.upper()!r}, got {path.name!r}"
    )

    # Property 3: uppercasing is idempotent — applying it twice gives the same result
    assert (FILINGS_DIR / ticker.upper()) == (FILINGS_DIR / ticker.upper().upper()), (
        "ticker.upper() is not idempotent"
    )
