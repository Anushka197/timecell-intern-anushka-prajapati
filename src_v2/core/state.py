"""
core/state.py — JSON-based state persistence for portfolio and watchlists.
"""

import json
import dataclasses
from pathlib import Path
from typing import Optional

from core.models import (
    Asset,
    Portfolio,
    WatchlistEntry,
    CorrelationWatchlist,
    CompanyWatchlistEntry,
    ReportWatchlist,
)

STATE_DIR = Path("../data/state")

_PORTFOLIO_FILE = STATE_DIR / "portfolio.json"
_CORRELATION_WATCHLIST_FILE = STATE_DIR / "correlation_watchlist.json"
_REPORT_WATCHLIST_FILE = STATE_DIR / "report_watchlist.json"


# ── Helpers ────────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    """Create parent directories for *path* if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ── Portfolio ──────────────────────────────────────────────────────────────

def save_portfolio(portfolio: Portfolio, path: Optional[Path] = None) -> None:
    """Serialize *portfolio* to JSON at *path* (default: portfolio.json)."""
    target = path if path is not None else _PORTFOLIO_FILE
    _ensure_dir(target)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(portfolio), fh, indent=2)


def load_portfolio(path: Optional[Path] = None) -> Portfolio:
    """
    Deserialize Portfolio from JSON at *path* (default: portfolio.json).
    Returns an empty Portfolio() if the file is missing.
    """
    target = path if path is not None else _PORTFOLIO_FILE
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return Portfolio()

    assets = [Asset(**a) for a in data.get("assets", [])]
    return Portfolio(
        assets=assets,
        total_value=data.get("total_value", 0.0),
        monthly_expenses=data.get("monthly_expenses", 0.0),
    )


# ── Correlation Watchlist ──────────────────────────────────────────────────

def save_correlation_watchlist(
    watchlist: CorrelationWatchlist,
    path: Optional[Path] = None,
) -> None:
    """Serialize *watchlist* to JSON at *path* (default: correlation_watchlist.json)."""
    target = path if path is not None else _CORRELATION_WATCHLIST_FILE
    _ensure_dir(target)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(watchlist), fh, indent=2)


def load_correlation_watchlist(
    path: Optional[Path] = None,
) -> CorrelationWatchlist:
    """
    Deserialize CorrelationWatchlist from JSON at *path*.
    Returns an empty CorrelationWatchlist() if the file is missing.
    """
    target = path if path is not None else _CORRELATION_WATCHLIST_FILE
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return CorrelationWatchlist()

    entries = [WatchlistEntry(**e) for e in data.get("entries", [])]
    return CorrelationWatchlist(entries=entries)


# ── Report Watchlist ───────────────────────────────────────────────────────

def save_report_watchlist(
    watchlist: ReportWatchlist,
    path: Optional[Path] = None,
) -> None:
    """Serialize *watchlist* to JSON at *path* (default: report_watchlist.json)."""
    target = path if path is not None else _REPORT_WATCHLIST_FILE
    _ensure_dir(target)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(watchlist), fh, indent=2)


def load_report_watchlist(
    path: Optional[Path] = None,
) -> ReportWatchlist:
    """
    Deserialize ReportWatchlist from JSON at *path*.
    Returns an empty ReportWatchlist() if the file is missing.
    """
    target = path if path is not None else _REPORT_WATCHLIST_FILE
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return ReportWatchlist()

    entries = [CompanyWatchlistEntry(**e) for e in data.get("entries", [])]
    return ReportWatchlist(entries=entries)
