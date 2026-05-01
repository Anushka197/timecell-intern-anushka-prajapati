# core/models.py — shared data models used across all modules

from dataclasses import dataclass, field
from typing import Optional


# ── Portfolio ──────────────────────────────────────────────────────────────

@dataclass
class Asset:
    name: str
    allocation_pct: float        # 0.0 – 100.0
    crash_pct: float = 0.0       # -100.0 to 0.0 (user-defined crash for custom scenarios)


@dataclass
class Portfolio:
    assets: list[Asset] = field(default_factory=list)
    total_value: float = 0.0
    monthly_expenses: float = 0.0

    def total_allocation(self) -> float:
        return sum(a.allocation_pct for a in self.assets)

    def is_valid(self) -> bool:
        return abs(self.total_allocation() - 100.0) <= 1.0


# ── Scenarios ─────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    asset_crashes: dict[str, float]  # asset_name -> crash_pct (-100 to 0)
    is_preset: bool = False


@dataclass
class ScenarioResult:
    scenario_name: str
    post_crash_value: float
    runway_months: float
    ruin_test: str               # "PASS" | "FAIL"
    largest_risk_asset: str
    concentration_warning: bool
    asset_losses: dict[str, float]  # asset_name -> loss_pct applied


# ── Correlation Watchlist ─────────────────────────────────────────────────

@dataclass
class WatchlistEntry:
    ticker: str                  # e.g. "AAPL", "BTC"
    is_crypto: bool              # True = use CoinGecko, False = use yfinance
    added_at: str                # ISO 8601 datetime string


@dataclass
class CorrelationWatchlist:
    entries: list[WatchlistEntry] = field(default_factory=list)

    def tickers(self) -> list[str]:
        return [e.ticker for e in self.entries]

    def crypto_tickers(self) -> set[str]:
        return {e.ticker for e in self.entries if e.is_crypto}


# ── Rebalancing ───────────────────────────────────────────────────────────

@dataclass
class RebalancingPlan:
    goal: str
    tone: str                    # "beginner" | "experienced" | "expert"
    new_allocations: dict[str, float]  # asset_name -> new allocation_pct
    rationale: str               # LLM explanation
    critic_review: str           # Second LLM call output

    def allocations_sum(self) -> float:
        return sum(self.new_allocations.values())

    def is_valid(self) -> bool:
        return abs(self.allocations_sum() - 100.0) <= 1.0


# ── Report Monitor ────────────────────────────────────────────────────────

@dataclass
class CompanyWatchlistEntry:
    ticker: str                  # e.g. "AAPL"
    company_name: str            # e.g. "Apple Inc."
    added_at: str                # ISO 8601 datetime string
    last_indexed_date: Optional[str] = None  # ISO date of most recent indexed filing


@dataclass
class ReportWatchlist:
    entries: list[CompanyWatchlistEntry] = field(default_factory=list)

    def tickers(self) -> list[str]:
        return [e.ticker for e in self.entries]

    def has_ticker(self, ticker: str) -> bool:
        return ticker.upper() in {e.ticker.upper() for e in self.entries}


@dataclass
class FilingRecord:
    ticker: str
    company_name: str
    filing_type: str             # "10-K" | "10-Q"
    filing_date: str             # ISO date string
    accession_number: str
    local_path: str              # relative path under data/filings/
    ingested: bool = False
    download_status: str = "Pending"  # "Downloaded" | "Already Exists" | "Failed"
