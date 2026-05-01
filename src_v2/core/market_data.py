"""
core/market_data.py — Market Data Fetcher

Fetches 365-day daily price history for stocks (via yfinance) and
cryptocurrencies (via CoinGecko REST API).

Implements: Requirements 4.2, 4.3, 4.4, 4.5
"""

import time

import pandas as pd
import requests
import yfinance as yf
from dataclasses import dataclass
from typing import Optional

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
HISTORY_DAYS = 365

# Mapping from user-entered crypto ticker to CoinGecko coin_id
CRYPTO_ID_MAP: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "ADA": "cardano",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "MATIC": "matic-network",
}


@dataclass
class PriceHistory:
    ticker: str
    prices: pd.Series          # DatetimeIndex -> float (daily close)
    source: str                # "yfinance" or "coingecko"
    error: Optional[str]       # None if successful


def fetch_stock_history(ticker: str) -> PriceHistory:
    """
    Fetches 365 days of daily closing prices via yfinance.

    Returns PriceHistory with error set if fetch fails.
    """
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        close_series: pd.Series = hist["Close"]
        return PriceHistory(
            ticker=ticker,
            prices=close_series,
            source="yfinance",
            error=None,
        )
    except Exception as e:
        return PriceHistory(
            ticker=ticker,
            prices=pd.Series(dtype=float),
            source="yfinance",
            error=str(e),
        )


def fetch_crypto_history(ticker: str) -> PriceHistory:
    """
    Fetches 365 days of daily closing prices via CoinGecko /market_chart endpoint.

    Looks up coin_id from CRYPTO_ID_MAP; falls back to lowercase ticker if not found.
    Implements exponential backoff retry (3 attempts: 2s, 4s, 8s) on HTTP 429.
    Returns PriceHistory with error set if fetch fails.
    """
    coin_id = CRYPTO_ID_MAP.get(ticker.upper(), ticker.lower())
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": 365, "interval": "daily"}

    delays = [2, 4, 8]
    last_error: Optional[str] = None

    for attempt, delay in enumerate(delays):
        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 429:
                last_error = f"HTTP 429 Too Many Requests (attempt {attempt + 1})"
                time.sleep(delay)
                continue

            response.raise_for_status()
            data = response.json()

            raw_prices = data["prices"]  # list of [timestamp_ms, price]
            timestamps = [entry[0] for entry in raw_prices]
            prices = [entry[1] for entry in raw_prices]

            index = pd.to_datetime(timestamps, unit="ms")
            price_series = pd.Series(prices, index=index, dtype=float)

            return PriceHistory(
                ticker=ticker,
                prices=price_series,
                source="coingecko",
                error=None,
            )

        except Exception as e:
            last_error = str(e)
            # Only retry on 429; for other errors, break immediately
            if "429" not in str(e):
                break

    return PriceHistory(
        ticker=ticker,
        prices=pd.Series(dtype=float),
        source="coingecko",
        error=last_error,
    )


def fetch_watchlist_history(
    tickers: list[str],
    crypto_tickers: set[str],
) -> tuple[dict[str, PriceHistory], list[str]]:
    """
    Fetches history for all tickers. crypto_tickers identifies which use CoinGecko.

    Returns (results_dict, failed_tickers).
    Failed tickers are excluded from results_dict.
    """
    results: dict[str, PriceHistory] = {}
    failed_tickers: list[str] = []

    for ticker in tickers:
        if ticker in crypto_tickers:
            result = fetch_crypto_history(ticker)
        else:
            result = fetch_stock_history(ticker)

        if result.error is not None:
            failed_tickers.append(ticker)
        else:
            results[ticker] = result

    return results, failed_tickers


def build_price_dataframe(histories: dict[str, PriceHistory]) -> pd.DataFrame:
    """
    Aligns all price series on a common DatetimeIndex (inner join).

    Returns DataFrame with tickers as columns, dates as index.
    Rows where any ticker has NaN are dropped.
    """
    if not histories:
        return pd.DataFrame()

    series_dict = {ticker: history.prices for ticker, history in histories.items()}
    df = pd.DataFrame(series_dict)

    # Inner join: drop rows where any ticker has NaN
    df = df.dropna()

    return df
