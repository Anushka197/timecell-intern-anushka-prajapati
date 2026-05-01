"""
Tests for core/market_data.py — market data fetcher.

Task 7.1: Unit and integration tests for market data fetcher
  Validates: Requirements 4.2, 4.3, 4.4, 4.5
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src_v2.core.market_data import (
    CRYPTO_ID_MAP,
    PriceHistory,
    build_price_dataframe,
    fetch_crypto_history,
    fetch_stock_history,
    fetch_watchlist_history,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_price_series(dates: list[str], prices: list[float]) -> pd.Series:
    """Build a pd.Series with a DatetimeIndex from string dates."""
    index = pd.to_datetime(dates)
    return pd.Series(prices, index=index, dtype=float)


def _make_coingecko_response(prices: list[float], start_ms: int = 1_700_000_000_000) -> dict:
    """Build a minimal CoinGecko market_chart response."""
    step_ms = 86_400_000  # 1 day in ms
    return {
        "prices": [[start_ms + i * step_ms, p] for i, p in enumerate(prices)]
    }


# ── test_fetch_stock_history_calls_yfinance ────────────────────────────────

def test_fetch_stock_history_calls_yfinance() -> None:
    """
    fetch_stock_history must call yf.Ticker with the correct ticker symbol
    and use period='1y' when calling .history().
    """
    mock_close = _make_price_series(["2024-01-01", "2024-01-02"], [150.0, 152.0])
    mock_hist_df = pd.DataFrame({"Close": mock_close})

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = mock_hist_df

    with patch("src_v2.core.market_data.yf.Ticker", return_value=mock_ticker_instance) as mock_ticker_cls:
        result = fetch_stock_history("AAPL")

    # Verify Ticker was constructed with the correct symbol
    mock_ticker_cls.assert_called_once_with("AAPL")
    # Verify history was called with period="1y"
    mock_ticker_instance.history.assert_called_once_with(period="1y")

    assert result.ticker == "AAPL"
    assert result.source == "yfinance"
    assert result.error is None
    assert len(result.prices) == 2


# ── test_fetch_stock_history_returns_error_on_failure ─────────────────────

def test_fetch_stock_history_returns_error_on_failure() -> None:
    """
    When yfinance raises an exception, fetch_stock_history must return a
    PriceHistory with error set and an empty prices Series.
    """
    with patch("src_v2.core.market_data.yf.Ticker", side_effect=RuntimeError("network error")):
        result = fetch_stock_history("BADTICKER")

    assert result.ticker == "BADTICKER"
    assert result.source == "yfinance"
    assert result.error is not None
    assert "network error" in result.error
    assert len(result.prices) == 0
    assert result.prices.dtype == float


def test_fetch_stock_history_returns_error_on_history_failure() -> None:
    """
    When .history() raises an exception, fetch_stock_history must return
    a PriceHistory with error set.
    """
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.side_effect = ValueError("no data")

    with patch("src_v2.core.market_data.yf.Ticker", return_value=mock_ticker_instance):
        result = fetch_stock_history("XYZ")

    assert result.error is not None
    assert "no data" in result.error
    assert len(result.prices) == 0


# ── test_fetch_crypto_history_uses_coin_id_map ────────────────────────────

def test_fetch_crypto_history_uses_coin_id_map() -> None:
    """
    fetch_crypto_history must look up the coin_id from CRYPTO_ID_MAP.
    For 'BTC', the URL must contain 'bitcoin'.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _make_coingecko_response([30000.0, 31000.0])
    mock_response.raise_for_status = MagicMock()

    with patch("src_v2.core.market_data.requests.get", return_value=mock_response) as mock_get:
        result = fetch_crypto_history("BTC")

    # Verify the URL contains the mapped coin_id "bitcoin"
    call_args = mock_get.call_args
    url = call_args[0][0]
    assert "bitcoin" in url, f"Expected 'bitcoin' in URL, got: {url}"

    assert result.ticker == "BTC"
    assert result.source == "coingecko"
    assert result.error is None
    assert len(result.prices) == 2


def test_fetch_crypto_history_eth_uses_ethereum() -> None:
    """
    fetch_crypto_history for 'ETH' must use 'ethereum' in the CoinGecko URL.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _make_coingecko_response([2000.0, 2100.0])
    mock_response.raise_for_status = MagicMock()

    with patch("src_v2.core.market_data.requests.get", return_value=mock_response) as mock_get:
        fetch_crypto_history("ETH")

    url = mock_get.call_args[0][0]
    assert "ethereum" in url


# ── test_fetch_crypto_history_falls_back_to_lowercase ─────────────────────

def test_fetch_crypto_history_falls_back_to_lowercase() -> None:
    """
    For an unknown ticker 'XYZ' not in CRYPTO_ID_MAP, the URL must use
    the lowercase ticker 'xyz' as the coin_id.
    """
    assert "XYZ" not in CRYPTO_ID_MAP, "XYZ should not be in CRYPTO_ID_MAP for this test"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _make_coingecko_response([1.0, 1.1])
    mock_response.raise_for_status = MagicMock()

    with patch("src_v2.core.market_data.requests.get", return_value=mock_response) as mock_get:
        result = fetch_crypto_history("XYZ")

    url = mock_get.call_args[0][0]
    assert "xyz" in url, f"Expected 'xyz' in URL, got: {url}"
    assert result.ticker == "XYZ"


def test_fetch_crypto_history_uppercase_ticker_maps_correctly() -> None:
    """
    fetch_crypto_history must handle uppercase tickers correctly via CRYPTO_ID_MAP.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _make_coingecko_response([100.0])
    mock_response.raise_for_status = MagicMock()

    with patch("src_v2.core.market_data.requests.get", return_value=mock_response) as mock_get:
        fetch_crypto_history("sol")  # lowercase input

    url = mock_get.call_args[0][0]
    assert "solana" in url


# ── test_fetch_watchlist_history_separates_failed_tickers ─────────────────

def test_fetch_watchlist_history_separates_failed_tickers() -> None:
    """
    fetch_watchlist_history must:
    - Exclude failed tickers from the results dict
    - Include failed tickers in the failed_tickers list
    """
    good_prices = _make_price_series(["2024-01-01", "2024-01-02"], [150.0, 152.0])
    good_history = PriceHistory(ticker="AAPL", prices=good_prices, source="yfinance", error=None)
    bad_history = PriceHistory(
        ticker="BADTICKER",
        prices=pd.Series(dtype=float),
        source="yfinance",
        error="fetch failed",
    )

    def mock_fetch_stock(ticker: str) -> PriceHistory:
        if ticker == "AAPL":
            return good_history
        return bad_history

    with patch("src_v2.core.market_data.fetch_stock_history", side_effect=mock_fetch_stock):
        results, failed = fetch_watchlist_history(
            tickers=["AAPL", "BADTICKER"],
            crypto_tickers=set(),
        )

    assert "AAPL" in results
    assert "BADTICKER" not in results
    assert "BADTICKER" in failed
    assert "AAPL" not in failed


def test_fetch_watchlist_history_routes_crypto_to_coingecko() -> None:
    """
    Tickers in crypto_tickers must be fetched via fetch_crypto_history,
    not fetch_stock_history.
    """
    btc_prices = _make_price_series(["2024-01-01"], [40000.0])
    btc_history = PriceHistory(ticker="BTC", prices=btc_prices, source="coingecko", error=None)

    with patch("src_v2.core.market_data.fetch_crypto_history", return_value=btc_history) as mock_crypto, \
         patch("src_v2.core.market_data.fetch_stock_history") as mock_stock:
        results, failed = fetch_watchlist_history(
            tickers=["BTC"],
            crypto_tickers={"BTC"},
        )

    mock_crypto.assert_called_once_with("BTC")
    mock_stock.assert_not_called()
    assert "BTC" in results
    assert len(failed) == 0


def test_fetch_watchlist_history_all_failed() -> None:
    """
    When all tickers fail, results dict must be empty and all tickers in failed list.
    """
    bad_history = PriceHistory(
        ticker="X",
        prices=pd.Series(dtype=float),
        source="yfinance",
        error="error",
    )

    with patch("src_v2.core.market_data.fetch_stock_history", return_value=bad_history):
        results, failed = fetch_watchlist_history(
            tickers=["A", "B", "C"],
            crypto_tickers=set(),
        )

    assert results == {}
    assert set(failed) == {"A", "B", "C"}


# ── test_build_price_dataframe_inner_join ─────────────────────────────────

def test_build_price_dataframe_inner_join() -> None:
    """
    build_price_dataframe must perform an inner join on the DatetimeIndex.
    Only dates present in ALL series must appear in the result.
    """
    # AAPL: 3 dates
    aapl_prices = _make_price_series(
        ["2024-01-01", "2024-01-02", "2024-01-03"],
        [150.0, 152.0, 153.0],
    )
    # MSFT: 2 dates (subset)
    msft_prices = _make_price_series(
        ["2024-01-02", "2024-01-03"],
        [300.0, 302.0],
    )

    histories = {
        "AAPL": PriceHistory(ticker="AAPL", prices=aapl_prices, source="yfinance", error=None),
        "MSFT": PriceHistory(ticker="MSFT", prices=msft_prices, source="yfinance", error=None),
    }

    df = build_price_dataframe(histories)

    # Inner join: only 2024-01-02 and 2024-01-03 are in both
    assert len(df) == 2
    assert "AAPL" in df.columns
    assert "MSFT" in df.columns
    assert pd.Timestamp("2024-01-01") not in df.index
    assert pd.Timestamp("2024-01-02") in df.index
    assert pd.Timestamp("2024-01-03") in df.index


def test_build_price_dataframe_no_overlap_returns_empty() -> None:
    """
    When two series have no overlapping dates, the result must be an empty DataFrame.
    """
    s1 = _make_price_series(["2024-01-01", "2024-01-02"], [100.0, 101.0])
    s2 = _make_price_series(["2024-02-01", "2024-02-02"], [200.0, 201.0])

    histories = {
        "A": PriceHistory(ticker="A", prices=s1, source="yfinance", error=None),
        "B": PriceHistory(ticker="B", prices=s2, source="yfinance", error=None),
    }

    df = build_price_dataframe(histories)
    assert len(df) == 0


def test_build_price_dataframe_single_ticker() -> None:
    """
    build_price_dataframe with a single ticker must return all rows (no NaN to drop).
    """
    prices = _make_price_series(["2024-01-01", "2024-01-02", "2024-01-03"], [10.0, 11.0, 12.0])
    histories = {
        "X": PriceHistory(ticker="X", prices=prices, source="yfinance", error=None),
    }

    df = build_price_dataframe(histories)
    assert len(df) == 3
    assert list(df.columns) == ["X"]


def test_build_price_dataframe_empty_histories() -> None:
    """
    build_price_dataframe with an empty dict must return an empty DataFrame.
    """
    df = build_price_dataframe({})
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_build_price_dataframe_columns_are_tickers() -> None:
    """
    Column names in the resulting DataFrame must match the ticker keys.
    """
    s1 = _make_price_series(["2024-01-01"], [100.0])
    s2 = _make_price_series(["2024-01-01"], [200.0])

    histories = {
        "AAPL": PriceHistory(ticker="AAPL", prices=s1, source="yfinance", error=None),
        "BTC": PriceHistory(ticker="BTC", prices=s2, source="coingecko", error=None),
    }

    df = build_price_dataframe(histories)
    assert set(df.columns) == {"AAPL", "BTC"}


# ── Smoke test: no external LLM imports ───────────────────────────────────

def test_no_external_llm_imports() -> None:
    """
    No .py file under src_v2/ must import openai, google.genai, anthropic,
    or openrouter.
    Validates: Requirement 12.3
    """
    import ast
    from pathlib import Path

    forbidden = {"openai", "google.genai", "anthropic", "openrouter"}
    src_root = Path("src_v2")

    violations: list[str] = []
    for py_file in src_root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden:
                        violations.append(f"{py_file}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden or any(module.startswith(f"{f}.") for f in forbidden):
                    violations.append(f"{py_file}: from {module} import ...")

    assert violations == [], f"Forbidden LLM imports found:\n" + "\n".join(violations)


# ── Compliance: ChromaDB path ──────────────────────────────────────────────

def test_chromadb_path() -> None:
    """
    CHROMA_DIR in ingest.py must equal Path("src_v2/data/chromadb").
    Validates: Requirement 12.2
    """
    from pathlib import Path
    from src_v2.core.rag.ingest import CHROMA_DIR
    assert CHROMA_DIR == Path("src_v2/data/chromadb"), (
        f"Expected CHROMA_DIR == Path('src_v2/data/chromadb'), got {CHROMA_DIR!r}"
    )


# ── Compliance: Ollama model names ────────────────────────────────────────

def test_ollama_model_names() -> None:
    """
    CHAT_MODEL must be 'llama3.1:8b' and EMBED_MODEL must be 'nomic-embed-text'.
    Validates: Requirement 12.1, 12.2
    """
    from src_v2.core.llm import CHAT_MODEL, EMBED_MODEL
    assert CHAT_MODEL == "llama3.1:8b", (
        f"Expected CHAT_MODEL == 'llama3.1:8b', got {CHAT_MODEL!r}"
    )
    assert EMBED_MODEL == "nomic-embed-text", (
        f"Expected EMBED_MODEL == 'nomic-embed-text', got {EMBED_MODEL!r}"
    )
