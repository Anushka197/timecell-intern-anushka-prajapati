"""
Tests for src_v2/core/correlation.py

Covers:
- Unit tests for compute_diversification_score, find_most_correlated,
  find_least_correlated, compute_correlation_matrix
- Property-based tests for correlation matrix diagonal/symmetry (Property 3)
  and diversification score range invariant (Property 4)
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src_v2.core.correlation import (
    compute_correlation_matrix,
    compute_diversification_score,
    find_least_correlated,
    find_most_correlated,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_corr_matrix(data: dict) -> pd.DataFrame:
    """Build a correlation-like DataFrame from a dict of lists."""
    return pd.DataFrame(data, index=list(data.keys()))


def random_price_df(n_tickers: int, n_days: int, seed: int = 42) -> pd.DataFrame:
    """Generate a DataFrame of random lognormal prices."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.01, size=(n_days, n_tickers))
    prices = np.exp(np.cumsum(log_returns, axis=0))
    tickers = [f"T{i}" for i in range(n_tickers)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ── Unit Tests ─────────────────────────────────────────────────────────────────

class TestComputeDiversificationScore:
    def test_zero_for_uncorrelated(self):
        """
        A 2×2 identity-like correlation matrix (diagonal=1, off-diagonal=0)
        should yield a diversification score of 0.0.
        """
        corr = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        score = compute_diversification_score(corr)
        assert score == 0.0

    def test_one_for_perfectly_correlated(self):
        """
        A 2×2 matrix where off-diagonal = 1.0 should yield score = 1.0.
        """
        corr = pd.DataFrame(
            [[1.0, 1.0], [1.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        score = compute_diversification_score(corr)
        assert score == pytest.approx(1.0)

    def test_single_asset_returns_zero(self):
        """A 1×1 matrix has no pairs, so score should be 0.0."""
        corr = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
        score = compute_diversification_score(corr)
        assert score == 0.0

    def test_score_between_zero_and_one(self):
        """Score for a typical correlation matrix should be in [0, 1]."""
        price_df = random_price_df(n_tickers=4, n_days=100)
        corr = compute_correlation_matrix(price_df)
        score = compute_diversification_score(corr)
        assert 0.0 <= score <= 1.0

    def test_partial_correlation(self):
        """
        A 3×3 matrix with known off-diagonal values should yield the correct mean.
        Off-diagonal absolute values: 0.5, 0.3, 0.8 → mean = (0.5 + 0.3 + 0.8) / 3
        """
        corr = pd.DataFrame(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.8],
                [0.3, 0.8, 1.0],
            ],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        expected = (0.5 + 0.3 + 0.8) / 3
        score = compute_diversification_score(corr)
        assert score == pytest.approx(expected)


class TestFindMostCorrelated:
    def test_returns_highest_mean_abs_corr(self):
        """
        In a 3×3 matrix where 'B' has the highest mean absolute off-diagonal
        correlation, find_most_correlated should return 'B'.

        A: mean(|0.2|, |0.3|) = 0.25
        B: mean(|0.2|, |0.9|) = 0.55  ← highest
        C: mean(|0.3|, |0.9|) = 0.60  ← actually highest here
        """
        corr = pd.DataFrame(
            [
                [1.0, 0.2, 0.3],
                [0.2, 1.0, 0.9],
                [0.3, 0.9, 1.0],
            ],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        # A: mean(0.2, 0.3) = 0.25
        # B: mean(0.2, 0.9) = 0.55
        # C: mean(0.3, 0.9) = 0.60  ← highest
        result = find_most_correlated(corr)
        assert result == "C"

    def test_clear_winner(self):
        """
        Construct a matrix where 'HIGH' clearly has the highest mean abs corr.
        """
        corr = pd.DataFrame(
            [
                [1.0, 0.9, 0.8],
                [0.9, 1.0, 0.85],
                [0.8, 0.85, 1.0],
            ],
            index=["HIGH", "MED", "LOW"],
            columns=["HIGH", "MED", "LOW"],
        )
        # HIGH: mean(0.9, 0.8) = 0.85
        # MED:  mean(0.9, 0.85) = 0.875  ← highest
        # LOW:  mean(0.8, 0.85) = 0.825
        result = find_most_correlated(corr)
        assert result == "MED"

    def test_explicit_highest(self):
        """
        Construct a matrix where 'X' has clearly the highest mean abs corr.
        X: mean(0.95, 0.90) = 0.925
        Y: mean(0.95, 0.10) = 0.525
        Z: mean(0.90, 0.10) = 0.500
        """
        corr = pd.DataFrame(
            [
                [1.0, 0.95, 0.90],
                [0.95, 1.0, 0.10],
                [0.90, 0.10, 1.0],
            ],
            index=["X", "Y", "Z"],
            columns=["X", "Y", "Z"],
        )
        result = find_most_correlated(corr)
        assert result == "X"


class TestFindLeastCorrelated:
    def test_returns_lowest_mean_abs_corr(self):
        """
        In a 3×3 matrix, find_least_correlated should return the ticker
        with the lowest mean absolute off-diagonal correlation.
        """
        corr = pd.DataFrame(
            [
                [1.0, 0.95, 0.90],
                [0.95, 1.0, 0.10],
                [0.90, 0.10, 1.0],
            ],
            index=["X", "Y", "Z"],
            columns=["X", "Y", "Z"],
        )
        # X: mean(0.95, 0.90) = 0.925
        # Y: mean(0.95, 0.10) = 0.525
        # Z: mean(0.90, 0.10) = 0.500  ← lowest
        result = find_least_correlated(corr)
        assert result == "Z"

    def test_uncorrelated_asset(self):
        """
        An asset with near-zero correlations to all others should be least correlated.
        """
        corr = pd.DataFrame(
            [
                [1.0, 0.8, 0.01],
                [0.8, 1.0, 0.02],
                [0.01, 0.02, 1.0],
            ],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        # A: mean(0.8, 0.01) = 0.405
        # B: mean(0.8, 0.02) = 0.41
        # C: mean(0.01, 0.02) = 0.015  ← lowest
        result = find_least_correlated(corr)
        assert result == "C"


class TestComputeCorrelationMatrix:
    def test_diagonal_is_one(self):
        """All diagonal values of a computed correlation matrix should be 1.0."""
        price_df = random_price_df(n_tickers=4, n_days=100)
        corr = compute_correlation_matrix(price_df)
        diagonal = np.diag(corr.values)
        np.testing.assert_allclose(diagonal, 1.0, atol=1e-10)

    def test_is_symmetric(self):
        """The correlation matrix should be symmetric: corr[i,j] == corr[j,i]."""
        price_df = random_price_df(n_tickers=5, n_days=120)
        corr = compute_correlation_matrix(price_df)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_shape_matches_tickers(self):
        """The correlation matrix shape should be (n_tickers, n_tickers)."""
        n = 6
        price_df = random_price_df(n_tickers=n, n_days=90)
        corr = compute_correlation_matrix(price_df)
        assert corr.shape == (n, n)

    def test_columns_and_index_match_tickers(self):
        """Columns and index of the correlation matrix should match the price_df columns."""
        price_df = random_price_df(n_tickers=3, n_days=60)
        corr = compute_correlation_matrix(price_df)
        assert list(corr.columns) == list(price_df.columns)
        assert list(corr.index) == list(price_df.columns)

    def test_values_in_range(self):
        """All correlation values should be in [-1, 1]."""
        price_df = random_price_df(n_tickers=5, n_days=200)
        corr = compute_correlation_matrix(price_df)
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()


# ── Property-Based Tests ───────────────────────────────────────────────────────

@settings(max_examples=100)
@given(
    n_tickers=st.integers(min_value=2, max_value=10),
    n_days=st.integers(min_value=30, max_value=365),
)
def test_correlation_matrix_diagonal_and_symmetry(n_tickers: int, n_days: int):
    """
    **Property 3: Correlation Matrix Diagonal and Symmetry**
    **Validates: Requirements 5.1, 5.7**

    For any random lognormal price data:
    - Shape is (n_tickers, n_tickers)
    - All diagonal values equal 1.0 within 1e-10
    - Matrix is symmetric
    """
    price_df = random_price_df(n_tickers=n_tickers, n_days=n_days)
    corr = compute_correlation_matrix(price_df)

    # Shape check
    assert corr.shape == (n_tickers, n_tickers)

    # Diagonal check
    diagonal = np.diag(corr.values)
    np.testing.assert_allclose(diagonal, 1.0, atol=1e-10)

    # Symmetry check
    np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)


@settings(max_examples=100)
@given(
    n_tickers=st.integers(min_value=2, max_value=10),
    n_days=st.integers(min_value=30, max_value=365),
)
def test_diversification_score_range(n_tickers: int, n_days: int):
    """
    **Property 4: Diversification Score Range Invariant**
    **Validates: Requirements 5.3, 5.4**

    For any valid correlation matrix computed from random price data,
    the diversification score must be in [0.0, 1.0].
    """
    price_df = random_price_df(n_tickers=n_tickers, n_days=n_days)
    corr = compute_correlation_matrix(price_df)
    score = compute_diversification_score(corr)

    assert 0.0 <= score <= 1.0
