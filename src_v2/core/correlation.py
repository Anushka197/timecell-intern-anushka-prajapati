"""
Correlation computation helpers for the Multi-Asset Correlation Analyzer.

Provides Pearson correlation matrix computation, diversification scoring,
and most/least correlated asset identification.
"""

import pandas as pd
import numpy as np


def compute_correlation_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes full pairwise Pearson correlation matrix.

    Args:
        price_df: DataFrame with tickers as columns and dates as index.

    Returns:
        DataFrame with tickers as both index and columns, values in [-1, 1].
    """
    return price_df.corr(method="pearson")


def compute_diversification_score(corr_matrix: pd.DataFrame) -> float:
    """
    Computes the mean of all unique pairwise absolute correlation coefficients
    (upper triangle, excluding diagonal).

    A lower score indicates better diversification (assets are less correlated).

    Args:
        corr_matrix: Square correlation matrix DataFrame.

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 for a 1×1 matrix (no pairs).
    """
    values = corr_matrix.values
    upper_triangle = np.triu(values, k=1)

    # Extract only the non-zero upper triangle elements
    # (k=1 excludes diagonal; off-diagonal zeros in a real corr matrix are valid,
    # so we need to identify the actual upper-triangle positions)
    n = values.shape[0]
    if n < 2:
        return 0.0

    # Get indices of upper triangle (excluding diagonal)
    rows, cols = np.triu_indices(n, k=1)
    upper_values = values[rows, cols]

    if len(upper_values) == 0:
        return 0.0

    return float(np.mean(np.abs(upper_values)))


def find_most_correlated(corr_matrix: pd.DataFrame) -> str:
    """
    Returns the ticker name with the highest mean absolute off-diagonal correlation
    with all other assets.

    Args:
        corr_matrix: Square correlation matrix DataFrame with ticker labels.

    Returns:
        Ticker name (str) with the highest mean absolute off-diagonal correlation.
    """
    tickers = corr_matrix.columns.tolist()
    n = len(tickers)
    values = corr_matrix.values

    mean_abs_corr = {}
    for i, ticker in enumerate(tickers):
        # All off-diagonal elements for this ticker
        off_diag = [abs(values[i, j]) for j in range(n) if j != i]
        mean_abs_corr[ticker] = np.mean(off_diag) if off_diag else 0.0

    return max(mean_abs_corr, key=mean_abs_corr.get)


def find_least_correlated(corr_matrix: pd.DataFrame) -> str:
    """
    Returns the ticker name with the lowest mean absolute off-diagonal correlation
    with all other assets.

    Args:
        corr_matrix: Square correlation matrix DataFrame with ticker labels.

    Returns:
        Ticker name (str) with the lowest mean absolute off-diagonal correlation.
    """
    tickers = corr_matrix.columns.tolist()
    n = len(tickers)
    values = corr_matrix.values

    mean_abs_corr = {}
    for i, ticker in enumerate(tickers):
        # All off-diagonal elements for this ticker
        off_diag = [abs(values[i, j]) for j in range(n) if j != i]
        mean_abs_corr[ticker] = np.mean(off_diag) if off_diag else 0.0

    return min(mean_abs_corr, key=mean_abs_corr.get)
