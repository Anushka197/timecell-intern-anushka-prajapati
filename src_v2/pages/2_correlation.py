"""
pages/2_correlation.py — Multi-Asset Correlation Analyzer

Two-column layout (35/65 split):
  Left  (35%): Watchlist Editor
  Right (65%): Fetch Data + Correlation Results

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from core.models import WatchlistEntry, CorrelationWatchlist
from core import state
from core.market_data import fetch_watchlist_history, build_price_dataframe
from core.correlation import (
    compute_correlation_matrix,
    compute_diversification_score,
    find_most_correlated,
    find_least_correlated,
)

st.set_page_config(layout="wide", page_title="Correlation Analyzer")
st.title("📈 Multi-Asset Correlation Analyzer")

# ── Session state initialisation ───────────────────────────────────────────

if "corr_watchlist" not in st.session_state:
    st.session_state.corr_watchlist = state.load_correlation_watchlist()
if "price_df" not in st.session_state:
    st.session_state.price_df = None
if "corr_matrix" not in st.session_state:
    st.session_state.corr_matrix = None

# ── Layout ─────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([0.35, 0.65])

# ══════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Watchlist Editor
# ══════════════════════════════════════════════════════════════════════════

with left_col:
    st.subheader("Watchlist Editor")

    watchlist: CorrelationWatchlist = st.session_state.corr_watchlist

    # ── Current entries table ──────────────────────────────────────────────

    if watchlist.entries:
        to_remove: int | None = None
        for idx, entry in enumerate(watchlist.entries):
            ticker_col, type_col, btn_col = st.columns([0.45, 0.35, 0.20])
            with ticker_col:
                st.markdown(f"**{entry.ticker}**")
            with type_col:
                st.markdown("Crypto" if entry.is_crypto else "Stock")
            with btn_col:
                if st.button("✕", key=f"remove_ticker_{idx}"):
                    to_remove = idx

        if to_remove is not None:
            watchlist.entries.pop(to_remove)
            st.session_state.corr_watchlist = watchlist
            state.save_correlation_watchlist(watchlist)
            st.rerun()
    else:
        st.info("No tickers in watchlist. Add tickers below.")

    st.divider()

    # ── Add Ticker form ────────────────────────────────────────────────────

    st.markdown("**Add Ticker**")

    with st.form("add_ticker_form", clear_on_submit=True):
        new_ticker = st.text_input("Ticker Symbol", placeholder="e.g. AAPL, BTC")
        asset_type = st.selectbox("Asset Type", options=["Stock", "Crypto"])
        submitted = st.form_submit_button("➕ Add")

    if submitted:
        ticker_clean = new_ticker.strip().upper()
        if not ticker_clean:
            st.error("Please enter a ticker symbol.")
        else:
            # Check for duplicates
            existing_tickers = {e.ticker.upper() for e in watchlist.entries}
            if ticker_clean in existing_tickers:
                st.warning(f"'{ticker_clean}' is already in the watchlist.")
            else:
                is_crypto = asset_type == "Crypto"
                new_entry = WatchlistEntry(
                    ticker=ticker_clean,
                    is_crypto=is_crypto,
                    added_at=datetime.now().isoformat(),
                )
                watchlist.entries.append(new_entry)
                st.session_state.corr_watchlist = watchlist
                state.save_correlation_watchlist(watchlist)
                st.success(f"Added {ticker_clean} ({'Crypto' if is_crypto else 'Stock'})")
                st.rerun()

    # ── Minimum ticker count notice ────────────────────────────────────────

    ticker_count = len(watchlist.entries)
    if ticker_count < 2:
        st.info(f"Minimum 2 tickers required. Currently: {ticker_count}.")


# ══════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Fetch + Results
# ══════════════════════════════════════════════════════════════════════════

with right_col:

    # ── Fetch Data button ──────────────────────────────────────────────────

    watchlist_for_fetch: CorrelationWatchlist = st.session_state.corr_watchlist
    tickers = watchlist_for_fetch.tickers()
    crypto_tickers = watchlist_for_fetch.crypto_tickers()

    if st.button("Fetch 365-Day Data ▶", type="primary"):
        if len(tickers) < 2:
            st.error(
                "At least 2 tickers are required to compute a correlation matrix. "
                "Please add more tickers in the Watchlist Editor."
            )
        else:
            with st.spinner("Fetching price history…"):
                histories, failed = fetch_watchlist_history(tickers, crypto_tickers)

            for failed_ticker in failed:
                st.warning(f"⚠️ Failed to fetch data for **{failed_ticker}**. It will be excluded from analysis.")

            if histories:
                price_df = build_price_dataframe(histories)
                if price_df.empty or price_df.shape[1] < 2:
                    st.error(
                        "Not enough valid price data to compute correlations. "
                        "Ensure at least 2 tickers have overlapping price history."
                    )
                    st.session_state.price_df = None
                    st.session_state.corr_matrix = None
                else:
                    st.session_state.price_df = price_df
                    st.session_state.corr_matrix = compute_correlation_matrix(price_df)
                    st.success(
                        f"Fetched data for {price_df.shape[1]} tickers "
                        f"({price_df.shape[0]} trading days)."
                    )
            else:
                st.error("All tickers failed to fetch. Please check your watchlist.")
                st.session_state.price_df = None
                st.session_state.corr_matrix = None

    # ── Correlation Analysis ───────────────────────────────────────────────

    price_df: pd.DataFrame | None = st.session_state.price_df
    corr_matrix: pd.DataFrame | None = st.session_state.corr_matrix

    if price_df is None or corr_matrix is None:
        st.info(
            "Add at least 2 tickers to the watchlist and click **Fetch 365-Day Data ▶** "
            "to compute the correlation matrix."
        )
    else:
        # ── Diversification Score ──────────────────────────────────────────

        st.subheader("Diversification Score")

        div_score = compute_diversification_score(corr_matrix)
        score_col, label_col = st.columns([0.3, 0.7])
        with score_col:
            st.metric(label="Score", value=f"{div_score:.3f}")
        with label_col:
            st.markdown("*Lower is better* — a score closer to 0 means assets are less correlated and the portfolio is more diversified.")

        st.divider()

        # ── Hidden Concentration Risk Warning ─────────────────────────────

        tickers_in_matrix = corr_matrix.columns.tolist()
        n = len(tickers_in_matrix)
        high_corr_pairs: list[tuple[str, str, float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append(
                        (tickers_in_matrix[i], tickers_in_matrix[j], corr_val)
                    )

        if high_corr_pairs:
            with st.expander("⚠️ Hidden Concentration Risk Detected", expanded=True):
                st.markdown(
                    "The following asset pairs have high absolute correlation (|corr| > 0.70), "
                    "indicating potential concentration risk:"
                )
                for t1, t2, corr_val in high_corr_pairs:
                    direction = "positively" if corr_val > 0 else "negatively"
                    st.markdown(
                        f"- **{t1} ↔ {t2}**: {corr_val:.3f} ({direction} correlated)"
                    )

        # ── Most / Least Correlated ────────────────────────────────────────

        if n >= 2:
            most_corr = find_most_correlated(corr_matrix)
            least_corr = find_least_correlated(corr_matrix)

            mc_col, lc_col = st.columns(2)
            with mc_col:
                st.metric(
                    label="Most Correlated Asset",
                    value=most_corr,
                    help="Asset with the highest mean absolute correlation to all others.",
                )
            with lc_col:
                st.metric(
                    label="Least Correlated Asset",
                    value=least_corr,
                    help="Asset with the lowest mean absolute correlation to all others — best diversifier.",
                )

        st.divider()

        # ── Correlation Heatmap ────────────────────────────────────────────

        st.subheader("Correlation Matrix Heatmap")

        z_values = corr_matrix.values.tolist()
        ticker_labels = corr_matrix.columns.tolist()

        # Build annotation text (rounded to 2 decimal places)
        annotation_text = [
            [f"{corr_matrix.iloc[i, j]:.2f}" for j in range(n)]
            for i in range(n)
        ]

        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=z_values,
                x=ticker_labels,
                y=ticker_labels,
                colorscale="RdBu",
                zmid=0,
                zmin=-1.0,
                zmax=1.0,
                text=annotation_text,
                texttemplate="%{text}",
                hovertemplate=(
                    "<b>%{y} ↔ %{x}</b><br>"
                    "Correlation: %{z:.4f}<br>"
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title="Pearson r",
                    tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"],
                ),
            )
        )

        cell_size = max(60, min(120, 600 // n))
        fig_size = max(400, cell_size * n + 100)

        heatmap_fig.update_layout(
            xaxis_title="Ticker",
            yaxis_title="Ticker",
            height=fig_size,
            margin=dict(t=40, b=80, l=80, r=40),
        )

        st.plotly_chart(heatmap_fig, use_container_width=True)

        # ── Raw Correlation Table ──────────────────────────────────────────

        with st.expander("📋 Raw Correlation Values", expanded=False):
            st.dataframe(
                corr_matrix.style.format("{:.4f}").background_gradient(
                    cmap="RdBu_r", vmin=-1.0, vmax=1.0
                ),
                use_container_width=True,
            )
