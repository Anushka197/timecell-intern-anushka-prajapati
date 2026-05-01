"""
pages/1_stress_test.py — Portfolio Stress-Testing Dashboard

Two-column layout (40/60 split):
  Left  (40%): Portfolio Editor + Scenario Builder
  Right (60%): Results (bar chart, heatmap, results table)

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.models import Asset, Portfolio, Scenario
from core import state
from core.risk import compute_all_scenarios, build_loss_matrix, validate_crash_pct
from core.scenarios import get_preset_names, get_preset

st.set_page_config(layout="wide", page_title="Stress-Testing Dashboard")
st.title("📊 Portfolio Stress-Testing Dashboard")

# ── Session state initialisation ───────────────────────────────────────────

if "portfolio" not in st.session_state:
    st.session_state.portfolio = state.load_portfolio()
if "active_scenarios" not in st.session_state:
    st.session_state.active_scenarios = []
if "stress_results" not in st.session_state:
    st.session_state.stress_results = []

# ── Layout ─────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([0.4, 0.6])

# ══════════════════════════════════════════════════════════════════════════
# LEFT COLUMN
# ══════════════════════════════════════════════════════════════════════════

with left_col:

    # ── Portfolio Editor ───────────────────────────────────────────────────

    st.subheader("Portfolio Editor")

    portfolio: Portfolio = st.session_state.portfolio

    # Build a DataFrame from the current portfolio assets
    if portfolio.assets:
        assets_df = pd.DataFrame(
            [
                {
                    "name": a.name,
                    "allocation_pct": a.allocation_pct,
                    "crash_pct": a.crash_pct,
                }
                for a in portfolio.assets
            ]
        )
    else:
        assets_df = pd.DataFrame(
            columns=["name", "allocation_pct", "crash_pct"],
            data=[],
        )

    # Portfolio-level numeric inputs
    col_tv, col_me = st.columns(2)
    with col_tv:
        total_value = st.number_input(
            "Total Portfolio Value ($)",
            min_value=0.0,
            value=float(portfolio.total_value),
            step=10000.0,
            format="%.2f",
            key="portfolio_total_value",
        )
    with col_me:
        monthly_expenses = st.number_input(
            "Monthly Expenses ($)",
            min_value=0.0,
            value=float(portfolio.monthly_expenses),
            step=1000.0,
            format="%.2f",
            key="portfolio_monthly_expenses",
        )

    # Editable assets table
    edited_df = st.data_editor(
        assets_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Asset Name", required=True),
            "allocation_pct": st.column_config.NumberColumn(
                "Allocation %",
                min_value=0.0,
                max_value=100.0,
                format="%.1f",
            ),
            "crash_pct": st.column_config.NumberColumn(
                "Crash %",
                min_value=-100.0,
                max_value=0.0,
                format="%.1f",
            ),
        },
        key="assets_editor",
    )

    # "Add Asset" button — appends a blank row
    if st.button("➕ Add Asset"):
        new_row = pd.DataFrame(
            [{"name": "New Asset", "allocation_pct": 0.0, "crash_pct": 0.0}]
        )
        edited_df = pd.concat([edited_df, new_row], ignore_index=True)

    # Reconstruct Portfolio from edited DataFrame and persist on any change
    new_assets = []
    for _, row in edited_df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        try:
            alloc = float(row.get("allocation_pct", 0.0))
            crash = float(row.get("crash_pct", 0.0))
        except (TypeError, ValueError):
            alloc, crash = 0.0, 0.0
        new_assets.append(Asset(name=name, allocation_pct=alloc, crash_pct=crash))

    new_portfolio = Portfolio(
        assets=new_assets,
        total_value=total_value,
        monthly_expenses=monthly_expenses,
    )

    # Persist whenever the portfolio changes
    if new_portfolio != st.session_state.portfolio:
        st.session_state.portfolio = new_portfolio
        state.save_portfolio(new_portfolio)

    # Show allocation total as a quick sanity check
    total_alloc = sum(a.allocation_pct for a in new_assets)
    if new_assets:
        alloc_color = "green" if abs(total_alloc - 100.0) <= 1.0 else "red"
        st.markdown(
            f"Total allocation: :{alloc_color}[**{total_alloc:.1f}%**]"
            + (" ✓" if abs(total_alloc - 100.0) <= 1.0 else " ⚠ should sum to 100%")
        )

    st.divider()

    # ── Preset Scenarios ───────────────────────────────────────────────────

    st.subheader("Preset Scenarios")

    preset_names = get_preset_names()
    selected_preset = st.selectbox(
        "Select a preset scenario",
        options=preset_names,
        key="preset_selector",
    )

    if st.button("Load Preset"):
        existing_names = {s.name for s in st.session_state.active_scenarios}
        if selected_preset in existing_names:
            st.warning(f"'{selected_preset}' is already in the active scenarios list.")
        else:
            preset_scenario = get_preset(selected_preset)
            st.session_state.active_scenarios.append(preset_scenario)
            st.success(f"Loaded preset: {selected_preset}")

    st.divider()

    # ── Custom Scenario ────────────────────────────────────────────────────

    st.subheader("Custom Scenario")

    custom_name = st.text_input("Scenario Name", key="custom_scenario_name")

    custom_crashes: dict[str, float] = {}
    validation_errors: list[str] = []

    current_assets = st.session_state.portfolio.assets
    if current_assets:
        st.markdown("**Per-asset crash % (must be between -100 and 0):**")
        for asset in current_assets:
            crash_val = st.number_input(
                f"{asset.name} crash %",
                min_value=-200.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                format="%.1f",
                key=f"custom_crash_{asset.name}",
            )
            if not validate_crash_pct(crash_val):
                validation_errors.append(
                    f"'{asset.name}': {crash_val}% is out of range (must be -100 to 0)."
                )
            custom_crashes[asset.name] = crash_val
    else:
        st.info("Add assets to the portfolio above to define per-asset crash percentages.")

    for err in validation_errors:
        st.error(err)

    if st.button("➕ Add Custom Scenario"):
        if not custom_name.strip():
            st.error("Please enter a scenario name.")
        elif validation_errors:
            st.error("Fix the validation errors above before adding the scenario.")
        else:
            existing_names = {s.name for s in st.session_state.active_scenarios}
            if custom_name.strip() in existing_names:
                st.warning(f"A scenario named '{custom_name.strip()}' already exists.")
            else:
                new_scenario = Scenario(
                    name=custom_name.strip(),
                    asset_crashes=dict(custom_crashes),
                    is_preset=False,
                )
                st.session_state.active_scenarios.append(new_scenario)
                st.success(f"Added custom scenario: {custom_name.strip()}")

    st.divider()

    # ── Active Scenarios Panel ─────────────────────────────────────────────

    st.subheader("Active Scenarios")

    if not st.session_state.active_scenarios:
        st.info("No active scenarios. Load a preset or add a custom scenario above.")
    else:
        # Display each scenario as a tag with a remove button
        to_remove: int | None = None
        for idx, scenario in enumerate(st.session_state.active_scenarios):
            tag_col, btn_col = st.columns([0.8, 0.2])
            with tag_col:
                badge = "🔖" if scenario.is_preset else "✏️"
                st.markdown(f"{badge} **{scenario.name}**")
            with btn_col:
                if st.button("✕", key=f"remove_scenario_{idx}"):
                    to_remove = idx

        if to_remove is not None:
            removed = st.session_state.active_scenarios.pop(to_remove)
            st.success(f"Removed scenario: {removed.name}")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN
# ══════════════════════════════════════════════════════════════════════════

with right_col:

    # ── Run Stress Test ────────────────────────────────────────────────────

    run_disabled = (
        not st.session_state.portfolio.assets
        or not st.session_state.active_scenarios
    )

    if st.button("▶ Run Stress Test", disabled=run_disabled, type="primary"):
        results = compute_all_scenarios(
            st.session_state.portfolio,
            st.session_state.active_scenarios,
        )
        st.session_state.stress_results = results

    if run_disabled and not st.session_state.stress_results:
        if not st.session_state.portfolio.assets:
            st.info("Add assets to your portfolio in the left panel to get started.")
        elif not st.session_state.active_scenarios:
            st.info("Load or create at least one scenario in the left panel, then run the stress test.")

    # ── Results ────────────────────────────────────────────────────────────

    results = st.session_state.stress_results

    if results:
        portfolio_for_results = st.session_state.portfolio
        active_scenarios_for_results = st.session_state.active_scenarios

        # ── Bar Chart: Post-Crash Portfolio Value per Scenario ─────────────

        st.subheader("Post-Crash Portfolio Value by Scenario")

        scenario_names = [r.scenario_name for r in results]
        post_crash_values = [r.post_crash_value for r in results]
        ruin_tests = [r.ruin_test for r in results]

        bar_colors = [
            "#ff4b4b" if rt == "FAIL" else "#1f77b4" for rt in ruin_tests
        ]

        bar_fig = go.Figure(
            data=[
                go.Bar(
                    x=scenario_names,
                    y=post_crash_values,
                    marker_color=bar_colors,
                    text=[f"${v:,.0f}" for v in post_crash_values],
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Post-Crash Value: $%{y:,.0f}<br>"
                        "<extra></extra>"
                    ),
                )
            ]
        )
        bar_fig.update_layout(
            xaxis_title="Scenario",
            yaxis_title="Post-Crash Value ($)",
            showlegend=False,
            height=350,
            margin=dict(t=30, b=60, l=60, r=20),
        )
        # Add a reference line for the original portfolio value
        if portfolio_for_results.total_value > 0:
            bar_fig.add_hline(
                y=portfolio_for_results.total_value,
                line_dash="dash",
                line_color="gray",
                annotation_text="Original Value",
                annotation_position="top right",
            )
        st.plotly_chart(bar_fig, use_container_width=True)

        # ── Heatmap: Assets × Scenarios ────────────────────────────────────

        st.subheader("Asset Loss Heatmap (% crash applied)")

        # Only include scenarios that are still active
        active_scenario_names = {s.name for s in active_scenarios_for_results}
        heatmap_scenarios = [
            s for s in active_scenarios_for_results
            if s.name in active_scenario_names
        ]

        if portfolio_for_results.assets and heatmap_scenarios:
            loss_matrix = build_loss_matrix(portfolio_for_results, heatmap_scenarios)

            asset_names_hm = [a.name for a in portfolio_for_results.assets]
            scenario_names_hm = [s.name for s in heatmap_scenarios]

            # Build z matrix: rows = assets, columns = scenarios
            z_values = [
                [loss_matrix[asset][scenario] for scenario in scenario_names_hm]
                for asset in asset_names_hm
            ]

            heatmap_fig = go.Figure(
                data=go.Heatmap(
                    z=z_values,
                    x=scenario_names_hm,
                    y=asset_names_hm,
                    colorscale="RdBu",
                    zmid=0,
                    text=[
                        [f"{v:.1f}%" for v in row] for row in z_values
                    ],
                    texttemplate="%{text}",
                    hovertemplate=(
                        "Asset: <b>%{y}</b><br>"
                        "Scenario: <b>%{x}</b><br>"
                        "Crash %: %{z:.1f}%<br>"
                        "<extra></extra>"
                    ),
                    colorbar=dict(title="Crash %"),
                )
            )
            heatmap_fig.update_layout(
                xaxis_title="Scenario",
                yaxis_title="Asset",
                height=max(300, 60 * len(asset_names_hm)),
                margin=dict(t=30, b=80, l=120, r=20),
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("No assets or scenarios available for heatmap.")

        # ── Results Table ──────────────────────────────────────────────────

        st.subheader("Stress Test Results")

        table_rows = []
        for r in results:
            runway_display = (
                "∞" if r.runway_months == float("inf") else f"{r.runway_months:.1f}"
            )
            table_rows.append(
                {
                    "Scenario": r.scenario_name,
                    "Post-Crash Value": f"${r.post_crash_value:,.0f}",
                    "Runway (months)": runway_display,
                    "Ruin Test": r.ruin_test,
                    "Largest Risk Asset": r.largest_risk_asset or "—",
                }
            )

        results_df = pd.DataFrame(table_rows)

        def _highlight_fail(row: pd.Series) -> list[str]:
            """Apply red background to FAIL rows."""
            if row["Ruin Test"] == "FAIL":
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        styled_df = results_df.style.apply(_highlight_fail, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Concentration warnings
        warnings = [r for r in results if r.concentration_warning]
        if warnings:
            st.warning(
                "⚠️ Concentration Warning: One or more assets exceed 40% allocation. "
                "Consider diversifying to reduce single-asset risk."
            )
