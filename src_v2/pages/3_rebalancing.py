"""
pages/3_rebalancing.py — Natural Language Rebalancing Advisor

Single-column layout (wide mode):
  1. Tone selector (Beginner / Experienced / Expert)
  2. Goal text area
  3. Generate Rebalancing Plan button
  4. Before/After allocation table
  5. Post-rebalancing risk metrics
  6. Critic review

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

import json
import re

import pandas as pd
import streamlit as st

from core import llm, state
from core.llm import (
    OllamaConnectionError,
    OllamaTimeoutError,
    build_advisor_prompt,
    build_critic_prompt,
)
from core.models import Asset, Portfolio, Scenario
from core.rebalancing import validate_allocations
from core.risk import compute_scenario

st.set_page_config(layout="wide", page_title="Rebalancing Advisor")
st.title("🤖 Natural Language Rebalancing Advisor")

# ── Session state initialisation ───────────────────────────────────────────

if "portfolio" not in st.session_state:
    st.session_state.portfolio = state.load_portfolio()
if "active_scenarios" not in st.session_state:
    st.session_state.active_scenarios = []
if "selected_tone" not in st.session_state:
    st.session_state.selected_tone = "experienced"
if "rebalancing_plan" not in st.session_state:
    st.session_state.rebalancing_plan = None
if "critic_review" not in st.session_state:
    st.session_state.critic_review = None

# ── Ollama health warning ──────────────────────────────────────────────────

ollama_ok = llm.check_health()
if not ollama_ok:
    st.warning(
        "⚠️ **Ollama is unreachable.** The rebalancing advisor requires Ollama to be running "
        "at `localhost:11434`. Start Ollama and ensure `llama3.1:8b` is pulled before using "
        "this page."
    )

# ── Tone selector ──────────────────────────────────────────────────────────

st.subheader("Communication Tone")

tone_options = ["Beginner", "Experienced", "Expert"]
# Map display label → internal key
tone_map = {
    "Beginner": "beginner",
    "Experienced": "experienced",
    "Expert": "expert",
}
# Reverse map for finding the current index
reverse_tone_map = {v: k for k, v in tone_map.items()}

current_tone_label = reverse_tone_map.get(
    st.session_state.selected_tone, "Experienced"
)
selected_tone_label = st.radio(
    "Select explanation tone:",
    options=tone_options,
    index=tone_options.index(current_tone_label),
    horizontal=True,
    key="tone_radio",
)
st.session_state.selected_tone = tone_map[selected_tone_label]

tone_descriptions = {
    "Beginner": "Simple language, no jargon, uses analogies.",
    "Experienced": "Standard financial terminology, concise.",
    "Expert": "Advanced concepts, quantitative reasoning, precise language.",
}
st.caption(f"ℹ️ {tone_descriptions[selected_tone_label]}")

st.divider()

# ── Goal text area ─────────────────────────────────────────────────────────

st.subheader("Rebalancing Goal")

goal = st.text_area(
    "Describe your rebalancing goal:",
    placeholder=(
        "e.g. Reduce crypto to under 20%, increase bonds to at least 25%, "
        "keep maximum drawdown under 15%."
    ),
    height=120,
    key="rebalancing_goal",
)

# ── Generate Rebalancing Plan button ──────────────────────────────────────

portfolio: Portfolio = st.session_state.portfolio

if st.button("▶ Generate Rebalancing Plan", type="primary", disabled=not ollama_ok):
    # Validate goal is non-empty
    if not goal.strip():
        st.error("Please enter a rebalancing goal before submitting.")
        st.stop()

    # Validate portfolio has assets
    if not portfolio.assets:
        st.error(
            "Your portfolio has no assets. Add assets in the "
            "Stress-Testing Dashboard first."
        )
        st.stop()

    with st.spinner("Generating rebalancing plan…"):
        try:
            messages = build_advisor_prompt(
                portfolio=portfolio,
                goal=goal.strip(),
                tone=st.session_state.selected_tone,
            )
            response = llm.chat(messages)
        except OllamaTimeoutError:
            st.error(
                "⏱️ The request timed out. The model took too long to respond."
            )
            if st.button("🔄 Retry", key="retry_advisor"):
                st.rerun()
            st.stop()
        except OllamaConnectionError:
            st.error(
                "🔌 Could not connect to Ollama. Please ensure Ollama is running."
            )
            st.stop()

    # ── Parse JSON allocation plan from response ───────────────────────────

    allocation_plan: dict[str, float] | None = None

    # Strategy 1: look for a ```json ... ``` code block
    json_block_match = re.search(
        r"```json\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE
    )
    if json_block_match:
        try:
            allocation_plan = json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            allocation_plan = None

    # Strategy 2: look for any ``` ... ``` code block containing a JSON object
    if allocation_plan is None:
        code_block_match = re.search(r"```\s*(\{.*?\})\s*```", response, re.DOTALL)
        if code_block_match:
            try:
                allocation_plan = json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                allocation_plan = None

    # Strategy 3: find the first top-level JSON object in the response
    if allocation_plan is None:
        json_obj_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_obj_match:
            try:
                allocation_plan = json.loads(json_obj_match.group(0))
            except json.JSONDecodeError:
                allocation_plan = None

    if allocation_plan is None:
        st.error(
            "❌ The model returned an invalid plan — no JSON allocation block was found. "
            "Please retry."
        )
        st.stop()

    # Ensure all values are floats
    try:
        allocation_plan = {k: float(v) for k, v in allocation_plan.items()}
    except (TypeError, ValueError):
        st.error(
            "❌ The model returned an allocation plan with non-numeric values. "
            "Please retry."
        )
        st.stop()

    # ── Validate allocations sum to ~100% ──────────────────────────────────

    if not validate_allocations(allocation_plan):
        total = sum(allocation_plan.values())
        st.error(
            f"❌ The proposed allocations sum to **{total:.2f}%**, which is not within "
            f"±1% of 100%. The plan cannot be used. Please retry."
        )
        st.stop()

    # Store the plan
    st.session_state.rebalancing_plan = {
        "allocations": allocation_plan,
        "raw_response": response,
        "goal": goal.strip(),
    }

    # ── Critic review ──────────────────────────────────────────────────────

    with st.spinner("Running critic review…"):
        try:
            critic_messages = build_critic_prompt(
                plan=allocation_plan,
                goal=goal.strip(),
            )
            critic_response = llm.chat(critic_messages)
            st.session_state.critic_review = critic_response
        except OllamaTimeoutError:
            st.session_state.critic_review = (
                "⏱️ Critic review timed out. Please regenerate the plan to retry."
            )
        except OllamaConnectionError:
            st.session_state.critic_review = (
                "🔌 Could not connect to Ollama for critic review."
            )

# ── Display results ────────────────────────────────────────────────────────

plan_data = st.session_state.rebalancing_plan

if plan_data is not None:
    allocation_plan = plan_data["allocations"]
    raw_response = plan_data["raw_response"]
    saved_goal = plan_data["goal"]

    st.divider()

    # ── Before / After allocation table ───────────────────────────────────

    st.subheader("Before / After Allocation")

    # Build a lookup of current allocations by asset name
    current_alloc: dict[str, float] = {
        a.name: a.allocation_pct for a in portfolio.assets
    }

    # Collect all asset names (union of current and proposed)
    all_asset_names = sorted(
        set(current_alloc.keys()) | set(allocation_plan.keys())
    )

    table_rows = []
    for asset_name in all_asset_names:
        current_pct = current_alloc.get(asset_name, 0.0)
        proposed_pct = allocation_plan.get(asset_name, 0.0)
        change = proposed_pct - current_pct
        change_str = f"▲ +{change:.1f}%" if change > 0 else (
            f"▼ {change:.1f}%" if change < 0 else "— 0.0%"
        )
        table_rows.append(
            {
                "Asset": asset_name,
                "Current %": f"{current_pct:.1f}%",
                "Proposed %": f"{proposed_pct:.1f}%",
                "Change": change_str,
            }
        )

    allocation_df = pd.DataFrame(table_rows)
    st.dataframe(allocation_df, use_container_width=True, hide_index=True)

    # ── LLM rationale (text outside the JSON block) ────────────────────────

    # Strip the JSON block from the response to show only the rationale text
    rationale = re.sub(r"```json\s*\{.*?\}\s*```", "", raw_response, flags=re.DOTALL | re.IGNORECASE)
    rationale = re.sub(r"```\s*\{.*?\}\s*```", "", rationale, flags=re.DOTALL)
    rationale = rationale.strip()

    if rationale:
        with st.expander("📝 Advisor Rationale", expanded=True):
            st.markdown(rationale)

    st.divider()

    # ── Post-rebalancing risk metrics ──────────────────────────────────────

    st.subheader("Post-Rebalancing Risk Metrics")

    active_scenarios = st.session_state.active_scenarios

    if not active_scenarios:
        st.info(
            "ℹ️ No active scenarios found. Go to the **Stress-Testing Dashboard** to load "
            "or create a scenario, then return here to see post-rebalancing risk metrics."
        )
    else:
        # Use the most recently added active scenario
        active_scenario: Scenario = active_scenarios[-1]

        # Build a new Portfolio with the proposed allocations
        proposed_assets = []
        for asset_name, proposed_pct in allocation_plan.items():
            # Preserve crash_pct from the original asset if it exists
            original_asset = next(
                (a for a in portfolio.assets if a.name == asset_name), None
            )
            crash_pct = original_asset.crash_pct if original_asset else 0.0
            proposed_assets.append(
                Asset(
                    name=asset_name,
                    allocation_pct=proposed_pct,
                    crash_pct=crash_pct,
                )
            )

        proposed_portfolio = Portfolio(
            assets=proposed_assets,
            total_value=portfolio.total_value,
            monthly_expenses=portfolio.monthly_expenses,
        )

        # Run the scenario on the proposed portfolio
        result = compute_scenario(proposed_portfolio, active_scenario)

        st.markdown(f"**Scenario:** {active_scenario.name}")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                label="Post-Crash Value",
                value=f"${result.post_crash_value:,.0f}",
            )

        with metric_col2:
            runway_display = (
                "∞ months"
                if result.runway_months == float("inf")
                else f"{result.runway_months:.1f} months"
            )
            st.metric(label="Runway", value=runway_display)

        with metric_col3:
            ruin_icon = "✅" if result.ruin_test == "PASS" else "❌"
            st.metric(
                label="Ruin Test",
                value=f"{ruin_icon} {result.ruin_test}",
            )

        if result.concentration_warning:
            st.warning(
                "⚠️ **Concentration Warning:** One or more proposed assets exceed 40% "
                "allocation. Consider diversifying to reduce single-asset risk."
            )

        if result.largest_risk_asset:
            st.caption(
                f"Largest risk asset under this scenario: **{result.largest_risk_asset}**"
            )

    st.divider()

    # ── Critic review ──────────────────────────────────────────────────────

    st.subheader("Critic Review")

    critic_text = st.session_state.critic_review
    if critic_text:
        st.markdown(critic_text)
    else:
        st.info("Critic review will appear here after generating a plan.")
