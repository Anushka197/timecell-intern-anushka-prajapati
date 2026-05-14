"""
app.py — Portfolio Intelligence Engine
=======================================
Professional Streamlit dashboard for the Family Office Portfolio Intelligence Engine.

Features:
  - Sidebar: Index status for all 6 Apple 10-K filings.
  - Chat interface with full ReAct thought-process display.
  - Source citations with fiscal year and page traceability.
  - Suggested starter questions for onboarding.

Run with:
    streamlit run app.py
"""

import logging
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PortfolioApp")

# ---------------------------------------------------------------------------
# Page Configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Intelligence Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ── Global ── */
        .main { background-color: #0f1117; }

        /* ── Header ── */
        .pie-header {
            background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
            border: 1px solid #2d3561;
            border-radius: 12px;
            padding: 24px 32px;
            margin-bottom: 24px;
        }
        .pie-header h1 { color: #e8eaf6; margin: 0; font-size: 1.8rem; }
        .pie-header p  { color: #9fa8da; margin: 4px 0 0; font-size: 0.95rem; }

        /* ── Thought Process ── */
        .thought-container {
            background: #1a1f2e;
            border-left: 3px solid #5c6bc0;
            border-radius: 0 8px 8px 0;
            padding: 12px 16px;
            margin: 8px 0;
            font-size: 0.85rem;
            color: #b0bec5;
        }
        .thought-step-label {
            color: #7986cb;
            font-weight: 600;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        /* ── Source Citations ── */
        .source-card {
            background: #1e2a3a;
            border: 1px solid #2d4a6e;
            border-radius: 8px;
            padding: 10px 14px;
            margin: 6px 0;
            font-size: 0.82rem;
        }
        .source-year-badge {
            background: #1565c0;
            color: #e3f2fd;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 8px;
        }
        .source-detail { color: #90a4ae; }

        /* ── Status Badges ── */
        .status-indexed   { color: #66bb6a; font-weight: 600; }
        .status-pending   { color: #ef5350; font-weight: 600; }

        /* ── Chat Messages ── */
        .stChatMessage { border-radius: 10px; }

        /* ── Metric Cards ── */
        .metric-card {
            background: #1a1f2e;
            border: 1px solid #2d3561;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }
        .metric-value { font-size: 1.6rem; font-weight: 700; color: #7986cb; }
        .metric-label { font-size: 0.8rem; color: #9fa8da; margin-top: 4px; }

        /* ── Divider ── */
        hr { border-color: #2d3561; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "messages": [],          # Chat history: list of {role, content, metadata}
        "agent": None,           # ReActAgent instance
        "debug_handler": None,   # LlamaDebugHandler for thought steps
        "ingestor": None,        # PortfolioIngestor instance
        "index": None,           # VectorStoreIndex
        "engine_ready": False,   # Whether the agent is initialised
        "init_error": None,      # Startup error message if any
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Engine Initialisation (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialise_engine():
    """
    Load the index and build the agent.
    Cached so it only runs once per Streamlit session.
    Returns (ingestor, agent, debug_handler) or raises on failure.
    """
    from ingest import PortfolioIngestor
    from engine import build_agent

    ingestor = PortfolioIngestor()
    index = ingestor.load_index()
    agent, debug_handler = build_agent(index)
    return ingestor, agent, debug_handler


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(ingestor) -> None:
    with st.sidebar:
        st.markdown("## 📁 Document Index Status")
        # st.markdown("---")

        if ingestor is None:
            st.warning("Engine not initialised yet.")
            return

        filing_statuses = ingestor.get_indexed_files()
        indexed_count = sum(1 for f in filing_statuses if f["indexed"])

        # Summary metric
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indexed", f"{indexed_count}/6")
        with col2:
            st.metric("Coverage", f"{int(indexed_count/6*100)}%")

        # st.markdown("---")
        # st.markdown("**Apple Inc. 10-K Filings**")

        # for filing in filing_statuses:
        #     status_icon = "🟢" if filing["indexed"] else "🔴"
        #     status_text = "Indexed" if filing["indexed"] else "Pending"
        #     st.markdown(
        #         f"{status_icon} **FY{filing['year']}** — "
        #         f"<span class='{'status-indexed' if filing['indexed'] else 'status-pending'}'>"
        #         f"{status_text}</span>",
        #         unsafe_allow_html=True,
        #     )

        # st.markdown("---")
        # st.markdown("## ⚙️ System Info")
        # st.markdown(
        #     """
        #     - **LLM**: GPT-4o via OpenRouter
        #     - **Embeddings**: text-embedding-ada-002
        #     - **Vector DB**: ChromaDB (local)
        #     - **Parser**: LlamaParse (Markdown)
        #     - **Agent**: ReAct (Reasoning + Acting)
        #     """
        # )

        # st.markdown("---")
        st.markdown("## 💡 Sample Questions")
        sample_questions = [
            "What was Apple's total revenue in FY2024?",
            "Compare R&D spend from 2021 to 2024 and calculate the % change.",
            "What are Apple's main risk factors in the 2023 10-K?",
            "Show the trend in gross margin from 2020 to 2024.",
            "How did iPhone revenue change between FY2022 and FY2023?",
            # "What was Apple's net income CAGR from 2020 to 2024?",
        ]
        for q in sample_questions:
            if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
                st.session_state["prefill_question"] = q
                st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Thought Process Renderer
# ---------------------------------------------------------------------------
def render_thought_process(steps: list) -> None:
    """
    Display the ReAct agent's reasoning steps in a collapsible expander.
    Each step shows: Thought → Action → Action Input → Observation.
    """
    if not steps:
        return

    with st.expander("🧠 Agent Reasoning Process", expanded=False):
        for i, step in enumerate(steps, 1):
            st.markdown(
                f"<div class='thought-container'>"
                f"<div class='thought-step-label'>Step {i}</div>"
                f"{_format_step(step)}"
                f"</div>",
                unsafe_allow_html=True,
            )


def _format_step(step: dict) -> str:
    """Format a single ReAct step dict into HTML."""
    parts = []
    if thought := step.get("thought"):
        parts.append(f"<b>💭 Thought:</b> {_escape(thought)}")
    if action := step.get("action"):
        parts.append(f"<b>🔧 Action:</b> <code>{_escape(action)}</code>")
    if action_input := step.get("action_input"):
        parts.append(f"<b>📥 Input:</b> {_escape(str(action_input))}")
    if observation := step.get("observation"):
        # Truncate very long observations for readability
        obs_text = str(observation)
        if len(obs_text) > 600:
            obs_text = obs_text[:600] + "… [truncated]"
        parts.append(f"<b>👁️ Observation:</b> {_escape(obs_text)}")
    return "<br>".join(parts) if parts else "No details available."


def _escape(text: str) -> str:
    """Basic HTML escaping for safe rendering."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Source Citations Renderer
# ---------------------------------------------------------------------------
def render_source_citations(source_nodes: list) -> None:
    """
    Display source nodes as citation cards with fiscal year badges.
    """
    if not source_nodes:
        return

    with st.expander(f"📚 Source Citations ({len(source_nodes)} chunks)", expanded=True):
        seen_sources: set[str] = set()

        for node in source_nodes:
            meta = node.metadata if hasattr(node, "metadata") else {}
            year = meta.get("fiscal_year", "Unknown")
            page = meta.get("page_number", "N/A")
            source_file = meta.get("source_file", "Unknown")
            filing_type = meta.get("filing_type", "10-K")
            score = getattr(node, "score", None)

            # Deduplicate by year + page
            dedup_key = f"{year}-{page}"
            if dedup_key in seen_sources:
                continue
            seen_sources.add(dedup_key)

            score_str = f" · Relevance: {score:.2f}" if score is not None else ""

            # Preview of the chunk text
            text_preview = ""
            if hasattr(node, "text") and node.text:
                preview = node.text[:200].replace("\n", " ").strip()
                if len(node.text) > 200:
                    preview += "…"
                text_preview = f"<br><span class='source-detail'><i>{_escape(preview)}</i></span>"

            st.markdown(
                f"<div class='source-card'>"
                f"<span class='source-year-badge'>FY{year}</span>"
                f"<b>Apple Inc. {filing_type}</b> · Page {page}{score_str}"
                f"{text_preview}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Extract ReAct Steps from Debug Handler
# ---------------------------------------------------------------------------
def extract_react_steps(debug_handler) -> list[dict]:
    """
    Parse the LlamaDebugHandler event log to extract ReAct reasoning steps.
    Returns a list of step dicts with keys: thought, action, action_input, observation.
    """
    steps = []
    if debug_handler is None:
        return steps

    try:
        events = debug_handler.get_events()
        current_step: dict = {}

        for event in events:
            event_type = str(getattr(event, "event_type", "")).lower()
            payload = getattr(event, "payload", {}) or {}

            if "agent_step" in event_type or "react" in event_type:
                if payload:
                    step = {}
                    if "thought" in payload:
                        step["thought"] = str(payload["thought"])
                    if "action" in payload:
                        step["action"] = str(payload["action"])
                    if "action_input" in payload:
                        step["action_input"] = payload["action_input"]
                    if "observation" in payload:
                        step["observation"] = str(payload["observation"])
                    if step:
                        steps.append(step)

            # Also capture tool call events as steps
            elif "function_call" in event_type or "tool" in event_type:
                if payload:
                    step = {
                        "action": payload.get("tool_name", payload.get("name", "Tool")),
                        "action_input": payload.get("tool_kwargs", payload.get("input", "")),
                    }
                    if step["action"] or step["action_input"]:
                        steps.append(step)

    except Exception as exc:
        logger.warning("Could not extract ReAct steps: %s", exc)

    return steps


# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------
def render_chat(agent, debug_handler) -> None:
    """Render the main chat interface and handle user input."""

    # Display existing messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "📊"):
            st.markdown(msg["content"])

            # Re-render thought process and citations for assistant messages
            if msg["role"] == "assistant":
                if steps := msg.get("steps"):
                    render_thought_process(steps)
                if sources := msg.get("sources"):
                    render_source_citations(sources)

    # Handle prefilled question from sidebar buttons
    prefill = st.session_state.pop("prefill_question", None)

    # Chat input
    user_input = st.chat_input(
        "Ask a financial question about Apple's 10-K filings…",
        key="chat_input",
    )

    # Use prefill if no direct input
    query = prefill or user_input
    if not query:
        return

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
    st.session_state["messages"].append({"role": "user", "content": query})

    # Generate response
    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("Analysing filings…"):
            start_time = time.time()

            try:
                # Clear previous debug events
                if debug_handler:
                    debug_handler.flush_event_logs()

                # Run the agent
                response = agent.chat(query)
                elapsed = time.time() - start_time

                # Extract response text
                response_text = str(response)

                # Display the answer
                st.markdown(response_text)
                st.caption(f"⏱️ Response generated in {elapsed:.1f}s")

                # Extract and display thought process
                steps = extract_react_steps(debug_handler)
                render_thought_process(steps)

                # Extract and display source citations
                source_nodes = []
                if hasattr(response, "source_nodes"):
                    source_nodes = response.source_nodes or []
                elif hasattr(response, "metadata") and response.metadata:
                    # Some response types wrap sources in metadata
                    source_nodes = response.metadata.get("source_nodes", [])

                render_source_citations(source_nodes)

                # Persist to session state
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "steps": steps,
                        "sources": source_nodes,
                    }
                )

            except Exception as exc:
                error_msg = f"⚠️ An error occurred: {exc}"
                st.error(error_msg)
                logger.error("Agent error: %s", exc, exc_info=True)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": error_msg}
                )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class='pie-header'>
            <h1>📊 Portfolio Intelligence Engine</h1>
            <p>Family Office · Apple Inc. 10-K Analysis · FY2020–FY2025 · Powered by GPT-4o + LlamaIndex</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Engine Initialisation ────────────────────────────────────────────────
    if not st.session_state["engine_ready"]:
        with st.spinner("🔄 Initialising Portfolio Intelligence Engine…"):
            try:
                try:
                    ingestor, agent, debug_handler = initialise_engine()
                except AttributeError as e:
                    st.error("🏗️ **Engine Setup Error**: There is a version mismatch with the Agent tools.")
                    st.info("Try running: `pip install --upgrade llama-index-core llama-index-agent-openai`")
                    st.stop() # This prevents the rest of the app from running and showing more errors
                except Exception as e:
                    st.error(f"⚠️ **Unexpected Error**: {e}")
                    st.stop()
                st.session_state["ingestor"] = ingestor
                st.session_state["agent"] = agent
                st.session_state["debug_handler"] = debug_handler
                st.session_state["engine_ready"] = True
                st.session_state["init_error"] = None
                logger.info("Engine initialised successfully.")
            except Exception as exc:
                st.session_state["init_error"] = str(exc)
                logger.error("Engine initialisation failed: %s", exc, exc_info=True)

    # ── Error State ──────────────────────────────────────────────────────────
    if st.session_state["init_error"]:
        st.error(
            f"**Engine Initialisation Failed**\n\n"
            f"```\n{st.session_state['init_error']}\n```\n\n"
            "Please check your `.env` file and ensure all API keys are set correctly."
        )
        render_sidebar(None)
        return

    # ── Sidebar ──────────────────────────────────────────────────────────────
    render_sidebar(st.session_state["ingestor"])

    # ── Status Banner ────────────────────────────────────────────────────────
    if st.session_state["engine_ready"]:
        st.success(
            "Engine ready — 6 Apple 10-K filings indexed and searchable.",
            icon="✅",
        )

    # ── Intro / Empty State ──────────────────────────────────────────────────
    if not st.session_state["messages"]:
        st.markdown("---")
        st.markdown("### 🚀 Get Started")
        st.markdown(
            "Ask any financial question about Apple's annual filings. "
            "The agent will search the documents, reason through the data, "
            "and show you exactly where each answer comes from."
        )

        col1, col2, col3 = st.columns(3)
        example_cards = [
            ("📈", "Revenue Trends", "What was Apple's revenue growth from FY2020 to FY2024?"),
            ("🔬", "R&D Analysis", "Compare Apple's R&D spend from 2021 to 2024 and calculate the % change."),
            ("💰", "Profitability", "Show Apple's gross margin trend from 2020 to 2024."),
        ]
        for col, (icon, title, question) in zip([col1, col2, col3], example_cards):
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div style='font-size:2rem'>{icon}</div>"
                    f"<div class='metric-value' style='font-size:1rem;margin-top:8px'>{title}</div>"
                    f"<div class='metric-label'>{question}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("---")

    # ── Chat Interface ───────────────────────────────────────────────────────
    render_chat(
        agent=st.session_state["agent"],
        debug_handler=st.session_state["debug_handler"],
    )


if __name__ == "__main__":
    main()
