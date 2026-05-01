import streamlit as st
from core import llm, state
from core.models import Portfolio, CorrelationWatchlist, ReportWatchlist

# Page config must be the first Streamlit call
st.set_page_config(
    layout="wide",
    page_title="TimeCell Family Office Intelligence Platform",
    page_icon="🏦",
)

CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

def check_ollama_status() -> bool:
    """Returns True if Ollama is reachable at localhost:11434."""
    return llm.check_health()

@st.cache_resource
def load_all_state() -> None:
    """
    Loads portfolio.json, correlation_watchlist.json, report_watchlist.json
    into st.session_state on first run.
    Initializes empty defaults if files are missing.
    """
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = state.load_portfolio()
    if "corr_watchlist" not in st.session_state:
        st.session_state.corr_watchlist = state.load_correlation_watchlist()
    if "report_watchlist" not in st.session_state:
        st.session_state.report_watchlist = state.load_report_watchlist()
    if "active_scenarios" not in st.session_state:
        st.session_state.active_scenarios = []
    if "stress_results" not in st.session_state:
        st.session_state.stress_results = []

def render_sidebar(ollama_ok: bool) -> None:
    """
    Renders sidebar nav, Ollama status badge, model names, and portfolio summary.
    """
    with st.sidebar:
        st.title("🏦 TimeCell")
        st.caption("Family Office Intelligence Platform")
        st.divider()
        
        # Navigation
        st.subheader("Navigation")
        st.page_link("app.py", label="🏠 Home")
        st.page_link("pages/1_stress_test.py", label="📊 Stress-Testing Dashboard")
        st.page_link("pages/2_correlation.py", label="📈 Correlation Analyzer")
        st.page_link("pages/3_rebalancing.py", label="🤖 Rebalancing Advisor")
        st.page_link("pages/4_report_monitor.py", label="📋 Report Monitor")
        st.divider()
        
        # System Status
        st.subheader("System Status")
        if ollama_ok:
            st.success("🟢 Ollama: Connected")
        else:
            st.error("🔴 Ollama: Unreachable")
        st.caption(f"Model: {CHAT_MODEL}")
        st.caption(f"Embed: {EMBED_MODEL}")
        st.divider()
        
        # Portfolio Summary
        st.subheader("Portfolio")
        portfolio = st.session_state.get("portfolio", Portfolio())
        st.metric("Total Value", f"{portfolio.total_value:,.2f}")
        st.metric("Assets", len(portfolio.assets))

def main():
    load_all_state()
    ollama_ok = check_ollama_status()
    render_sidebar(ollama_ok)
    
    # Home page content
    st.title("🏦 TimeCell Family Office Intelligence Platform")
    st.markdown("""
    Welcome to the TimeCell Family Office Intelligence Platform.
    Use the sidebar to navigate between the four analytical tools:
    
    - **📊 Stress-Testing Dashboard** — Multi-scenario portfolio crash simulation
    - **📈 Correlation Analyzer** — Pearson correlation matrix with diversification scoring
    - **🤖 Rebalancing Advisor** — LLM-driven rebalancing with critic review
    - **📋 Report Monitor** — SEC EDGAR filing ingestion and RAG-powered Q&A
    """)
    
    if not ollama_ok:
        st.warning(
            "⚠️ Ollama server is unreachable at localhost:11434. "
            "LLM-dependent features (Rebalancing Advisor, Report Monitor) will not work. "
            "Please start Ollama and refresh the page."
        )

if __name__ == "__main__":
    main()
