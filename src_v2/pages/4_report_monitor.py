"""
pages/4_report_monitor.py — Automated Financial Report Monitor

Three-tab layout:
  Tab 1 — Watchlist: manage company watchlist
  Tab 2 — Filing Check: download and ingest SEC filings
  Tab 3 — Query & Analysis: RAG Q&A and "What Changed?" analysis
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from core.models import CompanyWatchlistEntry, ReportWatchlist
from core import state
from core.rag.ingest import ingest_filing, is_filing_ingested, get_or_create_collection
from core.rag.engine import (
    query as rag_query,
    what_changed,
    get_indexed_filing_dates,
)
from core.llm import OllamaTimeoutError, OllamaConnectionError

FILINGS_DIR = Path("../data/filings")

st.set_page_config(layout="wide", page_title="Report Monitor")
st.title("📋 Automated Financial Report Monitor")

# ── Session state initialisation ───────────────────────────────────────────

if "report_watchlist" not in st.session_state:
    st.session_state.report_watchlist = state.load_report_watchlist()

if "filing_records" not in st.session_state:
    st.session_state.filing_records = []  # list[dict] — status table rows


# ── Helper: download filings via sec-edgar-downloader ─────────────────────

def download_filings_for_company(ticker: str, company_name: str) -> list[dict]:
    """
    Download 10-K and 10-Q filings for a company using sec-edgar-downloader.
    Returns a list of filing record dicts.
    """
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        return [
            {
                "ticker": ticker.upper(),
                "filing_type": ft,
                "accession_number": "N/A",
                "local_path": "",
                "status": "Failed: sec-edgar-downloader not installed",
            }
            for ft in ["10-K", "10-Q"]
        ]

    dl = Downloader(company_name, "research@timecell.ai")
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for filing_type in ["10-K", "10-Q"]:
        try:
            dl.get(filing_type, ticker, limit=3, download_folder=str(FILINGS_DIR))
            # sec-edgar-downloader saves to:
            # {download_folder}/sec-edgar-filings/{ticker}/{filing_type}/{accession}/
            edgar_dir = (
                FILINGS_DIR / "sec-edgar-filings" / ticker.upper() / filing_type
            )
            if edgar_dir.exists():
                for accession_dir in sorted(edgar_dir.iterdir()):
                    if not accession_dir.is_dir():
                        continue
                    # Look for primary document: prefer .htm/.html, then .txt
                    htm_files = (
                        list(accession_dir.glob("*.htm"))
                        + list(accession_dir.glob("*.html"))
                    )
                    txt_files = list(accession_dir.glob("*.txt"))
                    pdf_files = list(accession_dir.glob("*.pdf"))
                    filing_files = pdf_files or htm_files or txt_files
                    if filing_files:
                        records.append(
                            {
                                "ticker": ticker.upper(),
                                "filing_type": filing_type,
                                "accession_number": accession_dir.name,
                                "local_path": str(filing_files[0]),
                                "status": "Downloaded",
                            }
                        )
            else:
                records.append(
                    {
                        "ticker": ticker.upper(),
                        "filing_type": filing_type,
                        "accession_number": "N/A",
                        "local_path": "",
                        "status": "No filings found",
                    }
                )
        except Exception as exc:
            records.append(
                {
                    "ticker": ticker.upper(),
                    "filing_type": filing_type,
                    "accession_number": "N/A",
                    "local_path": "",
                    "status": f"Failed: {str(exc)[:80]}",
                }
            )
    return records


# ── Helper: ingest a single filing record ─────────────────────────────────

def _ingest_record(record: dict, company_name: str, progress_bar) -> int:
    """
    Ingest a downloaded filing into ChromaDB.
    Updates progress_bar while embedding chunks.
    Returns number of chunks ingested (0 if skipped or failed).
    """
    local_path = record.get("local_path", "")
    if not local_path or record["status"].startswith("Failed") or record["status"] == "No filings found":
        return 0

    filing_path = Path(local_path)
    if not filing_path.exists():
        return 0

    # Only PDF ingestion is supported by ingest_filing; skip non-PDF files
    if filing_path.suffix.lower() not in (".pdf",):
        return 0

    ticker = record["ticker"]
    accession = record["accession_number"]

    if is_filing_ingested(ticker, accession):
        record["status"] = "Already Ingested"
        return 0

    chunks_done = [0]
    total_chunks = [1]  # will be updated by callback

    def _progress(done: int, total: int) -> None:
        chunks_done[0] = done
        total_chunks[0] = total
        if total > 0:
            progress_bar.progress(done / total, text=f"Embedding {done}/{total} chunks…")

    try:
        # Derive a filing date from the accession number directory name if possible
        # Accession numbers look like: 0000320193-24-000123
        # We use today's date as a fallback
        filing_date = datetime.today().strftime("%Y-%m-%d")

        n = ingest_filing(
            filing_path=filing_path,
            ticker=ticker,
            company_name=company_name,
            filing_type=record["filing_type"],
            filing_date=filing_date,
            accession_number=accession,
            progress_callback=_progress,
        )
        return n
    except Exception as exc:
        record["status"] = f"Ingest failed: {str(exc)[:60]}"
        return 0


# ── Helper: update last_indexed_date for a watchlist entry ────────────────

def _refresh_last_indexed(ticker: str) -> None:
    """Update last_indexed_date for the given ticker in session state."""
    dates = get_indexed_filing_dates(ticker)
    if dates:
        wl: ReportWatchlist = st.session_state.report_watchlist
        for entry in wl.entries:
            if entry.ticker.upper() == ticker.upper():
                entry.last_indexed_date = dates[-1]
                break
        state.save_report_watchlist(wl)


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["📋 Watchlist", "📥 Filing Check", "🔍 Query & Analysis"])


# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — WATCHLIST
# ─────────────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Company Watchlist")

    wl: ReportWatchlist = st.session_state.report_watchlist

    if not wl.entries:
        st.info("No companies in watchlist. Add one below.")
    else:
        # Display watchlist as a table with remove buttons
        for i, entry in enumerate(list(wl.entries)):
            col_ticker, col_name, col_date, col_remove = st.columns([1, 3, 2, 1])
            with col_ticker:
                st.write(entry.ticker)
            with col_name:
                st.write(entry.company_name)
            with col_date:
                st.write(entry.last_indexed_date or "—")
            with col_remove:
                if st.button("✕ Remove", key=f"remove_company_{i}_{entry.ticker}"):
                    wl.entries = [e for e in wl.entries if e.ticker != entry.ticker]
                    state.save_report_watchlist(wl)
                    st.rerun()

        # Column headers (shown above the rows via a header row)
        st.caption("Columns: Ticker | Company Name | Last Indexed Date | Action")

    st.divider()

    # ── Add Company form ──────────────────────────────────────────────────
    st.subheader("Add Company")
    with st.form("add_company_form", clear_on_submit=True):
        new_ticker = st.text_input("Ticker Symbol", placeholder="e.g. AAPL").strip().upper()
        new_name = st.text_input("Company Name", placeholder="e.g. Apple Inc.").strip()
        submitted = st.form_submit_button("➕ Add Company")

    if submitted:
        if not new_ticker:
            st.error("Ticker symbol is required.")
        elif not new_name:
            st.error("Company name is required.")
        elif wl.has_ticker(new_ticker):
            st.error(f"Ticker **{new_ticker}** is already in the watchlist.")
        else:
            entry = CompanyWatchlistEntry(
                ticker=new_ticker,
                company_name=new_name,
                added_at=datetime.now().isoformat(timespec="seconds"),
                last_indexed_date=None,
            )
            wl.entries.append(entry)
            state.save_report_watchlist(wl)
            st.success(f"Added **{new_ticker}** ({new_name}) to watchlist.")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — FILING CHECK
# ─────────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Filing Check")

    wl: ReportWatchlist = st.session_state.report_watchlist

    if not wl.entries:
        st.warning("Add companies to the watchlist first (Tab 1).")
    else:
        if st.button("🔍 Check for New Filings", type="primary"):
            all_records: list[dict] = []

            for entry in wl.entries:
                st.write(f"**{entry.ticker}** — {entry.company_name}")
                with st.spinner(f"Downloading filings for {entry.ticker}…"):
                    records = download_filings_for_company(
                        entry.ticker, entry.company_name
                    )

                # Ingest each downloaded filing
                for record in records:
                    if (
                        record["status"] == "Downloaded"
                        and record["local_path"]
                        and Path(record["local_path"]).suffix.lower() == ".pdf"
                    ):
                        progress_bar = st.progress(0, text="Preparing ingestion…")
                        chunks = _ingest_record(record, entry.company_name, progress_bar)
                        progress_bar.empty()
                        if chunks > 0:
                            record["chunks"] = chunks
                            record["status"] = f"Ingested ({chunks} chunks)"
                            _refresh_last_indexed(entry.ticker)
                        elif record["status"] == "Already Ingested":
                            record["chunks"] = 0
                        else:
                            record["chunks"] = 0
                    else:
                        record.setdefault("chunks", 0)

                all_records.extend(records)

            st.session_state.filing_records = all_records

        # ── Status table ──────────────────────────────────────────────────
        if st.session_state.filing_records:
            st.subheader("Filing Status")
            df = pd.DataFrame(st.session_state.filing_records)
            display_cols = ["ticker", "filing_type", "accession_number", "status", "chunks"]
            # Only show columns that exist
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — QUERY & ANALYSIS
# ─────────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Query & Analysis")

    wl: ReportWatchlist = st.session_state.report_watchlist

    if not wl.entries:
        st.warning("Add companies to the watchlist first (Tab 1).")
    else:
        # ── Company selector ──────────────────────────────────────────────
        ticker_options = [e.ticker for e in wl.entries]
        selected_ticker = st.selectbox(
            "Select Company",
            options=ticker_options,
            format_func=lambda t: next(
                (f"{e.ticker} — {e.company_name}" for e in wl.entries if e.ticker == t),
                t,
            ),
        )

        # Determine how many filings are indexed for the selected company
        indexed_dates = get_indexed_filing_dates(selected_ticker) if selected_ticker else []
        has_two_filings = len(indexed_dates) >= 2

        if selected_ticker:
            if indexed_dates:
                st.caption(
                    f"Indexed filings: {len(indexed_dates)} "
                    f"(dates: {', '.join(indexed_dates)})"
                )
            else:
                st.caption("No filings indexed yet. Use the Filing Check tab to download and ingest filings.")

        st.divider()

        # ── Free-form Q&A ─────────────────────────────────────────────────
        st.markdown("#### Ask a Question")
        question = st.text_area(
            "Question",
            placeholder="e.g. What were the main revenue drivers in the latest 10-K?",
            label_visibility="collapsed",
        )

        ask_disabled = not selected_ticker or not indexed_dates
        if st.button("💬 Ask", disabled=ask_disabled, type="primary"):
            if not question.strip():
                st.error("Please enter a question.")
            else:
                with st.spinner("Querying filings…"):
                    try:
                        response = rag_query(question.strip(), selected_ticker)
                        st.markdown("**Answer:**")
                        st.write(response.answer)

                        if response.sources:
                            st.markdown("**Sources:**")
                            seen = set()
                            for src in response.sources:
                                key = (src.filing_type, src.filing_date, src.accession_number)
                                if key not in seen:
                                    seen.add(key)
                                    st.caption(
                                        f"• {src.filing_type} filed {src.filing_date} "
                                        f"(accession: {src.accession_number})"
                                    )
                    except OllamaTimeoutError:
                        st.error(
                            "⏱ Ollama request timed out. Check that Ollama is running "
                            "and try again."
                        )
                    except OllamaConnectionError:
                        st.error(
                            "🔌 Cannot connect to Ollama. Make sure the Ollama server "
                            "is running at localhost:11434."
                        )
                    except Exception as exc:
                        st.error(f"Query failed: {exc}")

        st.divider()

        # ── What Changed? ─────────────────────────────────────────────────
        st.markdown("#### What Changed?")

        if not has_two_filings:
            st.info(
                "At least two filings required for comparison. "
                "Index more filings in the Filing Check tab."
            )

        what_changed_disabled = not has_two_filings
        if st.button(
            "📊 What Changed?",
            disabled=what_changed_disabled,
            help="Compares the two most recent indexed filings for this company.",
        ):
            with st.spinner("Analysing changes between filings…"):
                try:
                    wc_response = what_changed(selected_ticker)

                    # Parse structured sections from the LLM response
                    answer_text = wc_response.answer

                    # Display the full structured answer
                    st.markdown("### Filing Comparison Summary")

                    # Try to extract labelled sections; fall back to full text
                    sections = {
                        "Revenue Changes": None,
                        "Risk Factor Changes": None,
                        "Guidance Changes": None,
                    }

                    import re

                    # Match numbered or bold section headers
                    pattern = re.compile(
                        r"(?:(?:\d+\.\s*)|(?:\*{1,2}))"
                        r"(Revenue Changes|Risk Factor Changes|Guidance Changes)"
                        r"(?:\*{1,2})?\s*[:\n]"
                        r"(.*?)(?=(?:\d+\.\s*|\*{1,2})"
                        r"(?:Revenue Changes|Risk Factor Changes|Guidance Changes)"
                        r"|\Z)",
                        re.DOTALL | re.IGNORECASE,
                    )

                    matches = list(pattern.finditer(answer_text))
                    if matches:
                        for match in matches:
                            section_name = match.group(1).strip()
                            section_body = match.group(2).strip()
                            # Normalise to canonical key
                            for key in sections:
                                if key.lower() == section_name.lower():
                                    sections[key] = section_body
                                    break

                    # Render each section
                    for section_title, section_body in sections.items():
                        with st.expander(f"📌 {section_title}", expanded=True):
                            if section_body:
                                st.write(section_body)
                            else:
                                # Section not parsed — show full answer under first section
                                if section_title == "Revenue Changes":
                                    st.write(answer_text)
                                else:
                                    st.write("See Revenue Changes section above.")

                    # Source citations
                    if wc_response.sources:
                        st.markdown("**Sources:**")
                        seen = set()
                        for src in wc_response.sources:
                            key = (src.filing_type, src.filing_date, src.accession_number)
                            if key not in seen:
                                seen.add(key)
                                st.caption(
                                    f"• {src.filing_type} filed {src.filing_date} "
                                    f"(accession: {src.accession_number})"
                                )

                except ValueError as exc:
                    st.error(str(exc))
                except OllamaTimeoutError:
                    st.error(
                        "⏱ Ollama request timed out. Check that Ollama is running "
                        "and try again."
                    )
                except OllamaConnectionError:
                    st.error(
                        "🔌 Cannot connect to Ollama. Make sure the Ollama server "
                        "is running at localhost:11434."
                    )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
