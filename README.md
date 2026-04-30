# Timecell Assessment

A collection of wealth management tools that help investors calculate portfolio risks during market crashes, track live asset prices, and receive AI-generated financial advice in simple, everyday language.

## Submission Links

- **Loom Video Walkthrough:** [Link to the Loom/video here]
- **GitHub Repository:** https://github.com/Anushka197/timecell-intern-anushka-prajapati

---

## Setup and Installation

This project requires **Python 3.10+**.

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

---

## Task 1: Portfolio Risk Calculator

A tool that computes risk metrics for a given portfolio under two scenarios: a **severe crash** and a **moderate crash**.

### Approach
Implemented a modular function that calculates the post-crash value for each asset based on its allocation and expected crash percentage. It also identifies the asset with the highest risk and checks for concentration warnings.

### Features
- Calculates **Runway Months** based on monthly expenses.
- Performs a **Ruin Test** to check if the portfolio lasts more than 12 months.
- Visualizes asset allocation using a simple CLI bar chart.

### Edge Cases
Handles cases where total value is zero or no assets are provided to prevent mathematical errors.

---

## Task 2: Live Market Data Fetch

A script that pulls real-time pricing for stock indices and cryptocurrencies.

### APIs Used
- **yfinance** — for the NIFTY 50 index and Reliance stock.
- **CoinGecko public API** — for Bitcoin.

### Error Handling
Uses a `try-except` structure for each asset. If one API call fails, it logs the error and continues fetching the remaining assets to ensure the tool remains functional.

### Output
Results are displayed in a clean ASCII table with timestamps and currency details.

---

## Task 3: AI-Powered Portfolio Explainer

Uses a Large Language Model (LLM) to provide a plain-English risk assessment of the user's portfolio.

### API Choice
**Google Gemini** (`gemini-2.5-flash`) — chosen for its high speed and excellent performance with structured JSON output.

### Prompt Engineering Approach

| Stage | Description |
|---|---|
| **Initial Approach** | Started with a simple summary request, but output was inconsistent. |
| **Refinement** | Switched to XML-like tags (`<CLIENT_PROFILE>`, `<PORTFOLIO_DATA>`) to clearly separate instructions from data. |
| **JSON Enforcement** | Forced the model to return a structured JSON object for reliable parsing of the summary, specific advice, and final verdict. |

### Bonuses

- **Configurable Tone** — The prompt dynamically adjusts its instructions based on whether the user is a `beginner`, `experienced`, or `expert` investor.
- **Senior Critique** — A second LLM call acts as a "Senior Risk Officer" to critique the initial advisor's response for accuracy.

---

## Task 4: RAG-Powered Portfolio Intelligence Engine

A Retrieval-Augmented Generation (RAG) system that lets users query Apple Inc.'s 10-K annual filings (FY2020–FY2025) through natural language, powered by a ReAct agent with source citations and a critic review layer.

### Approach

The system is built as a two-stage pipeline:

1. **Parse** — PDFs are converted to structured Markdown using LlamaParse, with fiscal year metadata injected per document section.
2. **Ingest** — Parsed documents are chunked and embedded into a local ChromaDB vector store using OpenAI's `text-embedding-ada-002` model.

At query time, a **ReAct agent** (via LlamaIndex) reasons step-by-step, calling two tools:
- `AppleFilingsSearch` — semantic vector search over the indexed 10-K chunks.
- `Calculator` — sandboxed Python `eval` for precise arithmetic (percentage changes, CAGRs, margins).

A **Critic-in-the-Loop** layer runs a second LLM call after every response, acting as a Chief Investment Officer who scores the draft for missing context, hidden assumptions, and mathematical traceability.

### Tech Stack

| Component | Choice |
|---|---|
| **LLM** | GPT-4o-mini via OpenRouter |
| **Embeddings** | text-embedding-ada-002 via OpenRouter |
| **Vector DB** | ChromaDB (local persistent store) |
| **PDF Parser** | LlamaParse (Markdown output) |
| **Agent Framework** | LlamaIndex ReActAgent |
| **UI** | Streamlit |

### Features

- **Streamlit dashboard** (`app.py`) with a dark-themed chat interface, sidebar showing index status for all 6 filings, and collapsible agent reasoning steps.
- **CLI interface** (`cli.py`) for terminal-based querying with the same Critic-in-the-Loop workflow.
- **Source citations** — every answer surfaces the fiscal year, page number, and a text preview of the retrieved chunks.
- **Deterministic math** — the system prompt strictly forbids the LLM from doing mental arithmetic; all calculations must go through the Calculator tool.
- **V2 ingestion pipeline** (`parse_v2.py` / `ingest_v2.py`) with table-aware chunking using `MarkdownElementNodeParser`, keeping financial tables fully intact rather than splitting them mid-row.

### Running Task 4

**Step 1 — Parse the PDFs** (requires a LlamaCloud API key):
```bash
cd src/Task4
python parse.py
```

**Step 2 — Ingest into ChromaDB** (only needed once; skipped automatically if the index already exists):
```bash
python ingest.py
```

**Step 3a — Launch the Streamlit UI:**
```bash
streamlit run app.py
```

**Step 3b — Or use the CLI:**
```bash
python cli.py
```

### Required API Keys (`.env` in `src/Task4/`)

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLAMA_CLOUD_API_KEY=your_key_here
```

### Design Decisions

**Why OpenRouter instead of direct OpenAI?** OpenRouter provides a unified API gateway, making it easy to swap models (e.g., GPT-4o → Claude) without changing code.

**Why a local ChromaDB?** Keeps the vector store self-contained with no external service dependency. The persisted index means embeddings are only computed once.

**Why a Critic layer?** Financial analysis is high-stakes. A second LLM pass that explicitly hunts for missing context and assigns a conviction score adds a useful sanity check before the answer reaches the user.

---

## Hardest Part

The most challenging aspect was ensuring the AI-generated explanation remained consistent and followed a strict JSON format. LLMs can sometimes include conversational filler or markdown formatting that breaks code parsers.

**Solution:** Implemented a robust prompt using XML delimiters and utilized Gemini's `response_mime_type="application/json"` configuration. Also added a retry mechanism with **exponential backoff** to handle `503` errors during high-demand periods.

---

## AI Usage

AI coding tools (Gemini Pro/Claude) were used to assist with boilerplate code generation and to help refine the terminal table formatting. All logic was manually reviewed and adjusted to ensure mathematical correctness and proper error handling.
