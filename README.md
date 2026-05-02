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
 
## Task 4: The Open Problem - RAG for Apple 10-K Filings (20 pts)
 
### The Problem (Self-Defined)
Build an intelligent document retrieval system that can answer complex financial questions across 6 years of Apple's annual reports.
 
### Why I Chose This
 
Looking at Timecell's product, I noticed they need to synthesize information across multiple filings, compare trends over time, and extract both structured (tables) and unstructured (narrative) data. A **Retrieval-Augmented Generation (RAG)** system solves all three problems.
 
### System Architecture
 
```
PDF Files (2020-2025.pdf)
    ↓
LlamaParse (Markdown Extraction)
    ↓
Table Detection Fork
    ↓                    ↓
Text Chunks         JSON Tables
    ↓                    ↓
Embeddings          Structured Storage
    ↓                    ↓
ChromaDB           tables.json
    ↓
ReActAgent (Tools: Vector Search + Table Search + Calculator)
    ↓
Natural Language Answers
```
 
#### Phase 1: Document Parsing (`parse.py`)
 
**Challenge**: 10-K filings are complex PDFs with multi-column layouts, nested tables, and footnotes. Standard PDF extraction gives garbage.
 
**Solution**: LlamaParse with a custom system prompt:
 
```python
parser = LlamaParse(
    result_type="markdown",
    system_prompt=(
        "This is an Apple Inc. Annual Report (10-K). "
        "Extract all financial tables, footnotes, and narrative sections. "
        "Preserve table formatting using Markdown pipe syntax."
    ),
)
```
 
**Metadata Injection**: Each document chunk gets tagged:
```python
doc.metadata.update({
    "source_file": "2023.pdf",
    "fiscal_year": "2023",
    "company": "Apple Inc.",
    "filing_type": "10-K",
    "page_number": i + 1,
})
```
 
This enables filtering queries like *"Find revenue data from 2022 filings only"*.
 
#### Phase 2: Ingestion Pipeline (`ingest.py`)
 
**The Hard Part**: Embedding 6 years of filings = ~100MB of text = 50,000+ chunks = 50,000 API calls.
 
**Production-Grade Solutions I Implemented:**
 
**1. Checkpoint-Based Recovery**
```python
def save_checkpoint(batch_idx):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_batch": batch_idx}, f)
```
If ingestion crashes at batch 347, resume from batch 347—not batch 0. This saved me hours during development when I hit rate limits.
 
**2. Embedding Cache**
```python
class EmbeddingCache:
    def get(self, text):
        return self.cache.get(text)
    
    def set(self, text, embedding):
        self.cache[text] = embedding
        self.save()
```
Identical text chunks (e.g., boilerplate legal disclaimers) get embedded once, then cached. Reduces API calls by ~30%.
 
**3. Table Extraction Fork**
 
Tables are structurally different from prose. I detect Markdown tables:
```python
def is_markdown_table(text: str):
    return "|" in text and "---" in text
```
 
Parse them into structured JSON:
```python
{
  "headers": ["Fiscal Year", "Revenue", "Net Income"],
  "rows": [
    {"Fiscal Year": "2023", "Revenue": "$394.3B", "Net Income": "$97.0B"},
    ...
  ],
  "metadata": {"fiscal_year": "2023", "source_file": "2023.pdf"}
}
```
 
Store separately in `tables.json` because:
- Vector search is bad at exact numeric queries
- Structured data needs structured retrieval
**4. Retry Logic with Exponential Backoff**
```python
@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def _embed(self, texts: List[str]):
    return self.embed_model.get_text_embedding_batch(texts)
```
 
Google's embedding API has rate limits. `tenacity` library handles retries automatically.
 
#### Phase 3: Agent Engine (`engine.py`)
 
**The Brain**: A ReActAgent with three tools:
 
**Tool 1: Vector Search (AppleFilingsSearch)**
```python
query_engine = index.as_query_engine(
    similarity_top_k=20,
    response_mode="tree_summarize",
)
```
For qualitative questions: *"What risks did Apple identify in 2022?"*
 
**Tool 2: Table Search (TableSearch)**
```python
class TableQueryEngine:
    def query(self, question: str) -> str:
        # Match fiscal year + keywords
        # Rank by relevance score
        # Return top 3 tables with metadata
```
For structured queries: *"Show me revenue breakdown by product line in 2023"*
 
**Tool 3: Calculator**
```python
def calculate(expression: str) -> str:
    safe_globals = {"sqrt": math.sqrt, "log": math.log, ...}
    result = eval(expression.strip(), safe_globals, {})
```
For computations: *"What's the CAGR of Apple's market cap from 2020 to 2025?"*
 
**ReAct Workflow**:
```
User: "Compute CAGR of Apple's revenue 2020-2025"
  → Agent: [Thought] I need revenue data for both years
  → Agent: [Action] TableSearch("revenue 2020")
  → Tool: Returns 2020 revenue table
  → Agent: [Action] TableSearch("revenue 2025")  
  → Tool: Returns 2025 revenue table
  → Agent: [Action] Calculator("((394.3/274.5)**(1/5) - 1) * 100")
  → Tool: "Result: 7.5234%"
  → Agent: [Answer] "Apple's revenue CAGR from 2020 to 2025 is 7.52%..."
```
 
#### Technical Challenges Solved
 
**Challenge 1: Async Event Loop Issues**
 
LlamaIndex's `ReActAgent` uses workflow-based execution with async internals. Running in a synchronous CLI caused:
```
RuntimeError: no running event loop
```
 
**Solution**: `nest_asyncio` to allow nested event loops:
```python
import nest_asyncio
nest_asyncio.apply()
 
def run_agent(agent, query):
    async def _execute():
        handler = agent.run(query)
        result = await handler
        return str(result.response)
    
    return asyncio.run(_execute())
```
 
**Challenge 2: OpenAI Quota Exhaustion**
 
Initial implementation used OpenAI embeddings. Hit rate limits immediately.
 
**Solution**: Switched to Google Gemini embeddings (`text-embedding-004`) which I was already using in ingestion. Consistent embedding model = better retrieval quality anyway.
 
**Challenge 3: Agent Not Retrieving Data**
 
Early versions returned *"I don't have access to that data"* despite having 50K+ chunks indexed.
 
**Root cause**: Poor tool descriptions. The agent didn't know WHEN to use each tool.
 
**Solution**: Explicit tool descriptions with examples:
```python
ToolMetadata(
    name="TableSearch",
    description=(
        "Search structured financial tables from Apple 10-K filings. "
        "Use for balance sheets, income statements, debt schedules. "
        "Specify fiscal year if known."
    ),
)
```
 
#### Why This Task Matters
 
**Document intelligence is a killer app for LLMs**: GPT-4 can read a 10-K, but it can't remember 6 years of filings. RAG bridges that gap.
 
**Hybrid retrieval is the future**: Pure vector search fails on exact queries (*"What was Q3 2022 revenue?"*). Pure SQL fails on semantic queries (*"What strategic risks did management discuss?"*). Combining both = 10x better UX.
 
**Production infrastructure matters**: The difference between a demo and a product is error handling, caching, checkpointing, and observability. This task forced me to think like a platform engineer, not just an ML experimenter.
 
---
 
## Overall Technical Stack
 
| Component | Technology | Why I Chose It |
|-----------|-----------|----------------|
| Language | Python 3.11 | Industry standard for data/AI |
| Risk Calculation | Pure Python + NumPy patterns | No dependencies = easier deployment |
| Market Data | `yfinance`, `requests` | Battle-tested, free tier available |
| LLM APIs | Google Gemini, OpenRouter | Gemini for cost, OpenRouter for model flexibility |
| Document Parsing | LlamaParse | Best-in-class for complex PDFs |
| Vector DB | ChromaDB | Lightweight, persistent, OSS |
| Embeddings | Google `text-embedding-004` | Fast, cheap, good quality |
| Agent Framework | LlamaIndex ReActAgent | Tool-calling + reasoning out of the box |
| CLI Tables | `tabulate` | Clean ASCII formatting |
| Logging | Python `logging` | Structured, filterable, production-ready |
 
---

## Hardest Part

The most challenging aspect was ensuring the AI-generated explanation remained consistent and followed a strict JSON format. LLMs can sometimes include conversational filler or markdown formatting that breaks code parsers.

**Solution:** Implemented a robust prompt using XML delimiters and utilized Gemini's `response_mime_type="application/json"` configuration. Also added a retry mechanism with **exponential backoff** to handle `503` errors during high-demand periods.

---

## AI Usage

AI coding tools (Gemini Pro/Claude) were used to assist with boilerplate code generation and to help refine the terminal table formatting. All logic was manually reviewed and adjusted to ensure mathematical correctness and proper error handling.
