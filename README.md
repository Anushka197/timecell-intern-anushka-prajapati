# imecell-intern-anushka-prajapati

A collection of wealth management tools that help investors calculate portfolio risks during market crashes, track live asset prices, and receive AI-generated financial advice in simple, everyday language.

## Submission Links

- **Loom Video Walkthrough:** [Link to the Loom/video here]
- **GitHub Repository:** https://github.com/your-username/timecell-intern-anushka-prajapati

---

## Setup and Installation

This project requires **Python 3.10+**.

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .lvenv
   ```
3. Activate the environment:
   ```bash
   source .lvenv/bin/activate
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

## Hardest Part

The most challenging aspect was ensuring the AI-generated explanation remained consistent and followed a strict JSON format. LLMs can sometimes include conversational filler or markdown formatting that breaks code parsers.

**Solution:** Implemented a robust prompt using XML delimiters and utilized Gemini's `response_mime_type="application/json"` configuration. Also added a retry mechanism with **exponential backoff** to handle `503` errors during high-demand periods.

---

## AI Usage

AI coding tools (Claude/Copilot) were used to assist with boilerplate code generation and to help refine the terminal table formatting. All logic was manually reviewed and adjusted to ensure mathematical correctness and proper error handling.