"""
engine.py — Portfolio Intelligence Engine
==========================================
Builds the ReActAgent with two tools:
  1. VectorIndexTool  — semantic search over Apple 10-K filings.
  2. CalculatorTool   — precise arithmetic for financial metrics.

Usage:
    from engine import build_agent
    agent = build_agent(index)
    response = agent.chat("What was Apple's R&D spend in 2023?")
"""

import os
import math
import logging
from typing import Any

from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms.openrouter import OpenRouter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("PortfolioEngine")

load_dotenv()

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

def build_llm() -> OpenRouter:
    """
    Instantiate the OpenRouter-backed LLM.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set in the environment.")

    llm = OpenRouter(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        temperature=0.1, 
        max_tokens=4096,
        context_window=128000,
        additional_kwargs={
            "extra_headers": {
                "HTTP-Referer": "https://portfolio-intelligence-engine",
                "X-Title": "Portfolio Intelligence Engine",
            }
        },
    )
    logger.info("LLM configured: openai/gpt-4o-mini via OpenRouter.")
    return llm


# ---------------------------------------------------------------------------
# Calculator Tool
# ---------------------------------------------------------------------------

def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression string and return the result.

    Supports standard arithmetic operators (+, -, *, /, **, %) and common
    math functions (sqrt, log, abs, round, etc.).

    Args:
        expression: A Python-compatible math expression string.
                    Examples:
                      "((29915 - 21914) / 21914) * 100"   → percentage change
                      "sqrt(144)"                          → 12.0
                      "round(1234567 / 1000000, 2)"        → 1.23

    Returns:
        A string with the numeric result or an error message.
    """
    # Whitelist of safe names available inside eval
    safe_globals: dict[str, Any] = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    try:
        # Strip whitespace and validate the expression is not empty
        expression = expression.strip()
        if not expression:
            return "Error: Empty expression provided."

        result = eval(expression, safe_globals, {})  # noqa: S307 — sandboxed eval

        # Format large numbers with commas for readability
        if isinstance(result, float):
            formatted = f"{result:,.4f}".rstrip("0").rstrip(".")
        elif isinstance(result, int):
            formatted = f"{result:,}"
        else:
            formatted = str(result)

        logger.info("Calculator: '%s' = %s", expression, formatted)
        return f"Result: {formatted}"

    except ZeroDivisionError:
        return "Error: Division by zero."
    except (SyntaxError, NameError, TypeError) as exc:
        return f"Error: Invalid expression — {exc}"
    except Exception as exc:
        return f"Error: {exc}"


def build_calculator_tool() -> FunctionTool:
    """Wrap the calculate function as a LlamaIndex FunctionTool."""
    return FunctionTool.from_defaults(
        fn=calculate,
        name="Calculator",
        description=(
            "Use this tool to perform precise mathematical calculations. "
            "Input must be a valid Python arithmetic expression as a string. "
            "Use this for: percentage changes, growth rates, CAGR, margins, "
            "ratios, and any numeric computation derived from financial data. "
            "Example inputs: '((29915 - 21914) / 21914) * 100' for % change, "
            "'(26.0 / 394.3) * 100' for a margin percentage."
        ),
    )


# ---------------------------------------------------------------------------
# Query Engine Tool
# ---------------------------------------------------------------------------

def build_query_engine_tool(index: VectorStoreIndex) -> QueryEngineTool:
    """
    Create a retrieval tool from the VectorStoreIndex.

    Retrieves the top-8 most relevant chunks and applies a metadata filter
    so the agent can target specific fiscal years when needed.
    """
    query_engine = index.as_query_engine(
        similarity_top_k=8,
        response_mode="tree_summarize",   # Synthesises across multiple chunks
        verbose=True,
    )

    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="AppleFilingsSearch",
            description=(
                "Search Apple Inc. 10-K annual filings (fiscal years 2020–2025). "
                "Use this tool to retrieve financial data, metrics, narratives, "
                "and tables from the filings. "
                "For trend analysis across multiple years, make separate queries "
                "per year (e.g., 'Apple R&D expense 2021', 'Apple R&D expense 2024') "
                "then use the Calculator tool to compute changes. "
                "Always include the fiscal year in your query for precision."
            ),
        ),
    )


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

def build_agent(index: VectorStoreIndex) -> ReActAgent:
    """
    Assemble the ReActAgent with:
      - AppleFilingsSearch tool (vector retrieval)
      - Calculator tool (precise arithmetic)
    """
    llm = build_llm()

    # Wire the LLM into global Settings so all components use it
    Settings.llm = llm

    # Debug handler captures every reasoning step for the UI
    debug_handler = LlamaDebugHandler(print_trace_on_end=False)
    callback_manager = CallbackManager(handlers=[debug_handler])
    Settings.callback_manager = callback_manager

    tools = [
        build_query_engine_tool(index),
        build_calculator_tool(),
    ]

    system_prompt = """You are a Senior Financial Analyst AI for a Family Office, specialising in Apple Inc. equity research.

    You have access to Apple's 10-K annual filings from fiscal years 2020 through 2025.

    ## Your Analytical Approach
    1. **Retrieve first**: Always search the filings for exact figures before calculating.
    2. **Calculate precisely**: Use the Calculator tool for ALL arithmetic — never compute numbers in your head.
    3. **Cite sources**: Every data point must reference the fiscal year and, where available, the page/section.
    4. **Trend analysis**: For multi-year comparisons, query each year separately, then calculate the change.
    5. **Be thorough**: For complex questions, break them into sub-questions and answer each systematically.

    ## Output Format
    - Lead with a concise executive summary (2-3 sentences).
    - Follow with detailed findings, organised by year where relevant.
    - CRITICAL: Present all numerical data, multi-year comparisons, and financial metrics in clean, visually appealing Markdown tables.
    - CRITICAL: You MUST include a blank empty line immediately before and after every table so the UI renders it properly.
    - End with a "Sources" section listing fiscal year, document section, and page numbers.
    - Flag any data limitations or assumptions clearly.

    ## Important Notes
    - Apple's fiscal year ends in late September (e.g., FY2023 ended September 30, 2023).
    - All dollar figures are in millions USD unless stated otherwise.
    - R&D is reported as "Research and Development" in the filings.
    """

    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=15,
        system_prompt=system_prompt,
        callback_manager=callback_manager,
    )

    logger.info(
        "ReActAgent built with %d tools: %s",
        len(tools),
        [t.metadata.name for t in tools],
    )
    return agent, debug_handler