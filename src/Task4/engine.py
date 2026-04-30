"""
engine.py — Portfolio Intelligence Engine
==========================================
Builds the ReActAgent with two tools:
  1. VectorIndexTool  — semantic search over Apple 10-K filings.
  2. CalculatorTool   — precise arithmetic for financial metrics.
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
from llama_index.core.base.llms.types import ChatMessage, MessageRole
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
    safe_globals: dict[str, Any] = {
        "__builtins__": {}, "abs": abs, "round": round, "min": min,
        "max": max, "sum": sum, "pow": pow, "sqrt": math.sqrt,
        "log": math.log, "log10": math.log10, "exp": math.exp,
        "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor,
    }

    try:
        expression = expression.strip()
        if not expression:
            return "Error: Empty expression provided."

        result = eval(expression, safe_globals, {})  # noqa: S307

        if isinstance(result, float):
            formatted = f"{result:,.4f}".rstrip("0").rstrip(".")
        elif isinstance(result, int):
            formatted = f"{result:,}"
        else:
            formatted = str(result)

        logger.info("Calculator: '%s' = %s", expression, formatted)
        return f"Result: {formatted}"

    except Exception as exc:
        return f"Error: {exc}"

def build_calculator_tool() -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=calculate,
        name="Calculator",
        description=(
            "Use this tool to perform precise mathematical calculations. "
            "Input must be a valid Python arithmetic expression as a string."
        ),
    )

# ---------------------------------------------------------------------------
# Query Engine Tool
# ---------------------------------------------------------------------------

def build_query_engine_tool(index: VectorStoreIndex) -> QueryEngineTool:
    query_engine = index.as_query_engine(
        similarity_top_k=8,
        response_mode="tree_summarize",
        verbose=False,
    )

    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="AppleFilingsSearch",
            description=(
                "Search Apple Inc. 10-K annual filings (fiscal years 2020–2025). "
                "Always include the fiscal year in your query for precision."
            ),
        ),
    )

# ---------------------------------------------------------------------------
# ReAct Agent & Critic Loop
# ---------------------------------------------------------------------------

def build_agent(index: VectorStoreIndex) -> tuple[ReActAgent, LlamaDebugHandler]:
    llm = build_llm()
    Settings.llm = llm

    debug_handler = LlamaDebugHandler(print_trace_on_end=False)
    callback_manager = CallbackManager(handlers=[debug_handler])
    Settings.callback_manager = callback_manager

    tools = [
        build_query_engine_tool(index),
        build_calculator_tool(),
    ]

    # REFINED CALCULATOR ENFORCEMENT: The system prompt dictates deterministic math.
    system_prompt = """You are a Senior Financial Analyst AI for a Family Office.
    You have access to Apple's 10-K annual filings from 2020 through 2025.

    ## DETERMINISTIC MATH REQUIREMENT (CRITICAL)
    You are strictly forbidden from performing mental arithmetic.
    For ANY calculation (subtraction, percentage change, margins, CAGRs), you MUST use the `Calculator` tool. 
    If you output a calculated number without using the tool, you have failed the integrity check.

    ## Output Format
    - Executive summary (2-3 sentences).
    - Present numerical data in clean Markdown tables (include blank lines before/after tables).
    - Cite sources (Fiscal Year, page/section).
    """

    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=False,
        max_iterations=30,
        system_prompt=system_prompt,
        callback_manager=callback_manager,
    )

    return agent, debug_handler

def query_with_critic(user_query: str, agent: ReActAgent) -> str:
    """
    Executes the Critic-in-the-Loop workflow.
    1. Generates a draft response using the tools.
    2. Passes the draft to a Critic LLM to identify assumptions and assign conviction.
    """
    # Step 1: Draft Analysis
    draft_response = agent.chat(user_query)
    
    # Step 2: The Scrutiny Layer
    llm = Settings.llm
    critic_prompt = f"""
    You are a ruthless but constructive Chief Investment Officer reviewing a subordinate's analysis.
    
    USER QUESTION: {user_query}
    DRAFT ANALYSIS: 
    {draft_response}
    
    Identify the weaknesses in this draft. Format your response exactly as follows:
    
    ### 🛑 Critic's Scrutiny
    **1. Missing Context:** (Identify any potential blind spots or footnotes likely missing from this analysis).
    **2. Critical Assumption:** (State one underlying assumption the draft makes that, if wrong, would invalidate the conclusion).
    **3. Conviction Score:** (Assign a score from 0-100% based on mathematical traceability and data completeness. Be harsh).
    """
    
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a precise financial critic."),
        ChatMessage(role=MessageRole.USER, content=critic_prompt)
    ]
    
    critic_feedback = llm.chat(messages)
    
    # Step 3: Combine for the final output
    return f"{draft_response}\n\n{critic_feedback.message.content}"