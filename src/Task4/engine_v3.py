"""
engine_v3.py — 100% Local Portfolio Intelligence Engine (with Critic Loop)
==========================================================================
"""

import math
import logging
from typing import Any

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from llama_index.llms.ollama import Ollama

logger = logging.getLogger("PortfolioEngine_V3")


# ---------------------------------------------------------------------------
# LLM Configuration (LOCAL)
# ---------------------------------------------------------------------------

def build_llm() -> Ollama:
    llm = Ollama(
        model="llama3.1",
        request_timeout=300.0,  # increased for local inference
    )
    logger.info("LLM configured: Local Llama 3.1 via Ollama.")
    return llm


# ---------------------------------------------------------------------------
# Calculator Tool
# ---------------------------------------------------------------------------

def calculate(expression: str) -> str:
    safe_globals: dict[str, Any] = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "sqrt": math.sqrt,
        "pi": math.pi, "ceil": math.ceil, "floor": math.floor,
    }

    try:
        result = eval(expression, safe_globals, {})

        if isinstance(result, float):
            formatted = f"{result:,.6f}".rstrip("0").rstrip(".")
        else:
            formatted = str(result)

        logger.info("Calculator: %s = %s", expression, formatted)
        return f"Result: {formatted}"

    except Exception as exc:
        return f"Error: {exc}"


def build_calculator_tool() -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=calculate,
        name="Calculator",
        description="Use for ALL math. Input must be a valid Python expression.",
    )


# ---------------------------------------------------------------------------
# Query Tool
# ---------------------------------------------------------------------------

def build_query_tool(index: VectorStoreIndex) -> QueryEngineTool:
    return QueryEngineTool(
        query_engine=index.as_query_engine(similarity_top_k=8),
        metadata=ToolMetadata(
            name="AppleFilings",
            description="Search Apple 10-K filings (2020–2025). Always include year.",
        ),
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def build_agent(index: VectorStoreIndex) -> ReActAgent:
    Settings.llm = build_llm()

    tools = [
        build_query_tool(index),
        build_calculator_tool(),
    ]

    system_prompt = """
You are a Local Financial Analyst AI.

STRICT RULES:
1. You MUST use AppleFilings to retrieve financial data.
2. You MUST use Calculator for ALL calculations.
3. NEVER estimate or hallucinate numbers.
4. Final answers MUST match calculator outputs exactly.

Output:
- Clear reasoning
- Final numerical answer
"""

    return ReActAgent(
        tools=tools,
        llm=Settings.llm,
        verbose=True,
        max_iterations=20,
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Critic-in-the-Loop (ASYNC)
# ---------------------------------------------------------------------------

async def query_with_critic(user_query: str, agent: ReActAgent) -> str:
    """
    Async Critic Loop:
    1. Run agent (tool-using)
    2. Critic evaluates output
    3. Return combined response
    """

    # Step 1: Draft response (agent)
    draft_result = await agent.run(user_query)
    draft_response = (
        draft_result if isinstance(draft_result, str)
        else draft_result.response
    )

    # Step 2: Critic
    llm = Settings.llm

    critic_prompt = f"""
You are a ruthless but precise financial critic.

USER QUESTION:
{user_query}

DRAFT ANSWER:
{draft_response}

Analyze strictly:

### 🛑 Critic's Scrutiny
1. Missing Context:
2. Critical Assumption:
3. Numerical Consistency Check:
4. Conviction Score (0-100%):
"""

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a strict financial auditor."),
        ChatMessage(role=MessageRole.USER, content=critic_prompt),
    ]

    critic_response = await llm.achat(messages)

    # Step 3: Combine
    return f"{draft_response}\n\n{critic_response.message.content}"