"""
core/llm.py — Ollama client wrapper

All LLM and embedding calls go through this module.
No other module calls Ollama directly.
"""

import json
import requests

from core.models import Portfolio

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_TIMEOUT = 90  # seconds

# ── Custom exceptions ──────────────────────────────────────────────────────

class OllamaTimeoutError(Exception):
    """Raised when an Ollama request exceeds the configured timeout."""
    ...


class OllamaConnectionError(Exception):
    """Raised when the Ollama server is unreachable."""
    ...


# ── Tone instruction map ───────────────────────────────────────────────────

_TONE_INSTRUCTIONS: dict[str, str] = {
    "beginner": "Explain in simple terms, avoid jargon, use analogies.",
    "experienced": "Use standard financial terminology and be concise.",
    "expert": "Use advanced financial concepts, quantitative reasoning, and precise language.",
}


# ── Health check ───────────────────────────────────────────────────────────

def check_health() -> bool:
    """
    GET /api/tags — returns True if Ollama server is reachable, False otherwise.
    Never raises; all exceptions are caught and converted to False.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


# ── Chat ───────────────────────────────────────────────────────────────────

def chat(
    messages: list[dict[str, str]],
    model: str = CHAT_MODEL,
    temperature: float = 0.3,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """
    POST /api/chat with stream=False.
    Returns the assistant message content as a plain string.
    Raises OllamaTimeoutError after `timeout` seconds.
    Raises OllamaConnectionError if server is unreachable.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.Timeout:
        raise OllamaTimeoutError(
            f"Ollama chat request timed out after {timeout} seconds."
        )
    except requests.ConnectionError as exc:
        raise OllamaConnectionError(
            f"Could not connect to Ollama server at {OLLAMA_BASE_URL}: {exc}"
        )


# ── Embeddings ─────────────────────────────────────────────────────────────

def embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    """
    POST /api/embeddings with body {"model": model, "prompt": text}.
    Returns the embedding vector as a list of floats.
    """
    payload = {"model": model, "prompt": text}
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json=payload,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Calls embed() for each text. Returns list of embedding vectors."""
    return [embed(text, model=model) for text in texts]


# ── Prompt builders ────────────────────────────────────────────────────────

def build_advisor_prompt(
    portfolio: Portfolio,
    goal: str,
    tone: str,
) -> list[dict[str, str]]:
    """
    Constructs the LLM messages list for the rebalancing advisor.

    Each tone maps to a unique instruction string:
    - "beginner":    "Explain in simple terms, avoid jargon, use analogies."
    - "experienced": "Use standard financial terminology and be concise."
    - "expert":      "Use advanced financial concepts, quantitative reasoning, and precise language."

    The system message contains ONLY the tone-specific instruction.
    The user message contains the portfolio summary and the user's goal.

    Returns a list of message dicts: [{"role": "system", ...}, {"role": "user", ...}]
    """
    tone_instruction = _TONE_INSTRUCTIONS.get(
        tone,
        _TONE_INSTRUCTIONS["experienced"],  # safe default
    )

    # Build a human-readable portfolio summary
    asset_lines = "\n".join(
        f"  - {asset.name}: {asset.allocation_pct:.1f}%"
        for asset in portfolio.assets
    )
    portfolio_summary = (
        f"Total portfolio value: {portfolio.total_value:,.2f}\n"
        f"Monthly expenses: {portfolio.monthly_expenses:,.2f}\n"
        f"Current allocations:\n{asset_lines}"
    )

    user_message = (
        f"Here is my current portfolio:\n\n"
        f"{portfolio_summary}\n\n"
        f"My rebalancing goal: {goal}\n\n"
        f"Please provide a rebalancing plan with new allocation percentages for each asset "
        f"(they must sum to 100%) and a clear rationale. "
        f"Return the allocations as a JSON object with asset names as keys and percentages as values, "
        f"followed by your explanation."
    )

    return [
        {"role": "system", "content": tone_instruction},
        {"role": "user", "content": user_message},
    ]


def build_critic_prompt(plan: dict, goal: str) -> list[dict[str, str]]:
    """
    Constructs the LLM messages list for the critic review.

    Asks the LLM to review the rebalancing plan for mathematical consistency
    and flag any allocation that violates the user's stated constraints.

    Returns a list of message dicts.
    """
    plan_text = json.dumps(plan, indent=2)

    system_message = (
        "You are a rigorous financial risk analyst. "
        "Your job is to critically review rebalancing plans for mathematical consistency "
        "and constraint violations. Be precise and concise."
    )

    user_message = (
        f"Please review the following rebalancing plan:\n\n"
        f"{plan_text}\n\n"
        f"The user's stated goal was: {goal}\n\n"
        f"Check for:\n"
        f"1. Mathematical consistency: do the allocations sum to 100%?\n"
        f"2. Constraint violations: does any allocation violate the user's stated constraints?\n"
        f"3. Risk concerns: are there any concentration risks or other issues?\n\n"
        f"Provide a concise critique with specific findings."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
