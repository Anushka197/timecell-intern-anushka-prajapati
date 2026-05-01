"""
Tests for rebalancing allocation validation and tone prompt injection.

Covers:
  - Property 5: Rebalancing Allocation Validation (Requirements 6.3, 6.4)
  - Property 6: Tone Selector Prompt Injection (Requirements 7.4)
  - Unit tests for validate_allocations
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src_v2.core.rebalancing import validate_allocations
from src_v2.core.llm import build_advisor_prompt
from src_v2.core.models import Portfolio, Asset


# ── Property 5: Rebalancing Allocation Validation ─────────────────────────

@given(allocations=st.dictionaries(
    keys=st.text(min_size=1),
    values=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=10
))
@settings(max_examples=100)
def test_property_5_validate_allocations(allocations):
    """
    Property 5: Rebalancing Allocation Validation
    Validates: Requirements 6.3, 6.4
    Assert validate_allocations(d) is True iff 99.0 <= sum(d.values()) <= 101.0
    """
    total = sum(allocations.values())
    result = validate_allocations(allocations)
    expected = 99.0 <= total <= 101.0
    assert result == expected


# ── Property 6: Tone Selector Prompt Injection ────────────────────────────

TONE_INSTRUCTIONS = {
    "beginner": "Explain in simple terms, avoid jargon, use analogies.",
    "experienced": "Use standard financial terminology and be concise.",
    "expert": "Use advanced financial concepts, quantitative reasoning, and precise language.",
}


@given(tone=st.sampled_from(["beginner", "experienced", "expert"]))
@settings(max_examples=100)
def test_property_6_tone_prompt_injection(tone):
    """
    Property 6: Tone Selector Prompt Injection
    Validates: Requirements 7.4
    Assert the constructed prompt contains the tone-specific instruction text.
    Assert the prompt does NOT contain instruction text belonging to the other two tones.
    """
    portfolio = Portfolio(
        assets=[Asset("BTC", 50.0, -80.0), Asset("CASH", 50.0, 0.0)],
        total_value=100000.0,
        monthly_expenses=5000.0,
    )
    messages = build_advisor_prompt(portfolio, "reduce risk", tone)

    # Combine all message content for searching
    full_prompt = " ".join(m["content"] for m in messages)

    # The tone-specific instruction must be present
    assert TONE_INSTRUCTIONS[tone] in full_prompt, (
        f"Expected tone instruction for '{tone}' not found in prompt"
    )

    # The other tones' instructions must NOT be present
    other_tones = [t for t in TONE_INSTRUCTIONS if t != tone]
    for other_tone in other_tones:
        assert TONE_INSTRUCTIONS[other_tone] not in full_prompt, (
            f"Instruction for '{other_tone}' should not appear in prompt for tone '{tone}'"
        )


# ── Unit tests for validate_allocations ──────────────────────────────────

def test_validate_allocations_exactly_100():
    """Single asset at 100% is valid."""
    assert validate_allocations({"A": 100.0}) is True


def test_validate_allocations_exactly_99():
    """Sum of exactly 99.0 is valid (lower boundary)."""
    assert validate_allocations({"A": 99.0}) is True


def test_validate_allocations_exactly_101():
    """Sum of exactly 101.0 is valid (upper boundary)."""
    assert validate_allocations({"A": 101.0}) is True


def test_validate_allocations_below_99():
    """Sum of 98.9 is invalid (below lower boundary)."""
    assert validate_allocations({"A": 98.9}) is False


def test_validate_allocations_above_101():
    """Sum of 101.1 is invalid (above upper boundary)."""
    assert validate_allocations({"A": 101.1}) is False


def test_validate_allocations_empty_dict():
    """Empty dict has sum 0.0, which is invalid."""
    assert validate_allocations({}) is False
