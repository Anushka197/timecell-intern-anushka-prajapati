"""
core/rebalancing.py — Rebalancing allocation validation helpers

Validates: Requirements 6.3, 6.4
"""


def validate_allocations(allocations: dict[str, float]) -> bool:
    """
    Returns True iff 99.0 <= sum(allocations.values()) <= 101.0

    Validates: Requirements 6.3, 6.4
    """
    total = sum(allocations.values())
    return 99.0 <= total <= 101.0
