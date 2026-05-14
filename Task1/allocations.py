# allocations.py

from typing import Dict, Any

def is_valid_portfolio_structure(p_data: Dict[str, Any]) -> bool:
    total_value = p_data.get("total_value_inr", 0)
    expenses = p_data.get("monthly_expenses_inr", 0)
    assets = p_data.get("assets", [])

    if not isinstance(total_value, (int, float)) or total_value <= 0:
        return False
    if not isinstance(expenses, (int, float)) or expenses < 0:
        return False
    if not assets or not isinstance(assets, list):
        return False

    # Check for negative allocations and calculate total
    total_alloc = 0
    for a in assets:
        alloc = a.get("allocation_pct", 0)
        if alloc < 0:
            return False  # No negative allocations allowed
        total_alloc += alloc

    # Check if total is exactly 100%
    if abs(total_alloc - 100) > 0.01:
        return False

    return True