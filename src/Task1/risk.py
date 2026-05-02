# risk.py

from typing import Dict, Any

SAFE_RUNWAY_MONTHS = 12
CONCENTRATION_LIMIT = 40

def _build_empty_response() -> Dict[str, Any]:
    return {
        "post_crash_value": 0,
        "runway_months": 0,
        "ruin_test": "FAIL",
        "largest_risk_asset": "None",
        "concentration_warning": False,
        "drawdown_pct": 0,
        "safety_ratio": 0,
        "moderate_scenario": {"post_crash_value": 0, "runway_months": 0},
    }

def classify_risk(runway: float) -> str:
    if runway < 6:
        return "HIGH RISK"
    elif runway < SAFE_RUNWAY_MONTHS:
        return "MODERATE RISK"
    else:
        return "SAFE"

def compute_risk_metrics(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    total_value = portfolio.get("total_value_inr", 0)
    expenses = portfolio.get("monthly_expenses_inr", 0)
    assets = portfolio.get("assets", [])

    if not assets or total_value <= 0:
        return _build_empty_response()

    total_alloc = sum(a.get("allocation_pct", 0) for a in assets)
    if total_alloc == 0:
        return _build_empty_response()

    severe_post_crash_value = 0.0
    moderate_post_crash_value = 0.0

    max_risk_value = -1
    largest_risk_asset = "None"
    concentration_warning = False

    for asset in assets:
        name = asset.get("name", "Unknown")

        alloc_pct = asset.get("allocation_pct", 0)
        crash_pct = abs(asset.get("expected_crash_pct", 0))

        if alloc_pct > CONCENTRATION_LIMIT:
            concentration_warning = True

        asset_value = total_value * (alloc_pct / 100)
        risk_value = asset_value * (crash_pct / 100)

        if risk_value > max_risk_value:
            max_risk_value = risk_value
            largest_risk_asset = name

        post_value = asset_value * (1 - crash_pct / 100)
        severe_post_crash_value += max(post_value, 0)
        moderate_post_crash_value += asset_value * (1 - (crash_pct / 2) / 100)

    severe_runway = max(severe_post_crash_value / expenses if expenses > 0 else float("inf"), 0)
    moderate_runway = moderate_post_crash_value / expenses if expenses > 0 else float("inf")

    ruin_test = classify_risk(severe_runway)

    drawdown = min(((total_value - severe_post_crash_value) / total_value) * 100, 100)
    safety_ratio = severe_post_crash_value / (12 * expenses) if expenses > 0 else float("inf")

    return {
        "post_crash_value": round(severe_post_crash_value, 2),
        "runway_months": round(severe_runway, 2),
        "ruin_test": ruin_test,
        "largest_risk_asset": largest_risk_asset,
        "concentration_warning": concentration_warning,
        "drawdown_pct": round(drawdown, 2),
        "safety_ratio": round(safety_ratio, 2),
        "moderate_scenario": {
            "post_crash_value": round(moderate_post_crash_value, 2),
            "runway_months": round(moderate_runway, 2),
        },
    }