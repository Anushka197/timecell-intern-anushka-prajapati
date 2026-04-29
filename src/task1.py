from typing import Dict, Any, List

def compute_risk_metrics(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes key risk metrics for a given portfolio under severe and moderate crash scenarios.
    Runs in O(N) time where N is the number of assets.
    """
    total_value = portfolio.get("total_value_inr", 0)
    expenses = portfolio.get("monthly_expenses_inr", 0)
    assets = portfolio.get("assets", [])

    # Edge case: Empty portfolio
    if not assets or total_value <= 0:
        return _build_empty_response()

    severe_post_crash_value = 0.0
    moderate_post_crash_value = 0.0
    
    max_risk_score = -1.0
    largest_risk_asset = "None"
    concentration_warning = False

    for asset in assets:
        name = asset.get("name", "Unknown")
        alloc_pct = asset.get("allocation_pct", 0)
        crash_pct = asset.get("expected_crash_pct", 0)

        if alloc_pct > 40:
            concentration_warning = True

        crash_magnitude = abs(crash_pct)
        risk_score = alloc_pct * crash_magnitude
        
        if risk_score > max_risk_score:
            max_risk_score = risk_score
            largest_risk_asset = name

        asset_value = total_value * (alloc_pct / 100.0)
        
        severe_loss = asset_value * (crash_magnitude / 100.0)
        severe_post_crash_value += (asset_value - severe_loss)
        
        moderate_loss = asset_value * ((crash_magnitude / 2.0) / 100.0)
        moderate_post_crash_value += (asset_value - moderate_loss)

    severe_runway = (severe_post_crash_value / expenses) if expenses > 0 else float('inf')
    moderate_runway = (moderate_post_crash_value / expenses) if expenses > 0 else float('inf')

    ruin_test = 'PASS' if severe_runway > 12 else 'FAIL'

    return {
        "post_crash_value": round(severe_post_crash_value, 2),
        "runway_months": round(severe_runway, 2),
        "ruin_test": ruin_test,
        "largest_risk_asset": largest_risk_asset,
        "concentration_warning": concentration_warning,
        "moderate_scenario": {
            "post_crash_value": round(moderate_post_crash_value, 2),
            "runway_months": round(moderate_runway, 2)
        }
    }

def _build_empty_response() -> Dict[str, Any]:
    """Helper to return a safe default if the portfolio is empty or invalid."""
    return {
        "post_crash_value": 0, "runway_months": 0, "ruin_test": 'FAIL',
        "largest_risk_asset": "None", "concentration_warning": False,
        "moderate_scenario": {"post_crash_value": 0, "runway_months": 0}
    }

def generate_cli_bar_chart(assets: List[Dict[str, Any]]) -> str:
    """Generates a simple text-based bar chart for asset allocation."""
    if not assets:
        return "No assets to display."
        
    lines = ["\n--- Portfolio Allocation ---"]
    for asset in assets:
        name = asset.get("name", "Unknown")
        alloc = asset.get("allocation_pct", 0)
        bar = "█" * int(alloc // 2) 
        lines.append(f"{name:<10} | {bar:<25} {alloc}%")
    
    return "\n".join(lines)

def display_risk_report(portfolio: Dict[str, Any]):
    """Formats and prints the final risk report, showing scenarios side-by-side."""
    metrics = compute_risk_metrics(portfolio)
    
    print("\n" + "="*55)
    print(f"{'PORTFOLIO RISK REPORT':^55}")
    print("="*55)
    
    print(f"\n{'METRIC':<25} | {'SEVERE CRASH':<12} | {'MODERATE CRASH':<12}")
    print("-" * 55)
    
    severe_val = f"₹{metrics['post_crash_value']:,.2f}"
    mod_val = f"₹{metrics['moderate_scenario']['post_crash_value']:,.2f}"
    print(f"{'Post-Crash Value':<25} | {severe_val:<12} | {mod_val:<12}")
    
    severe_runway = f"{metrics['runway_months']} mo"
    mod_runway = f"{metrics['moderate_scenario']['runway_months']} mo"
    print(f"{'Runway':<25} | {severe_runway:<12} | {mod_runway:<12}")
    
    print("-" * 55)
    print(f"Ruin Test (>12 mo runway) : {metrics['ruin_test']}")
    print(f"Largest Risk Asset        : {metrics['largest_risk_asset']}")
    print(f"Concentration Warning     : {metrics['concentration_warning']} (>40% in one asset)")
    
    print(generate_cli_bar_chart(portfolio.get("assets", [])))
    print("="*55 + "\n")

if __name__ == "__main__":
    sample_portfolio = {
        "total_value_inr": 10_000_000, 
        "monthly_expenses_inr": 80_000,
        "assets": [
            {"name": "BTC", "allocation_pct": 30, "expected_crash_pct": -80},
            {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
            {"name": "GOLD", "allocation_pct": 20, "expected_crash_pct": -15},
            {"name": "CASH", "allocation_pct": 10, "expected_crash_pct": 0},
        ]
    }
    
    display_risk_report(sample_portfolio)