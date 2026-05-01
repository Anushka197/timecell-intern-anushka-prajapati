# display.py

from typing import Dict, Any, List
from utils import format_inr
from risk import compute_risk_metrics

def generate_cli_bar_chart(assets: List[Dict[str, Any]]) -> str:
    lines = ["\n--- Portfolio Allocation ---"]
    max_width = 40

    for asset in assets:
        name = asset.get("name", "Unknown")
        alloc = asset.get("allocation_pct", 0)
        bar = "█" * int((alloc / 100) * max_width)
        lines.append(f"{name:<10} | {bar:<40} {alloc:.2f}%")

    return "\n".join(lines)

def display_risk_report(portfolio: Dict[str, Any]):
    metrics = compute_risk_metrics(portfolio)

    print("\n" + "=" * 60)
    print(f"{'PORTFOLIO RISK REPORT':^60}")
    print("=" * 60)

    print(f"\n{'METRIC':<25} | {'SEVERE':<12} | {'MODERATE':<12}")
    print("-" * 60)

    print(f"{'Post-Crash Value':<25} | {format_inr(metrics['post_crash_value']):<12} | {format_inr(metrics['moderate_scenario']['post_crash_value']):<12}")
    print(f"{'Runway':<25} | {metrics['runway_months']} mo{'':<6} | {metrics['moderate_scenario']['runway_months']} mo")

    print("-" * 60)
    print(f"Risk Level              : {metrics['ruin_test']}")
    print(f"Largest Risk Asset      : {metrics['largest_risk_asset']}")
    print(f"Concentration Warning   : {metrics['concentration_warning']}")
    print(f"Drawdown %              : {metrics['drawdown_pct']}%")
    print(f"Safety Ratio            : {metrics['safety_ratio']}")

    print(generate_cli_bar_chart(portfolio.get("assets", [])))
    print("=" * 60 + "\n")