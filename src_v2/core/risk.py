"""
core/risk.py — Stress Engine

Implements scenario-based portfolio stress testing.
Requirements: 3.1, 3.2, 3.7
"""

from core.models import Asset, Portfolio, Scenario, ScenarioResult


def validate_crash_pct(crash_pct: float) -> bool:
    """Returns True iff -100.0 <= crash_pct <= 0.0 (inclusive on both ends)."""
    return -100.0 <= crash_pct <= 0.0


def compute_scenario(portfolio: Portfolio, scenario: Scenario) -> ScenarioResult:
    """
    Applies scenario crash percentages to portfolio assets.

    Formula:
        asset_value = total_value * allocation_pct / 100
        post_value  = asset_value * (1 - abs(crash_pct) / 100)
        post_crash_value = sum of all post_values

    Assets not in scenario.asset_crashes receive crash_pct = 0 (no loss).

    - runway_months = post_crash_value / monthly_expenses
                      (float('inf') if monthly_expenses == 0)
    - ruin_test = "PASS" if runway_months > 12 else "FAIL"
    - largest_risk_asset = asset name with highest absolute dollar loss
                           (empty string if no assets)
    - concentration_warning = True if any asset has allocation_pct > 40
    - asset_losses = {asset_name: crash_pct_applied} for each asset
    """
    post_crash_value = 0.0
    asset_losses: dict[str, float] = {}
    largest_loss_amount = 0.0
    largest_risk_asset = ""
    concentration_warning = False

    for asset in portfolio.assets:
        # Check concentration warning
        if asset.allocation_pct > 40:
            concentration_warning = True

        # Determine crash percentage for this asset
        crash_pct = scenario.asset_crashes.get(asset.name, 0.0)

        # Compute asset value and post-crash value
        asset_value = portfolio.total_value * asset.allocation_pct / 100.0
        post_value = asset_value * (1.0 - abs(crash_pct) / 100.0)
        post_crash_value += post_value

        # Track the crash pct applied
        asset_losses[asset.name] = crash_pct

        # Track largest absolute dollar loss
        dollar_loss = asset_value - post_value  # always >= 0
        if dollar_loss > largest_loss_amount:
            largest_loss_amount = dollar_loss
            largest_risk_asset = asset.name

    # Compute runway months
    if portfolio.monthly_expenses == 0:
        runway_months = float("inf")
    else:
        runway_months = post_crash_value / portfolio.monthly_expenses

    # Ruin test: float('inf') > 12 is True, so inf → PASS
    ruin_test = "PASS" if runway_months > 12 else "FAIL"

    return ScenarioResult(
        scenario_name=scenario.name,
        post_crash_value=post_crash_value,
        runway_months=runway_months,
        ruin_test=ruin_test,
        largest_risk_asset=largest_risk_asset,
        concentration_warning=concentration_warning,
        asset_losses=asset_losses,
    )


def compute_all_scenarios(
    portfolio: Portfolio, scenarios: list[Scenario]
) -> list[ScenarioResult]:
    """Runs compute_scenario for each scenario. Returns results in same order."""
    return [compute_scenario(portfolio, scenario) for scenario in scenarios]


def build_loss_matrix(
    portfolio: Portfolio, scenarios: list[Scenario]
) -> dict[str, dict[str, float]]:
    """
    Returns nested dict: asset_name -> scenario_name -> crash_pct_applied.
    Used to populate the Plotly heatmap.
    """
    matrix: dict[str, dict[str, float]] = {}

    for asset in portfolio.assets:
        matrix[asset.name] = {}
        for scenario in scenarios:
            crash_pct = scenario.asset_crashes.get(asset.name, 0.0)
            matrix[asset.name][scenario.name] = crash_pct

    return matrix
