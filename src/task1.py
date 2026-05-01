from typing import Dict, Any, List

SAFE_RUNWAY_MONTHS = 12
CONCENTRATION_LIMIT = 40


# ================= CORE LOGIC =================

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

    severe_runway = max(severe_post_crash_value / expenses if expenses > 0 else float("inf"),0)
    moderate_runway = moderate_post_crash_value / expenses if expenses > 0 else float("inf")

    ruin_test = classify_risk(severe_runway)

    drawdown = min(((total_value - severe_post_crash_value) / total_value) * 100,100)
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


def classify_risk(runway: float) -> str:
    if runway < 6:
        return "HIGH RISK"
    elif runway < SAFE_RUNWAY_MONTHS:
        return "MODERATE RISK"
    else:
        return "SAFE"


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


# ================= DISPLAY =================

def format_inr(value: float) -> str:
    return f"₹{value:,.2f}"


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


# ================= INPUT HANDLING =================

def parse_percentage(value: str) -> float:
    return float(value.strip().replace("%", ""))


def get_valid_number(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid number. Try again.")


def get_valid_percentage(prompt: str) -> float:
    while True:
        try:
            val = parse_percentage(input(prompt))
            if val < 0:
                print("Cannot be negative.")
                continue
            return val
        except:
            print("Invalid input. Example: 40 or 40%")


def get_valid_crash(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()

        if raw == "":
            return 0.0

        try:
            val = parse_percentage(raw)

            if val > 0:
                print("Warning: crash should be negative. Converting.")
                val = -abs(val)

            if val < -100:
                print("Crash cannot exceed -100%. Capping at -100%.")
                val = -100

            return val

        except:
            print("Invalid crash value. Example: -40 or -25%")


# ================= ALLOCATION FIX SYSTEM =================

def validate_and_fix_allocations(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    while True:
        total_alloc = sum(a["allocation_pct"] for a in assets)

        if abs(total_alloc - 100) < 0.01:
            return assets

        print(f"\nWarning: Total allocation = {total_alloc}%")

        print(generate_cli_bar_chart(assets))

        print("\nOptions:")
        print("1. Edit an asset")
        print("2. Auto-normalize")
        print("3. Re-enter all")

        choice = input("Choose (1/2/3): ")

        if choice == "1":
            edit_asset(assets)
        elif choice == "2":
            return normalize_allocations(assets)
        elif choice == "3":
            return reenter_assets(len(assets))
        else:
            print("Invalid choice.")


def edit_asset(assets: List[Dict[str, Any]]):
    for i, a in enumerate(assets):
        print(f"{i+1}. {a['name']} ({a['allocation_pct']}%)")

    try:
        idx = int(input("Select asset #: ")) - 1
        assets[idx]["allocation_pct"] = get_valid_percentage("New allocation: ")
    except:
        print("Invalid selection.")


def normalize_allocations(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = sum(a["allocation_pct"] for a in assets)
    for a in assets:
        a["allocation_pct"] = (a["allocation_pct"] / total) * 100
    return assets


def reenter_assets(n: int) -> List[Dict[str, Any]]:
    assets = []
    for i in range(n):
        print(f"\nAsset {i+1}")
        name = input("Name: ")
        alloc = get_valid_percentage("Allocation %: ")
        crash = get_valid_crash("Crash %: ")

        assets.append({
            "name": name,
            "allocation_pct": alloc,
            "expected_crash_pct": crash
        })
    return assets


# ================= MAIN INPUT =================

def get_portfolio_from_user() -> Dict[str, Any]:
    print("\n--- Enter Portfolio Details ---")

    total_value = get_valid_number("Total value: ")
    expenses = get_valid_number("Monthly expenses: ")

    n = int(get_valid_number("Number of assets: "))

    assets = []
    for i in range(n):
        print(f"\nAsset {i+1}")
        name = input("Name: ")
        alloc = get_valid_percentage("Allocation %: ")
        crash = get_valid_crash("Crash %: ")

        assets.append({
            "name": name,
            "allocation_pct": alloc,
            "expected_crash_pct": crash
        })

    assets = validate_and_fix_allocations(assets)

    return {
        "total_value_inr": total_value,
        "monthly_expenses_inr": expenses,
        "assets": assets
    }


# ================= RUN =================

if __name__ == "__main__":
    portfolio = get_portfolio_from_user()
    display_risk_report(portfolio)