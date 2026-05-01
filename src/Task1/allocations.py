# allocations.py

from typing import Dict, Any, List
from utils import get_valid_percentage, get_valid_crash
from display import generate_cli_bar_chart

def edit_asset(assets: List[Dict[str, Any]]):
    for i, a in enumerate(assets):
        print(f"{i+1}. {a['name']} ({a['allocation_pct']}%)")

    try:
        idx = int(input("Select asset #: ")) - 1
        assets[idx]["allocation_pct"] = get_valid_percentage("New allocation: ")
    except Exception:
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