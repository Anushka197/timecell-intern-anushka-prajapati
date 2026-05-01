# main.py

from typing import Dict, Any
from risk import _build_empty_response
from allocations import validate_and_fix_allocations
from display import display_risk_report

# config.py

SAFE_RUNWAY_MONTHS = 12
CONCENTRATION_LIMIT = 40

def get_portfolio_from_file(filepath: str = "input/t1.txt") -> Dict[str, Any]:
    print(f"\n--- Loading Portfolio Details from {filepath} ---")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Parse the Python dictionary from the file contents
        local_scope = {}
        exec(content, {}, local_scope)
        portfolio_data = local_scope.get("portfolio", {})
        
        total_value = portfolio_data.get("total_value_inr", 0)
        expenses = portfolio_data.get("monthly_expenses_inr", 0)
        assets = portfolio_data.get("assets", [])
        
        # Keep exact verification systems active
        assets = validate_and_fix_allocations(assets)
        
        return {
            "total_value_inr": total_value,
            "monthly_expenses_inr": expenses,
            "assets": assets
        }
        
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please ensure it exists.")
        return _build_empty_response()
    except Exception as e:
        print(f"Error parsing '{filepath}': {e}")
        return _build_empty_response()

if __name__ == "__main__":
    portfolio = get_portfolio_from_file("input.txt")
    display_risk_report(portfolio)