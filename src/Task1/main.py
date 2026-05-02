# main.py

from typing import Dict, Any, List
from allocations import is_valid_portfolio_structure
from display import display_risk_report

def get_portfolios_from_file(filepath: str = "input.txt") -> List[Dict[str, Any]]:
    print(f"\n--- Loading Portfolios from {filepath} ---")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        local_scope = {}
        exec(content, {}, local_scope)
        return local_scope.get("portfolios", [])
        
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        return []
    except Exception as e:
        print(f"Error parsing '{filepath}': {e}")
        return []

if __name__ == "__main__":
    raw_portfolios = get_portfolios_from_file("input.txt")
    
    if not raw_portfolios:
        print("Warning: No 'portfolios' list found in file.")
        
    for p_data in raw_portfolios:
        p_name = p_data.get("name", "Unnamed")
        print(f"\nEvaluating: {p_name}")
        
        # 1. Check if portfolio is valid
        if not is_valid_portfolio_structure(p_data):
            print(f"Warning [{p_name}]: Invalid portfolio. Allocations must sum exactly to 100% with no negative values.")
            # Clear assets so risk.py safely generates a FAIL report without crashing
            p_data["assets"] = []
        
        # 2. Package for the risk report
        processed_portfolio = {
            "name": p_name,
            "total_value_inr": p_data.get("total_value_inr", 0),
            "monthly_expenses_inr": p_data.get("monthly_expenses_inr", 0),
            "assets": p_data.get("assets", [])
        }
        
        display_risk_report(processed_portfolio)