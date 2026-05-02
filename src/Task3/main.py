# main.py
import logging
import argparse
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"), # Saves logs to a file
        logging.StreamHandler()         # Also prints to the terminal
    ]
)

from core.validator import check_api_keys, load_and_validate_portfolio
from llm.main_model import generate_portfolio_explanation
from llm.critic_model import critique_explanation

logger = logging.getLogger(__name__)

# Load .env file variables before importing models that require keys
load_dotenv()

def main():
    # 1. Check API Keys
    if not check_api_keys():
        return

    # 2. Set up Command Line Arguments for User Input
    parser = argparse.ArgumentParser(description="AI-Powered Portfolio Explainer")
    parser.add_argument("--file", type=str, default="data/sample_portfolio.json", help="Path to the portfolio JSON file")
    parser.add_argument("--tone", type=str, choices=["beginner", "experienced", "expert"], default="beginner", help="The tone of the advisor")
    
    args = parser.parse_args()

    # 3. Load the User's Data
    print(f"Loading portfolio from {args.file}...")
    portfolio = load_and_validate_portfolio(args.file)
    if not portfolio:
        return

    # 4. Run the Main LLM
    explanation = generate_portfolio_explanation(portfolio, tone=args.tone)
    if not explanation:
        print("\n[ERROR] Failed to generate portfolio explanation.")
        return

    print("\n" + "="*50)
    print(f"EXTRACTED STRUCTURED OUTPUT ({args.tone.upper()} TONE):")
    print("="*50)
    for key, value in explanation.items():
        print(f"[{key.upper()}]\n{value}\n")
        
    # 5. Run the Critique LLM
    critique = critique_explanation(explanation, portfolio)
    print("="*50)
    print("SENIOR OFFICER CRITIQUE (OpenAI):")
    print("="*50)
    print(critique)
    print("\n")

if __name__ == "__main__":
    main()