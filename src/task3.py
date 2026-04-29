import os
import json
from typing import Dict, Any
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv
import time

load_dotenv()

client = genai.Client()

def build_advisor_prompt(portfolio: Dict[str, Any], tone: str) -> str:
    """
    Constructs the prompt using XML-like tags to clearly separate instructions from data.
    Adjusts the instruction complexity based on the desired tone.
    """
    
    tone_instructions = {
        "beginner": "Avoid all financial jargon. Use simple analogies (like buckets or safety nets). Keep it very encouraging.",
        "experienced": "Use standard financial terms (diversification, volatility, liquidity) but explain their specific impact on this portfolio.",
        "expert": "Speak like a peer. Discuss specific risk factors, correlation assumptions, and drawdown metrics directly."
    }
    
    selected_tone = tone_instructions.get(tone.lower(), tone_instructions["beginner"])

    prompt = f"""You are a friendly, honest, and highly capable wealth manager. 
Your task is to analyze the provided portfolio and explain its risk profile to your client.

<CLIENT_PROFILE>
Tone/Experience Level: {tone.upper()}
Instruction: {selected_tone}
</CLIENT_PROFILE>

<PORTFOLIO_DATA>
{json.dumps(portfolio, indent=2)}
</PORTFOLIO_DATA>

<OUTPUT_REQUIREMENTS>
You must respond with ONLY a valid JSON object matching this exact structure:
{{
  "summary": "A 3-4 sentence plain-English summary of the portfolio's overall risk level.",
  "doing_well": "One specific thing the investor is doing well based on their asset allocation.",
  "needs_change": "One specific thing the investor should consider changing, and why.",
  "verdict": "Must be exactly one of: 'Aggressive', 'Balanced', or 'Conservative'"
}}
</OUTPUT_REQUIREMENTS>
"""
    return prompt

def generate_portfolio_explanation(portfolio: Dict[str, Any], tone: str = "beginner") -> Dict[str, Any]:
    """Calls the LLM to generate the initial explanation and parses the JSON."""
    
    prompt = build_advisor_prompt(portfolio, tone)
    
    print(f"\n[{tone.upper()} TONE] Sending request to Gemini API...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                ),
            )
            raw_text = response.text
            break # Exit the loop if successful

        except errors.ServerError as e:
            if "503" in str(e) and attempt < max_retries - 1:
                print(f"\n[WARNING] Google API is busy (Attempt {attempt + 1}/{max_retries}). Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"\n[ERROR] API completely failed after {max_retries} attempts: {e}")
                return {}
        except Exception as e:
             print(f"\n[ERROR] An unexpected error occurred: {e}")
             return {}
    
    # Print the RAW output as required by the assignment
    print("\n" + "="*50)
    print("RAW API RESPONSE:")
    print("="*50)
    print(raw_text)
    
    # Parse the structured output
    try:
        parsed_output = json.loads(raw_text)
        return parsed_output
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] Failed to parse LLM output as JSON: {e}")
        return {}

def critique_explanation(original_explanation: Dict[str, Any], portfolio: Dict[str, Any]) -> str:
    """Bonus: A second LLM call to act as a senior partner critiquing the advisor's advice."""
    
    prompt = f"""You are a Senior Risk Officer at an elite wealth management firm.
Review the following portfolio and the advice given by a junior advisor. 

<PORTFOLIO>
{json.dumps(portfolio, indent=2)}
</PORTFOLIO>

<JUNIOR_ADVISOR_OUTPUT>
{json.dumps(original_explanation, indent=2)}
</JUNIOR_ADVISOR_OUTPUT>

Provide a brief, 2-sentence critique. Is the 'verdict' accurate? Is the advice to change sound? 
Respond in plain text, speaking directly to the junior advisor.
"""

    print("\n[CRITIQUE] Requesting Senior Officer critique...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3) # Lower temp for a stricter, more analytical critique
    )
    
    return response.text

def main():
    # Sample portfolio from Task 1
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
    
    # 1. Generate the initial explanation (Requirement 1-4)
    # We pass 'beginner' to test the configurable tone bonus
    explanation = generate_portfolio_explanation(sample_portfolio, tone="beginner")
    
    if not explanation:
        return

    # Print the EXTRACTED structured output as required
    print("\n" + "="*50)
    print("EXTRACTED STRUCTURED OUTPUT:")
    print("="*50)
    for key, value in explanation.items():
        print(f"[{key.upper()}]\n{value}\n")
        
    # 2. Run the Critique (Bonus)
    critique = critique_explanation(explanation, sample_portfolio)
    print("="*50)
    print("SENIOR OFFICER CRITIQUE (Bonus):")
    print("="*50)
    print(critique)
    print("\n")

if __name__ == "__main__":
    # Safety check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    else:
        main()