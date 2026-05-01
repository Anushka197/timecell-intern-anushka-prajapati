import json
from typing import Dict, Any

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