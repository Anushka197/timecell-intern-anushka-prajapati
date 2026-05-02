import os
import json
from typing import Dict, Any
from openai import OpenAI

def get_openai_client():
    """Initializes the client using the environment variable."""
    return OpenAI(
      base_url=os.environ.get("OPENROUTER_BASE_URL"),
      api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

def critique_explanation(original_explanation: Dict[str, Any], portfolio: Dict[str, Any]) -> str:
    """Uses OpenAI (GPT-4o-mini) to act as a senior partner critiquing the advisor's advice."""
    
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

    print("\n[CRITIQUE] Requesting Senior Officer critique from OpenAI (GPT-4o-mini)...")
    
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3 # Lower temp for a stricter, more analytical critique
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] Failed to get critique from OpenAI: {e}"