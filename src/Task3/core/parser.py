import json
import re
from typing import Dict, Any

def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """Safely extracts and parses JSON from an LLM response."""
    
    # Strip markdown code blocks if the LLM added them
    clean_text = re.sub(r"```(?:json)?", "", raw_text).strip()
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] Failed to parse LLM output as JSON: {e}")
        print(f"Raw Output was:\n{raw_text}")
        return {}