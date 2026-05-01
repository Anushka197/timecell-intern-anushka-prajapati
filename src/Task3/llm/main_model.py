import logging
import time

logger = logging.getLogger(__name__)

from typing import Dict, Any
from google import genai
from google.genai import types
from google.genai import errors
from core.prompt_builder import build_advisor_prompt
from core.parser import parse_llm_json

# Client initializes automatically using the GEMINI_API_KEY env variable
gemini_client = genai.Client()

def generate_portfolio_explanation(portfolio: Dict[str, Any], tone: str = "beginner") -> Dict[str, Any]:
    """Calls the Gemini API to generate the initial explanation and parses the JSON."""
    
    prompt = build_advisor_prompt(portfolio, tone)
    
    logger.info(f"\n[{tone.upper()} TONE] Sending request to Gemini API (Junior Advisor)...")
    
    max_retries = 3
    raw_text = ""
    
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                ),
            )
            raw_text = response.text
            break

        except errors.ServerError as e:
            if "503" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Google API 503 (Busy). Attempt {attempt + 1} retrying...")
                time.sleep(5)
            else:
                logger.error(f"API failed after {max_retries} attempts: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during API call: {e}")
            return {}
    
    # Do not print raw text output to CLI anymore, parsing handles it silently unless there's an error
    if not raw_text:
        return {}
        
    return parse_llm_json(raw_text)