import logging
import os
import json

# Create a logger for this specific module
logger = logging.getLogger(__name__)

def check_api_keys() -> bool:
    missing_keys = [k for k in ["GEMINI_API_KEY", "OPENROUTER_API_KEY"] if not os.environ.get(k)]
    if missing_keys:
        logger.error(f"Missing environment variables: {', '.join(missing_keys)}")
        return False
    return True

def load_and_validate_portfolio(filepath: str):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            logger.info(f"Successfully loaded portfolio from {filepath}")
            return data
    except Exception as e:
        logger.exception(f"Failed to load portfolio at {filepath}") # .exception adds the traceback!
        return {}