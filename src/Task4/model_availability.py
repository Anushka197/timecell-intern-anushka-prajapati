import requests
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/models"

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

response = requests.get(url, headers=headers)
data = response.json()

print("\n=== FREE MODELS ===\n")

free_models = []

for model in data["data"]:
    pricing = model.get("pricing", {})
    
    # Check if both prompt + completion are free
    if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
        free_models.append(model["id"])

# Sort for readability
free_models = sorted(free_models)

for m in free_models:
    print(m)

print(f"\nTotal free models: {len(free_models)}")